from __future__ import annotations
import json
import logging
import re
import sqlite3
import threading
import time
import warnings
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

try:
    from rdkit import Chem
    from rdkit.Chem import (
        Crippen, Descriptors, GraphDescriptors, Lipinski, QED, rdMolDescriptors,
        AllChem, Descriptors3D, rdPartialCharges, Fragments
    )
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from flask import Blueprint, render_template, request, jsonify, send_file, current_app
import os
import traceback
import uuid
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")
tqdm.pandas(desc="Processing")

dataset_bp = Blueprint('dataset', __name__, url_prefix='/dataset')

def load_chemicals_from_xml(xml_path: str = None) -> Dict[str, Dict]:
    chemicals = {}
    if xml_path is None:
        script_dir = Path(__file__).parent
        xml_path = script_dir / "common_chemicals.xml"
    if not Path(xml_path).exists():
        return chemicals
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for chemical in root.findall('chemical'):
            name_elem = chemical.find('name')
            smiles_elem = chemical.find('smiles')
            if name_elem is None or smiles_elem is None:
                continue
            name_text = name_elem.text.strip().lower()
            smiles_text = smiles_elem.text.strip()
            chemical_data = {
                'smiles': smiles_text,
                'category': chemical.get('category', 'unknown')
            }
            aliases_elem = chemical.find('aliases')
            if aliases_elem is not None:
                alias_list = [a.text.strip().lower() for a in aliases_elem.findall('alias') if a.text]
                chemical_data['aliases'] = alias_list
                for alias in alias_list:
                    if alias != name_text:
                        chemicals[alias] = {
                            'smiles': smiles_text,
                            'alias_of': name_text,
                            'category': chemical.get('category', 'unknown')
                        }
            chemicals[name_text] = chemical_data
        return chemicals
    except Exception:
        return {}

def load_solvent_properties_from_xml(xml_path: str = None) -> Dict[str, Dict[str, float]]:
    properties = {}
    if xml_path is None:
        script_dir = Path(__file__).parent
        xml_path = script_dir / "solvent_properties.xml"
    if not Path(xml_path).exists():
        return properties
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for solvent in root.findall('solvent'):
            name = solvent.get('name')
            if not name:
                continue
            props = {}
            for child in solvent:
                try:
                    value = float(child.text.strip())
                    props[child.tag] = value
                except (ValueError, AttributeError):
                    continue
            if props:
                properties[name.lower()] = props
        return properties
    except Exception:
        return {}

@dataclass
class Config:
    input_csv: str = "suzuki_dataset.csv"
    output_csv: str = "suzuki_dataset_COMPLETE.csv"
    db_path: str = "universal_chemical.db"
    log_path: str = "suzuki_build.log"
    checkpoint_csv: str = "checkpoint.csv"
    reaction_columns: List[str] = field(
        default_factory=lambda: ["subs1", "subs2", "product", "catalizor", "base", "solv1", "solv2"]
    )
    smiles_workers: int = 8
    property_workers: int = 8
    retry_after_days: int = 7
    checkpoint_every_column: bool = True
    ml_test_fraction: float = 0.2
    ml_min_rows_for_model: int = 20
    compute_3d_descriptors: bool = True
    target_column: Optional[str] = None

CONFIG = Config()

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("suzuki_builder")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger(CONFIG.log_path)
if not RDKIT_AVAILABLE:
    logger.warning("RDKit not available. Install: pip install rdkit")

PROPERTY_SCHEMA: List[str] = [
    "formula", "mw", "inchikey", "pubchem_cid",
    "logp", "tpsa", "refractivity",
    "heavy_atoms", "total_atoms", "hba", "hbd", "rot_bonds",
    "rings", "aromatic_rings", "aliphatic_rings", "hetero_rings",
    "saturated_rings", "carbocyclic_rings", "heterocyclic_rings",
    "spiro_atoms", "bridgehead_atoms",
    "branch_nodes", "kappa1", "kappa2", "kappa3",
    "fr_methoxy", "fr_nitro", "fr_Ar_halide", "fr_alkyl_halide",
    "c_count", "n_count", "o_count", "s_count", "p_count",
    "f_count", "cl_count", "br_count", "i_count", "b_count", "si_count",
    "metal_count", "halogen_count", "hetero_count",
    "aromatic_bonds", "single_bonds", "double_bonds", "triple_bonds", "total_bonds",
    "chiral_centers_defined", "chiral_centers_undefined",
    "qed", "complexity_bertz", "fraction_csp3", "num_amide_bonds", "fragment_count",
    "asphericity"
]

_DEFAULT_INT_FIELDS = {
    "heavy_atoms", "total_atoms", "hba", "hbd", "rot_bonds", "rings",
    "aromatic_rings", "aliphatic_rings", "hetero_rings", "saturated_rings",
    "carbocyclic_rings", "heterocyclic_rings", "spiro_atoms", "bridgehead_atoms",
    "c_count", "n_count", "o_count", "s_count", "p_count", "f_count", "cl_count",
    "br_count", "i_count", "b_count", "si_count", "metal_count", "halogen_count",
    "hetero_count", "aromatic_bonds", "single_bonds", "double_bonds", "triple_bonds",
    "total_bonds", "chiral_centers_defined", "chiral_centers_undefined",
    "num_amide_bonds", "fragment_count", "branch_nodes",
    "fr_methoxy", "fr_nitro", "fr_Ar_halide", "fr_alkyl_halide"
}

def default_properties() -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for p in PROPERTY_SCHEMA:
        if p == "formula":
            d[p] = "Unknown"
        elif p in ("inchikey", "pubchem_cid"):
            d[p] = None
        elif p in _DEFAULT_INT_FIELDS:
            d[p] = 0
        else:
            d[p] = 0.0
    return d

_COMMON_CHEMICALS_XML = load_chemicals_from_xml()

if not _COMMON_CHEMICALS_XML:
    def _build_common_chemicals_hardcoded() -> Dict[str, Dict]:
        common: Dict[str, Dict] = {}
        def add(d):
            common.update(d)
        add({
            'water': {'smiles': 'O', 'category': 'solvent'},
            'ethanol': {'smiles': 'CCO', 'category': 'solvent'},
            'methanol': {'smiles': 'CO', 'category': 'solvent'},
            'isopropanol': {'smiles': 'CC(C)O', 'category': 'solvent'},
            '2-propanol': {'smiles': 'CC(C)O', 'category': 'solvent'},
            'toluene': {'smiles': 'Cc1ccccc1', 'category': 'solvent'},
            'dmf': {'smiles': 'CN(C)C=O', 'category': 'solvent'},
            'n,n-dimethylformamide': {'smiles': 'CN(C)C=O', 'category': 'solvent'},
            'dmso': {'smiles': 'CS(C)=O', 'category': 'solvent'},
            'thf': {'smiles': 'C1CCOC1', 'category': 'solvent'},
            'tetrahydrofuran': {'smiles': 'C1CCOC1', 'category': 'solvent'},
            'dioxane': {'smiles': 'C1COCCO1', 'category': 'solvent'},
            '1,4-dioxane': {'smiles': 'C1COCCO1', 'category': 'solvent'},
            'acetone': {'smiles': 'CC(=O)C', 'category': 'solvent'},
            'acetonitrile': {'smiles': 'CC#N', 'category': 'solvent'},
            'dichloromethane': {'smiles': 'ClCCl', 'category': 'solvent'},
            'dcm': {'smiles': 'ClCCl', 'category': 'solvent'},
            'chloroform': {'smiles': 'ClC(Cl)Cl', 'category': 'solvent'},
            'hexane': {'smiles': 'CCCCCC', 'category': 'solvent'},
            'cyclohexane': {'smiles': 'C1CCCCC1', 'category': 'solvent'},
            'ethyl acetate': {'smiles': 'CCOC(=O)C', 'category': 'solvent'},
            'diethyl ether': {'smiles': 'CCOCC', 'category': 'solvent'},
            'benzene': {'smiles': 'c1ccccc1', 'category': 'solvent'},
            'pyridine': {'smiles': 'c1ccncc1', 'category': 'solvent'},
            'nmp': {'smiles': 'CN1CCCC1=O', 'category': 'solvent'},
            'dme': {'smiles': 'COCCOC', 'category': 'solvent'},
            '1,2-dimethoxyethane': {'smiles': 'COCCOC', 'category': 'solvent'},
        })
        add({
            'deuterium oxide': {'smiles': '[2H]O[2H]', 'category': 'solvent'},
            'd2o': {'smiles': '[2H]O[2H]', 'category': 'solvent'},
            'deuterated chloroform': {'smiles': '[2H]C(Cl)(Cl)Cl', 'category': 'solvent'},
            'cdcl3': {'smiles': '[2H]C(Cl)(Cl)Cl', 'category': 'solvent'},
            'deuterated dmso': {'smiles': '[2H]C([2H])([2H])S(=O)C([2H])([2H])[2H]', 'category': 'solvent'},
            'dmso-d6': {'smiles': '[2H]C([2H])([2H])S(=O)C([2H])([2H])[2H]', 'category': 'solvent'},
            'deuterated methanol': {'smiles': '[2H]OC([2H])([2H])[2H]', 'category': 'solvent'},
            'cd3od': {'smiles': '[2H]OC([2H])([2H])[2H]', 'category': 'solvent'},
            'deuterated acetonitrile': {'smiles': '[2H]C([2H])([2H])C#N', 'category': 'solvent'},
            'cd3cn': {'smiles': '[2H]C([2H])([2H])C#N', 'category': 'solvent'},
            'deuterated acetone': {'smiles': '[2H]C([2H])([2H])C(=O)C([2H])([2H])[2H]', 'category': 'solvent'},
            'acetone-d6': {'smiles': '[2H]C([2H])([2H])C(=O)C([2H])([2H])[2H]', 'category': 'solvent'},
            'deuterated benzene': {'smiles': '[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]', 'category': 'solvent'},
            'c6d6': {'smiles': '[2H]c1c([2H])c([2H])c([2H])c([2H])c1[2H]', 'category': 'solvent'},
            'deuterated pyridine': {'smiles': '[2H]c1c([2H])c([2H])n([2H])c([2H])c1[2H]', 'category': 'solvent'},
            'pyridine-d5': {'smiles': '[2H]c1c([2H])c([2H])n([2H])c([2H])c1[2H]', 'category': 'solvent'},
            'deuterated thf': {'smiles': '[2H]C1C([2H])([2H])C([2H])([2H])OC1([2H])[2H]', 'category': 'solvent'},
            'thf-d8': {'smiles': '[2H]C1C([2H])([2H])C([2H])([2H])OC1([2H])[2H]', 'category': 'solvent'},
            'deuterated dichloromethane': {'smiles': '[2H]C(Cl)(Cl)[2H]', 'category': 'solvent'},
            'cd2cl2': {'smiles': '[2H]C(Cl)(Cl)[2H]', 'category': 'solvent'},
            'deuterated toluene': {'smiles': '[2H]C([2H])([2H])c1c([2H])c([2H])c([2H])c([2H])c1[2H]', 'category': 'solvent'},
            'toluene-d8': {'smiles': '[2H]C([2H])([2H])c1c([2H])c([2H])c([2H])c([2H])c1[2H]', 'category': 'solvent'},
        })
        add({
            'potassium carbonate': {'smiles': 'O=C([O-])[O-].[K+].[K+]', 'category': 'base'},
            'k2co3': {'smiles': 'O=C([O-])[O-].[K+].[K+]', 'category': 'base'},
            'cesium carbonate': {'smiles': 'O=C([O-])[O-].[Cs+].[Cs+]', 'category': 'base'},
            'cs2co3': {'smiles': 'O=C([O-])[O-].[Cs+].[Cs+]', 'category': 'base'},
            'sodium carbonate': {'smiles': 'O=C([O-])[O-].[Na+].[Na+]', 'category': 'base'},
            'sodium bicarbonate': {'smiles': 'OC([O-])=O.[Na+]', 'category': 'base'},
            'sodium hydroxide': {'smiles': '[Na+].[OH-]', 'category': 'base'},
            'naoh': {'smiles': '[Na+].[OH-]', 'category': 'base'},
            'potassium hydroxide': {'smiles': '[K+].[OH-]', 'category': 'base'},
            'koh': {'smiles': '[K+].[OH-]', 'category': 'base'},
            'tripotassium phosphate': {'smiles': '[K+].[K+].[K+].[O-]P([O-])([O-])=O', 'category': 'base'},
            'k3po4': {'smiles': '[K+].[K+].[K+].[O-]P([O-])([O-])=O', 'category': 'base'},
            'potassium phosphate': {'smiles': '[K+].[K+].[K+].[O-]P([O-])([O-])=O', 'category': 'base'},
            'sodium tert-butoxide': {'smiles': 'CC(C)(C)[O-].[Na+]', 'category': 'base'},
            'potassium tert-butoxide': {'smiles': 'CC(C)(C)[O-].[K+]', 'category': 'base'},
            'triethylamine': {'smiles': 'CCN(CC)CC', 'category': 'base'},
            'diisopropylethylamine': {'smiles': 'CC(C)N(CC)C(C)C', 'category': 'base'},
            'dipea': {'smiles': 'CC(C)N(CC)C(C)C', 'category': 'base'},
            'cesium fluoride': {'smiles': '[F-].[Cs+]', 'category': 'base'},
            'potassium fluoride': {'smiles': '[F-].[K+]', 'category': 'base'},
            'potassium acetate': {'smiles': 'CC(=O)[O-].[K+]', 'category': 'base'},
            'barium hydroxide': {'smiles': '[OH-].[OH-].[Ba+2]', 'category': 'base'},
        })
        add({
            'tetrakis(triphenylphosphine)palladium(0)': {
                'smiles': 'c1ccc(cc1)P(c1ccccc1)c1ccccc1.c1ccc(cc1)P(c1ccccc1)c1ccccc1.c1ccc(cc1)P(c1ccccc1)c1ccccc1.c1ccc(cc1)P(c1ccccc1)c1ccccc1.[Pd]',
                'category': 'catalyst'
            },
            'pd(pph3)4': {
                'smiles': 'c1ccc(cc1)P(c1ccccc1)c1ccccc1.c1ccc(cc1)P(c1ccccc1)c1ccccc1.c1ccc(cc1)P(c1ccccc1)c1ccccc1.c1ccc(cc1)P(c1ccccc1)c1ccccc1.[Pd]',
                'category': 'catalyst'
            },
            'palladium(ii) acetate': {'smiles': 'CC(=O)O[Pd]OC(C)=O', 'category': 'catalyst'},
            'pd(oac)2': {'smiles': 'CC(=O)O[Pd]OC(C)=O', 'category': 'catalyst'},
            'bis(triphenylphosphine)palladium(ii) dichloride': {
                'smiles': 'Cl[Pd](Cl)(P(c1ccccc1)(c1ccccc1)c1ccccc1)P(c1ccccc1)(c1ccccc1)c1ccccc1',
                'category': 'catalyst'
            },
            'pdcl2(pph3)2': {
                'smiles': 'Cl[Pd](Cl)(P(c1ccccc1)(c1ccccc1)c1ccccc1)P(c1ccccc1)(c1ccccc1)c1ccccc1',
                'category': 'catalyst'
            },
            'pd(dppf)cl2': {'smiles': 'Cl[Pd]Cl.c1ccc(cc1)P(c1ccccc1)[c]1cccc1', 'category': 'catalyst'},
            'palladium(ii) chloride': {'smiles': 'Cl[Pd]Cl', 'category': 'catalyst'},
            'pdcl2': {'smiles': 'Cl[Pd]Cl', 'category': 'catalyst'},
            'palladium on carbon': {'smiles': '[Pd]', 'category': 'catalyst'},
            'pd/c': {'smiles': '[Pd]', 'category': 'catalyst'},
        })
        add({
            'triphenylphosphine': {'smiles': 'c1ccc(cc1)P(c1ccccc1)c1ccccc1', 'category': 'ligand'},
            'pph3': {'smiles': 'c1ccc(cc1)P(c1ccccc1)c1ccccc1', 'category': 'ligand'},
            'sphos': {'smiles': 'COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1', 'category': 'ligand'},
            'xphos': {'smiles': 'CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1', 'category': 'ligand'},
            'dppf': {'smiles': 'c1ccc(cc1)P(c1ccccc1)[c]1cccc1', 'category': 'ligand'},
        })
        add({
            'bromobenzene': {'smiles': 'Brc1ccccc1', 'category': 'substrate'},
            'chlorobenzene': {'smiles': 'Clc1ccccc1', 'category': 'substrate'},
            'iodobenzene': {'smiles': 'Ic1ccccc1', 'category': 'substrate'},
            '4-bromotoluene': {'smiles': 'Cc1ccc(Br)cc1', 'category': 'substrate'},
            '4-bromoanisole': {'smiles': 'COc1ccc(Br)cc1', 'category': 'substrate'},
            '2-bromopyridine': {'smiles': 'Brc1ccccn1', 'category': 'substrate'},
        })
        add({
            'phenylboronic acid': {'smiles': 'OB(O)c1ccccc1', 'category': 'boronic_acid'},
            'phenylboronic acid pinacol ester': {'smiles': 'CC1(C)OB(OC1(C)C)c1ccccc1', 'category': 'boronic_acid'},
            '4-methylphenylboronic acid': {'smiles': 'Cc1ccc(cc1)B(O)O', 'category': 'boronic_acid'},
            '4-methoxyphenylboronic acid': {'smiles': 'COc1ccc(cc1)B(O)O', 'category': 'boronic_acid'},
        })
        return common
    _COMMON_CHEMICALS = _build_common_chemicals_hardcoded()
else:
    _COMMON_CHEMICALS = _COMMON_CHEMICALS_XML

_SOLVENT_PROPERTIES_XML = load_solvent_properties_from_xml()

if not _SOLVENT_PROPERTIES_XML:
    SOLVENT_PHYSICS: Dict[str, Dict[str, float]] = {
        "water": {"dielectric": 80.1, "bp_c": 100.0, "polarity_index": 10.2, "donor_number": 18.0},
        "methanol": {"dielectric": 32.7, "bp_c": 64.7, "polarity_index": 5.1, "donor_number": 19.0},
        "ethanol": {"dielectric": 24.6, "bp_c": 78.4, "polarity_index": 4.3, "donor_number": 19.2},
        "isopropanol": {"dielectric": 19.9, "bp_c": 82.3, "polarity_index": 3.9, "donor_number": 18.5},
        "acetone": {"dielectric": 20.7, "bp_c": 56.1, "polarity_index": 5.1, "donor_number": 17.0},
        "acetonitrile": {"dielectric": 37.5, "bp_c": 82.0, "polarity_index": 5.8, "donor_number": 14.1},
        "dmso": {"dielectric": 46.7, "bp_c": 189.0, "polarity_index": 7.2, "donor_number": 29.8},
        "dmf": {"dielectric": 36.7, "bp_c": 153.0, "polarity_index": 6.4, "donor_number": 26.6},
        "thf": {"dielectric": 7.5, "bp_c": 66.0, "polarity_index": 4.0, "donor_number": 20.0},
        "dioxane": {"dielectric": 2.2, "bp_c": 101.0, "polarity_index": 4.8, "donor_number": 14.8},
        "toluene": {"dielectric": 2.4, "bp_c": 110.6, "polarity_index": 2.4, "donor_number": 0.1},
        "benzene": {"dielectric": 2.3, "bp_c": 80.1, "polarity_index": 2.7, "donor_number": 0.1},
        "dichloromethane": {"dielectric": 8.9, "bp_c": 39.6, "polarity_index": 3.1, "donor_number": 0.0},
        "chloroform": {"dielectric": 4.8, "bp_c": 61.2, "polarity_index": 4.1, "donor_number": 0.0},
        "hexane": {"dielectric": 1.9, "bp_c": 68.7, "polarity_index": 0.1, "donor_number": 0.0},
        "cyclohexane": {"dielectric": 2.0, "bp_c": 80.7, "polarity_index": 0.2, "donor_number": 0.0},
        "ethyl acetate": {"dielectric": 6.0, "bp_c": 77.1, "polarity_index": 4.4, "donor_number": 14.0},
        "diethyl ether": {"dielectric": 4.3, "bp_c": 34.6, "polarity_index": 2.8, "donor_number": 19.2},
        "pyridine": {"dielectric": 12.3, "bp_c": 115.2, "polarity_index": 5.3, "donor_number": 33.1},
        "nmp": {"dielectric": 32.2, "bp_c": 202.0, "polarity_index": 6.7, "donor_number": 27.3},
        "dme": {"dielectric": 7.2, "bp_c": 85.0, "polarity_index": 3.5, "donor_number": 19.5},
        "deuterium oxide": {"dielectric": 78.3, "bp_c": 101.4, "polarity_index": 10.2, "donor_number": 18.0},
        "deuterated chloroform": {"dielectric": 4.8, "bp_c": 60.9, "polarity_index": 4.1, "donor_number": 0.0},
        "deuterated dmso": {"dielectric": 46.5, "bp_c": 189.0, "polarity_index": 7.2, "donor_number": 29.8},
        "deuterated methanol": {"dielectric": 32.7, "bp_c": 65.4, "polarity_index": 5.1, "donor_number": 19.0},
        "deuterated acetonitrile": {"dielectric": 37.5, "bp_c": 81.6, "polarity_index": 5.8, "donor_number": 14.1},
        "deuterated acetone": {"dielectric": 20.7, "bp_c": 55.5, "polarity_index": 5.1, "donor_number": 17.0},
        "deuterated benzene": {"dielectric": 2.3, "bp_c": 79.1, "polarity_index": 2.7, "donor_number": 0.1},
        "deuterated pyridine": {"dielectric": 12.3, "bp_c": 115.0, "polarity_index": 5.3, "donor_number": 33.1},
        "deuterated thf": {"dielectric": 7.5, "bp_c": 66.0, "polarity_index": 4.0, "donor_number": 20.0},
        "deuterated dichloromethane": {"dielectric": 8.9, "bp_c": 40.0, "polarity_index": 3.1, "donor_number": 0.0},
        "deuterated toluene": {"dielectric": 2.4, "bp_c": 110.0, "polarity_index": 2.4, "donor_number": 0.1},
    }
else:
    SOLVENT_PHYSICS = _SOLVENT_PROPERTIES_XML

class RateLimiter:
    def __init__(self, rate: float = 4.0):
        self.min_interval = 1.0 / rate
        self.lock = threading.Lock()
        self.last = 0.0
    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last = time.time()

class ChemicalDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.name_memory: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self._init_db()
        self._load_name_memory()
        self.common: Dict[str, Dict] = _COMMON_CHEMICALS
        self._persist_common()
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn
    def _init_db(self):
        conn = self._connect()
        try:
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS name_cache (name TEXT PRIMARY KEY, smiles TEXT, source TEXT, confidence INTEGER DEFAULT 1, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            c.execute("CREATE TABLE IF NOT EXISTS properties_cache (canonical_smiles TEXT PRIMARY KEY, props_json TEXT, last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            c.execute("CREATE TABLE IF NOT EXISTS failed_lookups (name TEXT PRIMARY KEY, attempts INTEGER DEFAULT 1, reason TEXT, last_attempt TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            c.execute("CREATE TABLE IF NOT EXISTS resolution_lineage (id INTEGER PRIMARY KEY AUTOINCREMENT, query_name TEXT, normalized_key TEXT, winning_source TEXT, canonical_smiles TEXT, inchikey TEXT, fragment_count INTEGER, attempted_sources TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_name ON name_cache(name)")
            conn.commit()
        finally:
            conn.close()
    def _load_name_memory(self):
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("SELECT name, smiles, source FROM name_cache")
            for name, smiles, source in c.fetchall():
                self.name_memory[name] = {"smiles": smiles, "source": source}
            conn.close()
        except Exception:
            pass
    def _persist_common(self):
        conn = self._connect()
        try:
            c = conn.cursor()
            for name, data in self.common.items():
                if 'smiles' in data:
                    c.execute("INSERT OR IGNORE INTO name_cache (name, smiles, source, confidence) VALUES (?, ?, 'builtin', 5)", (name, data["smiles"]))
            conn.commit()
        finally:
            conn.close()
    @staticmethod
    def normalize(name: str) -> str:
        if not name:
            return ""
        return re.sub(r"\s+", " ", name.lower().strip())
    def get_name(self, name: str) -> Optional[str]:
        if not name:
            return None
        norm = self.normalize(name)
        raw = name.lower().strip()
        for key in (norm, raw):
            if key in self.common:
                return self.common[key].get("smiles")
            if key in self.name_memory:
                return self.name_memory[key]["smiles"]
        return None
    def save_name(self, name: str, smiles: str, source: str):
        key = self.normalize(name)
        with self.lock:
            self.name_memory[key] = {"smiles": smiles, "source": source}
        threading.Thread(target=self._save_name_async, args=(key, smiles, source), daemon=True).start()
    def _save_name_async(self, name: str, smiles: str, source: str):
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("INSERT INTO name_cache (name, smiles, source, confidence) VALUES (?, ?, ?, 1) ON CONFLICT(name) DO UPDATE SET smiles=excluded.smiles, source=excluded.source, confidence=name_cache.confidence + 1, last_updated=CURRENT_TIMESTAMP", (name, smiles, source))
            conn.commit()
            conn.close()
        except Exception:
            pass
    def get_properties(self, canonical_smiles: str) -> Optional[Dict]:
        if not canonical_smiles:
            return None
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("SELECT props_json FROM properties_cache WHERE canonical_smiles = ?", (canonical_smiles,))
            row = c.fetchone()
            conn.close()
            if row and row[0]:
                return json.loads(row[0])
        except Exception:
            pass
        return None
    def save_properties(self, canonical_smiles: str, props: Dict):
        if not canonical_smiles:
            return
        threading.Thread(target=self._save_properties_async, args=(canonical_smiles, props), daemon=True).start()
    def _save_properties_async(self, canonical_smiles: str, props: Dict):
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("INSERT INTO properties_cache (canonical_smiles, props_json, last_updated) VALUES (?, ?, CURRENT_TIMESTAMP) ON CONFLICT(canonical_smiles) DO UPDATE SET props_json=excluded.props_json, last_updated=CURRENT_TIMESTAMP", (canonical_smiles, json.dumps(props)))
            conn.commit()
            conn.close()
        except Exception:
            pass
    def is_recently_failed(self, name: str, retry_after_days: int = 7) -> bool:
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("SELECT last_attempt FROM failed_lookups WHERE name = ?", (self.normalize(name),))
            row = c.fetchone()
            conn.close()
            if not row:
                return False
            return datetime.now() - datetime.fromisoformat(row[0]) < timedelta(days=retry_after_days)
        except Exception:
            return False
    def mark_failed(self, name: str, reason: str):
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("INSERT INTO failed_lookups (name, attempts, reason, last_attempt) VALUES (?, 1, ?, CURRENT_TIMESTAMP) ON CONFLICT(name) DO UPDATE SET attempts = failed_lookups.attempts + 1, reason = excluded.reason, last_attempt = CURRENT_TIMESTAMP", (self.normalize(name), reason))
            conn.commit()
            conn.close()
        except Exception:
            pass
    def log_lineage(self, record):
        try:
            conn = self._connect()
            c = conn.cursor()
            c.execute("INSERT INTO resolution_lineage (query_name, normalized_key, winning_source, canonical_smiles, inchikey, fragment_count, attempted_sources) VALUES (?, ?, ?, ?, ?, ?, ?)", (record.query_name, self.normalize(record.query_name), record.winning_source, record.canonical_smiles, record.inchikey, record.fragment_count, json.dumps(record.attempted_sources)))
            conn.commit()
            conn.close()
        except Exception:
            pass

@dataclass
class ResolutionRecord:
    query_name: str
    winning_source: Optional[str] = None
    canonical_smiles: Optional[str] = None
    inchikey: Optional[str] = None
    fragment_count: int = 0
    fragments: List[Dict[str, Any]] = field(default_factory=list)
    attempted_sources: List[str] = field(default_factory=list)
    status: str = "unresolved"

class SMILESResolver:
    SOURCE_ORDER = [
        ("pubchem_name", "PubChem"),
        ("cas_common_chemistry", "CAS"),
        ("chembl", "ChEMBL"),
        ("opsin", "OPSIN"),
        ("pubchem_synonym", "PubChem Synonym"),
        ("cactus", "CACTUS"),
    ]
    def __init__(self, db: ChemicalDatabase, cfg: Config = CONFIG):
        self.db = db
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "suzuki-dataset-builder/4.2"})
        retry = Retry(total=3, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.limiters = {
            "pubchem_name": RateLimiter(4.0),
            "pubchem_synonym": RateLimiter(4.0),
            "cas_common_chemistry": RateLimiter(3.0),
            "chembl": RateLimiter(5.0),
            "opsin": RateLimiter(8.0),
            "cactus": RateLimiter(5.0),
        }
        self.stats = defaultdict(int)
        self.lock = threading.Lock()
        self.lineage: Dict[str, ResolutionRecord] = {}
    def _is_smiles(self, s: str) -> bool:
        if not s or not s.strip():
            return False
        s = s.strip()
        if " " in s or len(s) < 2:
            return False
        if not re.match(r"^[A-Za-z0-9@+\-\[\]()=#$/\\.%:]+$", s):
            return False
        if RDKIT_AVAILABLE:
            try:
                return Chem.MolFromSmiles(s) is not None
            except Exception:
                return False
        return True
    def _src_pubchem_name(self, name: str) -> Optional[str]:
        self.limiters["pubchem_name"].wait()
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/CanonicalSMILES/TXT"
        r = self.session.get(url, timeout=6)
        if r.ok and r.text.strip():
            return r.text.strip().splitlines()[0].strip()
        return None
    def _src_pubchem_synonym(self, name: str) -> Optional[str]:
        self.limiters["pubchem_synonym"].wait()
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/cids/TXT"
        r = self.session.get(url, timeout=6)
        if not (r.ok and r.text.strip()):
            return None
        cid = r.text.strip().splitlines()[0].strip()
        self.limiters["pubchem_synonym"].wait()
        url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
        r2 = self.session.get(url2, timeout=6)
        if r2.ok and r2.text.strip():
            return r2.text.strip().splitlines()[0].strip()
        return None
    def _src_cas_common_chemistry(self, name: str) -> Optional[str]:
        self.limiters["cas_common_chemistry"].wait()
        url = f"https://commonchemistry.cas.org/api/search?q={quote(name)}"
        r = self.session.get(url, timeout=6)
        if not r.ok:
            return None
        data = r.json()
        results = data.get("results", [])
        if not results:
            return None
        cas_rn = results[0].get("rn")
        if not cas_rn:
            return None
        self.limiters["cas_common_chemistry"].wait()
        url2 = f"https://commonchemistry.cas.org/api/detail?cas_rn={quote(cas_rn)}"
        r2 = self.session.get(url2, timeout=6)
        if r2.ok:
            detail = r2.json()
            return detail.get("canonicalSmile") or detail.get("smile")
        return None
    def _src_chembl(self, name: str) -> Optional[str]:
        self.limiters["chembl"].wait()
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/search?q={quote(name)}&format=json&limit=1"
        r = self.session.get(url, timeout=6)
        if r.ok:
            data = r.json()
            mols = data.get("molecules", [])
            if mols:
                struct = mols[0].get("molecule_structures", {}) or {}
                return struct.get("canonical_smiles") or struct.get("standard_smiles")
        return None
    def _src_opsin(self, name: str) -> Optional[str]:
        self.limiters["opsin"].wait()
        url = f"https://opsin.ch.cam.ac.uk/opsin/{quote(name)}.json"
        r = self.session.get(url, timeout=5)
        if r.ok:
            return r.json().get("smiles")
        return None
    def _src_cactus(self, name: str) -> Optional[str]:
        self.limiters["cactus"].wait()
        url = f"https://cactus.nci.nih.gov/chemical/structure/{quote(name)}/smiles"
        r = self.session.get(url, timeout=6)
        if r.ok and r.text.strip() and "not found" not in r.text.lower() and "<html" not in r.text.lower():
            return r.text.strip().splitlines()[0].strip()
        return None
    def _get_source_fn(self, key: str):
        return {
            "pubchem_name": self._src_pubchem_name,
            "pubchem_synonym": self._src_pubchem_synonym,
            "cas_common_chemistry": self._src_cas_common_chemistry,
            "chembl": self._src_chembl,
            "opsin": self._src_opsin,
            "cactus": self._src_cactus,
        }[key]
    def resolve(self, name: str) -> Tuple[Optional[str], ResolutionRecord]:
        record = ResolutionRecord(query_name=name)
        if pd.isna(name) or not str(name).strip():
            record.status = "empty_input"
            return None, record
        name = str(name).strip()
        if self._is_smiles(name):
            record.winning_source = "input_smiles"
            record.canonical_smiles = name
            record.status = "ok"
            with self.lock:
                self.stats["direct_smiles"] += 1
            return name, record
        cached = self.db.get_name(name)
        if cached:
            record.winning_source = "cache"
            record.canonical_smiles = cached
            record.status = "ok"
            with self.lock:
                self.stats["cache_hit"] += 1
            return cached, record
        if self.db.is_recently_failed(name, self.cfg.retry_after_days):
            record.status = "skipped_recent_fail"
            with self.lock:
                self.stats["skipped_recent_fail"] += 1
            return None, record
        with self.lock:
            self.stats["api_lookup"] += 1
        for key, label in self.SOURCE_ORDER:
            record.attempted_sources.append(key)
            try:
                fn = self._get_source_fn(key)
                result = fn(name)
                if result and self._is_smiles(result):
                    self.db.save_name(name, result, key)
                    record.winning_source = key
                    record.canonical_smiles = result
                    record.status = "ok"
                    with self.lock:
                        self.stats["resolved"] += 1
                        self.stats[f"resolved_via_{key}"] += 1
                    return result, record
            except Exception:
                pass
        self.db.mark_failed(name, reason="no_source_found")
        record.status = "no_source_found"
        with self.lock:
            self.stats["failed"] += 1
        return None, record
    def resolve_batch(self, values: List[str], workers: Optional[int] = None) -> List[Optional[str]]:
        workers = workers or self.cfg.smiles_workers
        unique = list({str(v) for v in values if pd.notna(v) and str(v).strip()})
        results: Dict[str, Optional[str]] = {}
        remaining = []
        for v in unique:
            cached = self.db.get_name(v)
            if cached:
                results[v] = cached
                self.lineage[v] = ResolutionRecord(v, "cache", cached, status="ok")
                with self.lock:
                    self.stats["cache_hit"] += 1
            elif self._is_smiles(v):
                results[v] = v
                self.lineage[v] = ResolutionRecord(v, "input_smiles", v, status="ok")
                with self.lock:
                    self.stats["direct_smiles"] += 1
            else:
                remaining.append(v)
        if remaining:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(self.resolve, v): v for v in remaining}
                for future in tqdm(as_completed(futures), total=len(futures), desc="SMILES Resolution"):
                    v = futures[future]
                    try:
                        smiles, record = future.result(timeout=30)
                        results[v] = smiles
                        self.lineage[v] = record
                    except Exception:
                        results[v] = None
                        self.lineage[v] = ResolutionRecord(v, status="exception")
        return [results.get(str(v)) if pd.notna(v) else None for v in values]

_METAL_ATOMIC_NUMS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 87, 88,
    89, 104, 105, 106, 107, 108, 109, 110, 111, 112,
}

class PropertyCalculator:
    def __init__(self, db: ChemicalDatabase, cfg: Config = CONFIG):
        self.db = db
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "suzuki-dataset-builder/4.2"})
        self.pubchem_limiter = RateLimiter(4.0)
        self.stats = defaultdict(int)
        self.lock = threading.Lock()
    def _largest_fragment(self, mol):
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if not frags:
                return mol
            return sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)[0]
        except Exception:
            try:
                frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                return sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True)[0] if frags else mol
            except Exception:
                return mol
    def fragment_breakdown(self, smiles: str) -> List[Dict[str, Any]]:
        out = []
        if not RDKIT_AVAILABLE or not smiles:
            return out
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return out
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            metals = _METAL_ATOMIC_NUMS
            for frag in sorted(frags, key=lambda m: m.GetNumAtoms(), reverse=True):
                atoms = [a.GetAtomicNum() for a in frag.GetAtoms()]
                is_metal = any(z in metals for z in atoms)
                role = "metal_or_organometallic" if is_metal else ("organic_core" if frag.GetNumAtoms() > 3 else "counterion_or_small_fragment")
                try:
                    frag_smiles = Chem.MolToSmiles(frag)
                except Exception:
                    frag_smiles = "?"
                out.append({"smiles": frag_smiles, "atom_count": frag.GetNumAtoms(), "role": role})
        except Exception:
            pass
        return out
    def _canonical_key(self, smiles: str) -> str:
        if not smiles:
            return ""
        try:
            if RDKIT_AVAILABLE:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            pass
        return smiles.strip()
    def _calc_rdkit(self, smiles: str) -> Tuple[Dict[str, Any], str]:
        props = default_properties()
        if not RDKIT_AVAILABLE or not smiles:
            return props, "no_rdkit_or_empty"
        mol_full = Chem.MolFromSmiles(smiles)
        if not mol_full:
            return props, "parse_failed"
        mol = self._largest_fragment(mol_full) if "." in smiles else mol_full
        if mol is None or mol.GetNumAtoms() == 0:
            mol = mol_full
        try:
            props["mw"] = round(Descriptors.ExactMolWt(mol_full), 4)
            props["formula"] = CalcMolFormula(mol_full)
            props["heavy_atoms"] = Descriptors.HeavyAtomCount(mol_full)
            props["total_atoms"] = mol_full.GetNumAtoms()
        except Exception:
            pass
        try:
            props["inchikey"] = Chem.MolToInchiKey(mol_full)
        except Exception:
            pass
        try:
            props["logp"] = round(Descriptors.MolLogP(mol), 4)
            props["tpsa"] = round(Descriptors.TPSA(mol), 4)
            props["rot_bonds"] = Descriptors.NumRotatableBonds(mol)
            props["refractivity"] = round(Crippen.MolMR(mol), 4)
        except Exception:
            pass
        try:
            props["hba"] = Lipinski.NumHAcceptors(mol)
            props["hbd"] = Lipinski.NumHDonors(mol)
        except Exception:
            pass
        try:
            ri = mol.GetRingInfo()
            props["rings"] = ri.NumRings()
            ar = al = het = sat = carbocyclic = heterocyclic = 0
            for ring in ri.AtomRings():
                is_ar = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                is_het = any(mol.GetAtomWithIdx(i).GetAtomicNum() not in (6, 1) for i in ring)
                ar += int(is_ar)
                al += int(not is_ar)
                het += int(is_het)
                sat += int(all(not mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
                if is_het:
                    heterocyclic += 1
                else:
                    carbocyclic += 1
            props.update(aromatic_rings=ar, aliphatic_rings=al, hetero_rings=het, saturated_rings=sat, carbocyclic_rings=carbocyclic, heterocyclic_rings=heterocyclic)
        except Exception:
            pass
        try:
            props["spiro_atoms"] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            props["bridgehead_atoms"] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        except Exception:
            pass
        try:
            props["branch_nodes"] = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
            props["kappa1"] = round(GraphDescriptors.Kappa1(mol), 4)
            props["kappa2"] = round(GraphDescriptors.Kappa2(mol), 4)
            props["kappa3"] = round(GraphDescriptors.Kappa3(mol), 4)
        except Exception:
            pass
        try:
            props["fr_methoxy"] = Fragments.fr_methoxy(mol)
            props["fr_nitro"] = Fragments.fr_nitro(mol)
            props["fr_Ar_halide"] = Fragments.fr_Ar_halide(mol)
            props["fr_alkyl_halide"] = Fragments.fr_alkyl_halide(mol)
        except Exception:
            pass
        try:
            counts = defaultdict(int)
            for atom in mol_full.GetAtoms():
                counts[atom.GetAtomicNum()] += 1
            props.update(c_count=counts.get(6, 0), n_count=counts.get(7, 0), o_count=counts.get(8, 0), s_count=counts.get(16, 0), p_count=counts.get(15, 0), f_count=counts.get(9, 0), cl_count=counts.get(17, 0), br_count=counts.get(35, 0), i_count=counts.get(53, 0), b_count=counts.get(5, 0), si_count=counts.get(14, 0))
            props["metal_count"] = sum(counts.get(z, 0) for z in _METAL_ATOMIC_NUMS)
            props["hetero_count"] = props["n_count"] + props["o_count"] + props["s_count"] + props["p_count"]
            props["halogen_count"] = props["f_count"] + props["cl_count"] + props["br_count"] + props["i_count"]
        except Exception:
            pass
        try:
            bc = defaultdict(int)
            for bond in mol.GetBonds():
                bc[bond.GetBondType()] += 1
            props["aromatic_bonds"] = bc.get(Chem.rdchem.BondType.AROMATIC, 0)
            props["single_bonds"] = bc.get(Chem.rdchem.BondType.SINGLE, 0)
            props["double_bonds"] = bc.get(Chem.rdchem.BondType.DOUBLE, 0)
            props["triple_bonds"] = bc.get(Chem.rdchem.BondType.TRIPLE, 0)
            props["total_bonds"] = mol.GetNumBonds()
        except Exception:
            pass
        try:
            props["chiral_centers_defined"] = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
            props["chiral_centers_undefined"] = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
        except Exception:
            pass
        try:
            props["qed"] = round(QED.qed(mol), 4)
        except Exception:
            pass
        try:
            props["fragment_count"] = len(Chem.GetMolFrags(mol_full))
        except Exception:
            pass
        try:
            props["complexity_bertz"] = round(GraphDescriptors.BertzCT(mol), 2)
        except Exception:
            pass
        try:
            props["fraction_csp3"] = round(Descriptors.FractionCSP3(mol), 4)
        except Exception:
            pass
        try:
            amide = Chem.MolFromSmarts("[NX3][CX3](=O)")
            props["num_amide_bonds"] = len(mol.GetSubstructMatches(amide)) if amide else 0
        except Exception:
            pass
        if self.cfg.compute_3d_descriptors:
            try:
                mol_3d = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol_3d, maxAttempts=10, randomSeed=42) == 0:
                    props["asphericity"] = round(Descriptors3D.Asphericity(mol_3d), 4)
            except Exception:
                pass
        return props, "ok"
    def _get_pubchem_cid(self, smiles: str) -> Optional[str]:
        try:
            self.pubchem_limiter.wait()
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/cids/TXT?limit=1"
            r = self.session.get(url, timeout=5)
            if r.ok and r.text.strip():
                return r.text.strip().splitlines()[0].strip()
        except Exception:
            pass
        return None
    def calculate(self, smiles: str) -> Tuple[Dict[str, Any], str]:
        if not smiles or not smiles.strip():
            return default_properties(), "empty_input"
        key = self._canonical_key(smiles)
        if not key:
            return default_properties(), "parse_failed"
        cached = self.db.get_properties(key)
        if cached is not None:
            merged = default_properties()
            merged.update({k: v for k, v in cached.items() if k in merged})
            with self.lock:
                self.stats["cache_hit"] += 1
            return merged, "ok_cached"
        props, status = self._calc_rdkit(key)
        if status == "ok" and not props.get("pubchem_cid"):
            props["pubchem_cid"] = self._get_pubchem_cid(key)
        if status == "ok":
            self.db.save_properties(key, props)
            with self.lock:
                self.stats["computed"] += 1
        else:
            with self.lock:
                self.stats[status] += 1
        return props, status
    def calculate_batch(self, smiles_list: List[str], workers: Optional[int] = None) -> Dict[str, Tuple[Dict[str, Any], str]]:
        workers = workers or self.cfg.property_workers
        unique = list({str(s).strip() for s in smiles_list if s and str(s).strip()})
        results: Dict[str, Tuple[Dict[str, Any], str]] = {}
        remaining = []
        for s in unique:
            key = self._canonical_key(s)
            cached = self.db.get_properties(key) if key else None
            if cached is not None:
                merged = default_properties()
                merged.update({k: v for k, v in cached.items() if k in merged})
                results[s] = (merged, "ok_cached")
            else:
                remaining.append(s)
        if remaining:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(self.calculate, s): s for s in remaining}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Property Calculation"):
                    s = futures[future]
                    try:
                        results[s] = future.result(timeout=30)
                    except Exception:
                        results[s] = (default_properties(), "exception")
        for s in unique:
            results.setdefault(s, (default_properties(), "missing"))
        return results

class MoleculeMapBuilder:
    def __init__(self, calc: PropertyCalculator):
        self.calc = calc
    def build(self, df: pd.DataFrame, column_map: Dict[str, str], resolver_lineage: Dict[str, ResolutionRecord], props_map: Dict[str, Dict[str, Tuple[Dict[str, Any], str]]]) -> Dict[str, Any]:
        molecule_map: Dict[str, Any] = {}
        for orig_col, smiles_col in column_map.items():
            entry: Dict[str, Any] = {}
            unique_originals = [v for v in df[orig_col].dropna().unique() if str(v).strip()]
            for orig_val in unique_originals:
                orig_val_s = str(orig_val)
                record = resolver_lineage.get(orig_val_s)
                smiles = record.canonical_smiles if record else None
                fragments = self.calc.fragment_breakdown(smiles) if smiles else []
                props, status = ({}, "unresolved")
                if smiles and smiles_col in props_map:
                    props, status = props_map[smiles_col].get(smiles, ({}, "missing"))
                entry[orig_val_s] = {
                    "smiles": smiles,
                    "source": record.winning_source if record else None,
                    "attempted_sources": record.attempted_sources if record else [],
                    "status": status,
                    "inchikey": props.get("inchikey"),
                    "fragment_count": len(fragments),
                    "fragments": fragments,
                }
            molecule_map[orig_col] = entry
        return molecule_map
    def print_tree(self, molecule_map: Dict[str, Any], max_per_column: int = 5):
        print("\n" + "="*80)
        print("MOLECULAR TREE AND FRAGMENT MAP")
        print("="*80)
        for col, molecules in molecule_map.items():
            print(f"\nColumn: {col.upper()} ({len(molecules)} Unique Molecules)")
            for i, (name, info) in enumerate(molecules.items()):
                if i >= max_per_column:
                    print(f"   ... and {len(molecules) - max_per_column} more molecules hidden.")
                    break
                marker = "+" if info["status"] in ("ok", "ok_cached") else "-"
                print(f"   +-- {marker} Name: {name}")
                print(f"   |   SMILES: {info['smiles']}")
                if info["fragments"]:
                    print("   |       Fragments:")
                    for idx, frag in enumerate(info["fragments"]):
                        is_last = (idx == len(info["fragments"]) - 1)
                        branch_char = "+--" if is_last else "|--"
                        role_icon = "[M]" if frag["role"] == "metal_or_organometallic" else ("[C]" if frag["role"] == "organic_core" else "[I]")
                        print(f"   |           {branch_char} {role_icon} {frag['smiles']} ({frag['atom_count']} Atoms)")
                else:
                    print("   |       No fragments or single fragment")
        print("="*80 + "\n")

SUBSTITUENT_HAMMETT: Dict[str, Dict[str, float]] = {
    '[NX2H2]': dict(sigma_m=-0.16, sigma_p=-0.66, taft_es=0.0),
    '[OX1H]': dict(sigma_m=0.12, sigma_p=-0.37, taft_es=-0.51),
    'C(#N)': dict(sigma_m=0.56, sigma_p=0.66, taft_es=-0.51),
    'N(=O)(=O)': dict(sigma_m=0.71, sigma_p=0.78, taft_es=-1.01),
    'C(F)(F)F': dict(sigma_m=0.43, sigma_p=0.54, taft_es=-2.40),
    '[CX3](=O)[OX1H]': dict(sigma_m=0.37, sigma_p=0.45, taft_es=-2.06),
}

ARYL_HALIDE_BDE: Dict[str, Dict[str, Any]] = {
    'Ar-I': dict(smarts='[c,C]-I', bde_kcal_mol=65.1, oxidative_addition_rank=4, halide_softness=9.2),
    'Ar-Br': dict(smarts='[c,C]-Br', bde_kcal_mol=81.3, oxidative_addition_rank=3, halide_softness=7.9),
    'Ar-OTf': dict(smarts='[c,C]OS(=O)(=O)C(F)(F)F', bde_kcal_mol=90.0, oxidative_addition_rank=2, halide_softness=4.1),
    'Ar-Cl': dict(smarts='[c,C]-Cl', bde_kcal_mol=96.0, oxidative_addition_rank=1, halide_softness=5.6),
}

LIGAND_PROPERTIES: Dict[str, Dict[str, Any]] = {
    'triphenylphosphine': dict(cone_angle=145.0, tep_cm1=2068.9, bite_angle=np.nan, pka_bh=2.73, denticity=1, ligand_class='triaryl_phosphine', softness=6.2),
    'pph3': dict(cone_angle=145.0, tep_cm1=2068.9, bite_angle=np.nan, pka_bh=2.73, denticity=1, ligand_class='triaryl_phosphine', softness=6.2),
    'sphos': dict(cone_angle=163.0, tep_cm1=2064.0, bite_angle=np.nan, pka_bh=7.70, denticity=1, ligand_class='dialkylbiaryl_phosphine', softness=7.8),
    'xphos': dict(cone_angle=180.0, tep_cm1=2062.4, bite_angle=np.nan, pka_bh=7.90, denticity=1, ligand_class='dialkylbiaryl_phosphine', softness=8.0),
    'dppf': dict(cone_angle=np.nan, tep_cm1=2065.3, bite_angle=99.07, pka_bh=4.50, denticity=2, ligand_class='bidentate_ferrocenyl', softness=7.1),
}

HSAB_PARAMETERS: Dict[str, Dict[str, float]] = {
    'Pd0': dict(hardness=3.8, softness=7.5, class_type='soft'),
    'PdII': dict(hardness=5.2, softness=5.8, class_type='borderline'),
    'I': dict(hardness=3.5, softness=9.2, class_type='soft'),
    'Br': dict(hardness=4.5, softness=7.9, class_type='soft'),
    'Cl': dict(hardness=5.8, softness=5.6, class_type='borderline'),
    'F': dict(hardness=7.5, softness=3.2, class_type='hard'),
}

def calculate_hammett_taft(smiles: str) -> Tuple[float, float, float]:
    if not RDKIT_AVAILABLE or not smiles or "." in smiles:
        return 0.0, 0.0, 0.0
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return 0.0, 0.0, 0.0
        sigma_m_sum = 0.0
        sigma_p_sum = 0.0
        taft_sum = 0.0
        match_count = 0
        for smarts, vals in SUBSTITUENT_HAMMETT.items():
            patt = Chem.MolFromSmarts(smarts)
            if patt and mol.HasSubstructMatch(patt):
                matches = len(mol.GetSubstructMatches(patt))
                sigma_m_sum += vals['sigma_m'] * matches
                sigma_p_sum += vals['sigma_p'] * matches
                taft_sum += vals['taft_es'] * matches
                match_count += matches
        if match_count == 0:
            return 0.0, 0.0, 0.0
        return round(sigma_m_sum / match_count, 3), round(sigma_p_sum / match_count, 3), round(taft_sum / match_count, 3)
    except:
        return 0.0, 0.0, 0.0

def detect_substituent_pattern(smiles: str) -> Dict[str, int]:
    pattern_counts = {"ortho": 0, "meta": 0, "para": 0}
    if not RDKIT_AVAILABLE or not smiles:
        return pattern_counts
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return pattern_counts
        rings = mol.GetRingInfo().AtomRings()
        aromatic_rings = [r for r in rings if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
        for ring in aromatic_rings:
            ring_atoms = set(ring)
            substituents = []
            for idx in ring:
                atom = mol.GetAtomWithIdx(idx)
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() not in ring_atoms:
                        substituents.append((idx, neighbor.GetIdx()))
            if len(substituents) >= 2:
                positions = sorted([pos for pos, _ in substituents])
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        diff = abs(positions[j] - positions[i])
                        ring_size = len(ring)
                        if diff == 1 or diff == ring_size - 1:
                            pattern_counts["ortho"] += 1
                        elif diff == 2 or diff == ring_size - 2:
                            pattern_counts["meta"] += 1
                        elif diff == 3 or diff == ring_size - 3:
                            pattern_counts["para"] += 1
    except:
        pass
    return pattern_counts

def calculate_stoichiometric_discrepancy(row: pd.Series) -> Dict[str, float]:
    metrics = {"pd_ligand_ratio": np.nan, "ligand_excess": np.nan, "catalyst_loading_mmol": np.nan, "ligand_pd_complexation_efficiency": np.nan, "pd_l_3_formation_indicator": 0.0}
    try:
        cat_name = str(row.get('catalizor', '')).lower()
        for lig_name, props in LIGAND_PROPERTIES.items():
            if lig_name.lower() in cat_name:
                metrics['ligand_class'] = props.get('ligand_class', 'unknown')
                metrics['ligand_denticity'] = props.get('denticity', 0)
                metrics['ligand_softness'] = props.get('softness', 5.0)
                break
        pd_smiles = row.get('catalizor_SMILES')
        if pd_smiles and RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(str(pd_smiles))
            if mol:
                pd_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 46]
                if pd_atoms:
                    metrics['pd_atoms_per_molecule'] = len(pd_atoms)
        if 'quantity' in row.index:
            qty = pd.to_numeric(row.get('quantity'), errors='coerce')
            if pd.notna(qty) and qty > 0:
                metrics['catalyst_loading_mmol'] = qty
        if metrics.get('pd_atoms_per_molecule') and metrics.get('ligand_denticity'):
            if metrics.get('pd_atoms_per_molecule', 0) > 0:
                metrics['pd_ligand_ratio'] = round(metrics.get('ligand_denticity', 1) / metrics.get('pd_atoms_per_molecule', 1), 3)
                metrics['ligand_excess'] = max(0, metrics.get('pd_ligand_ratio', 0) - 1)
                if metrics['pd_ligand_ratio'] >= 3.0:
                    metrics['pd_l_3_formation_indicator'] = 1.0
                elif metrics['pd_ligand_ratio'] >= 2.0:
                    metrics['pd_l_3_formation_indicator'] = 0.5
                metrics['ligand_pd_complexation_efficiency'] = min(1.0, metrics['pd_ligand_ratio'] / 2.0)
    except Exception:
        pass
    return metrics

def calculate_hsab_compatibility(smiles: str, catalyst: str, solvent: str) -> Dict[str, float]:
    metrics = {'hsab_pd_halide_mismatch': 10.0, 'hsab_pd_ligand_mismatch': 10.0, 'hsab_ligand_halide_mismatch': 10.0, 'hsab_overall_compatibility': 0.0, 'hsab_soft_soft_interaction_score': 0.0, 'hsab_hard_hard_interaction_score': 0.0, 'hsab_pearson_class_match': 0.0}
    if not RDKIT_AVAILABLE or not smiles:
        return metrics
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return metrics
        halide_softness = 5.0
        halide_type = 'borderline'
        for name, info in ARYL_HALIDE_BDE.items():
            patt = Chem.MolFromSmarts(info['smarts'])
            if patt and mol.HasSubstructMatch(patt):
                halide_softness = info['halide_softness']
                halide_type = 'soft' if halide_softness > 7.0 else ('hard' if halide_softness < 5.0 else 'borderline')
                break
        cat_str = str(catalyst).lower()
        cat_softness = 5.0
        cat_type = 'borderline'
        if 'pd' in cat_str or 'palladium' in cat_str:
            if '0' in cat_str or 'tetrakis' in cat_str:
                cat_softness = HSAB_PARAMETERS.get('Pd0', {}).get('softness', 7.5)
                cat_type = 'soft'
            elif 'ii' in cat_str or 'oac' in cat_str or 'cl2' in cat_str:
                cat_softness = HSAB_PARAMETERS.get('PdII', {}).get('softness', 5.8)
                cat_type = 'borderline'
            else:
                cat_softness = 6.8
                cat_type = 'soft'
        solv_str = str(solvent).lower()
        solv_softness = 5.0
        solv_type = 'borderline'
        for solv_name, params in HSAB_PARAMETERS.items():
            if solv_name in solv_str:
                solv_softness = params.get('softness', 5.0)
                solv_type = params.get('class_type', 'borderline')
                break
        metrics['hsab_pd_halide_mismatch'] = abs(cat_softness - halide_softness)
        metrics['hsab_pd_ligand_mismatch'] = abs(cat_softness - solv_softness)
        metrics['hsab_ligand_halide_mismatch'] = abs(solv_softness - halide_softness)
        if cat_type == 'soft' and halide_type == 'soft':
            metrics['hsab_soft_soft_interaction_score'] = 1.0 - metrics['hsab_pd_halide_mismatch'] / 10.0
        elif cat_type == 'hard' and halide_type == 'hard':
            metrics['hsab_hard_hard_interaction_score'] = 1.0 - metrics['hsab_pd_halide_mismatch'] / 10.0
        if cat_type == halide_type:
            metrics['hsab_pearson_class_match'] = 1.0
        elif cat_type == 'borderline' or halide_type == 'borderline':
            metrics['hsab_pearson_class_match'] = 0.5
        metrics['hsab_overall_compatibility'] = round((1 - metrics['hsab_pd_halide_mismatch'] / 10.0) * 0.5 + (1 - metrics['hsab_pd_ligand_mismatch'] / 10.0) * 0.3 + (1 - metrics['hsab_ligand_halide_mismatch'] / 10.0) * 0.2 + metrics['hsab_pearson_class_match'] * 0.1, 3)
        for key in ['hsab_pd_halide_mismatch', 'hsab_pd_ligand_mismatch', 'hsab_ligand_halide_mismatch']:
            metrics[key] = round(metrics[key], 3)
    except Exception:
        pass
    return metrics

def add_domain_features(df: pd.DataFrame, cfg: Config = CONFIG) -> Tuple[pd.DataFrame, List[str]]:
    added: List[str] = []
    out = df.copy()
    for solv_col in ("solv1", "solv2"):
        if solv_col not in out.columns:
            continue
        def clean_solvent_name(val):
            if pd.isna(val) or not val:
                return np.nan
            val = str(val).lower().strip()
            if val in ('o', 'o=o', 'unknown', 'none', 'null', '', 'nan'):
                return np.nan
            return val
        norm = out[solv_col].astype(str).apply(clean_solvent_name)
        for prop in ("dielectric", "bp_c", "polarity_index", "donor_number"):
            new_col = f"{solv_col}_phys_{prop}"
            out[new_col] = norm.map(lambda n: SOLVENT_PHYSICS.get(n, {}).get(prop, np.nan) if pd.notna(n) else np.nan)
            added.append(new_col)
    smiles_col = "subs1_SMILES" if "subs1_SMILES" in out.columns else None
    if smiles_col and RDKIT_AVAILABLE:
        _LEAVING_GROUP_SMARTS = [("Ar-I", "c-I", 4.0), ("Ar-Br", "c-Br", 3.0), ("Ar-OTf", "cOS(=O)(=O)C(F)(F)F", 2.5), ("Ar-Cl", "c-Cl", 1.0), ("Ar-F", "c-F", 0.2)]
        def leaving_group_score(smi):
            if not isinstance(smi, str) or not smi:
                return 0.0
            try:
                mol = Chem.MolFromSmiles(smi)
                if not mol:
                    return 0.0
                best = 0.0
                for _, smarts, score in _LEAVING_GROUP_SMARTS:
                    patt = Chem.MolFromSmarts(smarts)
                    if patt and mol.HasSubstructMatch(patt):
                        best = max(best, score)
                return best
            except Exception:
                return 0.0
        out["subs1_leaving_group_reactivity"] = out[smiles_col].apply(leaving_group_score)
        added.append("subs1_leaving_group_reactivity")
    if "temp" in out.columns and "time" in out.columns:
        out["temp_time_product"] = pd.to_numeric(out["temp"], errors="coerce") * pd.to_numeric(out["time"], errors="coerce")
        added.append("temp_time_product")
    if "quantity" in out.columns:
        q = pd.to_numeric(out["quantity"], errors="coerce")
        out["quantity_log1p"] = np.log1p(q.clip(lower=0))
        added.append("quantity_log1p")
    mw1, mw2 = "subs1_SMILES_mw", "subs2_SMILES_mw"
    if mw1 in out.columns and mw2 in out.columns:
        denom = out[mw2].replace(0, np.nan)
        out["subs1_subs2_mw_ratio"] = out[mw1] / denom
        added.append("subs1_subs2_mw_ratio")
    return out, added

def add_advanced_academic_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    added_cols = []
    if "subs1_SMILES" in out.columns:
        ht_data = out["subs1_SMILES"].progress_apply(calculate_hammett_taft)
        out["subs1_sigma_m"] = ht_data.apply(lambda x: x[0])
        out["subs1_sigma_p"] = ht_data.apply(lambda x: x[1])
        out["subs1_taft_es"] = ht_data.apply(lambda x: x[2])
        added_cols.extend(["subs1_sigma_m", "subs1_sigma_p", "subs1_taft_es"])
        pattern_data = out["subs1_SMILES"].progress_apply(detect_substituent_pattern)
        for p in ["ortho", "meta", "para"]:
            col = f"subs1_{p}_count"
            out[col] = pattern_data.apply(lambda x: x.get(p, 0))
            added_cols.append(col)
        out["subs1_electron_donating_power"] = -out["subs1_sigma_p"]
        out["subs1_electron_withdrawing_power"] = out["subs1_sigma_p"]
        out["subs1_total_hammett_effect"] = out["subs1_sigma_m"] + out["subs1_sigma_p"]
        added_cols.extend(["subs1_electron_donating_power", "subs1_electron_withdrawing_power", "subs1_total_hammett_effect"])
    if "catalizor" in out.columns:
        stoich_data = out.progress_apply(calculate_stoichiometric_discrepancy, axis=1)
        for col in ["pd_ligand_ratio", "ligand_excess", "catalyst_loading_mmol", "ligand_pd_complexation_efficiency", "pd_l_3_formation_indicator"]:
            out[f"catalizor_{col}"] = stoich_data.apply(lambda x: x.get(col, np.nan))
            added_cols.append(f"catalizor_{col}")
    if "subs1_SMILES" in out.columns and "catalizor" in out.columns:
        hsab_data = out.progress_apply(lambda row: calculate_hsab_compatibility(row.get("subs1_SMILES", ""), row.get("catalizor", ""), row.get("solv1", "")), axis=1)
        hsab_cols = ["hsab_pd_halide_mismatch", "hsab_pd_ligand_mismatch", "hsab_ligand_halide_mismatch", "hsab_overall_compatibility", "hsab_soft_soft_interaction_score", "hsab_hard_hard_interaction_score", "hsab_pearson_class_match"]
        for col in hsab_cols:
            out[col] = hsab_data.apply(lambda x: x.get(col, np.nan))
            added_cols.append(col)
        out["hsab_total_mismatch"] = out["hsab_pd_halide_mismatch"] + out["hsab_pd_ligand_mismatch"] + out["hsab_ligand_halide_mismatch"]
        added_cols.append("hsab_total_mismatch")
        out["hsab_compatibility_grade"] = 1 - (out["hsab_total_mismatch"] / 30)
        out["hsab_compatibility_grade"] = out["hsab_compatibility_grade"].clip(0, 1)
        added_cols.append("hsab_compatibility_grade")
    if "subs1_sigma_p" in out.columns and "hsab_overall_compatibility" in out.columns:
        out["mechanistic_predictor_electronic_softness"] = out["subs1_sigma_p"] * 0.3 + out["hsab_soft_soft_interaction_score"].fillna(0) * 0.4 + out["hsab_overall_compatibility"].fillna(0) * 0.3
        added_cols.append("mechanistic_predictor_electronic_softness")
        out["mechanistic_oxidative_addition_liability"] = ((1 - out["subs1_sigma_p"]) * 0.4 + out["hsab_soft_soft_interaction_score"].fillna(0) * 0.4 + (1 - out["hsab_pd_halide_mismatch"].fillna(10) / 10) * 0.2).clip(0, 1)
        added_cols.append("mechanistic_oxidative_addition_liability")
    if "subs1_SMILES" in out.columns and "subs2_SMILES" in out.columns:
        out["transmetalation_steric_mismatch"] = abs(out.get("subs1_taft_es", 0) - out.get("subs2_taft_es", 0))
        added_cols.append("transmetalation_steric_mismatch")
    speed_cols = [c for c in added_cols if any(x in c for x in ["mechanistic_", "hsab_overall", "subs1_sigma"])]
    if speed_cols:
        out["reaction_rate_indicator"] = out[speed_cols].mean(axis=1)
        added_cols.append("reaction_rate_indicator")
    return out, added_cols

def add_all_academic_features(df: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    df, domain_cols = add_domain_features(df, cfg)
    df, academic_cols = add_advanced_academic_features(df)
    return df

ML_TARGET_CANDIDATES = ["yield", "verim", "conversion", "conversiyon"]

def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (bool, int, float, str)):
        return obj
    elif obj is None:
        return None
    else:
        return str(obj)

def read_universal_csv(file_path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1254", "iso-8859-9"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, engine="python", on_bad_lines="skip", quotechar='"')
            if len(df) > 0:
                return df
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"CSV read failed, attempted encodings: {last_err}")

def analyze_target_variable(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    if target_col is None:
        candidates = ML_TARGET_CANDIDATES
        for col in candidates:
            if col in df.columns:
                target_col = col
                break
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols:
                    if any(k in col.lower() for k in ["yield", "verim", "convers"]):
                        target_col = col
                        break
                if target_col is None:
                    target_col = numeric_cols[0] if numeric_cols else None
    if target_col is None:
        report["status"] = "no_target_found"
        report["message"] = f"Target variable not found. Searched: {ML_TARGET_CANDIDATES}"
        return make_json_serializable(report)
    if target_col not in df.columns:
        report["status"] = "column_not_found"
        report["message"] = f"{target_col} column not found in DataFrame"
        return make_json_serializable(report)
    report["target_column"] = target_col
    values = df[target_col].dropna()
    if values.empty:
        report["status"] = "empty"
        report["message"] = "Target variable is empty"
        return make_json_serializable(report)
    numeric_values = pd.to_numeric(values, errors='coerce').dropna()
    is_numeric = len(numeric_values) == len(values)
    if is_numeric:
        unique_count = values.nunique()
        if unique_count <= 10:
            report["type"] = "categorical"
            report["unique_values"] = sorted(values.unique().tolist())
            class_counts = values.value_counts().to_dict()
            report["class_counts"] = make_json_serializable(class_counts)
            if len(class_counts) >= 2:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                report["class_imbalance_ratio"] = round(max_count / min_count, 2) if min_count > 0 else float('inf')
                report["imbalanced"] = report["class_imbalance_ratio"] > 3.0
                if report["imbalanced"]:
                    report["recommendation"] = "Class imbalance detected. SMOTE, class_weight or stratified sampling recommended."
                else:
                    report["recommendation"] = "Classes appear balanced."
            else:
                report["class_imbalance_ratio"] = 1.0
                report["imbalanced"] = False
                report["recommendation"] = "Single class present (may be regression problem)."
        else:
            report["type"] = "continuous"
            report["unique_values"] = unique_count
            report["stats"] = {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean()), "std": float(values.std()), "q25": float(values.quantile(0.25)), "q50": float(values.quantile(0.50)), "q75": float(values.quantile(0.75))}
            report["range"] = report["stats"]["max"] - report["stats"]["min"]
            report["recommendation"] = "Continuous target variable detected. Suitable for regression models."
    else:
        report["type"] = "categorical_string"
        report["unique_values"] = sorted(values.unique().tolist())
        class_counts = values.value_counts().to_dict()
        report["class_counts"] = make_json_serializable(class_counts)
        if len(class_counts) >= 2:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            report["class_imbalance_ratio"] = round(max_count / min_count, 2) if min_count > 0 else float('inf')
            report["imbalanced"] = report["class_imbalance_ratio"] > 3.0
            if report["imbalanced"]:
                report["recommendation"] = "Class imbalance detected. SMOTE, class_weight or stratified sampling recommended."
            else:
                report["recommendation"] = "Classes appear balanced."
        else:
            report["class_imbalance_ratio"] = 1.0
            report["imbalanced"] = False
            report["recommendation"] = "Single class present."
    report["status"] = "success"
    report["n_samples"] = int(len(values))
    report["n_missing"] = int(df[target_col].isna().sum())
    report["data_coverage"] = round((len(values) / len(df)) * 100, 2)
    return make_json_serializable(report)

tasks = {}
tasks_lock = threading.Lock()

class TaskStatus:
    __slots__ = ['task_id', 'status', 'percentage', 'message', 'message_key', 'result', 'error', 'created_at', 'updated_at', 'logs', '_log_lock']
    def __init__(self, task_id):
        self.task_id = task_id
        self.status = 'pending'
        self.percentage = 0
        self.message = 'Starting...'
        self.message_key = 'starting'
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.logs = []
        self._log_lock = threading.Lock()
    def add_log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        with self._log_lock:
            self.logs.append(log_entry)
            if len(self.logs) > 50:
                self.logs = self.logs[-50:]
        print(log_entry)
    def update(self, percentage, message=None, message_key=None):
        self.percentage = min(percentage, 100)
        if message:
            self.message = message
        if message_key:
            self.message_key = message_key
        self.updated_at = datetime.now()
        if message:
            self.add_log(message)
    def complete(self, result):
        self.status = 'completed'
        self.percentage = 100
        self.result = result
        self.updated_at = datetime.now()
        self.add_log("Process completed!")
    def fail(self, error):
        self.status = 'error'
        self.error = error
        self.updated_at = datetime.now()
        self.add_log(f"Error: {error}")
    def to_dict(self):
        with self._log_lock:
            logs = self.logs[-10:]
        return {'status': self.status, 'percentage': self.percentage, 'message': self.message, 'message_key': self.message_key, 'error': self.error, 'result': self.result, 'task_id': self.task_id, 'logs': logs}

_thread_local = threading.local()

class TaskLoggingHandler(logging.Handler):
    def emit(self, record):
        task = getattr(_thread_local, 'task', None)
        if task is None:
            return
        try:
            task.add_log(self.format(record))
        except Exception:
            pass

_task_log_handler = TaskLoggingHandler()
_task_log_handler.setLevel(logging.INFO)
_task_log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_task_log_handler)

def sanitize_csv_filename(filename):
    filename = os.path.basename(filename)
    if not filename.lower().endswith('.csv'):
        filename += '.csv'
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    return filename

def get_datasets_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    datasets = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    datasets = [f for f in datasets if f.endswith('.csv')]
    return datasets

def run_conversion_pipeline(cfg: Config, task: TaskStatus) -> Dict[str, Any]:
    start = time.time()
    task.update(3, 'Database and resolvers initializing...', 'preparing')
    db = ChemicalDatabase(cfg.db_path)
    resolver = SMILESResolver(db, cfg)
    calc = PropertyCalculator(db, cfg)
    task.update(6, 'Reading CSV...', 'reading')
    df = read_universal_csv(cfg.input_csv)
    task.add_log(f"Columns: {list(df.columns)}")
    task.add_log(f"Data size: {df.shape[0]} rows, {df.shape[1]} columns")
    task.update(9, 'Analyzing target variable...', 'target_analysis')
    target_report = analyze_target_variable(df, cfg.target_column)
    if target_report.get("status") == "success":
        task.add_log(f"Target: {target_report['target_column']} ({target_report['type']}, coverage %{target_report['data_coverage']})")
        if target_report['type'] in ('categorical', 'categorical_string') and target_report.get('imbalanced'):
            task.add_log(f"Class imbalance: ratio={target_report.get('class_imbalance_ratio', 0)}")
    else:
        task.add_log(f"Target not found: {target_report.get('message', '')}")
    available = [c for c in cfg.reaction_columns if c in df.columns]
    if not available:
        raise ValueError("No expected reaction columns found (subs1, subs2, product, catalizor, base, solv1, solv2).")
    task.update(12, f'SMILES resolution starting: {len(available)} columns...', 'smiles')
    column_map: Dict[str, str] = {}
    n_cols = len(available)
    for idx, col in enumerate(available):
        task.add_log(f"Resolving SMILES: {col}")
        smiles_col = f"{col}_SMILES"
        df[smiles_col] = resolver.resolve_batch(df[col].tolist(), workers=cfg.smiles_workers)
        df[smiles_col] = df[smiles_col].fillna(df[col])
        column_map[col] = smiles_col
        if cfg.checkpoint_every_column:
            try:
                df.to_csv(cfg.checkpoint_csv, index=False, encoding="utf-8-sig")
            except Exception:
                pass
        pct = 12 + int(((idx + 1) / n_cols) * 33)
        task.update(pct, f'SMILES: {idx + 1}/{n_cols} columns done ({col})', 'smiles')
    smiles_cols = list(column_map.values())
    task.update(46, f'Molecular properties calculation: {len(smiles_cols)} columns...', 'properties')
    props_map_by_col: Dict[str, Dict[str, Tuple[Dict[str, Any], str]]] = {}
    n_sc = max(len(smiles_cols), 1)
    for idx, sc in enumerate(smiles_cols):
        task.add_log(f"Calculating properties: {sc}")
        unique = [s for s in df[sc].dropna().unique() if s and str(s).strip()]
        if not unique:
            task.add_log(f"No SMILES found for {sc}, skipping.")
            continue
        props_map = calc.calculate_batch(unique, workers=cfg.property_workers)
        props_map_by_col[sc] = props_map
        status_col = f"{sc}_status"
        df[status_col] = df[sc].progress_apply(lambda x: props_map.get(str(x).strip(), ({}, "no_input"))[1] if pd.notna(x) else "no_input")
        for prop in PROPERTY_SCHEMA:
            col_name = f"{sc}_{prop}"
            df[col_name] = df[sc].progress_apply(lambda x, p=prop: props_map.get(str(x).strip(), ({}, "missing"))[0].get(p) if pd.notna(x) else None)
            if df[col_name].isna().any():
                if prop == "formula":
                    df[col_name] = df[col_name].fillna("Unknown")
                elif prop in ("inchikey", "pubchem_cid"):
                    df[col_name] = df[col_name].fillna("Not_Found")
                else:
                    df[col_name] = df[col_name].fillna(0)
        n_ok = int((df[status_col].isin(["ok", "ok_cached"])).sum())
        task.add_log(f"{sc}: {len(PROPERTY_SCHEMA)} properties added | successfully calculated: {n_ok}/{len(df)}")
        if cfg.checkpoint_every_column:
            try:
                df.to_csv(cfg.checkpoint_csv, index=False, encoding="utf-8-sig")
            except Exception:
                pass
        pct = 46 + int(((idx + 1) / n_sc) * 34)
        task.update(pct, f'Properties: {idx + 1}/{n_sc} columns done ({sc})', 'properties')
    task.update(82, 'Applying academic feature engineering...', 'features')
    df = add_all_academic_features(df, cfg)
    task.update(88, 'Building molecular tree...', 'mapping')
    mapper = MoleculeMapBuilder(calc)
    molecule_map = mapper.build(df, column_map, resolver.lineage, props_map_by_col)
    mapper.print_tree(molecule_map)
    task.update(93, 'Checking empty cells...', 'cleanup')
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        task.add_log(f"{total_nulls} empty cells found, filling...")
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype == "float64":
                    df[col] = df[col].fillna(0.0)
                elif df[col].dtype == "int64":
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna("Unknown")
        task.add_log("Empty values filled.")
    task.update(97, 'Saving CSV...', 'saving')
    df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")
    elapsed = time.time() - start
    task.add_log(f"{os.path.basename(cfg.output_csv)} — {len(df.columns)} columns, {len(df)} rows")
    task.add_log(f"Duration: {elapsed:.2f} seconds")
    return {'output_file': os.path.basename(cfg.output_csv), 'columns_converted': available, 'rows_processed': len(df), 'total_columns_output': len(df.columns), 'elapsed_seconds': round(elapsed, 2), 'target_analysis': target_report}

@dataset_bp.route('/')
def index():
    return render_template('dataset.html')

@dataset_bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files allowed'}), 400
        app = current_app._get_current_object()
        filename = secure_filename(file.filename)
        upload_folder = app.config.get('UPLOAD_FOLDER', '/tmp')
        os.makedirs(upload_folder, exist_ok=True)
        temp_filename = f"temp_{uuid.uuid4().hex[:8]}_{filename}"
        filepath = os.path.join(upload_folder, temp_filename)
        file.save(filepath)
        task_id = uuid.uuid4().hex[:12]
        task = TaskStatus(task_id)
        task.add_log(f"File uploaded: {filename}")
        with tasks_lock:
            tasks[task_id] = task
        thread = threading.Thread(target=process_upload_task, args=(task_id, filepath, filename, app))
        thread.daemon = True
        thread.start()
        return jsonify({'success': True, 'task_id': task_id, 'message': 'File uploaded, processing started...'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def process_upload_task(task_id, filepath, original_filename, app):
    task = tasks.get(task_id)
    if not task:
        return
    _thread_local.task = task
    try:
        with app.app_context():
            output_dir = app.config.get('OUTPUTS_DIR', 'static/datasets')
            os.makedirs(output_dir, exist_ok=True)
            db_dir = app.config.get('CHEM_DB_DIR', output_dir)
            os.makedirs(db_dir, exist_ok=True)
            log_dir = app.config.get('LOGS_DIR', os.path.join(output_dir, 'logs'))
            os.makedirs(log_dir, exist_ok=True)
            output_filename = f"smiles_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original_filename}"
            output_path = os.path.join(output_dir, output_filename)
            cfg = Config(input_csv=filepath, output_csv=output_path, db_path=os.path.join(db_dir, 'universal_chemical.db'), log_path=os.path.join(log_dir, f"{task_id}.log"), checkpoint_csv=output_path + ".checkpoint.csv")
            result = run_conversion_pipeline(cfg, task)
            try:
                if os.path.exists(cfg.checkpoint_csv):
                    os.remove(cfg.checkpoint_csv)
            except Exception:
                pass
            try:
                os.remove(filepath)
            except Exception:
                pass
            task.complete(result)
    except Exception as e:
        traceback.print_exc()
        task.fail(str(e))
        try:
            os.remove(filepath)
        except Exception:
            pass
    finally:
        _thread_local.task = None

@dataset_bp.route('/progress/<task_id>', methods=['GET'])
def get_progress(task_id):
    with tasks_lock:
        task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task.to_dict())

@dataset_bp.route('/list', methods=['GET'])
def list_datasets():
    try:
        output_dir = current_app.config.get('OUTPUTS_DIR', 'static/datasets')
        files = get_datasets_from_folder(output_dir)
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@dataset_bp.route('/preview/<filename>', methods=['GET'])
def preview_dataset(filename):
    try:
        output_dir = current_app.config.get('OUTPUTS_DIR', 'static/datasets')
        filepath = os.path.join(output_dir, secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'}), 404
        df = pd.read_csv(filepath)
        return jsonify({'success': True, 'columns': df.columns.tolist(), 'head': df.head(10).to_dict('records'), 'shape': {'rows': len(df), 'cols': len(df.columns)}, 'dtypes': df.dtypes.astype(str).to_dict()})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@dataset_bp.route('/save', methods=['POST'])
def save_dataset():
    try:
        data = request.get_json()
        content = data.get('content', '')
        filename = data.get('filename', '')
        if not content:
            return jsonify({'success': False, 'message': 'Content empty'})
        if not filename:
            return jsonify({'success': False, 'message': 'Filename required'})
        filename = sanitize_csv_filename(filename)
        output_dir = current_app.config.get('OUTPUTS_DIR', 'static/datasets')
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return jsonify({'success': True, 'message': f'CSV saved: {filename}', 'filename': filename})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@dataset_bp.route('/delete/<filename>', methods=['DELETE'])
def delete_dataset(filename):
    try:
        output_dir = current_app.config.get('OUTPUTS_DIR', 'static/datasets')
        filepath = os.path.join(output_dir, secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'File not found'}), 404
        os.remove(filepath)
        return jsonify({'success': True, 'message': f'{filename} deleted'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@dataset_bp.route('/download/<filename>', methods=['GET'])
def download_dataset(filename):
    try:
        output_dir = current_app.config.get('OUTPUTS_DIR', 'static/datasets')
        filepath = os.path.join(output_dir, secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dataset_bp.route('/chemicals/xml', methods=['GET'])
def get_chemicals_xml():
    xml_path = Path(__file__).parent / "common_chemicals.xml"
    if xml_path.exists():
        return send_file(xml_path, as_attachment=False, mimetype='application/xml')
    return jsonify({'error': 'XML file not found'}), 404

@dataset_bp.route('/chemicals/xml', methods=['POST', 'PUT'])
def upload_chemicals_xml():
    if 'file' not in request.files:
        return jsonify({'error': 'XML file required'}), 400
    file = request.files['file']
    if not file.filename.endswith('.xml'):
        return jsonify({'error': 'Only XML files allowed'}), 400
    xml_path = Path(__file__).parent / "common_chemicals.xml"
    xml_path.parent.mkdir(exist_ok=True)
    file.save(xml_path)
    try:
        test_data = load_chemicals_from_xml(str(xml_path))
        return jsonify({'success': True, 'message': 'XML updated successfully', 'chemical_count': len(test_data)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@dataset_bp.route('/solvents/xml', methods=['GET'])
def get_solvents_xml():
    xml_path = Path(__file__).parent / "solvent_properties.xml"
    if xml_path.exists():
        return send_file(xml_path, as_attachment=False, mimetype='application/xml')
    return jsonify({'error': 'XML file not found'}), 404

@dataset_bp.route('/solvents/xml', methods=['POST', 'PUT'])
def upload_solvents_xml():
    if 'file' not in request.files:
        return jsonify({'error': 'XML file required'}), 400
    file = request.files['file']
    if not file.filename.endswith('.xml'):
        return jsonify({'error': 'Only XML files allowed'}), 400
    xml_path = Path(__file__).parent / "solvent_properties.xml"
    xml_path.parent.mkdir(exist_ok=True)
    file.save(xml_path)
    try:
        test_data = load_solvent_properties_from_xml(str(xml_path))
        SOLVENT_PHYSICS.update(test_data)
        return jsonify({'success': True, 'message': 'XML updated successfully', 'solvent_count': len(test_data)})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@dataset_bp.route('/chemicals/list', methods=['GET'])
def list_chemicals():
    return jsonify({'success': True, 'count': len(_COMMON_CHEMICALS), 'chemicals': list(_COMMON_CHEMICALS.keys())})

@dataset_bp.route('/chemicals/category/<category>', methods=['GET'])
def list_chemicals_by_category(category):
    result = {}
    for name, data in _COMMON_CHEMICALS.items():
        if data.get('category') == category:
            result[name] = data.get('smiles')
    return jsonify({'success': True, 'category': category, 'count': len(result), 'chemicals': result})

def cleanup_old_tasks():
    while True:
        try:
            now = datetime.now()
            with tasks_lock:
                for task_id in list(tasks.keys()):
                    task = tasks[task_id]
                    if (now - task.created_at).total_seconds() > 86400:
                        del tasks[task_id]
            time.sleep(3600)
        except Exception:
            pass

cleanup_thread = threading.Thread(target=cleanup_old_tasks, daemon=True)
cleanup_thread.start()