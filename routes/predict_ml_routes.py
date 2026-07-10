from flask import Blueprint, render_template, request, jsonify, session
import os
import sys
import json
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Generator
import numpy as np
import pandas as pd
import io
import base64
from functools import wraps
import time
import re
import pickle
import joblib
from pathlib import Path
import warnings
import random
import hashlib
from collections import defaultdict, OrderedDict
import itertools
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize, differential_evolution
from scipy.constants import R, h, k as boltzmann_k
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict,
    KFold, StratifiedKFold, LeaveOneOut, GroupKFold,
    GridSearchCV, RandomizedSearchCV, ParameterGrid
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, max_error, explained_variance_score,
    median_absolute_error, mean_squared_log_error
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer,
    PowerTransformer, OneHotEncoder, LabelEncoder, PolynomialFeatures
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    mutual_info_regression, VarianceThreshold, SelectKBest,
    RFE, SelectFromModel, SequentialFeatureSelector
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen, QED, AllChem
    from rdkit.Chem.Draw import IPythonConsole
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    PERM_IMP_AVAILABLE = True
except ImportError:
    PERM_IMP_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    ISOTONIC_AVAILABLE = True
except ImportError:
    ISOTONIC_AVAILABLE = False

try:
    from sklearn.calibration import CalibratedRegressorCV
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import anova_lm
    from statsmodels.formula.api import ols
    from scipy.stats import shapiro, levene, kruskal, f_oneway
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

predict_ml_bp = Blueprint('predict_ml', __name__, url_prefix='/predict_ml')

HAMMETT_SIGMA = {
    'H': {'sigma_m': 0.00, 'sigma_p': 0.00, 'taft_es': 0.00, 'sigma_plus': 0.00, 'sigma_minus': 0.00},
    'CH3': {'sigma_m': -0.07, 'sigma_p': -0.17, 'taft_es': 0.00, 'sigma_plus': -0.31, 'sigma_minus': -0.17},
    'OCH3': {'sigma_m': 0.12, 'sigma_p': -0.27, 'taft_es': -0.20, 'sigma_plus': -0.78, 'sigma_minus': -0.27},
    'OH': {'sigma_m': 0.12, 'sigma_p': -0.37, 'taft_es': -0.51, 'sigma_plus': -0.92, 'sigma_minus': -0.37},
    'F': {'sigma_m': 0.34, 'sigma_p': 0.06, 'taft_es': -0.46, 'sigma_plus': -0.07, 'sigma_minus': 0.06},
    'Cl': {'sigma_m': 0.37, 'sigma_p': 0.23, 'taft_es': -0.97, 'sigma_plus': 0.11, 'sigma_minus': 0.23},
    'Br': {'sigma_m': 0.39, 'sigma_p': 0.23, 'taft_es': -1.16, 'sigma_plus': 0.15, 'sigma_minus': 0.23},
    'I': {'sigma_m': 0.35, 'sigma_p': 0.18, 'taft_es': -1.40, 'sigma_plus': 0.14, 'sigma_minus': 0.18},
    'NO2': {'sigma_m': 0.71, 'sigma_p': 0.78, 'taft_es': -1.01, 'sigma_plus': 0.79, 'sigma_minus': 1.27},
    'CN': {'sigma_m': 0.56, 'sigma_p': 0.66, 'taft_es': -0.51, 'sigma_plus': 0.66, 'sigma_minus': 1.00},
    'CF3': {'sigma_m': 0.43, 'sigma_p': 0.54, 'taft_es': -2.40, 'sigma_plus': 0.61, 'sigma_minus': 0.54},
    'COOH': {'sigma_m': 0.37, 'sigma_p': 0.45, 'taft_es': -1.20, 'sigma_plus': 0.42, 'sigma_minus': 0.45},
    'COOCH3': {'sigma_m': 0.35, 'sigma_p': 0.39, 'taft_es': -1.10, 'sigma_plus': 0.32, 'sigma_minus': 0.39},
    'CHO': {'sigma_m': 0.36, 'sigma_p': 0.42, 'taft_es': -1.20, 'sigma_plus': 0.42, 'sigma_minus': 0.42},
    'NH2': {'sigma_m': -0.16, 'sigma_p': -0.66, 'taft_es': -0.20, 'sigma_plus': -1.30, 'sigma_minus': -0.66},
    'N(CH3)2': {'sigma_m': -0.15, 'sigma_p': -0.83, 'taft_es': -0.30, 'sigma_plus': -1.70, 'sigma_minus': -0.83},
    'SO2CH3': {'sigma_m': 0.60, 'sigma_p': 0.72, 'taft_es': -1.50, 'sigma_plus': 0.73, 'sigma_minus': 0.72},
    'B(OH)2': {'sigma_m': 0.04, 'sigma_p': -0.10, 'taft_es': 0.00, 'sigma_plus': -0.10, 'sigma_minus': -0.10},
    'Si(CH3)3': {'sigma_m': -0.04, 'sigma_p': -0.07, 'taft_es': -0.80, 'sigma_plus': -0.07, 'sigma_minus': -0.07},
    'C(CH3)3': {'sigma_m': -0.10, 'sigma_p': -0.20, 'taft_es': -1.54, 'sigma_plus': -0.26, 'sigma_minus': -0.20},
    'C6H5': {'sigma_m': 0.06, 'sigma_p': -0.01, 'taft_es': -1.20, 'sigma_plus': -0.18, 'sigma_minus': -0.01},
}

LIGAND_PROPERTIES_EXTENDED = {
    'triphenylphosphine': {
        'cone_angle': 145.0, 'tep': 2068.9, 'denticity': 1, 
        'class': 'triarylphosphine', 'pka_bh': 2.73, 'softness': 6.2,
        'electronic_donating': 0.45, 'steric_bulk': 1.8, 'tolman_angle': 145.0
    },
    'pph3': {
        'cone_angle': 145.0, 'tep': 2068.9, 'denticity': 1,
        'class': 'triarylphosphine', 'pka_bh': 2.73, 'softness': 6.2,
        'electronic_donating': 0.45, 'steric_bulk': 1.8, 'tolman_angle': 145.0
    },
    'sphos': {
        'cone_angle': 163.0, 'tep': 2064.0, 'denticity': 1,
        'class': 'dialkylbiarylphosphine', 'pka_bh': 7.70, 'softness': 7.8,
        'electronic_donating': 0.65, 'steric_bulk': 2.2, 'tolman_angle': 163.0
    },
    'xphos': {
        'cone_angle': 180.0, 'tep': 2062.4, 'denticity': 1,
        'class': 'dialkylbiarylphosphine', 'pka_bh': 7.90, 'softness': 8.0,
        'electronic_donating': 0.70, 'steric_bulk': 2.5, 'tolman_angle': 180.0
    },
    'dppf': {
        'cone_angle': 150.0, 'tep': 2065.3, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 4.50, 'softness': 7.1,
        'electronic_donating': 0.55, 'steric_bulk': 1.9, 'bite_angle': 99.07, 'tolman_angle': 150.0
    },
    'xantphos': {
        'cone_angle': 120.0, 'tep': 2060.0, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 5.20, 'softness': 7.3,
        'electronic_donating': 0.50, 'steric_bulk': 2.0, 'bite_angle': 110.0, 'tolman_angle': 120.0
    },
    'ipr': {
        'cone_angle': 170.0, 'tep': 2050.0, 'denticity': 1,
        'class': 'nhc', 'pka_bh': 8.50, 'softness': 8.5,
        'electronic_donating': 0.85, 'steric_bulk': 2.8, 'tolman_angle': 170.0
    },
    'imes': {
        'cone_angle': 160.0, 'tep': 2055.0, 'denticity': 1,
        'class': 'nhc', 'pka_bh': 8.20, 'softness': 8.3,
        'electronic_donating': 0.80, 'steric_bulk': 2.5, 'tolman_angle': 160.0
    },
    'sipr': {
        'cone_angle': 165.0, 'tep': 2052.0, 'denticity': 1,
        'class': 'nhc', 'pka_bh': 8.40, 'softness': 8.4,
        'electronic_donating': 0.82, 'steric_bulk': 2.6, 'tolman_angle': 165.0
    },
    'binap': {
        'cone_angle': 130.0, 'tep': 2065.0, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 4.80, 'softness': 6.8,
        'electronic_donating': 0.48, 'steric_bulk': 2.3, 'bite_angle': 90.0, 'tolman_angle': 130.0
    },
    'dppp': {
        'cone_angle': 140.0, 'tep': 2067.0, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 4.60, 'softness': 6.9,
        'electronic_donating': 0.50, 'steric_bulk': 1.7, 'bite_angle': 90.0, 'tolman_angle': 140.0
    },
}

BASE_PROPERTIES = {
    'k2co3': {'pka': 10.3, 'solubility': 0.1, 'cation_radius': 1.38, 'hygroscopic': False, 'class': 'carbonate', 'pkb': 3.7},
    'cs2co3': {'pka': 10.3, 'solubility': 2.6, 'cation_radius': 1.67, 'hygroscopic': True, 'class': 'carbonate', 'pkb': 3.7},
    'na2co3': {'pka': 10.3, 'solubility': 0.2, 'cation_radius': 1.02, 'hygroscopic': False, 'class': 'carbonate', 'pkb': 3.7},
    'k3po4': {'pka': 12.3, 'solubility': 0.5, 'cation_radius': 1.38, 'hygroscopic': True, 'class': 'phosphate', 'pkb': 1.7},
    'naoh': {'pka': 15.7, 'solubility': 1.1, 'cation_radius': 1.02, 'hygroscopic': True, 'class': 'hydroxide', 'pkb': -1.7},
    'koh': {'pka': 15.7, 'solubility': 1.2, 'cation_radius': 1.38, 'hygroscopic': True, 'class': 'hydroxide', 'pkb': -1.7},
    'tea': {'pka': 10.7, 'solubility': 0.8, 'cation_radius': None, 'hygroscopic': False, 'class': 'amine', 'pkb': 3.3},
    'dipea': {'pka': 11.4, 'solubility': 0.6, 'cation_radius': None, 'hygroscopic': False, 'class': 'amine', 'pkb': 2.6},
    'koac': {'pka': 4.8, 'solubility': 0.1, 'cation_radius': 1.38, 'hygroscopic': False, 'class': 'acetate', 'pkb': 9.2},
    'csf': {'pka': 3.2, 'solubility': 0.3, 'cation_radius': 1.67, 'hygroscopic': True, 'class': 'fluoride', 'pkb': 10.8},
    'kf': {'pka': 3.2, 'solubility': 0.2, 'cation_radius': 1.38, 'hygroscopic': True, 'class': 'fluoride', 'pkb': 10.8},
    'k2hpo4': {'pka': 12.3, 'solubility': 0.4, 'cation_radius': 1.38, 'hygroscopic': False, 'class': 'phosphate', 'pkb': 1.7},
    'nahco3': {'pka': 6.4, 'solubility': 0.1, 'cation_radius': 1.02, 'hygroscopic': False, 'class': 'bicarbonate', 'pkb': 7.6},
    'dbu': {'pka': 12.0, 'solubility': 0.5, 'cation_radius': None, 'hygroscopic': False, 'class': 'amidine', 'pkb': 2.0},
    'dabco': {'pka': 8.8, 'solubility': 0.4, 'cation_radius': None, 'hygroscopic': False, 'class': 'amine', 'pkb': 5.2},
    'pyridine': {'pka': 5.2, 'solubility': 0.3, 'cation_radius': None, 'hygroscopic': False, 'class': 'amine', 'pkb': 8.8},
}

SOLVENT_PHYSICS_ADVANCED = {
    'water': {
        'dielectric': 80.1, 'bp_c': 100.0, 'polarity_index': 10.2, 
        'donor_number': 18.0, 'reichardt_et30': 63.1, 
        'hildebrand_delta': 47.8, 'viscosity_cp': 0.89, 
        'acceptor_number': 54.8, 'class': 'protic',
        'alpha': 1.17, 'beta': 0.47, 'pi_star': 1.09
    },
    'methanol': {
        'dielectric': 32.7, 'bp_c': 64.7, 'polarity_index': 5.1,
        'donor_number': 19.0, 'reichardt_et30': 55.5,
        'hildebrand_delta': 29.7, 'viscosity_cp': 0.54,
        'acceptor_number': 41.5, 'class': 'protic',
        'alpha': 0.93, 'beta': 0.62, 'pi_star': 0.60
    },
    'ethanol': {
        'dielectric': 24.6, 'bp_c': 78.4, 'polarity_index': 4.3,
        'donor_number': 19.2, 'reichardt_et30': 51.9,
        'hildebrand_delta': 26.5, 'viscosity_cp': 1.08,
        'acceptor_number': 37.9, 'class': 'protic',
        'alpha': 0.83, 'beta': 0.77, 'pi_star': 0.54
    },
    'isopropanol': {
        'dielectric': 19.9, 'bp_c': 82.3, 'polarity_index': 3.9,
        'donor_number': 18.5, 'reichardt_et30': 48.6,
        'hildebrand_delta': 23.5, 'viscosity_cp': 2.04,
        'acceptor_number': 33.5, 'class': 'protic',
        'alpha': 0.76, 'beta': 0.84, 'pi_star': 0.48
    },
    'acetone': {
        'dielectric': 20.7, 'bp_c': 56.1, 'polarity_index': 5.1,
        'donor_number': 17.0, 'reichardt_et30': 42.2,
        'hildebrand_delta': 19.7, 'viscosity_cp': 0.32,
        'acceptor_number': 12.5, 'class': 'aprotic',
        'alpha': 0.08, 'beta': 0.48, 'pi_star': 0.71
    },
    'acetonitrile': {
        'dielectric': 37.5, 'bp_c': 82.0, 'polarity_index': 5.8,
        'donor_number': 14.1, 'reichardt_et30': 46.0,
        'hildebrand_delta': 24.3, 'viscosity_cp': 0.37,
        'acceptor_number': 18.9, 'class': 'aprotic',
        'alpha': 0.19, 'beta': 0.31, 'pi_star': 0.75
    },
    'dmso': {
        'dielectric': 46.7, 'bp_c': 189.0, 'polarity_index': 7.2,
        'donor_number': 29.8, 'reichardt_et30': 45.1,
        'hildebrand_delta': 26.7, 'viscosity_cp': 1.99,
        'acceptor_number': 19.3, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.76, 'pi_star': 1.00
    },
    'dmf': {
        'dielectric': 36.7, 'bp_c': 153.0, 'polarity_index': 6.4,
        'donor_number': 26.6, 'reichardt_et30': 43.8,
        'hildebrand_delta': 24.9, 'viscosity_cp': 0.82,
        'acceptor_number': 16.0, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88
    },
    'thf': {
        'dielectric': 7.5, 'bp_c': 66.0, 'polarity_index': 4.0,
        'donor_number': 20.0, 'reichardt_et30': 37.4,
        'hildebrand_delta': 18.5, 'viscosity_cp': 0.46,
        'acceptor_number': 8.0, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.55, 'pi_star': 0.58
    },
    'dioxane': {
        'dielectric': 2.2, 'bp_c': 101.0, 'polarity_index': 4.8,
        'donor_number': 14.8, 'reichardt_et30': 36.0,
        'hildebrand_delta': 19.9, 'viscosity_cp': 1.37,
        'acceptor_number': 10.8, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.37, 'pi_star': 0.49
    },
    'toluene': {
        'dielectric': 2.4, 'bp_c': 110.6, 'polarity_index': 2.4,
        'donor_number': 0.1, 'reichardt_et30': 33.9,
        'hildebrand_delta': 18.2, 'viscosity_cp': 0.59,
        'acceptor_number': 3.3, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.11, 'pi_star': 0.54
    },
    'benzene': {
        'dielectric': 2.3, 'bp_c': 80.1, 'polarity_index': 2.7,
        'donor_number': 0.1, 'reichardt_et30': 34.5,
        'hildebrand_delta': 18.6, 'viscosity_cp': 0.60,
        'acceptor_number': 8.2, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.10, 'pi_star': 0.59
    },
    'dichloromethane': {
        'dielectric': 8.9, 'bp_c': 39.6, 'polarity_index': 3.1,
        'donor_number': 0.0, 'reichardt_et30': 41.1,
        'hildebrand_delta': 20.2, 'viscosity_cp': 0.44,
        'acceptor_number': 20.4, 'class': 'aprotic',
        'alpha': 0.13, 'beta': 0.10, 'pi_star': 0.82
    },
    'chloroform': {
        'dielectric': 4.8, 'bp_c': 61.2, 'polarity_index': 4.1,
        'donor_number': 0.0, 'reichardt_et30': 39.1,
        'hildebrand_delta': 19.0, 'viscosity_cp': 0.54,
        'acceptor_number': 23.1, 'class': 'aprotic',
        'alpha': 0.44, 'beta': 0.00, 'pi_star': 0.58
    },
    'hexane': {
        'dielectric': 1.9, 'bp_c': 68.7, 'polarity_index': 0.1,
        'donor_number': 0.0, 'reichardt_et30': 31.0,
        'hildebrand_delta': 14.9, 'viscosity_cp': 0.29,
        'acceptor_number': 0.0, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.00, 'pi_star': 0.00
    },
    'cyclohexane': {
        'dielectric': 2.0, 'bp_c': 80.7, 'polarity_index': 0.2,
        'donor_number': 0.0, 'reichardt_et30': 31.2,
        'hildebrand_delta': 16.7, 'viscosity_cp': 0.89,
        'acceptor_number': 0.0, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.00, 'pi_star': 0.00
    },
    'ethyl acetate': {
        'dielectric': 6.0, 'bp_c': 77.1, 'polarity_index': 4.4,
        'donor_number': 14.0, 'reichardt_et30': 38.1,
        'hildebrand_delta': 18.2, 'viscosity_cp': 0.43,
        'acceptor_number': 9.3, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.45, 'pi_star': 0.55
    },
    'diethyl ether': {
        'dielectric': 4.3, 'bp_c': 34.6, 'polarity_index': 2.8,
        'donor_number': 19.2, 'reichardt_et30': 34.6,
        'hildebrand_delta': 15.4, 'viscosity_cp': 0.22,
        'acceptor_number': 3.9, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.47, 'pi_star': 0.27
    },
    'pyridine': {
        'dielectric': 12.3, 'bp_c': 115.2, 'polarity_index': 5.3,
        'donor_number': 33.1, 'reichardt_et30': 40.2,
        'hildebrand_delta': 21.8, 'viscosity_cp': 0.88,
        'acceptor_number': 14.2, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.64, 'pi_star': 0.87
    },
    'nmp': {
        'dielectric': 32.2, 'bp_c': 202.0, 'polarity_index': 6.7,
        'donor_number': 27.3, 'reichardt_et30': 42.0,
        'hildebrand_delta': 23.1, 'viscosity_cp': 1.67,
        'acceptor_number': 13.6, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.77, 'pi_star': 0.92
    },
    'dme': {
        'dielectric': 7.2, 'bp_c': 85.0, 'polarity_index': 3.5,
        'donor_number': 19.5, 'reichardt_et30': 36.5,
        'hildebrand_delta': 17.6, 'viscosity_cp': 0.46,
        'acceptor_number': 8.5, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.53, 'pi_star': 0.53
    },
}

class AcademicDFTCalculator:
    def __init__(self):
        self._fukui_cache = {}
        self._dft_cache = {}
    
    def calculate_fukui_indices(self, smiles: str) -> Dict[str, float]:
        """
        Calculates Fukui indices.
        f+ = nucleophilic attack (electron acceptance)
        f- = electrophilic attack (electron donation)
        f0 = radical attack
        """
        if smiles in self._fukui_cache:
            return self._fukui_cache[smiles]
        
        result = {'f_plus': 0.0, 'f_minus': 0.0, 'f_zero': 0.0, 'condensed_fukui': {}}
        
        if not RDKIT_AVAILABLE:
            return result
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return result
            
            atom_count = mol.GetNumAtoms()
            
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                degree = atom.GetDegree()
                valence = atom.GetTotalValence()
                
                symbol_factor = {
                    'C': 1.0, 'N': 1.2, 'O': 1.1, 'S': 1.3,
                    'P': 1.1, 'F': 1.4, 'Cl': 1.3, 'Br': 1.2,
                    'I': 1.1, 'B': 0.9, 'Si': 0.8
                }.get(symbol, 1.0)
                
                degree_factor = 1.0 if degree < 2 else (0.8 if degree < 3 else 0.6)
                valence_factor = 1.0 if valence < 4 else 0.7
                
                base_fukui = symbol_factor * degree_factor * valence_factor / (atom_count ** 0.5)
                
                f_plus = base_fukui * (1.0 - 0.1 * degree)
                f_minus = base_fukui * (1.0 + 0.1 * degree)
                f_zero = (f_plus + f_minus) / 2
                
                result['condensed_fukui'][atom.GetIdx()] = {
                    'f_plus': round(f_plus, 4),
                    'f_minus': round(f_minus, 4),
                    'f_zero': round(f_zero, 4)
                }
                
                result['f_plus'] += f_plus
                result['f_minus'] += f_minus
                result['f_zero'] += f_zero
            
            result['f_plus'] = round(result['f_plus'] / atom_count, 4)
            result['f_minus'] = round(result['f_minus'] / atom_count, 4)
            result['f_zero'] = round(result['f_zero'] / atom_count, 4)
            
            self._fukui_cache[smiles] = result
            
        except Exception as e:
            pass
        
        return result
    
    def calculate_homo_lumo(self, smiles: str) -> Dict[str, float]:
        """
        Estimates HOMO-LUMO energies.
        Not a real DFT calculation, but literature-based estimation.
        """
        if smiles in self._dft_cache:
            return self._dft_cache[smiles]
        
        result = {
            'homo': -6.5,
            'lumo': -1.5,
            'gap': 5.0,
            'chemical_potential': -4.0,
            'hardness': 2.5,
            'electronegativity': 4.0,
            'electrophilicity_index': 3.2
        }
        
        if not RDKIT_AVAILABLE:
            return result
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return result
            
            total_atoms = mol.GetNumAtoms()
            heavy_atoms = mol.GetNumHeavyAtoms()
            
            aromatic_rings = 0
            hetero_atoms = 0
            double_bonds = 0
            halogens = 0
            
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in ['N', 'O', 'S', 'P']:
                    hetero_atoms += 1
                if symbol in ['F', 'Cl', 'Br', 'I']:
                    halogens += 1
            
            for bond in mol.GetBonds():
                if bond.GetIsAromatic():
                    aromatic_rings += 1
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    double_bonds += 1
            
            ring_info = mol.GetRingInfo()
            ring_count = ring_info.NumRings()
            
            base_homo = -6.5
            base_lumo = -1.5
            base_gap = 5.0
            
            ring_effect = -0.1 * ring_count
            hetero_effect = -0.05 * hetero_atoms
            halogen_effect = 0.1 * halogens
            double_bond_effect = -0.03 * double_bonds
            
            if heavy_atoms > 10:
                delocalization_effect = -0.02 * (heavy_atoms - 10)
            else:
                delocalization_effect = 0.0
            
            homo = base_homo + ring_effect + hetero_effect + halogen_effect + double_bond_effect + delocalization_effect
            lumo = base_lumo + ring_effect * 0.5 + hetero_effect * 0.3 + halogen_effect * 0.2 + double_bond_effect * 0.5
            
            gap = lumo - homo
            
            result['homo'] = round(homo, 4)
            result['lumo'] = round(lumo, 4)
            result['gap'] = round(gap, 4)
            result['chemical_potential'] = round((homo + lumo) / 2, 4)
            result['hardness'] = round((lumo - homo) / 2, 4)
            result['electronegativity'] = round(-(homo + lumo) / 2, 4)
            
            if result['hardness'] > 0:
                result['electrophilicity_index'] = round((result['chemical_potential'] ** 2) / (2 * result['hardness']), 4)
            
            self._dft_cache[smiles] = result
            
        except Exception as e:
            pass
        
        return result
    
    def calculate_global_reactivity_descriptors(self, smiles: str) -> Dict[str, float]:
        """
        Calculates global reactivity descriptors:
        - Chemical potential (μ)
        - Absolute hardness (η)
        - Electronegativity (χ)
        - Electrophilicity index (ω)
        - Nucleophilicity index (N)
        """
        dft = self.calculate_homo_lumo(smiles)
        
        result = {
            'chemical_potential': dft['chemical_potential'],
            'absolute_hardness': dft['hardness'],
            'absolute_softness': 1 / dft['hardness'] if dft['hardness'] > 0 else 1.0,
            'electronegativity': dft['electronegativity'],
            'electrophilicity_index': dft['electrophilicity_index'],
            'nucleophilicity_index': -dft['chemical_potential'] if dft['chemical_potential'] < 0 else 0,
            'homo_energy': dft['homo'],
            'lumo_energy': dft['lumo'],
            'gap_energy': dft['gap']
        }
        
        return result
    
    def calculate_condensed_fukui_for_molecule(self, smiles: str) -> Dict:
        """
        Calculates condensed Fukui indices for a molecule.
        Used to identify reactive regions.
        """
        return self.calculate_fukui_indices(smiles)

class QSARCalculator:
    """
    QSAR/QSPR calculations:
    - 2D QSAR (topological, electronic, hydrophobic)
    - 3D QSAR (geometric, steric)
    - CoMFA-like field analyses
    """
    
    def __init__(self):
        self._qsar_cache = {}
    
    def calculate_2d_qsar_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculates 2D QSAR descriptors"""
        if smiles in self._qsar_cache:
            return self._qsar_cache[smiles]
        
        descriptors = {}
        
        if not RDKIT_AVAILABLE:
            return descriptors
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return descriptors
            
            descriptors['MW'] = Descriptors.ExactMolWt(mol)
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['MR'] = Descriptors.MolarRefractivity(mol)
            descriptors['HBA'] = Lipinski.NumHAcceptors(mol)
            descriptors['HBD'] = Lipinski.NumHDonors(mol)
            descriptors['RotBonds'] = Lipinski.NumRotatableBonds(mol)
            descriptors['HeavyAtoms'] = mol.GetNumHeavyAtoms()
            descriptors['Rings'] = mol.GetRingInfo().NumRings()
            descriptors['AromaticRings'] = mol.GetRingInfo().NumAromaticRings()
            descriptors['QED'] = QED.qed(mol)
            descriptors['FractionCsp3'] = Descriptors.FractionCsp3(mol)
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            
            descriptors['Kappa1'] = Descriptors.Kappa1(mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(mol)
            
            try:
                descriptors['StericVolume'] = rdMolDescriptors.CalcStericVolume(mol)
            except:
                descriptors['StericVolume'] = 0.0
            
            self._qsar_cache[smiles] = descriptors
            
        except Exception as e:
            pass
        
        return descriptors
    
    def calculate_drug_likeness(self, smiles: str) -> Dict[str, Any]:
        """
        Drug-likeness calculations:
        - Lipinski Rules
        - Ghose Rules
        - Veber Rules
        - QED (Quantitative Estimate of Drug-likeness)
        """
        result = {
            'lipinski_rules': {'passed': False, 'details': {}},
            'ghose_rules': {'passed': False, 'details': {}},
            'veber_rules': {'passed': False, 'details': {}},
            'qed_score': 0.0,
            'drug_likeness_score': 0.0
        }
        
        if not RDKIT_AVAILABLE:
            return result
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return result
            
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Lipinski.NumHDonors(mol)
            hba = Lipinski.NumHAcceptors(mol)
            rot_bonds = Lipinski.NumRotatableBonds(mol)
            tpsa = Descriptors.TPSA(mol)
            
            lipinski = {
                'MW <= 500': mw <= 500,
                'LogP <= 5': logp <= 5,
                'HBD <= 5': hbd <= 5,
                'HBA <= 10': hba <= 10
            }
            result['lipinski_rules']['details'] = lipinski
            result['lipinski_rules']['passed'] = all(lipinski.values())
            
            ghose = {
                '160 <= MW <= 480': 160 <= mw <= 480,
                '-0.4 <= LogP <= 5.6': -0.4 <= logp <= 5.6,
                '20 <= HeavyAtoms <= 70': 20 <= mol.GetNumHeavyAtoms() <= 70,
                'HBA <= 10': hba <= 10,
                'HBD <= 5': hbd <= 5
            }
            result['ghose_rules']['details'] = ghose
            result['ghose_rules']['passed'] = all(ghose.values())
            
            veber = {
                'RotBonds <= 10': rot_bonds <= 10,
                'TPSA <= 140': tpsa <= 140
            }
            result['veber_rules']['details'] = veber
            result['veber_rules']['passed'] = all(veber.values())
            
            result['qed_score'] = QED.qed(mol)
            
            drug_likeness = 0.0
            if result['lipinski_rules']['passed']:
                drug_likeness += 0.35
            if result['ghose_rules']['passed']:
                drug_likeness += 0.30
            if result['veber_rules']['passed']:
                drug_likeness += 0.20
            drug_likeness += result['qed_score'] * 0.15
            
            result['drug_likeness_score'] = round(drug_likeness, 4)
            
        except Exception as e:
            pass
        
        return result

class MolecularDockingCalculator:
    """
    Molecular docking-like calculations:
    - Protein-ligand interactions (estimated)
    - Binding affinity (estimated)
    - Molecular recognition
    """
    
    def __init__(self):
        self._docking_cache = {}
    
    def calculate_binding_affinity(self, smiles: str, target_protein: str = 'pd') -> Dict[str, float]:
        """
        Binding affinity estimation.
        Not real docking, but literature-based estimation.
        """
        if smiles in self._docking_cache:
            return self._docking_cache[smiles]
        
        result = {
            'binding_affinity': -7.5,
            'hydrophobic_contribution': -4.0,
            'electrostatic_contribution': -2.0,
            'hydrogen_bond_contribution': -1.5,
            'entropic_penalty': 0.5,
            'predicted_docking_score': -7.5
        }
        
        if not RDKIT_AVAILABLE:
            return result
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return result
            
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hba = Lipinski.NumHAcceptors(mol)
            hbd = Lipinski.NumHDonors(mol)
            rot_bonds = Lipinski.NumRotatableBonds(mol)
            
            hydrophobic = -0.8 * logp
            electrostatic = -0.3 * (hba + hbd)
            h_bond = -0.5 * min(hba, hbd)
            entropy = 0.1 * rot_bonds
            
            if target_protein == 'pd':
                hydrophobic *= 1.2
                electrostatic *= 1.1
            
            affinity = hydrophobic + electrostatic + h_bond + entropy
            
            result['hydrophobic_contribution'] = round(hydrophobic, 2)
            result['electrostatic_contribution'] = round(electrostatic, 2)
            result['hydrogen_bond_contribution'] = round(h_bond, 2)
            result['entropic_penalty'] = round(entropy, 2)
            result['predicted_docking_score'] = round(affinity, 2)
            result['binding_affinity'] = round(affinity, 2)
            
            self._docking_cache[smiles] = result
            
        except Exception as e:
            pass
        
        return result

PREDICTOR = None
CONFIG = None
DATA_INFO = None
CURRENT_FILE = None
CURRENT_MODEL = 'Ensemble'
MODEL_PERFORMANCE = {}
CACHE = {}
LAST_PREDICTION = None
FEATURE_IMPORTANCE = {}
MODEL_HISTORY = []
PREDICTION_HISTORY = []
CACHE_HIT = 0
CACHE_MISS = 0

DFT_CALCULATOR = AcademicDFTCalculator()
QSAR_CALCULATOR = QSARCalculator()
DOCKING_CALCULATOR = MolecularDockingCalculator()

REQUIRED_COLUMNS = ['yield', 'temp', 'time', 'quantity', 'catalizor', 'base', 'solv1']
OPTIONAL_COLUMNS = ['solv2', 'subs1', 'subs2', 'product']

ACADEMIC_FEATURE_COLUMNS = [
    'subs1_SMILES_logp', 'subs1_SMILES_sigma_p', 'subs1_SMILES_sigma_m', 
    'subs1_SMILES_taft_es', 'subs1_SMILES_hba', 'subs1_SMILES_hbd',
    'subs2_SMILES_logp', 'subs2_SMILES_sigma_p', 'subs2_SMILES_sigma_m',
    'subs2_SMILES_taft_es', 'subs2_SMILES_hba', 'subs2_SMILES_hbd',
    'hsab_overall_compatibility', 'hsab_pd_halide_mismatch',
    'mechanistic_predictor_electronic_softness', 'reaction_rate_indicator'
]

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, np.generic):
        return float(obj) if hasattr(obj, '__float__') else str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, set):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Index):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return obj.total_seconds()
    elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool)):
        try:
            return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
        except:
            return str(obj)
    else:
        return obj

def clean_feature_name(name: str) -> str:
    tr_map = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'İ': 'I', 'Ğ': 'G', 'Ü': 'U', 'Ş': 'S', 'Ö': 'O', 'Ç': 'C',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u'
    }
    for old, new in tr_map.items():
        name = name.replace(old, new)
    
    name = re.sub(r'[^a-zA-Z0-9_.]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    if len(name) > 50:
        parts = name.split('_')
        if len(parts) > 1:
            name = '_'.join(p[:3] + p[-3:] if len(p) > 6 else p for p in parts)
        else:
            name = name[:20] + '_' + name[-10:]
    
    return name if name else 'feature'

def format_size(bytes):
    if bytes == 0:
        return '0 B'
    k = 1024
    sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while bytes >= k and i < len(sizes) - 1:
        bytes /= k
        i += 1
    return f"{bytes:.1f} {sizes[i]}"

def secure_filename(filename):
    import re
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename

def is_enriched_dataset(df: pd.DataFrame) -> bool:
    cols = df.columns.tolist()
    academic_count = sum(1 for col in ACADEMIC_FEATURE_COLUMNS if col in cols)
    if academic_count >= 5:
        return True
    smiles_cols = [c for c in cols if '_SMILES_' in c]
    if len(smiles_cols) >= 10:
        return True
    sigma_cols = [c for c in cols if '_sigma_' in c]
    if len(sigma_cols) >= 4:
        return True
    return False

class Logger:
    def __init__(self):
        self.logs = []
        self.start_time = datetime.now()
        self.levels = {'INFO': 1, 'SUCCESS': 1, 'DEBUG': 0, 'WARNING': 2, 'ERROR': 3}
        self.current_level = 'INFO'
        self.colors = {
            'INFO': '\033[94m',
            'SUCCESS': '\033[92m',
            'DEBUG': '\033[90m',
            'WARNING': '\033[93m',
            'ERROR': '\033[91m'
        }
        self.reset = '\033[0m'
    
    def set_level(self, level: str):
        if level in self.levels:
            self.current_level = level
    
    def log(self, msg: str, level: str = 'INFO'):
        if self.levels.get(level, 1) < self.levels.get(self.current_level, 1):
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        elapsed = (datetime.now() - self.start_time).total_seconds()
        log_entry = {
            'timestamp': timestamp,
            'elapsed': elapsed,
            'level': level,
            'message': msg
        }
        self.logs.append(log_entry)
        
        color = self.colors.get(level, '')
        reset = self.reset
        print(f"{color}[{timestamp}] [{level}] [{elapsed:.1f}s] {msg}{reset}")
        
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]
    
    def info(self, msg: str): self.log(msg, 'INFO')
    def success(self, msg: str): self.log(msg, 'SUCCESS')
    def debug(self, msg: str): self.log(msg, 'DEBUG')
    def warning(self, msg: str): self.log(msg, 'WARNING')
    def error(self, msg: str): self.log(msg, 'ERROR')
    
    def get_logs(self, level: str = None) -> List[Dict]:
        if level:
            return [l for l in self.logs if l['level'] == level]
        return self.logs
    
    def clear(self):
        self.logs = []
    
    def get_summary(self) -> Dict:
        levels = {}
        for log in self.logs:
            levels[log['level']] = levels.get(log['level'], 0) + 1
        return {
            'total': len(self.logs),
            'by_level': levels,
            'start_time': self.start_time.isoformat(),
            'elapsed': (datetime.now() - self.start_time).total_seconds()
        }

logger = Logger()

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                'success': False,
                'message': str(e),
                'traceback': traceback.format_exc() if os.getenv('DEBUG') else None,
                'timestamp': datetime.now().isoformat()
            }), 500
    return wrapper

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper

def cache_result(ttl: int = 300, max_size: int = 100):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global CACHE_HIT, CACHE_MISS
            
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in kwargs.items())
            cache_key = hashlib.md5('_'.join(key_parts).encode()).hexdigest()
            
            if cache_key in CACHE:
                entry = CACHE[cache_key]
                if time.time() - entry['time'] < ttl:
                    CACHE_HIT += 1
                    logger.debug(f"Cache hit: {cache_key[:8]}... (TTL: {ttl}s)")
                    return entry['data']
                else:
                    del CACHE[cache_key]
            
            CACHE_MISS += 1
            result = func(*args, **kwargs)
            
            CACHE[cache_key] = {
                'data': result,
                'time': time.time(),
                'hits': 0
            }
            
            if len(CACHE) > max_size:
                oldest = min(CACHE.keys(), key=lambda k: CACHE[k]['time'])
                del CACHE[oldest]
                logger.debug(f"Cache evicted: {oldest[:8]}...")
            
            return result
        return wrapper
    return decorator

class ConfigManager:
    def __init__(self, config_path: str = 'config/info.xml'):
        self.config_path = config_path
        self.tree = None
        self.root = None
        self._cache = {}
        self.params = {}
        self.raw_xml = ""
        self._xml_hash = None
        self._ensure_dir()
        self._load_or_create()
        self._parse_all()
        self._validate_config()
        self._compute_xml_hash()
        logger.success(f"ConfigManager initialized with FULL XML parse")
        logger.info(f"   Config path: {config_path}")
        logger.info(f"   XML size: {len(self.raw_xml)} bytes")
        logger.info(f"   Parameters loaded: {len(self.params)} top-level sections")
    
    def _ensure_dir(self):
        dir_path = os.path.dirname(self.config_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    def _load_or_create(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.raw_xml = f.read()
                self.tree = ET.parse(self.config_path)
                self.root = self.tree.getroot()
                logger.info(f"Config loaded: {self.config_path}")
            else:
                self._create_default()
                logger.info(f"Default config created: {self.config_path}")
        except ET.ParseError as e:
            logger.error(f"XML Parse error: {e}")
            self._create_default()
        except Exception as e:
            logger.error(f"Config error: {e}")
            self._create_default()
    
    def _create_default(self):
        default = """<?xml version="1.0" encoding="UTF-8"?>
<suzuki_config version="3.1.0">
    <metadata>
        <version>3.1.0</version>
        <last_updated>2026-07-10</last_updated>
        <author>Molytica AI Team</author>
        <description>Ultimate Suzuki-Miyaura coupling predictor with full XML integration + Academic Features</description>
        <dataset_size>15</dataset_size>
        <target_variable>yield</target_variable>
        <target_range_min>5</target_range_min>
        <target_range_max>99</target_range_max>
    </metadata>
    <chemical_intuition>
        <temperature>
            <optimal_temp>85</optimal_temp>
            <temp_range>35</temp_range>
            <min_temp>40</min_temp>
            <max_temp>150</max_temp>
            <degradation_threshold>130</degradation_threshold>
            <too_low_threshold>50</too_low_threshold>
            <temp_coefficient>0.8</temp_coefficient>
            <activation_energy>45.2</activation_energy>
            <arrhenius_prefactor>1.2e12</arrhenius_prefactor>
            <gas_constant>8.314</gas_constant>
            <curve_steepness>0.15</curve_steepness>
            <curve_asymmetry>1.2</curve_asymmetry>
            <low_temp_penalty>0.65</low_temp_penalty>
            <high_temp_penalty>0.40</high_temp_penalty>
            <optimal_temp_bonus>1.15</optimal_temp_bonus>
            <solvent_bp_margin>15</solvent_bp_margin>
            <solvent_bp_penalty>0.85</solvent_bp_penalty>
            <eyring_prefactor>1.0e13</eyring_prefactor>
            <entropy_activation>-20.5</entropy_activation>
            <enthalpy_activation>42.8</enthalpy_activation>
        </temperature>
        <time>
            <optimal_time>18</optimal_time>
            <time_range>12</time_range>
            <min_time>1</min_time>
            <max_time>48</max_time>
            <time_coefficient>1.2</time_coefficient>
            <reaction_half_life>6.5</reaction_half_life>
            <rate_constant>0.107</rate_constant>
            <diffusion_limit>0.85</diffusion_limit>
            <saturation_point>24</saturation_point>
            <plateau_factor>0.92</plateau_factor>
            <diminishing_returns>0.35</diminishing_returns>
            <short_time_penalty>0.40</short_time_penalty>
            <long_time_penalty>0.70</long_time_penalty>
            <optimal_time_bonus>1.10</optimal_time_bonus>
            <reaction_order>1.5</reaction_order>
            <half_life_temperature_dependence>-0.12</half_life_temperature_dependence>
        </time>
        <catalyst>
            <k_m>0.003</k_m>
            <v_max>18</v_max>
            <min_quantity>0.0005</min_quantity>
            <max_quantity>0.08</max_quantity>
            <degradation_threshold>0.04</degradation_threshold>
            <quality_coefficient>0.6</quality_coefficient>
            <turnover_number>1200</turnover_number>
            <turnover_frequency>45.6</turnover_frequency>
            <catalyst_efficiency>0.78</catalyst_efficiency>
            <ligand_pd_ratio>4.0</ligand_pd_ratio>
            <ligand_bite_angle>102</ligand_bite_angle>
            <ligand_electron_donating>0.45</ligand_electron_donating>
            <ligand_steric_bulk>1.8</ligand_steric_bulk>
            <low_quantity_penalty>0.30</low_quantity_penalty>
            <high_quantity_penalty>0.50</high_quantity_penalty>
            <optimal_quantity_bonus>1.20</optimal_quantity_bonus>
            <monodentate_ligand_factor>0.90</monodentate_ligand_factor>
            <bidentate_ligand_factor>1.10</bidentate_ligand_factor>
            <bulky_ligand_factor>0.85</bulky_ligand_factor>
            <electron_rich_ligand_factor>1.15</electron_rich_ligand_factor>
            <pd_oxidation_state>2</pd_oxidation_state>
            <ligand_coordination_number>4</ligand_coordination_number>
        </catalyst>
        <steric>
            <threshold>0.35</threshold>
            <penalty_coefficient>7.5</penalty_coefficient>
            <ring_penalty_factor>0.3</ring_penalty_factor>
            <bulky_group_penalty>1.8</bulky_group_penalty>
            <ortho_substituent_penalty>2.5</ortho_substituent_penalty>
            <meta_substituent_penalty>1.2</meta_substituent_penalty>
            <para_substituent_penalty>0.8</para_substituent_penalty>
            <molecular_volume_threshold>250</molecular_volume_threshold>
            <volume_penalty_factor>0.02</volume_penalty_factor>
            <rotatable_bond_penalty>0.15</rotatable_bond_penalty>
            <rotatable_bond_threshold>6</rotatable_bond_threshold>
            <fused_ring_penalty>0.40</fused_ring_penalty>
            <bridged_ring_penalty>0.60</bridged_ring_penalty>
            <spiro_ring_penalty>0.50</spiro_ring_penalty>
            <heavy_atom_steric_factor>0.08</heavy_atom_steric_factor>
            <halogen_steric_factor>0.15</halogen_steric_factor>
            <a_value_methyl>1.74</a_value_methyl>
            <a_value_ethyl>1.75</a_value_ethyl>
            <a_value_isopropyl>2.21</a_value_isopropyl>
            <a_value_tertbutyl>4.9</a_value_tertbutyl>
        </steric>
        <electronic>
            <logp_coefficient>0.25</logp_coefficient>
            <hbd_penalty>2.0</hbd_penalty>
            <hba_bonus>1.5</hba_bonus>
            <hammett_coefficient>2.8</hammett_coefficient>
            <taft_coefficient>1.5</taft_coefficient>
            <sigma_m_electron_withdrawing>0.65</sigma_m_electron_withdrawing>
            <sigma_p_electron_withdrawing>0.78</sigma_p_electron_withdrawing>
            <sigma_m_electron_donating>-0.25</sigma_m_electron_donating>
            <sigma_p_electron_donating>-0.35</sigma_p_electron_donating>
            <electron_donating_bonus>1.25</electron_donating_bonus>
            <electron_withdrawing_penalty>0.75</electron_withdrawing_penalty>
            <conjugation_effect>1.10</conjugation_effect>
            <inductive_effect>0.95</inductive_effect>
            <resonance_effect>1.15</resonance_effect>
            <polarity_factor>0.12</polarity_factor>
            <solubility_threshold>-2.0</solubility_threshold>
            <solubility_penalty>0.60</solubility_penalty>
            <sigma_plus_coefficient>3.2</sigma_plus_coefficient>
            <sigma_minus_coefficient>2.5</sigma_minus_coefficient>
            <brown_sigma_plus_factor>1.2</brown_sigma_plus_factor>
            <hammett_reaction_constant>1.0</hammett_reaction_constant>
        </electronic>
        <hsab>
            <pd_softness>2.8</pd_softness>
            <halide_softness>3.2</halide_softness>
            <ligand_softness>2.5</ligand_softness>
            <base_softness>3.0</base_softness>
            <soft_soft_bonus>1.20</soft_soft_bonus>
            <hard_hard_bonus>1.10</hard_hard_bonus>
            <soft_hard_penalty>0.70</soft_hard_penalty>
            <mismatch_penalty>0.50</mismatch_penalty>
            <pd_halide_match>0.85</pd_halide_match>
            <pd_ligand_match>0.90</pd_ligand_match>
            <ligand_halide_match>0.75</ligand_halide_match>
            <overall_compatibility>0.80</overall_compatibility>
            <absolute_hardness_pd>3.8</absolute_hardness_pd>
            <absolute_hardness_halide>4.2</absolute_hardness_halide>
            <absolute_hardness_ligand>3.5</absolute_hardness_ligand>
            <pearson_softness_threshold>6.0</pearson_softness_threshold>
            <chemical_potential_pd>-5.2</chemical_potential_pd>
            <electronegativity_pd>5.2</electronegativity_pd>
        </hsab>
        <solvent>
            <dielectric_optimal>25.0</dielectric_optimal>
            <dielectric_range>15.0</dielectric_range>
            <dielectric_weight>0.15</dielectric_weight>
            <donor_optimal>20.0</donor_optimal>
            <donor_range>15.0</donor_range>
            <donor_weight>0.12</donor_weight>
            <polarity_optimal>4.0</polarity_optimal>
            <polarity_range>3.0</polarity_range>
            <polarity_weight>0.10</polarity_weight>
            <aprotic_solvent_bonus>1.10</aprotic_solvent_bonus>
            <protic_solvent_penalty>0.90</protic_solvent_penalty>
            <polar_solvent_bonus>1.05</polar_solvent_bonus>
            <nonpolar_solvent_penalty>0.95</nonpolar_solvent_penalty>
            <alpha_weight>0.08</alpha_weight>
            <beta_weight>0.08</beta_weight>
            <pi_star_weight>0.06</pi_star_weight>
            <reichardt_weight>0.05</reichardt_weight>
            <hildebrand_weight>0.04</hildebrand_weight>
            <solvent_mixtures>
                <toluene_ethanol>1.15</toluene_ethanol>
                <dioxane_water>1.10</dioxane_water>
                <thf_water>1.05</thf_water>
                <dme_water>1.08</dme_water>
            </solvent_mixtures>
        </solvent>
        <base>
            <pka_threshold>18.0</pka_threshold>
            <strong_base_bonus>1.15</strong_base_bonus>
            <weak_base_penalty>0.85</weak_base_penalty>
            <inorganic_base_factor>1.00</inorganic_base_factor>
            <organic_base_factor>0.95</organic_base_factor>
            <carbonate_base_factor>1.10</carbonate_base_factor>
            <phosphate_base_factor>1.05</phosphate_base_factor>
            <soluble_base_bonus>1.08</soluble_base_bonus>
            <insoluble_base_penalty>0.80</insoluble_base_penalty>
            <hygroscopic_base_penalty>0.90</hygroscopic_base_penalty>
            <cation_radius_effect>1.02</cation_radius_effect>
            <pka_effect>0.03</pka_effect>
        </base>
        <yield_parameters>
            <max_yield>98</max_yield>
            <min_yield>5</min_yield>
            <base_yield_offset>45</base_yield_offset>
            <random_variation>0.15</random_variation>
            <reproducibility_factor>0.92</reproducibility_factor>
            <scale_up_factor>0.88</scale_up_factor>
            <batch_variation>0.12</batch_variation>
            <yield_mean>72.5</yield_mean>
            <yield_std>18.3</yield_std>
            <excellent_threshold>85</excellent_threshold>
            <good_threshold>70</good_threshold>
            <moderate_threshold>50</moderate_threshold>
            <poor_threshold>30</poor_threshold>
            <confidence_interval_alpha>0.05</confidence_interval_alpha>
            <prediction_interval_alpha>0.10</prediction_interval_alpha>
        </yield_parameters>
        <mechanistic>
            <oxidative_addition_barrier>28.5</oxidative_addition_barrier>
            <oxidative_addition_rate>0.045</oxidative_addition_rate>
            <oxidative_addition_steric_sensitivity>0.65</oxidative_addition_steric_sensitivity>
            <oxidative_addition_electronic_sensitivity>0.85</oxidative_addition_electronic_sensitivity>
            <transmetalation_barrier>22.3</transmetalation_barrier>
            <transmetalation_rate>0.078</transmetalation_rate>
            <transmetalation_base_sensitivity>0.75</transmetalation_base_sensitivity>
            <transmetalation_boronic_sensitivity>0.70</transmetalation_boronic_sensitivity>
            <reductive_elimination_barrier>18.7</reductive_elimination_barrier>
            <reductive_elimination_rate>0.120</reductive_elimination_rate>
            <reductive_elimination_steric_sensitivity>0.90</reductive_elimination_steric_sensitivity>
            <reductive_elimination_electronic_sensitivity>0.60</reductive_elimination_electronic_sensitivity>
            <oa_weight>0.35</oa_weight>
            <tm_weight>0.35</tm_weight>
            <re_weight>0.30</re_weight>
            <transition_state_asymmetry>1.2</transition_state_asymmetry>
            <reaction_coordinate_step>0.1</reaction_coordinate_step>
            <intermediate_stability_factor>0.8</intermediate_stability_factor>
        </mechanistic>
    </chemical_intuition>
    <model_parameters>
        <Random_Forest><n_estimators>250</n_estimators><max_depth>12</max_depth><min_samples_split>4</min_samples_split><min_samples_leaf>2</min_samples_leaf><max_features>sqrt</max_features><bootstrap>true</bootstrap><oob_score>true</oob_score><random_state>42</random_state><n_jobs>1</n_jobs><ccp_alpha>0.0</ccp_alpha><max_samples>None</max_samples></Random_Forest>
        <Gradient_Boosting><n_estimators>300</n_estimators><max_depth>6</max_depth><min_samples_split>5</min_samples_split><min_samples_leaf>3</min_samples_leaf><learning_rate>0.08</learning_rate><subsample>0.8</subsample><max_features>sqrt</max_features><validation_fraction>0.15</validation_fraction><n_iter_no_change>10</n_iter_no_change><tol>0.001</tol><random_state>42</random_state><init>None</init><loss>squared_error</loss><criterion>friedman_mse</criterion></Gradient_Boosting>
        <Hist_Gradient_Boosting><max_iter>300</max_iter><max_depth>7</max_depth><min_samples_leaf>3</min_samples_leaf><learning_rate>0.1</learning_rate><max_bins>255</max_bins><l2_regularization>0.01</l2_regularization><early_stopping>true</early_stopping><scoring>neg_mean_squared_error</scoring><validation_fraction>0.15</validation_fraction><n_iter_no_change>10</n_iter_no_change><random_state>42</random_state><loss>squared_error</loss><max_leaf_nodes>31</max_leaf_nodes></Hist_Gradient_Boosting>
        <XGBoost><n_estimators>280</n_estimators><max_depth>6</max_depth><learning_rate>0.09</learning_rate><subsample>0.85</subsample><colsample_bytree>0.9</colsample_bytree><colsample_bylevel>0.8</colsample_bylevel><reg_alpha>0.1</reg_alpha><reg_lambda>1.0</reg_lambda><min_child_weight>3</min_child_weight><gamma>0.1</gamma><early_stopping_rounds>10</early_stopping_rounds><random_state>42</random_state><n_jobs>1</n_jobs><objective>reg:squarederror</objective><eval_metric>rmse</eval_metric><booster>gbtree</booster><tree_method>hist</tree_method><grow_policy>lossguide</grow_policy><max_leaves>31</max_leaves></XGBoost>
        <LightGBM><n_estimators>320</n_estimators><max_depth>8</max_depth><num_leaves>31</num_leaves><learning_rate>0.07</learning_rate><subsample>0.8</subsample><colsample_bytree>0.85</colsample_bytree><min_child_samples>5</min_child_samples><reg_alpha>0.1</reg_alpha><reg_lambda>0.1</reg_lambda><min_split_gain>0.01</min_split_gain><early_stopping_rounds>10</early_stopping_rounds><random_state>42</random_state><n_jobs>1</n_jobs><boosting_type>gbdt</boosting_type><objective>regression</objective><metric>rmse</metric><verbose>-1</verbose><bagging_freq>0</bagging_freq><cat_smooth>10.0</cat_smooth><cat_l2>10.0</cat_l2></LightGBM>
        <CatBoost><iterations>300</iterations><depth>6</depth><learning_rate>0.08</learning_rate><l2_leaf_reg>3</l2_leaf_reg><border_count>128</border_count><random_seed>42</random_seed><verbose>false</verbose><loss_function>RMSE</loss_function><eval_metric>RMSE</eval_metric><early_stopping_rounds>10</early_stopping_rounds><od_type>Iter</od_type><od_wait>20</od_wait></CatBoost>
        <Extra_Trees><n_estimators>200</n_estimators><max_depth>10</max_depth><min_samples_split>4</min_samples_split><min_samples_leaf>2</min_samples_leaf><max_features>sqrt</max_features><bootstrap>true</bootstrap><random_state>42</random_state><n_jobs>1</n_jobs><ccp_alpha>0.0</ccp_alpha></Extra_Trees>
        <KNN><n_neighbors>5</n_neighbors><weights>distance</weights><algorithm>auto</algorithm><leaf_size>30</leaf_size><p>2</p><metric>minkowski</metric></KNN>
        <Ridge><alpha>1.0</alpha><fit_intercept>true</fit_intercept><copy_X>true</copy_X><max_iter>None</max_iter><tol>0.001</tol><solver>auto</solver><random_state>42</random_state></Ridge>
        <Lasso><alpha>1.0</alpha><fit_intercept>true</fit_intercept><max_iter>1000</max_iter><tol>0.0001</tol><selection>cyclic</selection><random_state>42</random_state></Lasso>
        <ElasticNet><alpha>1.0</alpha><l1_ratio>0.5</l1_ratio><fit_intercept>true</fit_intercept><max_iter>1000</max_iter><tol>0.0001</tol><selection>cyclic</selection><random_state>42</random_state></ElasticNet>
        <SVR><kernel>rbf</kernel><C>1.2</C><epsilon>0.08</epsilon><gamma>scale</gamma><degree>3</degree><coef0>0.0</coef0><shrinking>true</shrinking><tol>0.001</tol><max_iter>-1</max_iter><cache_size>200</cache_size></SVR>
        <Neural_Network><hidden_layer_sizes>128,64,32</hidden_layer_sizes><activation>relu</activation><solver>adam</solver><alpha>0.001</alpha><learning_rate_init>0.001</learning_rate_init><max_iter>1000</max_iter><tol>0.0001</tol><momentum>0.9</momentum><nesterovs_momentum>true</nesterovs_momentum><early_stopping>true</early_stopping><validation_fraction>0.15</validation_fraction><beta_1>0.9</beta_1><beta_2>0.999</beta_2><epsilon>1e-08</epsilon><n_iter_no_change>10</n_iter_no_change><random_state>42</random_state><warm_start>false</warm_start></Neural_Network>
        <Gaussian_Process><kernel>1.0 * RBF(1.0)</kernel><alpha>1e-10</alpha><optimizer>fmin_l_bfgs_b</optimizer><n_restarts_optimizer>5</n_restarts_optimizer><normalize_y>true</normalize_y><random_state>42</random_state></Gaussian_Process>
        <Ensemble>
            <weights><Random_Forest>0.16</Random_Forest><Gradient_Boosting>0.12</Gradient_Boosting><Hist_Gradient_Boosting>0.16</Hist_Gradient_Boosting><XGBoost>0.12</XGBoost><LightGBM>0.08</LightGBM><CatBoost>0.08</CatBoost><Extra_Trees>0.04</Extra_Trees><SVR>0.02</SVR><Neural_Network>0.03</Neural_Network><Gaussian_Process>0.05</Gaussian_Process><Bayesian_Ridge>0.02</Bayesian_Ridge><PLS>0.01</PLS></weights>
            <stacking>true</stacking>
            <stacking_meta_model>Random_Forest</stacking_meta_model>
            <voting>soft</voting>
        </Ensemble>
    </model_parameters>
    <feature_importance>
        <temperature>0.22</temperature><time>0.16</time><catalyst_quantity>0.14</catalyst_quantity><substrate1_steric>0.09</substrate1_steric><substrate2_steric>0.09</substrate2_steric><solvent_effect>0.08</solvent_effect><base_effect>0.06</base_effect><electronic_effects>0.05</electronic_effects><hsab_effects>0.03</hsab_effects><mechanistic_effects>0.03</mechanistic_effects><hammett_effects>0.03</hammett_effects><taft_effects>0.02</taft_effects>
    </feature_importance>
    <data_processing>
        <missing_values><strategy>median_imputation</strategy><categorical_strategy>mode_imputation</categorical_strategy><threshold>0.30</threshold></missing_values>
        <normalization><numeric_method>standard_scaler</numeric_method><categorical_method>one_hot_encoding</categorical_method><target_scaling>minmax</target_scaling></normalization>
        <feature_selection><method>mutual_information</method><k_best>30</k_best><variance_threshold>0.01</variance_threshold><correlation_threshold>0.85</correlation_threshold></feature_selection>
        <augmentation><enabled>true</enabled><method>gaussian_noise</method><noise_level>0.05</noise_level><n_augmentations>50</n_augmentations><bootstrap_samples>1000</bootstrap_samples></augmentation>
        <split><test_size>0.20</test_size><validation_size>0.15</validation_size><stratify>true</stratify><random_state>42</random_state><shuffle>true</shuffle></split>
    </data_processing>
    <optimization>
        <top_candidates>10</top_candidates>
        <catalyst_search><min_quantity>0.0005</min_quantity><max_quantity>0.08</max_quantity><step_size>0.0005</step_size><n_candidates>10</n_candidates></catalyst_search>
        <grid_search><enabled>true</enabled><n_candidates>50</n_candidates><n_jobs>1</n_jobs><scoring>neg_mean_squared_error</scoring></grid_search>
        <bayesian><enabled>true</enabled><n_iterations>25</n_iterations><n_initial_points>5</n_initial_points><acquisition_function>ei</acquisition_function></bayesian>
        <genetic_algorithm><enabled>false</enabled><population_size>20</population_size><generations>30</generations><mutation_rate>0.1</mutation_rate><crossover_rate>0.8</crossover_rate></genetic_algorithm>
    </optimization>
    <performance_metrics>
        <metrics><r2>true</r2><mae>true</mae><rmse>true</rmse><mape>true</mape><max_error>true</max_error><explained_variance>true</explained_variance><mean_absolute_percentage_error>true</mean_absolute_percentage_error><median_absolute_error>true</median_absolute_error><mean_squared_log_error>true</mean_squared_log_error></metrics>
        <cross_validation><enabled>true</enabled><folds>5</folds><shuffle>true</shuffle><random_state>42</random_state><stratified>true</stratified></cross_validation>
        <learning_curve><enabled>true</enabled><train_sizes>0.1,0.3,0.5,0.7,0.9</train_sizes><n_jobs>1</n_jobs></learning_curve>
        <statistical_tests><anova>true</anova><tukey_hsd>true</tukey_hsd><levene>true</levene><shapiro_wilk>true</shapiro_wilk><kruskal_wallis>true</kruskal_wallis></statistical_tests>
    </performance_metrics>
    <visualization>
        <molecule_images><enabled>true</enabled><image_size>300</image_size><format>png</format><dpi>150</dpi><show_atoms>true</show_atoms><show_bonds>true</show_bonds></molecule_images>
        <plots><feature_importance>true</feature_importance><actual_vs_predicted>true</actual_vs_predicted><residuals>true</residuals><learning_curve>true</learning_curve><parity_plot>true</parity_plot><shap_values>true</shap_values><partial_dependence>true</partial_dependence></plots>
        <colors><primary>#2563EB</primary><secondary>#10B981</secondary><warning>#F59E0B</warning><danger>#EF4444</danger><background>#F8FAFC</background></colors>
    </visualization>
    <logging>
        <log_level>INFO</log_level><log_file>logs/predict_ml.log</log_file><max_log_size>10MB</max_log_size><backup_count>5</backup_count><console_output>true</console_output>
        <error_handling><retry_attempts>3</retry_attempts><retry_delay>1.0</retry_delay><fallback_model>Random_Forest</fallback_model></error_handling>
    </logging>
    <security>
        <file_upload><allowed_extensions>csv</allowed_extensions><max_file_size>50MB</max_file_size><max_files>10</max_files></file_upload>
        <api><rate_limit>100</rate_limit><rate_limit_period>60</rate_limit_period><max_payload_size>1MB</max_payload_size></api>
        <sanitization><strip_xss>true</strip_xss><strip_sql_injection>true</strip_sql_injection><validate_smiles>true</validate_smiles></sanitization>
    </security>
</suzuki_config>"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(default)
        self.tree = ET.parse(self.config_path)
        self.root = self.tree.getroot()
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.raw_xml = f.read()
    
    def _parse_all(self):
        self.params = self._parse_element(self.root)
        logger.debug(f"Parsed {len(self.params)} top-level sections")
        logger.debug(f"Top-level keys: {list(self.params.keys())}")
    
    def _parse_element(self, element) -> Dict:
        result = {}
        for child in element:
            if len(child) > 0:
                result[child.tag] = self._parse_element(child)
            elif child.text:
                val = child.text.strip()
                if val.lower() in ['true', 'false']:
                    val = val.lower() == 'true'
                elif val.lower() in ['none', 'null']:
                    val = None
                elif val.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                    if '.' in val or 'e' in val.lower():
                        try:
                            val = float(val)
                        except:
                            pass
                    else:
                        try:
                            val = int(val)
                        except:
                            pass
                result[child.tag] = val
        return result
    
    def _validate_config(self):
        required = [
            'chemical_intuition/temperature/optimal_temp',
            'chemical_intuition/time/optimal_time',
            'chemical_intuition/catalyst/k_m',
            'chemical_intuition/yield_parameters/max_yield'
        ]
        missing = []
        for path in required:
            val = self.get(path)
            if val is None:
                missing.append(path)
        if missing:
            logger.warning(f"Required parameters missing: {', '.join(missing)}")
        else:
            logger.debug("All required parameters present")
    
    def _compute_xml_hash(self):
        self._xml_hash = hashlib.md5(self.raw_xml.encode()).hexdigest()
        logger.debug(f"XML hash: {self._xml_hash[:8]}...")
    
    def get(self, path: str, default=None):
        try:
            if path in self._cache:
                return self._cache[path]
            keys = path.split('/')
            current = self.params
            for key in keys:
                if key in current:
                    current = current[key]
                else:
                    return default
            self._cache[path] = current
            return current
        except:
            return default
    
    def get_float(self, path: str, default: float = 0.0) -> float:
        try:
            val = self.get(path)
            return float(val) if val is not None else default
        except:
            return default
    
    def get_int(self, path: str, default: int = 0) -> int:
        try:
            val = self.get(path)
            return int(val) if val is not None else default
        except:
            return default
    
    def get_bool(self, path: str, default: bool = False) -> bool:
        try:
            val = self.get(path)
            return bool(val) if val is not None else default
        except:
            return default
    
    def get_dict(self, path: str) -> Dict:
        try:
            val = self.get(path)
            return val if isinstance(val, dict) else {}
        except:
            return {}
    
    def get_model_params(self, model_name: str) -> Dict:
        params = self.get_dict(f'model_parameters/{model_name}')
        if 'n_jobs' in params:
            try:
                if int(params['n_jobs']) < 1:
                    params['n_jobs'] = 1
            except:
                params['n_jobs'] = 1
        return params
    
    def get_chemical_params(self) -> Dict:
        return self.get_dict('chemical_intuition')
    
    def get_feature_importance(self) -> Dict:
        return self.get_dict('feature_importance')
    
    def get_models_list(self) -> List[str]:
        models = self.get_dict('model_parameters')
        return list(models.keys()) if models else []
    
    def reload(self) -> bool:
        try:
            self._cache.clear()
            self._load_or_create()
            self._parse_all()
            self._validate_config()
            self._compute_xml_hash()
            logger.info("Config reloaded")
            return True
        except Exception as e:
            logger.error(f"Reload error: {e}")
            return False
    
    def save(self) -> bool:
        try:
            self.tree.write(self.config_path, encoding='UTF-8', xml_declaration=True)
            self._cache.clear()
            self._parse_all()
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.raw_xml = f.read()
            self._compute_xml_hash()
            logger.success("Config saved successfully")
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False
    
    def get_raw_xml(self) -> str:
        return self.raw_xml
    
    def get_all_params(self) -> Dict:
        return self.params
    
    def get_xml_hash(self) -> str:
        return self._xml_hash
    
    def get_summary(self) -> Dict:
        return {
            'config_path': self.config_path,
            'xml_size': len(self.raw_xml),
            'xml_hash': self._xml_hash[:8] if self._xml_hash else None,
            'top_level_sections': len(self.params),
            'sections': list(self.params.keys()),
            'cache_size': len(self._cache),
            'last_modified': os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else None
        }

class ChemicalCalculator:
    def __init__(self, config: ConfigManager):
        self.config = config
        self._load_all_params()
        self._load_feature_importance()
        self._validate_params()
        
        self.dft_calc = AcademicDFTCalculator()
        self.qsar_calc = QSARCalculator()
        self.docking_calc = MolecularDockingCalculator()
        
        logger.success(f"ChemicalCalculator initialized with FULL XML params + Academic DFT/QSAR/Docking")
        logger.info(f"   Temperature optimal: {self.optimal_temp}C")
        logger.info(f"   Time optimal: {self.optimal_time}h")
        logger.info(f"   Catalyst Km: {self.k_m} mmol")
        logger.info(f"   Max yield: {self.max_yield}%")
    
    def _load_all_params(self):
        chem = self.config.get_chemical_params()
        logger.debug("Loading all chemical parameters from XML...")
        
        temp = chem.get('temperature', {})
        self.optimal_temp = temp.get('optimal_temp', 85)
        self.temp_range = temp.get('temp_range', 35)
        self.min_temp = temp.get('min_temp', 40)
        self.max_temp = temp.get('max_temp', 150)
        self.degradation_threshold = temp.get('degradation_threshold', 130)
        self.too_low_threshold = temp.get('too_low_threshold', 50)
        self.temp_coefficient = temp.get('temp_coefficient', 0.8)
        self.activation_energy = temp.get('activation_energy', 45.2)
        self.arrhenius_prefactor = temp.get('arrhenius_prefactor', 1.2e12)
        self.gas_constant = temp.get('gas_constant', 8.314)
        self.curve_steepness = temp.get('curve_steepness', 0.15)
        self.curve_asymmetry = temp.get('curve_asymmetry', 1.2)
        self.low_temp_penalty = temp.get('low_temp_penalty', 0.65)
        self.high_temp_penalty = temp.get('high_temp_penalty', 0.40)
        self.optimal_temp_bonus = temp.get('optimal_temp_bonus', 1.15)
        self.solvent_bp_margin = temp.get('solvent_bp_margin', 15)
        self.solvent_bp_penalty = temp.get('solvent_bp_penalty', 0.85)
        self.eyring_prefactor = temp.get('eyring_prefactor', 1.0e13)
        self.entropy_activation = temp.get('entropy_activation', -20.5)
        self.enthalpy_activation = temp.get('enthalpy_activation', 42.8)
        logger.debug(f"   Temperature params loaded: optimal={self.optimal_temp}, range={self.temp_range}")
        
        time_p = chem.get('time', {})
        self.optimal_time = time_p.get('optimal_time', 18)
        self.time_range = time_p.get('time_range', 12)
        self.min_time = time_p.get('min_time', 1)
        self.max_time = time_p.get('max_time', 48)
        self.time_coefficient = time_p.get('time_coefficient', 1.2)
        self.reaction_half_life = time_p.get('reaction_half_life', 6.5)
        self.rate_constant = time_p.get('rate_constant', 0.107)
        self.diffusion_limit = time_p.get('diffusion_limit', 0.85)
        self.saturation_point = time_p.get('saturation_point', 24)
        self.plateau_factor = time_p.get('plateau_factor', 0.92)
        self.diminishing_returns = time_p.get('diminishing_returns', 0.35)
        self.short_time_penalty = time_p.get('short_time_penalty', 0.40)
        self.long_time_penalty = time_p.get('long_time_penalty', 0.70)
        self.optimal_time_bonus = time_p.get('optimal_time_bonus', 1.10)
        self.reaction_order = time_p.get('reaction_order', 1.5)
        self.half_life_temperature_dependence = time_p.get('half_life_temperature_dependence', -0.12)
        logger.debug(f"   Time params loaded: optimal={self.optimal_time}, range={self.time_range}")
        
        cat = chem.get('catalyst', {})
        self.k_m = cat.get('k_m', 0.003)
        self.v_max = cat.get('v_max', 18)
        self.min_quantity = cat.get('min_quantity', 0.0005)
        self.max_quantity = cat.get('max_quantity', 0.08)
        self.degradation_threshold_cat = cat.get('degradation_threshold', 0.04)
        self.quality_coefficient = cat.get('quality_coefficient', 0.6)
        self.turnover_number = cat.get('turnover_number', 1200)
        self.turnover_frequency = cat.get('turnover_frequency', 45.6)
        self.catalyst_efficiency_xml = cat.get('catalyst_efficiency', 0.78)
        self.ligand_pd_ratio = cat.get('ligand_pd_ratio', 4.0)
        self.ligand_bite_angle = cat.get('ligand_bite_angle', 102)
        self.ligand_electron_donating = cat.get('ligand_electron_donating', 0.45)
        self.ligand_steric_bulk = cat.get('ligand_steric_bulk', 1.8)
        self.low_quantity_penalty = cat.get('low_quantity_penalty', 0.30)
        self.high_quantity_penalty = cat.get('high_quantity_penalty', 0.50)
        self.optimal_quantity_bonus = cat.get('optimal_quantity_bonus', 1.20)
        self.monodentate_ligand_factor = cat.get('monodentate_ligand_factor', 0.90)
        self.bidentate_ligand_factor = cat.get('bidentate_ligand_factor', 1.10)
        self.bulky_ligand_factor = cat.get('bulky_ligand_factor', 0.85)
        self.electron_rich_ligand_factor = cat.get('electron_rich_ligand_factor', 1.15)
        self.pd_sources = cat.get('pd_sources', {})
        self.pd_oxidation_state = cat.get('pd_oxidation_state', 2)
        self.ligand_coordination_number = cat.get('ligand_coordination_number', 4)
        logger.debug(f"   Catalyst params loaded: Km={self.k_m}, Vmax={self.v_max}")
        
        ster = chem.get('steric', {})
        self.steric_threshold = ster.get('threshold', 0.35)
        self.steric_penalty = ster.get('penalty_coefficient', 7.5)
        self.ring_penalty = ster.get('ring_penalty_factor', 0.3)
        self.bulky_penalty = ster.get('bulky_group_penalty', 1.8)
        self.ortho_penalty = ster.get('ortho_substituent_penalty', 2.5)
        self.meta_penalty = ster.get('meta_substituent_penalty', 1.2)
        self.para_penalty = ster.get('para_substituent_penalty', 0.8)
        self.molecular_volume_threshold = ster.get('molecular_volume_threshold', 250)
        self.volume_penalty_factor = ster.get('volume_penalty_factor', 0.02)
        self.rotatable_bond_penalty = ster.get('rotatable_bond_penalty', 0.15)
        self.rotatable_bond_threshold = ster.get('rotatable_bond_threshold', 6)
        self.fused_ring_penalty = ster.get('fused_ring_penalty', 0.40)
        self.bridged_ring_penalty = ster.get('bridged_ring_penalty', 0.60)
        self.spiro_ring_penalty = ster.get('spiro_ring_penalty', 0.50)
        self.heavy_atom_steric_factor = ster.get('heavy_atom_steric_factor', 0.08)
        self.halogen_steric_factor = ster.get('halogen_steric_factor', 0.15)
        self.a_value_methyl = ster.get('a_value_methyl', 1.74)
        self.a_value_ethyl = ster.get('a_value_ethyl', 1.75)
        self.a_value_isopropyl = ster.get('a_value_isopropyl', 2.21)
        self.a_value_tertbutyl = ster.get('a_value_tertbutyl', 4.9)
        logger.debug(f"   Steric params loaded: threshold={self.steric_threshold}, penalty={self.steric_penalty}")
        
        elec = chem.get('electronic', {})
        self.logp_coefficient = elec.get('logp_coefficient', 0.25)
        self.hbd_penalty = elec.get('hbd_penalty', 2.0)
        self.hba_bonus = elec.get('hba_bonus', 1.5)
        self.hammett_coeff = elec.get('hammett_coefficient', 2.8)
        self.taft_coeff = elec.get('taft_coefficient', 1.5)
        self.sigma_m_ew = elec.get('sigma_m_electron_withdrawing', 0.65)
        self.sigma_p_ew = elec.get('sigma_p_electron_withdrawing', 0.78)
        self.sigma_m_ed = elec.get('sigma_m_electron_donating', -0.25)
        self.sigma_p_ed = elec.get('sigma_p_electron_donating', -0.35)
        self.electron_donating_bonus = elec.get('electron_donating_bonus', 1.25)
        self.electron_withdrawing_penalty = elec.get('electron_withdrawing_penalty', 0.75)
        self.conjugation_effect = elec.get('conjugation_effect', 1.10)
        self.inductive_effect = elec.get('inductive_effect', 0.95)
        self.resonance_effect = elec.get('resonance_effect', 1.15)
        self.polarity_factor = elec.get('polarity_factor', 0.12)
        self.solubility_threshold = elec.get('solubility_threshold', -2.0)
        self.solubility_penalty = elec.get('solubility_penalty', 0.60)
        self.sigma_plus_coeff = elec.get('sigma_plus_coefficient', 3.2)
        self.sigma_minus_coeff = elec.get('sigma_minus_coefficient', 2.5)
        self.brown_sigma_plus_factor = elec.get('brown_sigma_plus_factor', 1.2)
        self.hammett_reaction_constant = elec.get('hammett_reaction_constant', 1.0)
        logger.debug(f"   Electronic params loaded: Hammett={self.hammett_coeff}, Taft={self.taft_coeff}")
        
        hsab = chem.get('hsab', {})
        self.pd_softness = hsab.get('pd_softness', 2.8)
        self.halide_softness = hsab.get('halide_softness', 3.2)
        self.ligand_softness = hsab.get('ligand_softness', 2.5)
        self.base_softness = hsab.get('base_softness', 3.0)
        self.soft_soft_bonus = hsab.get('soft_soft_bonus', 1.20)
        self.hard_hard_bonus = hsab.get('hard_hard_bonus', 1.10)
        self.soft_hard_penalty = hsab.get('soft_hard_penalty', 0.70)
        self.mismatch_penalty = hsab.get('mismatch_penalty', 0.50)
        self.pd_halide_match_xml = hsab.get('pd_halide_match', 0.85)
        self.pd_ligand_match_xml = hsab.get('pd_ligand_match', 0.90)
        self.ligand_halide_match_xml = hsab.get('ligand_halide_match', 0.75)
        self.overall_compatibility_xml = hsab.get('overall_compatibility', 0.80)
        self.absolute_hardness_pd = hsab.get('absolute_hardness_pd', 3.8)
        self.absolute_hardness_halide = hsab.get('absolute_hardness_halide', 4.2)
        self.absolute_hardness_ligand = hsab.get('absolute_hardness_ligand', 3.5)
        self.pearson_softness_threshold = hsab.get('pearson_softness_threshold', 6.0)
        self.chemical_potential_pd = hsab.get('chemical_potential_pd', -5.2)
        self.electronegativity_pd = hsab.get('electronegativity_pd', 5.2)
        logger.debug(f"   HSAB params loaded: Pd={self.pd_softness}, halide={self.halide_softness}")
        
        solv = chem.get('solvent', {})
        self.dielectric_optimal = solv.get('dielectric_optimal', 25.0)
        self.dielectric_range = solv.get('dielectric_range', 15.0)
        self.dielectric_weight = solv.get('dielectric_weight', 0.15)
        self.donor_optimal = solv.get('donor_optimal', 20.0)
        self.donor_range = solv.get('donor_range', 15.0)
        self.donor_weight = solv.get('donor_weight', 0.12)
        self.polarity_optimal = solv.get('polarity_optimal', 4.0)
        self.polarity_range = solv.get('polarity_range', 3.0)
        self.polarity_weight = solv.get('polarity_weight', 0.10)
        self.aprotic_solvent_bonus = solv.get('aprotic_solvent_bonus', 1.10)
        self.protic_solvent_penalty = solv.get('protic_solvent_penalty', 0.90)
        self.polar_solvent_bonus = solv.get('polar_solvent_bonus', 1.05)
        self.nonpolar_solvent_penalty = solv.get('nonpolar_solvent_penalty', 0.95)
        self.alpha_weight = solv.get('alpha_weight', 0.08)
        self.beta_weight = solv.get('beta_weight', 0.08)
        self.pi_star_weight = solv.get('pi_star_weight', 0.06)
        self.reichardt_weight = solv.get('reichardt_weight', 0.05)
        self.hildebrand_weight = solv.get('hildebrand_weight', 0.04)
        
        mixtures = solv.get('solvent_mixtures', {})
        self.toluene_ethanol = mixtures.get('toluene_ethanol', 1.15)
        self.dioxane_water = mixtures.get('dioxane_water', 1.10)
        self.thf_water = mixtures.get('thf_water', 1.05)
        self.dme_water = mixtures.get('dme_water', 1.08)
        logger.debug(f"   Solvent params loaded: dielectric={self.dielectric_optimal}, donor={self.donor_optimal}")
        
        base_p = chem.get('base', {})
        self.pka_threshold = base_p.get('pka_threshold', 18.0)
        self.strong_base_bonus = base_p.get('strong_base_bonus', 1.15)
        self.weak_base_penalty = base_p.get('weak_base_penalty', 0.85)
        self.inorganic_base_factor = base_p.get('inorganic_base_factor', 1.00)
        self.organic_base_factor = base_p.get('organic_base_factor', 0.95)
        self.carbonate_base_factor = base_p.get('carbonate_base_factor', 1.10)
        self.phosphate_base_factor = base_p.get('phosphate_base_factor', 1.05)
        self.soluble_base_bonus = base_p.get('soluble_base_bonus', 1.08)
        self.insoluble_base_penalty = base_p.get('insoluble_base_penalty', 0.80)
        self.hygroscopic_base_penalty = base_p.get('hygroscopic_base_penalty', 0.90)
        self.cation_radius_effect = base_p.get('cation_radius_effect', 1.02)
        self.pka_effect = base_p.get('pka_effect', 0.03)
        logger.debug(f"   Base params loaded: pKa threshold={self.pka_threshold}")
        
        yield_p = chem.get('yield_parameters', {})
        self.max_yield = yield_p.get('max_yield', 98)
        self.min_yield = yield_p.get('min_yield', 5)
        self.base_yield_offset = yield_p.get('base_yield_offset', 45)
        self.random_variation = yield_p.get('random_variation', 0.15)
        self.reproducibility = yield_p.get('reproducibility_factor', 0.92)
        self.scale_up_factor = yield_p.get('scale_up_factor', 0.88)
        self.batch_variation = yield_p.get('batch_variation', 0.12)
        self.yield_mean = yield_p.get('yield_mean', 72.5)
        self.yield_std = yield_p.get('yield_std', 18.3)
        self.excellent_threshold = yield_p.get('excellent_threshold', 85)
        self.good_threshold = yield_p.get('good_threshold', 70)
        self.moderate_threshold = yield_p.get('moderate_threshold', 50)
        self.poor_threshold = yield_p.get('poor_threshold', 30)
        self.confidence_interval_alpha = yield_p.get('confidence_interval_alpha', 0.05)
        self.prediction_interval_alpha = yield_p.get('prediction_interval_alpha', 0.10)
        logger.debug(f"   Yield params loaded: max={self.max_yield}, min={self.min_yield}")
        
        mech = chem.get('mechanistic', {})
        self.oa_barrier = mech.get('oxidative_addition_barrier', 28.5)
        self.oa_rate = mech.get('oxidative_addition_rate', 0.045)
        self.oa_steric_sens = mech.get('oxidative_addition_steric_sensitivity', 0.65)
        self.oa_electronic_sens = mech.get('oxidative_addition_electronic_sensitivity', 0.85)
        self.tm_barrier = mech.get('transmetalation_barrier', 22.3)
        self.tm_rate = mech.get('transmetalation_rate', 0.078)
        self.tm_base_sens = mech.get('transmetalation_base_sensitivity', 0.75)
        self.tm_boronic_sens = mech.get('transmetalation_boronic_sensitivity', 0.70)
        self.re_barrier = mech.get('reductive_elimination_barrier', 18.7)
        self.re_rate = mech.get('reductive_elimination_rate', 0.120)
        self.re_steric_sens = mech.get('reductive_elimination_steric_sensitivity', 0.90)
        self.re_electronic_sens = mech.get('reductive_elimination_electronic_sensitivity', 0.60)
        self.oa_weight = mech.get('oa_weight', 0.35)
        self.tm_weight = mech.get('tm_weight', 0.35)
        self.re_weight = mech.get('re_weight', 0.30)
        self.transition_state_asymmetry = mech.get('transition_state_asymmetry', 1.2)
        self.reaction_coordinate_step = mech.get('reaction_coordinate_step', 0.1)
        self.intermediate_stability_factor = mech.get('intermediate_stability_factor', 0.8)
        logger.debug(f"   Mechanistic params loaded: OA={self.oa_barrier}, TM={self.tm_barrier}, RE={self.re_barrier}")
        
        logger.info("All 200+ academic parameters loaded from XML")
    
    def _load_feature_importance(self):
        fi = self.config.get_feature_importance()
        self.temp_weight = fi.get('temperature', 0.22)
        self.time_weight = fi.get('time', 0.16)
        self.catalyst_weight = fi.get('catalyst_quantity', 0.14)
        self.substrate1_steric_weight = fi.get('substrate1_steric', 0.09)
        self.substrate2_steric_weight = fi.get('substrate2_steric', 0.09)
        self.solvent_weight = fi.get('solvent_effect', 0.08)
        self.base_weight = fi.get('base_effect', 0.06)
        self.electronic_weight = fi.get('electronic_effects', 0.05)
        self.hsab_weight = fi.get('hsab_effects', 0.03)
        self.mechanistic_weight = fi.get('mechanistic_effects', 0.03)
        self.hammett_weight = fi.get('hammett_effects', 0.03)
        self.taft_weight = fi.get('taft_effects', 0.02)
        logger.debug(f"Feature importance weights: temp={self.temp_weight}, time={self.time_weight}, catalyst={self.catalyst_weight}")
    
    def _validate_params(self):
        if self.optimal_temp < self.min_temp or self.optimal_temp > self.max_temp:
            logger.warning(f"Optimal temp ({self.optimal_temp}) outside min-max range")
        if self.optimal_time < self.min_time or self.optimal_time > self.max_time:
            logger.warning(f"Optimal time ({self.optimal_time}) outside min-max range")
        if self.k_m <= 0:
            logger.warning(f"Km must be positive: {self.k_m}")
        if self.min_yield >= self.max_yield:
            logger.warning(f"Min yield ({self.min_yield}) >= max yield ({self.max_yield})")
        total_weight = (self.temp_weight + self.time_weight + self.catalyst_weight + self.substrate1_steric_weight + self.substrate2_steric_weight + self.solvent_weight + self.base_weight + self.electronic_weight + self.hsab_weight + self.mechanistic_weight + self.hammett_weight + self.taft_weight)
        if abs(total_weight - 1.0) > 0.05:
            logger.warning(f"Feature importance weights sum to {total_weight:.2f}, not 1.0")
    
    def calculate_dft_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate DFT-like global reactivity descriptors"""
        return self.dft_calc.calculate_global_reactivity_descriptors(smiles)
    
    def calculate_fukui_indices(self, smiles: str) -> Dict[str, float]:
        """Calculate Fukui indices"""
        return self.dft_calc.calculate_fukui_indices(smiles)
    
    def calculate_qsar_descriptors(self, smiles: str) -> Dict[str, float]:
        """Calculate 2D QSAR descriptors"""
        return self.qsar_calc.calculate_2d_qsar_descriptors(smiles)
    
    def calculate_drug_likeness(self, smiles: str) -> Dict[str, Any]:
        """Calculate drug-likeness"""
        return self.qsar_calc.calculate_drug_likeness(smiles)
    
    def calculate_docking_score(self, smiles: str, target: str = 'pd') -> Dict[str, float]:
        """Calculate molecular docking-like binding affinity"""
        return self.docking_calc.calculate_binding_affinity(smiles, target)
    
    def calculate_hsab_absolute_hardness(self, ip: float, ea: float) -> float:
        """Pearson absolute hardness (η = IP - EA)"""
        return ip - ea
    
    def calculate_hsab_chemical_potential(self, ip: float, ea: float) -> float:
        """Chemical potential (μ = -(IP + EA)/2)"""
        return -(ip + ea) / 2
    
    def calculate_hsab_electronegativity(self, ip: float, ea: float) -> float:
        """Electronegativity (χ = (IP + EA)/2)"""
        return (ip + ea) / 2
    
    def calculate_solvent_kamlet_taft(self, alpha: float, beta: float, pi_star: float) -> float:
        """Kamlet-Taft solvent parameter"""
        return alpha * 0.3 + beta * 0.3 + pi_star * 0.4
    
    def calculate_steric_a_value(self, substituent: str) -> float:
        """Calculate A-value (steric effect)"""
        a_values = {
            'methyl': self.a_value_methyl,
            'ethyl': self.a_value_ethyl,
            'isopropyl': self.a_value_isopropyl,
            'tertbutyl': self.a_value_tertbutyl
        }
        return a_values.get(substituent.lower(), 1.74)
    
    def calculate_hammett_sigma_plus(self, substituent: str) -> float:
        """Hammett σ⁺ value (Brown σ⁺)"""
        sigma_plus = HAMMETT_SIGMA.get(substituent, {}).get('sigma_plus', 0)
        return sigma_plus * self.sigma_plus_coeff
    
    def calculate_hammett_sigma_minus(self, substituent: str) -> float:
        """Hammett σ⁻ value"""
        sigma_minus = HAMMETT_SIGMA.get(substituent, {}).get('sigma_minus', 0)
        return sigma_minus * self.sigma_minus_coeff
    
    def calculate_taft_steric_parameter(self, substituent: str) -> float:
        """Taft steric parameter (Eₛ)"""
        return HAMMETT_SIGMA.get(substituent, {}).get('taft_es', 0)
    
    def calculate_eyring_rate(self, temp: float, delta_h: float, delta_s: float) -> float:
        """Calculate reaction rate using Eyring equation"""
        T = temp + 273.15
        return (boltzmann_k / h) * T * np.exp(-(delta_h * 1000) / (R * T)) * np.exp(delta_s / R)
    
    def calculate_gibbs_energy(self, temp: float, delta_h: float, delta_s: float) -> float:
        """Calculate Gibbs free energy"""
        T = temp + 273.15
        return delta_h - T * delta_s / 1000
    
    def calculate_equilibrium_constant(self, temp: float, delta_g: float) -> float:
        """Calculate equilibrium constant"""
        T = temp + 273.15
        return np.exp(-delta_g * 1000 / (R * T))
    
    def calculate_lfer(self, sigma: float, rho: float) -> float:
        """Linear Free Energy Relationship (LFER)"""
        return np.exp(rho * sigma)
    
    def temperature_factor(self, temp: float) -> float:
        if temp < self.min_temp:
            return self.low_temp_penalty
        if temp > self.max_temp:
            return self.high_temp_penalty
        if temp > self.degradation_threshold:
            return 0.3
        if temp < self.too_low_threshold:
            return 0.4
        
        if temp < self.optimal_temp:
            sigma = self.temp_range / 3 / self.curve_asymmetry
        else:
            sigma = self.temp_range / 3 * self.curve_asymmetry
        
        deviation = abs(temp - self.optimal_temp)
        factor = np.exp(-(deviation ** 2) / (2 * sigma ** 2))
        
        if temp > 0:
            R_val = self.gas_constant
            T_opt = self.optimal_temp + 273.15
            T_curr = temp + 273.15
            arrhenius = np.exp(-self.activation_energy * 1000 / R_val * (1/T_curr - 1/T_opt))
            factor = factor * arrhenius
        
        if abs(temp - self.optimal_temp) < 5:
            factor = factor * self.optimal_temp_bonus
        
        eyring_rate = self.calculate_eyring_rate(temp, self.enthalpy_activation, self.entropy_activation)
        eyring_rate_opt = self.calculate_eyring_rate(self.optimal_temp, self.enthalpy_activation, self.entropy_activation)
        eyring_factor = eyring_rate / eyring_rate_opt if eyring_rate_opt > 0 else 1.0
        factor = factor * (0.7 + 0.3 * eyring_factor)
        
        result = np.clip(factor * 1.2, 0.1, 1.3)
        return result
    
    def time_factor(self, time_hours: float) -> float:
        if time_hours < self.min_time:
            return self.short_time_penalty
        if time_hours > self.max_time:
            return self.long_time_penalty
        
        factor = 1 - np.exp(-self.rate_constant * time_hours)
        
        if factor > self.diffusion_limit:
            factor = factor * (1 - self.diminishing_returns * (factor - self.diffusion_limit))
        
        if time_hours > self.saturation_point:
            factor = factor * self.plateau_factor
        
        if abs(time_hours - self.optimal_time) < 2:
            factor = factor * self.optimal_time_bonus
        
        reaction_order_factor = time_hours ** (1 / self.reaction_order)
        factor = factor * (0.7 + 0.3 * reaction_order_factor / (self.optimal_time ** (1 / self.reaction_order)))
        
        half_life = self.reaction_half_life * np.exp(self.half_life_temperature_dependence * 0)
        half_life_factor = 1 - np.exp(-np.log(2) * time_hours / half_life)
        factor = factor * (0.85 + 0.15 * half_life_factor)
        
        result = np.clip(factor * 1.5, 0.1, 1.2)
        return result
    
    def catalyst_factor(self, quantity: float) -> float:
        if quantity < self.min_quantity:
            return self.low_quantity_penalty
        if quantity > self.max_quantity:
            return self.high_quantity_penalty
        if quantity > self.degradation_threshold_cat:
            return 0.3
        
        rate = self.v_max * quantity / (self.k_m + quantity)
        factor = rate / self.v_max * self.quality_coefficient
        
        ton_factor = min(1.0, quantity * self.turnover_number / 100)
        factor = factor * (0.8 + 0.2 * ton_factor)
        
        ligand_factor = 1.0
        if quantity > 0.01:
            ligand_factor = self.bidentate_ligand_factor
        elif quantity < 0.001:
            ligand_factor = self.monodentate_ligand_factor
        
        factor = factor * ligand_factor
        
        if abs(quantity - 0.005) < 0.001:
            factor = factor * self.optimal_quantity_bonus
        
        mm_factor = quantity / (self.k_m + quantity)
        factor = factor * (0.8 + 0.2 * mm_factor / (0.005 / (self.k_m + 0.005)))
        
        result = np.clip(factor, 0.1, 1.3)
        return result
    
    def steric_factor(self, conditions: Dict) -> float:
        ring_count = conditions.get('ring_count', 0)
        bulky_groups = conditions.get('bulky_groups', 0)
        rotatable_bonds = conditions.get('rotatable_bonds', 0)
        ortho_sub = conditions.get('ortho_substituents', 0)
        meta_sub = conditions.get('meta_substituents', 0)
        para_sub = conditions.get('para_substituents', 0)
        heavy_atoms = conditions.get('heavy_atoms', 0)
        halogens = conditions.get('halogen_count', 0)
        spiro_atoms = conditions.get('spiro_atoms', 0)
        
        ring_penalty = ring_count * self.ring_penalty
        fused_penalty = (ring_count - 1) * self.fused_ring_penalty if ring_count > 1 else 0
        bridged_penalty = (ring_count - 2) * self.bridged_ring_penalty if ring_count > 2 else 0
        spiro_penalty = spiro_atoms * self.spiro_ring_penalty
        bulky_penalty = bulky_groups * self.bulky_penalty
        sub_penalty = (ortho_sub * self.ortho_penalty + meta_sub * self.meta_penalty + para_sub * self.para_penalty)
        rot_penalty = max(0, (rotatable_bonds - self.rotatable_bond_threshold)) * self.rotatable_bond_penalty
        heavy_penalty = heavy_atoms * self.heavy_atom_steric_factor
        halogen_penalty = halogens * self.halogen_steric_factor
        volume_penalty = (heavy_atoms - self.molecular_volume_threshold / 10) * self.volume_penalty_factor if heavy_atoms > self.molecular_volume_threshold / 10 else 0
        
        a_value_factor = 1.0
        if bulky_groups >= 4:
            a_value_factor = self.a_value_tertbutyl / self.a_value_methyl
        elif bulky_groups >= 3:
            a_value_factor = self.a_value_isopropyl / self.a_value_methyl
        elif bulky_groups >= 2:
            a_value_factor = self.a_value_ethyl / self.a_value_methyl
        
        total_penalty = (ring_penalty + fused_penalty + bridged_penalty + spiro_penalty +
                        bulky_penalty + sub_penalty + rot_penalty + 
                        heavy_penalty + halogen_penalty + volume_penalty) * a_value_factor
        
        if total_penalty > self.steric_threshold:
            factor = 1 - self.steric_penalty * total_penalty
        else:
            factor = 1 - 0.5 * self.steric_penalty * total_penalty
        
        result = np.clip(factor, 0.1, 1.0)
        return result
    
    def electronic_factor(self, conditions: Dict) -> float:
        logp = conditions.get('logp', 0)
        hba = conditions.get('hba', 0)
        hbd = conditions.get('hbd', 0)
        sigma_m = conditions.get('sigma_m', 0)
        sigma_p = conditions.get('sigma_p', 0)
        sigma_plus = conditions.get('sigma_plus', sigma_p)
        sigma_minus = conditions.get('sigma_minus', sigma_p)
        es = conditions.get('taft_es', 0)
        
        factor = 1.0
        
        if logp > 0:
            factor += self.logp_coefficient * min(logp, 5)
        else:
            factor += self.logp_coefficient * max(logp, -2) * 0.5
        
        factor += self.hba_bonus * hba / 10
        factor -= self.hbd_penalty * hbd / 10
        
        sigma_total = sigma_m + sigma_p
        if sigma_total > 0:
            sigma_effect = (sigma_m / (abs(sigma_m) + 0.001) * self.sigma_m_ew + 
                          sigma_p / (abs(sigma_p) + 0.001) * self.sigma_p_ew) / 2
            factor += self.hammett_coeff * sigma_effect * self.electron_withdrawing_penalty
        else:
            sigma_effect = (sigma_m / (abs(sigma_m) + 0.001) * self.sigma_m_ed + 
                          sigma_p / (abs(sigma_p) + 0.001) * self.sigma_p_ed) / 2
            factor += self.hammett_coeff * sigma_effect * self.electron_donating_bonus
        
        sigma_plus_effect = self.sigma_plus_coeff * sigma_plus
        sigma_minus_effect = self.sigma_minus_coeff * sigma_minus
        factor += 0.1 * (sigma_plus_effect + sigma_minus_effect)
        
        brown_factor = self.brown_sigma_plus_factor * sigma_plus
        factor += 0.05 * brown_factor
        
        factor += self.taft_coeff * es / 2
        
        lfer_factor = self.calculate_lfer(sigma_p, self.hammett_reaction_constant)
        factor = factor * (0.8 + 0.2 * lfer_factor)
        
        if 'conjugation' in conditions:
            factor += self.conjugation_effect * conditions.get('conjugation', 0)
        if 'inductive' in conditions:
            factor += self.inductive_effect * conditions.get('inductive', 0)
        if 'resonance' in conditions:
            factor += self.resonance_effect * conditions.get('resonance', 0)
        
        if logp < self.solubility_threshold:
            factor *= self.solubility_penalty
        
        result = np.clip(factor, 0.1, 1.5)
        return result
    
    def hsab_factor(self, conditions: Dict) -> float:
        pd_soft = self.pd_softness
        halide_soft = self.halide_softness
        ligand_soft = self.ligand_softness
        base_soft = self.base_softness
        
        pd_halide_match_soft = 1 - abs(pd_soft - halide_soft) / 6
        pd_ligand_match_soft = 1 - abs(pd_soft - ligand_soft) / 6
        ligand_halide_match_soft = 1 - abs(ligand_soft - halide_soft) / 6
        
        pd_hard = self.absolute_hardness_pd
        halide_hard = self.absolute_hardness_halide
        ligand_hard = self.absolute_hardness_ligand
        
        pd_halide_match_hard = 1 - abs(pd_hard - halide_hard) / 8
        pd_ligand_match_hard = 1 - abs(pd_hard - ligand_hard) / 8
        ligand_halide_match_hard = 1 - abs(ligand_hard - halide_hard) / 8
        
        w1 = self.pd_halide_match_xml
        w2 = self.pd_ligand_match_xml
        w3 = self.ligand_halide_match_xml
        
        overall_soft = (pd_halide_match_soft * w1 + pd_ligand_match_soft * w2 + ligand_halide_match_soft * w3) / (w1 + w2 + w3)
        overall_hard = (pd_halide_match_hard * w1 + pd_ligand_match_hard * w2 + ligand_halide_match_hard * w3) / (w1 + w2 + w3)
        
        overall = overall_soft * 0.6 + overall_hard * 0.4
        
        chem_pot_effect = np.exp((self.chemical_potential_pd - self.electronegativity_pd) / 4)
        
        if overall > self.overall_compatibility_xml:
            factor = self.soft_soft_bonus * chem_pot_effect
        elif overall > 0.5:
            factor = 1.0 * chem_pot_effect
        else:
            factor = self.soft_hard_penalty * chem_pot_effect
        
        if overall < 0.3:
            factor *= self.mismatch_penalty
        
        result = np.clip(factor, 0.3, 1.3)
        return result
    
    def solvent_factor(self, solv1: str, solv2: str = '') -> float:
        solvent_effects = {
            'toluene': 1.10, 'benzene': 1.08, 'dioxane': 1.08,
            'THF': 1.05, 'DME': 1.08, 'DMF': 1.12,
            'DMSO': 1.15, 'NMP': 1.10, 'acetonitrile': 0.95,
            'ethanol': 0.90, 'methanol': 0.85, 'water': 0.70,
            'IPA': 0.88, 'ethyl acetate': 0.88, 'dichloromethane': 0.80,
            'chloroform': 0.75, 'hexane': 0.60, 'cyclohexane': 0.55,
            'Toluene': 1.10, 'Benzene': 1.08, 'Dioxane': 1.08
        }
        
        factor = 1.0
        solv1_lower = solv1.lower()
        
        aprotic = ['toluene', 'benzene', 'dioxane', 'thf', 'dme', 'dmf', 'dmso', 'nmp', 
                   'acetonitrile', 'ethyl acetate', 'dichloromethane', 'chloroform', 'hexane', 'cyclohexane']
        if any(s in solv1_lower for s in aprotic):
            factor *= self.aprotic_solvent_bonus
        
        protic = ['ethanol', 'methanol', 'water', 'ipa', 'propanol', 'butanol']
        if any(s in solv1_lower for s in protic):
            factor *= self.protic_solvent_penalty
        
        polar = ['dmf', 'dmso', 'nmp', 'acetonitrile', 'water']
        if any(s in solv1_lower for s in polar):
            factor *= self.polar_solvent_bonus
        
        nonpolar = ['hexane', 'cyclohexane', 'heptane', 'pentane']
        if any(s in solv1_lower for s in nonpolar):
            factor *= self.nonpolar_solvent_penalty
        
        for key, val in solvent_effects.items():
            if key.lower() in solv1_lower:
                factor *= val
                break
        
        if solv1_lower in SOLVENT_PHYSICS_ADVANCED:
            props = SOLVENT_PHYSICS_ADVANCED[solv1_lower]
            dielectric_factor = np.exp(-((props.get('dielectric', 25) - self.dielectric_optimal) ** 2) / (2 * self.dielectric_range ** 2))
            donor_factor = np.exp(-((props.get('donor_number', 20) - self.donor_optimal) ** 2) / (2 * self.donor_range ** 2))
            polarity_factor = np.exp(-((props.get('polarity_index', 4) - self.polarity_optimal) ** 2) / (2 * self.polarity_range ** 2))
            
            alpha = props.get('alpha', 0)
            beta = props.get('beta', 0)
            pi_star = props.get('pi_star', 0.5)
            reichardt = props.get('reichardt_et30', 40)
            hildebrand = props.get('hildebrand_delta', 20)
            
            kamlet_taft_factor = 1 + self.alpha_weight * alpha + self.beta_weight * beta + self.pi_star_weight * pi_star
            reichardt_factor = 1 + self.reichardt_weight * (reichardt - 40) / 10
            hildebrand_factor = 1 + self.hildebrand_weight * (hildebrand - 20) / 10
            
            factor = factor * (dielectric_factor * self.dielectric_weight + 
                              donor_factor * self.donor_weight + 
                              polarity_factor * self.polarity_weight + 
                              kamlet_taft_factor * 0.15 + 
                              reichardt_factor * 0.05 + 
                              hildebrand_factor * 0.05)
        
        if solv2 and solv2 != 'O':
            solv2_lower = solv2.lower()
            if 'toluene' in solv1_lower and 'ethanol' in solv2_lower:
                factor *= self.toluene_ethanol
            elif 'dioxane' in solv1_lower and 'water' in solv2_lower:
                factor *= self.dioxane_water
            elif 'thf' in solv1_lower and 'water' in solv2_lower:
                factor *= self.thf_water
            elif 'dme' in solv1_lower and 'water' in solv2_lower:
                factor *= self.dme_water
            else:
                factor = factor * 0.95
        
        result = np.clip(factor, 0.4, 1.3)
        return result
    
    def base_factor(self, base: str) -> float:
        base_lower = base.lower()
        
        base_props = BASE_PROPERTIES.get(base_lower, {})
        pka = base_props.get('pka', 10.3)
        solubility = base_props.get('solubility', 0.1)
        cation_radius = base_props.get('cation_radius', 1.38)
        hygroscopic = base_props.get('hygroscopic', False)
        base_class = base_props.get('class', 'carbonate')
        pkb = base_props.get('pkb', 3.7)
        
        factor = 1.0
        
        pka_effect = (pka - 10) * self.pka_effect
        factor *= np.exp(pka_effect)
        
        if pka > self.pka_threshold:
            factor *= self.strong_base_bonus
        else:
            factor *= self.weak_base_penalty
        
        if base_class == 'carbonate':
            factor *= self.carbonate_base_factor
        elif base_class == 'phosphate':
            factor *= self.phosphate_base_factor
        elif base_class == 'amine' or base_class == 'amidine':
            factor *= self.organic_base_factor
        else:
            factor *= self.inorganic_base_factor
        
        if solubility > 0.5:
            factor *= self.soluble_base_bonus
        else:
            factor *= self.insoluble_base_penalty
        
        if hygroscopic:
            factor *= self.hygroscopic_base_penalty
        
        if cation_radius:
            radius_effect = np.exp((cation_radius - 1.38) * self.cation_radius_effect)
            factor *= radius_effect
        
        pkb_effect = np.exp(-(pkb - 3.7) * 0.05)
        factor *= pkb_effect
        
        result = np.clip(factor, 0.4, 1.4)
        return result
    
    def mechanistic_factor(self, conditions: Dict) -> float:
        temp = conditions.get('temp', 80)
        time_hours = conditions.get('time', 24)
        steric_factor = conditions.get('steric_bulk', 0.5)
        electronic_factor = conditions.get('electronic_sensitivity', 1.0)
        base_strength = conditions.get('base_strength', 1.0)
        
        R_val = self.gas_constant
        T = temp + 273.15
        
        k_oa = self.oa_rate * np.exp(-self.oa_barrier * 1000 / (R_val * T))
        k_oa = k_oa * (1 - self.oa_steric_sens * steric_factor)
        k_oa = k_oa * (1 + self.oa_electronic_sens * electronic_factor)
        
        k_tm = self.tm_rate * np.exp(-self.tm_barrier * 1000 / (R_val * T))
        k_tm = k_tm * (1 + self.tm_base_sens * base_strength)
        k_tm = k_tm * (1 + self.tm_boronic_sens * 0.5)
        
        k_re = self.re_rate * np.exp(-self.re_barrier * 1000 / (R_val * T))
        k_re = k_re * (1 - self.re_steric_sens * steric_factor)
        k_re = k_re * (1 + self.re_electronic_sens * electronic_factor)
        
        intermediate_stability = self.intermediate_stability_factor * np.exp(-(self.oa_barrier + self.tm_barrier) * 1000 / (2 * R_val * T))
        
        rate = (self.oa_weight * k_oa + self.tm_weight * k_tm + self.re_weight * k_re) * intermediate_stability
        
        time_factor = 1 - np.exp(-rate * time_hours * 60)
        
        mechanistic_efficiency = (k_oa * k_tm * k_re) / (max(k_oa, 0.001) * max(k_tm, 0.001) * max(k_re, 0.001) + 0.001)
        
        transition_state_asymmetry_factor = np.exp(-self.transition_state_asymmetry * abs(k_oa - k_re) / (k_oa + k_re + 0.001))
        
        factor = time_factor * (0.8 + 0.2 * mechanistic_efficiency) * transition_state_asymmetry_factor
        
        result = np.clip(factor * 1.5, 0.1, 1.3)
        return result
    
    def calculate_yield(self, conditions: Dict) -> float:
        temp_factor = self.temperature_factor(conditions.get('temp', 80))
        time_factor = self.time_factor(conditions.get('time', 24))
        cat_factor = self.catalyst_factor(conditions.get('quantity', 0.0025))
        steric_factor_1 = self.steric_factor({**conditions, **{'substrate': 1}})
        steric_factor_2 = self.steric_factor({**conditions, **{'substrate': 2}})
        solvent_factor = self.solvent_factor(
            conditions.get('solv1', ''),
            conditions.get('solv2', '')
        )
        base_factor = self.base_factor(conditions.get('base', ''))
        electronic_factor = self.electronic_factor(conditions)
        hsab_factor = self.hsab_factor(conditions)
        mechanistic_factor = self.mechanistic_factor(conditions)
        
        total_weight = (self.temp_weight + self.time_weight + self.catalyst_weight + 
                       self.substrate1_steric_weight + self.substrate2_steric_weight +
                       self.solvent_weight + self.base_weight + self.electronic_weight +
                       self.hsab_weight + self.mechanistic_weight + 
                       self.hammett_weight + self.taft_weight)
        
        if total_weight == 0:
            total_weight = 1
        
        combined_factor = (
            (self.temp_weight / total_weight) * temp_factor +
            (self.time_weight / total_weight) * time_factor +
            (self.catalyst_weight / total_weight) * cat_factor +
            (self.substrate1_steric_weight / total_weight) * steric_factor_1 +
            (self.substrate2_steric_weight / total_weight) * steric_factor_2 * 0.8 +
            (self.solvent_weight / total_weight) * solvent_factor +
            (self.base_weight / total_weight) * base_factor +
            (self.electronic_weight / total_weight) * electronic_factor +
            (self.hsab_weight / total_weight) * hsab_factor +
            (self.mechanistic_weight / total_weight) * mechanistic_factor +
            (self.hammett_weight / total_weight) * electronic_factor * 0.5 +
            (self.taft_weight / total_weight) * electronic_factor * 0.3
        )
        
        raw_yield = self.base_yield_offset + (self.max_yield - self.base_yield_offset) * combined_factor
        
        noise = np.random.normal(0, self.random_variation * raw_yield * 0.1)
        final_yield = raw_yield + noise
        
        final_yield = final_yield * self.reproducibility
        final_yield = final_yield * self.scale_up_factor
        
        batch_noise = np.random.normal(1, self.batch_variation * 0.5)
        final_yield = final_yield * batch_noise
        
        result = np.clip(final_yield, self.min_yield, self.max_yield)
        return result
    
    def get_yield_class(self, yield_val: float) -> Tuple[str, str]:
        if yield_val >= self.excellent_threshold:
            return 'Excellent', '#10B981'
        elif yield_val >= self.good_threshold:
            return 'Good', '#3B82F6'
        elif yield_val >= self.moderate_threshold:
            return 'Moderate', '#F59E0B'
        elif yield_val >= self.poor_threshold:
            return 'Poor', '#EF4444'
        else:
            return 'Very Poor', '#DC2626'
    
    def get_yield_stats(self) -> Dict:
        return {
            'mean': self.yield_mean,
            'std': self.yield_std,
            'min': self.min_yield,
            'max': self.max_yield,
            'excellent_threshold': self.excellent_threshold,
            'good_threshold': self.good_threshold,
            'moderate_threshold': self.moderate_threshold,
            'poor_threshold': self.poor_threshold,
            'confidence_interval_alpha': self.confidence_interval_alpha,
            'prediction_interval_alpha': self.prediction_interval_alpha
        }

class FeatureEngineer:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.feature_columns = []
        self.scaler = None
        self.encoders = {}
        self.imputer = None
        self.selected_features = []
        self.pca = None
        self.feature_names = []
        self._load_params()
        
        self.dft_calc = AcademicDFTCalculator()
        self.qsar_calc = QSARCalculator()
        
        logger.success("FeatureEngineer initialized with FULL 500+ features + Academic DFT/QSAR")
    
    def _load_params(self):
        dp = self.config.get_dict('data_processing')
        
        mv = dp.get('missing_values', {})
        self.missing_strategy = mv.get('strategy', 'median_imputation')
        self.categorical_strategy = mv.get('categorical_strategy', 'mode_imputation')
        self.missing_threshold = mv.get('threshold', 0.30)
        
        norm = dp.get('normalization', {})
        self.numeric_method = norm.get('numeric_method', 'standard_scaler')
        self.categorical_method = norm.get('categorical_method', 'one_hot_encoding')
        self.target_scaling = norm.get('target_scaling', 'minmax')
        
        fs = dp.get('feature_selection', {})
        self.fs_method = fs.get('method', 'mutual_information')
        self.fs_k_best = fs.get('k_best', 30)
        self.fs_variance_threshold = fs.get('variance_threshold', 0.01)
        self.fs_correlation_threshold = fs.get('correlation_threshold', 0.85)
        
        aug = dp.get('augmentation', {})
        self.aug_enabled = aug.get('enabled', True)
        self.aug_method = aug.get('method', 'gaussian_noise')
        self.aug_noise_level = aug.get('noise_level', 0.05)
        self.aug_n_augmentations = aug.get('n_augmentations', 50)
        self.aug_bootstrap_samples = aug.get('bootstrap_samples', 1000)
        
        split = dp.get('split', {})
        self.test_size = split.get('test_size', 0.20)
        self.validation_size = split.get('validation_size', 0.15)
        self.split_stratify = split.get('stratify', True)
        self.split_random_state = split.get('random_state', 42)
        self.split_shuffle = split.get('shuffle', True)
        
        logger.debug(f"Feature engineering params loaded: k_best={self.fs_k_best}")
    
    def extract_smiles_features(self, smiles: str) -> Dict:
        features = {}
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen, QED
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return features
            
            features['mw'] = Descriptors.ExactMolWt(mol)
            features['logp'] = Descriptors.MolLogP(mol)
            features['tpsa'] = Descriptors.TPSA(mol)
            features['refractivity'] = Descriptors.MolarRefractivity(mol)
            features['heavy_atoms'] = mol.GetNumHeavyAtoms()
            features['total_atoms'] = mol.GetNumAtoms()
            
            features['hba'] = Lipinski.NumHAcceptors(mol)
            features['hbd'] = Lipinski.NumHDonors(mol)
            
            rings = mol.GetRingInfo()
            features['rings'] = rings.NumRings()
            features['aromatic_rings'] = rings.NumAromaticRings()
            features['aliphatic_rings'] = rings.NumAliphaticRings()
            features['saturated_rings'] = rings.NumSaturatedRings()
            
            features['rotatable_bonds'] = Lipinski.NumRotatableBonds(mol)
            
            features['kappa1'] = Descriptors.Kappa1(mol)
            features['kappa2'] = Descriptors.Kappa2(mol)
            features['kappa3'] = Descriptors.Kappa3(mol)
            
            try:
                features['steric_volume'] = rdMolDescriptors.CalcStericVolume(mol)
            except:
                features['steric_volume'] = 0.0
            
            features['qed'] = QED.qed(mol)
            features['complexity'] = Descriptors.BertzCT(mol)
            features['fraction_csp3'] = Descriptors.FractionCsp3(mol)
            
            features['c_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')
            features['n_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
            features['o_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
            features['s_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
            features['p_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'P')
            features['f_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'F')
            features['cl_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl')
            features['br_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'Br')
            features['i_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'I')
            features['b_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'B')
            
            metals = ['Pd', 'Pt', 'Ni', 'Cu', 'Ru', 'Rh', 'Ir', 'Au', 'Ag', 'Fe', 'Co', 'Mn', 'Cr', 'Mo', 'W']
            features['metal_count'] = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() in metals)
            
            features['halogen_count'] = features['f_count'] + features['cl_count'] + features['br_count'] + features['i_count']
            features['hetero_count'] = features['n_count'] + features['o_count'] + features['s_count'] + features['p_count'] + features['halogen_count']
            
            features['aromatic_bonds'] = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
            features['single_bonds'] = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.SINGLE)
            features['double_bonds'] = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
            features['triple_bonds'] = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)
            features['total_bonds'] = mol.GetNumBonds()
            
            features['chiral_centers_defined'] = rdMolDescriptors.CalcNumAtomStereoAtoms(mol)
            features['spiro_atoms'] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            features['bridgehead_atoms'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            features['branch_nodes'] = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
            
            features['sigma_m'] = 0.0
            features['sigma_p'] = 0.0
            features['sigma_plus'] = 0.0
            features['sigma_minus'] = 0.0
            features['taft_es'] = 0.0
            
            for group, vals in HAMMETT_SIGMA.items():
                if group in smiles:
                    features['sigma_m'] += vals.get('sigma_m', 0)
                    features['sigma_p'] += vals.get('sigma_p', 0)
                    features['sigma_plus'] += vals.get('sigma_plus', vals.get('sigma_p', 0))
                    features['sigma_minus'] += vals.get('sigma_minus', vals.get('sigma_p', 0))
                    features['taft_es'] += vals.get('taft_es', 0)
            
            try:
                dft = self.dft_calc.calculate_global_reactivity_descriptors(smiles)
                features['dft_homo'] = dft.get('homo_energy', -6.5)
                features['dft_lumo'] = dft.get('lumo_energy', -1.5)
                features['dft_gap'] = dft.get('gap_energy', 5.0)
                features['dft_chemical_potential'] = dft.get('chemical_potential', -4.0)
                features['dft_hardness'] = dft.get('absolute_hardness', 2.5)
                features['dft_electrophilicity'] = dft.get('electrophilicity_index', 3.2)
            except:
                pass
            
            try:
                qsar = self.qsar_calc.calculate_2d_qsar_descriptors(smiles)
                features['qsar_mr'] = qsar.get('MR', 0)
                features['qsar_drug_likeness'] = self.qsar_calc.calculate_drug_likeness(smiles).get('drug_likeness_score', 0)
            except:
                pass
            
        except Exception as e:
            pass
        
        return features
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.info("Starting feature engineering...")
        original_cols = len(df.columns)
        
        if 'temp' in df.columns and 'time' in df.columns:
            df['temp_time_product'] = df['temp'] * df['time']
            df['temp_time_ratio'] = df['temp'] / (df['time'] + 1)
            df['temp_time_sum'] = df['temp'] + df['time']
            df['temp_time_diff'] = df['temp'] - df['time']
            df['temp_time_interaction'] = df['temp'] * df['time'] / 100
            df['temp_log_time'] = df['temp'] * np.log1p(df['time'])
            df['time_log_temp'] = df['time'] * np.log1p(df['temp'])
        
        if 'quantity' in df.columns:
            df['quantity_log1p'] = np.log1p(df['quantity'])
            df['quantity_sqrt'] = np.sqrt(df['quantity'])
            df['quantity_squared'] = df['quantity'] ** 2
            df['quantity_inv'] = 1 / (df['quantity'] + 0.0001)
            df['quantity_exp'] = np.exp(df['quantity'])
            df['quantity_power3'] = df['quantity'] ** 3
        
        if 'temp' in df.columns and 'quantity' in df.columns:
            df['temp_quantity_product'] = df['temp'] * df['quantity']
            df['catalyst_loading'] = df['quantity'] / (df['temp'] + 1)
            df['temp_quantity_ratio'] = df['temp'] / (df['quantity'] + 0.0001)
        
        smiles_cols = ['subs1', 'subs2', 'product', 'catalizor', 'base', 'solv1', 'solv2']
        
        for col in smiles_cols:
            if col in df.columns:
                df[f'{col}_length'] = df[col].astype(str).str.len()
                
                features_list = []
                for smiles in df[col]:
                    features_list.append(self.extract_smiles_features(smiles))
                
                if features_list:
                    features_df = pd.DataFrame(features_list)
                    for fcol in features_df.columns:
                        df[f'{col}_{fcol}'] = features_df[fcol]
        
        if 'subs1_length' in df.columns and 'subs2_length' in df.columns:
            df['substrate_steric_sum'] = df['subs1_length'] + df['subs2_length']
            df['substrate_steric_diff'] = abs(df['subs1_length'] - df['subs2_length'])
            df['substrate_steric_ratio'] = df['subs1_length'] / (df['subs2_length'] + 1)
            df['substrate_steric_product'] = df['subs1_length'] * df['subs2_length']
            df['substrate_steric_euclidean'] = np.sqrt(df['subs1_length'] ** 2 + df['subs2_length'] ** 2)
        
        if 'subs1_logp' in df.columns and 'subs2_logp' in df.columns:
            df['substrate_logp_avg'] = (df['subs1_logp'] + df['subs2_logp']) / 2
            df['substrate_logp_diff'] = abs(df['subs1_logp'] - df['subs2_logp'])
            df['substrate_logp_sum'] = df['subs1_logp'] + df['subs2_logp']
            df['substrate_logp_product'] = df['subs1_logp'] * df['subs2_logp']
        
        if 'subs1_mw' in df.columns and 'subs2_mw' in df.columns:
            df['substrate_mw_avg'] = (df['subs1_mw'] + df['subs2_mw']) / 2
            df['substrate_mw_diff'] = abs(df['subs1_mw'] - df['subs2_mw'])
            df['substrate_mw_ratio'] = df['subs1_mw'] / (df['subs2_mw'] + 1)
            df['substrate_mw_sum'] = df['subs1_mw'] + df['subs2_mw']
        
        if 'subs1_rings' in df.columns and 'subs2_rings' in df.columns:
            df['total_rings'] = df['subs1_rings'] + df['subs2_rings']
            df['ring_diff'] = abs(df['subs1_rings'] - df['subs2_rings'])
            df['ring_product'] = df['subs1_rings'] * df['subs2_rings']
            df['aromatic_sum'] = df.get('subs1_aromatic_rings', 0) + df.get('subs2_aromatic_rings', 0)
            df['aromatic_ratio'] = df['aromatic_sum'] / (df['total_rings'] + 1)
        
        if 'subs1_sigma_p' in df.columns and 'subs2_sigma_p' in df.columns:
            df['sigma_p_sum'] = df['subs1_sigma_p'] + df['subs2_sigma_p']
            df['sigma_p_diff'] = abs(df['subs1_sigma_p'] - df['subs2_sigma_p'])
            df['sigma_p_avg'] = (df['subs1_sigma_p'] + df['subs2_sigma_p']) / 2
            df['sigma_p_product'] = df['subs1_sigma_p'] * df['subs2_sigma_p']
        
        if 'subs1_taft_es' in df.columns and 'subs2_taft_es' in df.columns:
            df['taft_es_sum'] = df['subs1_taft_es'] + df['subs2_taft_es']
            df['taft_es_diff'] = abs(df['subs1_taft_es'] - df['subs2_taft_es'])
            df['taft_es_avg'] = (df['subs1_taft_es'] + df['subs2_taft_es']) / 2
        
        if 'subs1_halogen_count' in df.columns and 'subs2_halogen_count' in df.columns:
            df['halogen_total'] = df['subs1_halogen_count'] + df['subs2_halogen_count']
            df['halogen_diff'] = abs(df['subs1_halogen_count'] - df['subs2_halogen_count'])
            df['halogen_product'] = df['subs1_halogen_count'] * df['subs2_halogen_count']
        
        if 'subs1_sigma_p' in df.columns:
            df['electronic_softness'] = df['subs1_sigma_p'] * 0.3
            if 'hsab_overall_compatibility' in df.columns:
                df['mechanistic_predictor'] = df['subs1_sigma_p'] * 0.3 + df['hsab_overall_compatibility'] * 0.7
            df['hammett_effect'] = np.exp(2.8 * df['subs1_sigma_p'])
            df['taft_effect'] = np.exp(1.5 * df['subs1_taft_es'] / 2)
        
        if 'subs1_dft_homo' in df.columns and 'subs2_dft_homo' in df.columns:
            df['dft_homo_sum'] = df['subs1_dft_homo'] + df['subs2_dft_homo']
            df['dft_homo_diff'] = abs(df['subs1_dft_homo'] - df['subs2_dft_homo'])
            df['dft_homo_avg'] = (df['subs1_dft_homo'] + df['subs2_dft_homo']) / 2
            df['dft_gap_sum'] = df.get('subs1_dft_gap', 5) + df.get('subs2_dft_gap', 5)
            df['dft_chemical_potential_avg'] = (df.get('subs1_dft_chemical_potential', -4) + df.get('subs2_dft_chemical_potential', -4)) / 2
        
        logger.info(f"Feature engineering complete: {len(df.columns)} columns (was {original_cols})")
        return df
    
    def select_features(self, df: pd.DataFrame, target: str = 'yield') -> pd.DataFrame:
        try:
            from sklearn.feature_selection import mutual_info_regression
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)
            
            if len(numeric_cols) <= 5:
                return df[numeric_cols] if numeric_cols else df
            
            logger.info(f"Selecting features from {len(numeric_cols)} numeric columns...")
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df[numeric_cols].fillna(0))
            
            mi_scores = mutual_info_regression(X_scaled, df[target].values, random_state=42)
            
            feature_scores = list(zip(numeric_cols, mi_scores))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            k_best = min(self.fs_k_best, len(numeric_cols) // 2)
            k_best = max(5, k_best)
            selected = [f for f, _ in feature_scores[:k_best]]
            
            corr_threshold = self.fs_correlation_threshold
            if len(selected) > 1:
                corr_matrix = df[selected].corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
                selected = [f for f in selected if f not in to_drop]
            
            self.selected_features = selected
            logger.info(f"Selected {len(selected)} features from {len(numeric_cols)}")
            
            return df[selected] if selected else df[numeric_cols]
            
        except Exception as e:
            logger.error(f"Feature selection error: {e}")
            return df.select_dtypes(include=[np.number])

class SuzukiPredictor:
    def __init__(self, config: ConfigManager):
        self.config = config
        self.chemical = ChemicalCalculator(config)
        self.fe = FeatureEngineer(config)
        self.df = None
        self.X = None
        self.y = None
        self.models = {}
        self.feature_columns = []
        self.is_trained = False
        self.model_performance = {}
        self.scaler = None
        self.ensemble_weights = self._load_weights()
        self.model_history = []
        self.best_model = None
        self.cv_results = {}
        self.feature_importance = {}
        self.residuals = None
        self.predictions = None
        self._model_instances = {}
        self.is_enriched = False
        
        self.gp_model = None
        self.shap_values = None
        self.shap_explainer = None
        self.cv_scores = None
        self.anova_results = None
        self.confidence_intervals = None
        self.prediction_intervals = None
        self.calibration_model = None
        
        logger.success("SuzukiPredictor initialized with FULL XML integration + Academic Features (GP, SHAP, ANOVA, CIs)")
    
    def _load_weights(self) -> Dict:
        try:
            w = self.config.get_dict('model_parameters/Ensemble/weights')
            if w:
                return {k: float(v) for k, v in w.items() if float(v) > 0}
        except Exception as e:
            logger.warning(f"Could not load ensemble weights: {e}")
        
        return {
            'Random_Forest': 0.16,
            'Gradient_Boosting': 0.12,
            'Hist_Gradient_Boosting': 0.16,
            'XGBoost': 0.12,
            'LightGBM': 0.08,
            'CatBoost': 0.08,
            'Extra_Trees': 0.04,
            'SVR': 0.02,
            'Neural_Network': 0.03,
            'Gaussian_Process': 0.05
        }
    
    def validate_csv(self, filepath: str) -> Tuple[bool, List[str]]:
        try:
            df = pd.read_csv(filepath, nrows=1)
            columns = df.columns.tolist()
            missing = []
            for col in REQUIRED_COLUMNS:
                if col not in columns:
                    missing.append(col)
            for opt in OPTIONAL_COLUMNS:
                if opt not in columns:
                    missing.append(f"{opt} (optional)")
            if missing:
                return False, missing
            return True, []
        except Exception as e:
            return False, [str(e)]
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        valid, missing = self.validate_csv(filepath)
        if not valid:
            missing_required = [m for m in missing if 'optional' not in m]
            if missing_required:
                raise ValueError(f"CSV validation failed. Missing required columns: {', '.join(missing_required)}")
        
        try:
            self.df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.df)} rows from {filepath}")
            
            self.is_enriched = is_enriched_dataset(self.df)
            
            if not self.is_enriched:
                logger.error("BASIC DATASET DETECTED! This file does NOT contain academic features.")
                logger.error("Please use dataset_routes.py to enrich your data first.")
                raise ValueError(
                    "This is a basic dataset without academic features.\n"
                    "Please use dataset_routes.py to enrich your data first.\n"
                    "Required: columns like subs1_SMILES_*, subs2_SMILES_*, hsab_*, etc."
                )
            
            logger.success(f"ENRICHED DATASET detected! {len(self.df.columns)} columns with academic features.")
            
            if 'yield' not in self.df.columns:
                raise ValueError("'yield' column not found")
            
            self.df = self.df.dropna(subset=['yield'])
            
            if len(self.df) < 5:
                raise ValueError(f"Dataset must have at least 5 rows. Current: {len(self.df)}")
            
            self.df = self.fe.engineer_features(self.df)
            
            self._prepare_features()
            return self.df
            
        except Exception as e:
            logger.error(f"Load error: {str(e)}")
            raise
    
    def _prepare_features(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'yield' in numeric_cols:
            numeric_cols.remove('yield')
        
        important_features = [
            'temp', 'time', 'quantity',
            'temp_time_product', 'temp_time_ratio', 'temp_time_sum',
            'temp_time_diff', 'temp_time_interaction', 'temp_log_time', 'time_log_temp',
            'catalyst_loading', 'temp_quantity_product', 'temp_quantity_ratio',
            'quantity_log1p', 'quantity_sqrt', 'quantity_squared',
            'quantity_inv', 'quantity_exp', 'quantity_power3',
            'subs1_length', 'subs2_length',
            'substrate_steric_sum', 'substrate_steric_diff', 'substrate_steric_ratio',
            'substrate_steric_product', 'substrate_steric_euclidean',
            'subs1_logp', 'subs2_logp',
            'substrate_logp_avg', 'substrate_logp_diff', 'substrate_logp_sum', 'substrate_logp_product',
            'subs1_mw', 'subs2_mw',
            'substrate_mw_avg', 'substrate_mw_diff', 'substrate_mw_ratio', 'substrate_mw_sum',
            'subs1_rings', 'subs2_rings',
            'total_rings', 'ring_diff', 'ring_product',
            'aromatic_sum', 'aromatic_ratio',
            'subs1_hba', 'subs2_hba',
            'subs1_hbd', 'subs2_hbd',
            'subs1_qed', 'subs2_qed',
            'subs1_complexity', 'subs2_complexity',
            'subs1_kappa1', 'subs2_kappa1',
            'subs1_tpsa', 'subs2_tpsa',
            'subs1_halogen_count', 'subs2_halogen_count',
            'halogen_total', 'halogen_diff', 'halogen_product',
            'subs1_hetero_count', 'subs2_hetero_count',
            'subs1_fraction_csp3', 'subs2_fraction_csp3',
            'subs1_rotatable_bonds', 'subs2_rotatable_bonds',
            'subs1_sigma_m', 'subs1_sigma_p', 'subs1_sigma_plus', 'subs1_sigma_minus', 'subs1_taft_es',
            'subs2_sigma_m', 'subs2_sigma_p', 'subs2_sigma_plus', 'subs2_sigma_minus', 'subs2_taft_es',
            'sigma_p_sum', 'sigma_p_diff', 'sigma_p_avg', 'sigma_p_product',
            'taft_es_sum', 'taft_es_diff', 'taft_es_avg',
            'electronic_softness', 'mechanistic_predictor',
            'hammett_effect', 'taft_effect',
            'hsab_overall_compatibility', 'hsab_pd_halide_mismatch',
            'hsab_soft_soft_interaction_score', 'hsab_pearson_class_match',
            'mechanistic_oxidative_addition_liability', 'reaction_rate_indicator',
            'subs1_dft_homo', 'subs1_dft_lumo', 'subs1_dft_gap', 'subs1_dft_chemical_potential', 'subs1_dft_hardness',
            'subs2_dft_homo', 'subs2_dft_lumo', 'subs2_dft_gap', 'subs2_dft_chemical_potential', 'subs2_dft_hardness',
            'dft_homo_sum', 'dft_homo_diff', 'dft_homo_avg', 'dft_gap_sum', 'dft_chemical_potential_avg',
            'subs1_qsar_mr', 'subs1_qsar_drug_likeness', 'subs2_qsar_mr', 'subs2_qsar_drug_likeness'
        ]
        
        available_features = [c for c in important_features if c in self.df.columns]
        logger.debug(f"Found {len(available_features)} important features")
        
        categorical_cols = ['catalizor', 'base', 'solv1', 'solv2']
        categorical_cols = [c for c in categorical_cols if c in self.df.columns]
        
        X_numeric = self.df[available_features].copy() if available_features else pd.DataFrame()
        for col in X_numeric.columns:
            if X_numeric[col].isnull().any():
                X_numeric[col] = X_numeric[col].fillna(X_numeric[col].median() if not X_numeric[col].empty else 0)
        
        X_categorical = pd.DataFrame()
        for col in categorical_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                X_categorical = pd.concat([X_categorical, dummies], axis=1)
        
        X_categorical.columns = [clean_feature_name(c) for c in X_categorical.columns]
        
        self.X = pd.concat([X_numeric, X_categorical], axis=1) if not X_categorical.empty else X_numeric
        self.y = self.df['yield'].values
        self.feature_columns = [clean_feature_name(c) for c in self.X.columns]
        self.X.columns = self.feature_columns
        
        if self.X.isnull().any().any():
            self.X = self.X.fillna(0)
        
        logger.info(f"Prepared {len(self.feature_columns)} features")
    
    def train(self, model_type: str = 'Ensemble') -> Dict:
        try:
            if self.X is None or len(self.X) == 0:
                raise ValueError("Data must be loaded first")
            
            if not self.is_enriched:
                raise ValueError(
                    "Cannot train ML model on basic dataset.\n"
                    "Please use dataset_routes.py to enrich your data first."
                )
            
            logger.info(f"Starting training with {len(self.X)} samples, {len(self.feature_columns)} features")
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
            
            imputer = SimpleImputer(strategy='mean')
            X_scaled = pd.DataFrame(imputer.fit_transform(X_scaled), columns=self.feature_columns)
            
            test_size = min(0.2, max(0.1, 3.0 / len(self.X))) if len(self.X) > 3 else 0.1
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.y, test_size=test_size, random_state=42
            )
            
            models = {}
            performances = {}
            
            if model_type == 'Ensemble' or model_type == 'all':
                model_names = [
                    'Random_Forest', 'Gradient_Boosting', 'Hist_Gradient_Boosting',
                    'XGBoost', 'LightGBM', 'CatBoost', 'Extra_Trees',
                    'KNN', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 'Neural_Network',
                    'Gaussian_Process'
                ]
                
                for name in model_names:
                    try:
                        logger.info(f"Training {name}...")
                        model = self._create_model(name)
                        if model:
                            model.fit(X_train, y_train)
                            models[name] = model
                            
                            y_pred = model.predict(X_test)
                            if len(y_pred) > 0 and not np.isnan(y_pred).all():
                                r2 = r2_score(y_test, y_pred)
                                mae = mean_absolute_error(y_test, y_pred)
                                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                                
                                performances[name] = {
                                    'r2': float(r2),
                                    'mae': float(mae),
                                    'rmse': float(rmse)
                                }
                                
                                logger.success(f"{name}: R2={r2:.3f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
                    except Exception as e:
                        logger.warning(f"Could not train {name}: {str(e)}")
                
                if models:
                    self.models = models
                    self.is_trained = True
                    self.model_performance = performances
                    
                    if performances:
                        best_name = max(performances.items(), key=lambda x: x[1].get('r2', 0))[0]
                        self.best_model = best_name
                        logger.success(f"Best model: {best_name} (R2={performances[best_name]['r2']:.3f})")
                    
                    self._run_academic_analyses(X_scaled, y_train, X_test, y_test)
                    
                    return {
                        'success': True,
                        'message': f"Ensemble trained ({len(models)} models)",
                        'performance': convert_to_serializable(performances),
                        'best_model': self.best_model,
                        'model_count': len(models),
                        'academic_analyses': {
                            'cv_scores': self.cv_scores,
                            'anova_results': self.anova_results,
                            'confidence_intervals': self.confidence_intervals,
                            'prediction_intervals': self.prediction_intervals
                        }
                    }
            
            return {'success': False, 'message': 'Training failed'}
            
        except Exception as e:
            logger.error(f"Train error: {str(e)}")
            raise
    
    def _run_academic_analyses(self, X_scaled, y_train, X_test, y_test):
        try:
            if self.models:
                model = list(self.models.values())[0]
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, X_scaled, y_train, cv=cv, scoring='r2')
                self.cv_scores = {
                    'mean': float(np.mean(cv_scores)),
                    'std': float(np.std(cv_scores)),
                    'min': float(np.min(cv_scores)),
                    'max': float(np.max(cv_scores)),
                    'scores': [float(s) for s in cv_scores]
                }
                logger.info(f"CV R2: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        except Exception as e:
            logger.warning(f"CV analysis failed: {e}")
        
        if SHAP_AVAILABLE:
            try:
                model = list(self.models.values())[0]
                self.shap_explainer = shap.TreeExplainer(model)
                self.shap_values = self.shap_explainer.shap_values(X_scaled[:100])
                logger.info("SHAP analysis completed")
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {e}")
        
        if STATSMODELS_AVAILABLE:
            try:
                from statsmodels.stats.anova import anova_lm
                from statsmodels.formula.api import ols
                
                df_temp = pd.DataFrame(X_scaled, columns=self.feature_columns)
                df_temp['yield'] = y_train
                
                top_features = self.feature_columns[:5]
                formula = 'yield ~ ' + ' + '.join(top_features)
                
                model_ols = ols(formula, data=df_temp).fit()
                anova_table = anova_lm(model_ols, typ=2)
                
                self.anova_results = {
                    'features': top_features,
                    'f_values': [float(anova_table.loc[f, 'F']) for f in top_features if f in anova_table.index],
                    'p_values': [float(anova_table.loc[f, 'PR(>F)']) for f in top_features if f in anova_table.index]
                }
                logger.info("ANOVA analysis completed")
            except Exception as e:
                logger.warning(f"ANOVA analysis failed: {e}")
        
        try:
            if self.models:
                model = list(self.models.values())[0]
                y_pred = model.predict(X_test)
                n = len(y_test)
                mse = mean_squared_error(y_test, y_pred)
                se = np.sqrt(mse / n)
                
                alpha = 0.05
                t_val = stats.t.ppf(1 - alpha/2, n - 2)
                
                self.confidence_intervals = {
                    'alpha': alpha,
                    't_value': float(t_val),
                    'standard_error': float(se),
                    'lower': float(np.mean(y_pred) - t_val * se),
                    'upper': float(np.mean(y_pred) + t_val * se),
                    'mean_prediction': float(np.mean(y_pred))
                }
                logger.info("Confidence intervals computed")
        except Exception as e:
            logger.warning(f"Confidence intervals failed: {e}")
        
        try:
            if self.models:
                model = list(self.models.values())[0]
                y_pred = model.predict(X_test)
                residuals = y_test - y_pred
                residual_std = np.std(residuals)
                
                alpha = 0.10
                z_val = stats.norm.ppf(1 - alpha/2)
                
                self.prediction_intervals = {
                    'alpha': alpha,
                    'z_value': float(z_val),
                    'residual_std': float(residual_std),
                    'lower': float(np.mean(y_pred) - z_val * residual_std),
                    'upper': float(np.mean(y_pred) + z_val * residual_std)
                }
                logger.info("Prediction intervals computed")
        except Exception as e:
            logger.warning(f"Prediction intervals failed: {e}")
    
    def _create_model(self, name: str):
        try:
            params = self.config.get_model_params(name)
            params = {k: self._parse_param(v) for k, v in params.items()}
            
            if name == 'Random_Forest':
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**params)
            elif name == 'Gradient_Boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor(**params)
            elif name == 'Hist_Gradient_Boosting':
                from sklearn.ensemble import HistGradientBoostingRegressor
                return HistGradientBoostingRegressor(**params)
            elif name == 'XGBoost':
                try:
                    from xgboost import XGBRegressor
                    return XGBRegressor(**params)
                except ImportError:
                    return None
            elif name == 'LightGBM':
                try:
                    from lightgbm import LGBMRegressor
                    return LGBMRegressor(**params)
                except ImportError:
                    return None
            elif name == 'CatBoost':
                try:
                    from catboost import CatBoostRegressor
                    return CatBoostRegressor(**params)
                except ImportError:
                    return None
            elif name == 'Extra_Trees':
                from sklearn.ensemble import ExtraTreesRegressor
                return ExtraTreesRegressor(**params)
            elif name == 'KNN':
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor(**params)
            elif name == 'Ridge':
                from sklearn.linear_model import Ridge
                return Ridge(**params)
            elif name == 'Lasso':
                from sklearn.linear_model import Lasso
                return Lasso(**params)
            elif name == 'ElasticNet':
                from sklearn.linear_model import ElasticNet
                return ElasticNet(**params)
            elif name == 'SVR':
                from sklearn.svm import SVR
                return SVR(**params)
            elif name == 'Neural_Network':
                from sklearn.neural_network import MLPRegressor
                return MLPRegressor(**params)
            elif name == 'Gaussian_Process':
                if GP_AVAILABLE:
                    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-10)
                    return GaussianProcessRegressor(kernel=kernel, **params)
                return None
            else:
                return None
        except Exception as e:
            logger.error(f"Model creation error for {name}: {e}")
            return None
    
    def _parse_param(self, val):
        if val is None:
            return None
        if isinstance(val, str):
            if val.lower() in ['true', 'false']:
                return val.lower() == 'true'
            if val.lower() == 'none':
                return None
            if ',' in val:
                try:
                    return tuple(int(x) for x in val.split(','))
                except:
                    return val
            try:
                return float(val) if '.' in val else int(val)
            except:
                return val
        return val
    
    def _ensemble_predict(self, X) -> np.ndarray:
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            if name in self.ensemble_weights:
                weight = self.ensemble_weights.get(name, 0)
                if weight > 0:
                    try:
                        pred = model.predict(X)
                        if not np.isnan(pred).all():
                            predictions.append(pred)
                            weights.append(weight)
                    except Exception as e:
                        pass
        
        if not predictions:
            return np.zeros(len(X))
        
        weights = np.array(weights) / np.sum(weights)
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += weight * pred
        
        return ensemble_pred
    
    def predict(self, conditions: Dict) -> Dict:
        try:
            logger.info("Making prediction...")
            
            chemical_yield = self.chemical.calculate_yield(conditions)
            logger.debug(f"Chemical yield: {chemical_yield:.3f}%")
            
            ml_yield = None
            ml_std = None
            if self.is_enriched and self.is_trained and self.models:
                try:
                    feature_vector = self._create_feature_vector(conditions)
                    if feature_vector is not None:
                        if len(self.models) > 1:
                            ml_pred = self._ensemble_predict([feature_vector])
                            ml_yield = float(ml_pred[0]) if len(ml_pred) > 0 else None
                        else:
                            model = list(self.models.values())[0]
                            ml_pred = model.predict([feature_vector])
                            ml_yield = float(ml_pred[0]) if len(ml_pred) > 0 else None
                            
                            if 'Gaussian_Process' in self.models and GP_AVAILABLE:
                                try:
                                    gp = self.models['Gaussian_Process']
                                    pred, std = gp.predict([feature_vector], return_std=True)
                                    ml_std = float(std[0])
                                except:
                                    pass
                        
                        logger.debug(f"ML yield: {ml_yield:.3f}%")
                except Exception as e:
                    logger.warning(f"ML prediction failed: {str(e)}")
            
            if ml_yield is not None and not np.isnan(ml_yield) and self.is_enriched:
                combined_yield = 0.6 * ml_yield + 0.4 * chemical_yield
                model_name = 'Ensemble'
            else:
                combined_yield = chemical_yield
                model_name = 'Chemical Intuition'
                logger.info("Using only Chemical Intuition (ML not available or data not enriched)")
            
            final_yield = np.clip(combined_yield, 0, 100)
            
            yield_class, color = self.chemical.get_yield_class(final_yield)
            
            confidence = self._calculate_confidence(ml_yield, chemical_yield, final_yield)
            
            logger.success(f"Final prediction: {final_yield:.1f}% ({yield_class})")
            
            academic_details = {}
            try:
                subs1_smiles = conditions.get('subs1_smiles', '')
                subs2_smiles = conditions.get('subs2_smiles', '')
                if subs1_smiles:
                    academic_details['subs1_dft'] = self.chemical.calculate_dft_descriptors(subs1_smiles)
                    academic_details['subs1_qsar'] = self.chemical.calculate_qsar_descriptors(subs1_smiles)
                    academic_details['subs1_docking'] = self.chemical.calculate_docking_score(subs1_smiles, 'pd')
                if subs2_smiles:
                    academic_details['subs2_dft'] = self.chemical.calculate_dft_descriptors(subs2_smiles)
                    academic_details['subs2_qsar'] = self.chemical.calculate_qsar_descriptors(subs2_smiles)
                    academic_details['subs2_docking'] = self.chemical.calculate_docking_score(subs2_smiles, 'pd')
            except Exception as e:
                logger.warning(f"Academic calculations failed: {e}")
            
            return {
                'success': True,
                'prediction': float(final_yield),
                'ml_prediction': float(ml_yield) if ml_yield is not None else None,
                'ml_std': ml_std,
                'chemical_prediction': float(chemical_yield),
                'model': model_name,
                'yield_class': yield_class,
                'yield_class_color': color,
                'confidence': float(confidence),
                'best_model': self.best_model,
                'model_count': len(self.models) if self.models else 0,
                'is_enriched': self.is_enriched,
                'academic_details': academic_details,
                'confidence_interval': self.confidence_intervals,
                'prediction_interval': self.prediction_intervals
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def _create_feature_vector(self, conditions: Dict) -> np.ndarray:
        if not self.feature_columns:
            return None
        
        f = {}
        
        f['temp'] = conditions.get('temp', 80)
        f['time'] = conditions.get('time', 24)
        f['quantity'] = conditions.get('quantity', 0.0025)
        
        f['temp_time_product'] = f['temp'] * f['time']
        f['temp_time_ratio'] = f['temp'] / (f['time'] + 1)
        f['temp_time_sum'] = f['temp'] + f['time']
        f['temp_time_diff'] = f['temp'] - f['time']
        f['temp_time_interaction'] = f['temp'] * f['time'] / 100
        f['temp_log_time'] = f['temp'] * np.log1p(f['time'])
        f['time_log_temp'] = f['time'] * np.log1p(f['temp'])
        
        f['quantity_log1p'] = np.log1p(f['quantity'])
        f['quantity_sqrt'] = np.sqrt(f['quantity'])
        f['quantity_squared'] = f['quantity'] ** 2
        f['quantity_inv'] = 1 / (f['quantity'] + 0.0001)
        f['quantity_exp'] = np.exp(f['quantity'])
        f['quantity_power3'] = f['quantity'] ** 3
        
        f['catalyst_loading'] = f['quantity'] / (f['temp'] + 1)
        f['temp_quantity_product'] = f['temp'] * f['quantity']
        f['temp_quantity_ratio'] = f['temp'] / (f['quantity'] + 0.0001)
        
        subs1 = conditions.get('subs1_smiles', '')
        subs2 = conditions.get('subs2_smiles', '')
        
        if subs1:
            mf = self.fe.extract_smiles_features(subs1)
            f['subs1_length'] = len(subs1)
            f['subs1_logp'] = mf.get('logp', 0)
            f['subs1_mw'] = mf.get('mw', 0)
            f['subs1_rings'] = mf.get('rings', 0)
            f['subs1_aromatic_rings'] = mf.get('aromatic_rings', 0)
            f['subs1_hba'] = mf.get('hba', 0)
            f['subs1_hbd'] = mf.get('hbd', 0)
            f['subs1_qed'] = mf.get('qed', 0)
            f['subs1_complexity'] = mf.get('complexity', 0)
            f['subs1_kappa1'] = mf.get('kappa1', 0)
            f['subs1_tpsa'] = mf.get('tpsa', 0)
            f['subs1_halogen_count'] = mf.get('halogen_count', 0)
            f['subs1_hetero_count'] = mf.get('hetero_count', 0)
            f['subs1_fraction_csp3'] = mf.get('fraction_csp3', 0)
            f['subs1_rotatable_bonds'] = mf.get('rotatable_bonds', 0)
            f['subs1_sigma_m'] = mf.get('sigma_m', 0)
            f['subs1_sigma_p'] = mf.get('sigma_p', 0)
            f['subs1_sigma_plus'] = mf.get('sigma_plus', 0)
            f['subs1_sigma_minus'] = mf.get('sigma_minus', 0)
            f['subs1_taft_es'] = mf.get('taft_es', 0)
            
            dft = self.chemical.calculate_dft_descriptors(subs1)
            f['subs1_dft_homo'] = dft.get('homo_energy', -6.5)
            f['subs1_dft_lumo'] = dft.get('lumo_energy', -1.5)
            f['subs1_dft_gap'] = dft.get('gap_energy', 5.0)
            f['subs1_dft_chemical_potential'] = dft.get('chemical_potential', -4.0)
            f['subs1_dft_hardness'] = dft.get('absolute_hardness', 2.5)
            
            qsar = self.chemical.calculate_qsar_descriptors(subs1)
            f['subs1_qsar_mr'] = qsar.get('MR', 0)
            f['subs1_qsar_drug_likeness'] = self.chemical.calculate_drug_likeness(subs1).get('drug_likeness_score', 0)
        else:
            f['subs1_length'] = 0
            f['subs1_logp'] = 0
            f['subs1_mw'] = 0
            f['subs1_rings'] = 0
            f['subs1_aromatic_rings'] = 0
            f['subs1_hba'] = 0
            f['subs1_hbd'] = 0
            f['subs1_qed'] = 0
            f['subs1_complexity'] = 0
            f['subs1_kappa1'] = 0
            f['subs1_tpsa'] = 0
            f['subs1_halogen_count'] = 0
            f['subs1_hetero_count'] = 0
            f['subs1_fraction_csp3'] = 0
            f['subs1_rotatable_bonds'] = 0
            f['subs1_sigma_m'] = 0
            f['subs1_sigma_p'] = 0
            f['subs1_sigma_plus'] = 0
            f['subs1_sigma_minus'] = 0
            f['subs1_taft_es'] = 0
            f['subs1_dft_homo'] = -6.5
            f['subs1_dft_lumo'] = -1.5
            f['subs1_dft_gap'] = 5.0
            f['subs1_dft_chemical_potential'] = -4.0
            f['subs1_dft_hardness'] = 2.5
            f['subs1_qsar_mr'] = 0
            f['subs1_qsar_drug_likeness'] = 0
        
        if subs2:
            mf = self.fe.extract_smiles_features(subs2)
            f['subs2_length'] = len(subs2)
            f['subs2_logp'] = mf.get('logp', 0)
            f['subs2_mw'] = mf.get('mw', 0)
            f['subs2_rings'] = mf.get('rings', 0)
            f['subs2_aromatic_rings'] = mf.get('aromatic_rings', 0)
            f['subs2_hba'] = mf.get('hba', 0)
            f['subs2_hbd'] = mf.get('hbd', 0)
            f['subs2_qed'] = mf.get('qed', 0)
            f['subs2_complexity'] = mf.get('complexity', 0)
            f['subs2_kappa1'] = mf.get('kappa1', 0)
            f['subs2_tpsa'] = mf.get('tpsa', 0)
            f['subs2_halogen_count'] = mf.get('halogen_count', 0)
            f['subs2_hetero_count'] = mf.get('hetero_count', 0)
            f['subs2_fraction_csp3'] = mf.get('fraction_csp3', 0)
            f['subs2_rotatable_bonds'] = mf.get('rotatable_bonds', 0)
            f['subs2_sigma_m'] = mf.get('sigma_m', 0)
            f['subs2_sigma_p'] = mf.get('sigma_p', 0)
            f['subs2_sigma_plus'] = mf.get('sigma_plus', 0)
            f['subs2_sigma_minus'] = mf.get('sigma_minus', 0)
            f['subs2_taft_es'] = mf.get('taft_es', 0)
            
            dft = self.chemical.calculate_dft_descriptors(subs2)
            f['subs2_dft_homo'] = dft.get('homo_energy', -6.5)
            f['subs2_dft_lumo'] = dft.get('lumo_energy', -1.5)
            f['subs2_dft_gap'] = dft.get('gap_energy', 5.0)
            f['subs2_dft_chemical_potential'] = dft.get('chemical_potential', -4.0)
            f['subs2_dft_hardness'] = dft.get('absolute_hardness', 2.5)
            
            qsar = self.chemical.calculate_qsar_descriptors(subs2)
            f['subs2_qsar_mr'] = qsar.get('MR', 0)
            f['subs2_qsar_drug_likeness'] = self.chemical.calculate_drug_likeness(subs2).get('drug_likeness_score', 0)
        else:
            f['subs2_length'] = 0
            f['subs2_logp'] = 0
            f['subs2_mw'] = 0
            f['subs2_rings'] = 0
            f['subs2_aromatic_rings'] = 0
            f['subs2_hba'] = 0
            f['subs2_hbd'] = 0
            f['subs2_qed'] = 0
            f['subs2_complexity'] = 0
            f['subs2_kappa1'] = 0
            f['subs2_tpsa'] = 0
            f['subs2_halogen_count'] = 0
            f['subs2_hetero_count'] = 0
            f['subs2_fraction_csp3'] = 0
            f['subs2_rotatable_bonds'] = 0
            f['subs2_sigma_m'] = 0
            f['subs2_sigma_p'] = 0
            f['subs2_sigma_plus'] = 0
            f['subs2_sigma_minus'] = 0
            f['subs2_taft_es'] = 0
            f['subs2_dft_homo'] = -6.5
            f['subs2_dft_lumo'] = -1.5
            f['subs2_dft_gap'] = 5.0
            f['subs2_dft_chemical_potential'] = -4.0
            f['subs2_dft_hardness'] = 2.5
            f['subs2_qsar_mr'] = 0
            f['subs2_qsar_drug_likeness'] = 0
        
        f['substrate_steric_sum'] = f['subs1_length'] + f['subs2_length']
        f['substrate_steric_diff'] = abs(f['subs1_length'] - f['subs2_length'])
        f['substrate_steric_ratio'] = f['subs1_length'] / (f['subs2_length'] + 1)
        f['substrate_steric_product'] = f['subs1_length'] * f['subs2_length']
        f['substrate_steric_euclidean'] = np.sqrt(f['subs1_length'] ** 2 + f['subs2_length'] ** 2)
        
        f['substrate_logp_avg'] = (f['subs1_logp'] + f['subs2_logp']) / 2
        f['substrate_logp_diff'] = abs(f['subs1_logp'] - f['subs2_logp'])
        f['substrate_logp_sum'] = f['subs1_logp'] + f['subs2_logp']
        f['substrate_logp_product'] = f['subs1_logp'] * f['subs2_logp']
        
        f['substrate_mw_avg'] = (f['subs1_mw'] + f['subs2_mw']) / 2
        f['substrate_mw_diff'] = abs(f['subs1_mw'] - f['subs2_mw'])
        f['substrate_mw_ratio'] = f['subs1_mw'] / (f['subs2_mw'] + 1)
        f['substrate_mw_sum'] = f['subs1_mw'] + f['subs2_mw']
        
        f['total_rings'] = f['subs1_rings'] + f['subs2_rings']
        f['ring_diff'] = abs(f['subs1_rings'] - f['subs2_rings'])
        f['ring_product'] = f['subs1_rings'] * f['subs2_rings']
        f['aromatic_sum'] = f['subs1_aromatic_rings'] + f['subs2_aromatic_rings']
        f['aromatic_ratio'] = f['aromatic_sum'] / (f['total_rings'] + 1)
        
        f['sigma_p_sum'] = f['subs1_sigma_p'] + f['subs2_sigma_p']
        f['sigma_p_diff'] = abs(f['subs1_sigma_p'] - f['subs2_sigma_p'])
        f['sigma_p_avg'] = (f['subs1_sigma_p'] + f['subs2_sigma_p']) / 2
        f['sigma_p_product'] = f['subs1_sigma_p'] * f['subs2_sigma_p']
        
        f['taft_es_sum'] = f['subs1_taft_es'] + f['subs2_taft_es']
        f['taft_es_diff'] = abs(f['subs1_taft_es'] - f['subs2_taft_es'])
        f['taft_es_avg'] = (f['subs1_taft_es'] + f['subs2_taft_es']) / 2
        
        f['halogen_total'] = f['subs1_halogen_count'] + f['subs2_halogen_count']
        f['halogen_diff'] = abs(f['subs1_halogen_count'] - f['subs2_halogen_count'])
        f['halogen_product'] = f['subs1_halogen_count'] * f['subs2_halogen_count']
        
        f['electronic_softness'] = f['subs1_sigma_p'] * 0.3
        f['mechanistic_predictor'] = f['subs1_sigma_p'] * 0.3 + 0.7
        f['hammett_effect'] = np.exp(2.8 * f['subs1_sigma_p'])
        f['taft_effect'] = np.exp(1.5 * f['subs1_taft_es'] / 2)
        
        f['dft_homo_sum'] = f['subs1_dft_homo'] + f['subs2_dft_homo']
        f['dft_homo_diff'] = abs(f['subs1_dft_homo'] - f['subs2_dft_homo'])
        f['dft_homo_avg'] = (f['subs1_dft_homo'] + f['subs2_dft_homo']) / 2
        f['dft_gap_sum'] = f['subs1_dft_gap'] + f['subs2_dft_gap']
        f['dft_chemical_potential_avg'] = (f['subs1_dft_chemical_potential'] + f['subs2_dft_chemical_potential']) / 2
        
        vector = []
        for col in self.feature_columns:
            vector.append(f.get(col, 0))
        
        if self.scaler is not None:
            try:
                vector = self.scaler.transform([vector])[0]
            except Exception as e:
                pass
        
        return np.array(vector).reshape(1, -1)
    
    def _calculate_confidence(self, ml_yield: float, chemical_yield: float, final_yield: float) -> float:
        confidence = 0.85
        
        if ml_yield is not None and not np.isnan(ml_yield):
            diff = abs(ml_yield - chemical_yield)
            consistency = 1 - min(diff / 30, 1)
            confidence = confidence * (0.7 + 0.3 * consistency)
        
        if self.models:
            model_count = len(self.models)
            if model_count > 3:
                confidence = confidence * (0.9 + 0.1 * min(model_count / 10, 1))
        
        if final_yield < 10 or final_yield > 90:
            confidence = confidence * 0.95
        
        if self.X is not None:
            data_size = len(self.X)
            if data_size < 10:
                confidence = confidence * 0.8
            elif data_size < 30:
                confidence = confidence * 0.9
        
        if self.cv_scores:
            cv_std = self.cv_scores.get('std', 0.1)
            confidence = confidence * (1 - min(cv_std, 0.2))
        
        return np.clip(confidence, 0.3, 0.98)
    
    def optimize_catalyst(self, conditions: Dict) -> List[Tuple[str, float]]:
        try:
            logger.info("Optimizing catalyst...")
            results = []
            
            catalysts = []
            if self.df is not None and 'catalizor' in self.df.columns:
                catalysts = self.df['catalizor'].unique().tolist()
            else:
                catalysts = [
                    'Pd(PPh3)4', 'PdCl2(dppf)', 'Pd(OAc)2', 'Pd2(dba)3',
                    'PdCl2(PPh3)2', 'Pd(PPh3)2Cl2', 'PdCl2', 'Pd(acac)2',
                    'Pd(PhCN)2Cl2', 'Pd(PPh3)4'
                ]
            
            for idx, catalyst in enumerate(catalysts[:20]):
                test_conditions = conditions.copy()
                test_conditions['catalizor'] = catalyst
                
                best_yield = 0
                best_qty = conditions.get('quantity', 0.0025)
                
                quantities = [0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04]
                for qty in quantities:
                    test_conditions['quantity'] = qty
                    result = self.predict(test_conditions)
                    if result['success'] and result['prediction'] > best_yield:
                        best_yield = result['prediction']
                        best_qty = qty
                
                results.append((catalyst, best_yield, best_qty))
            
            results.sort(key=lambda x: x[1], reverse=True)
            
            formatted = [(cat, float(yield_val)) for cat, yield_val, _ in results[:10]]
            
            return formatted
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return []
    
    def get_best_model(self) -> Tuple[str, Dict]:
        if not self.model_performance:
            return None, {}
        best = max(self.model_performance.items(), key=lambda x: x[1].get('r2', 0))
        return best[0], best[1]
    
    def get_feature_importance(self) -> Dict:
        if not self.is_trained or not self.models:
            return {}
        
        try:
            from sklearn.inspection import permutation_importance
            
            model = list(self.models.values())[0]
            
            X_scaled = self.scaler.transform(self.X)
            result = permutation_importance(model, X_scaled, self.y, n_repeats=10, random_state=42)
            
            importance_dict = {}
            for i, col in enumerate(self.feature_columns):
                importance_dict[col] = {
                    'importance': float(result.importances_mean[i]),
                    'std': float(result.importances_std[i])
                }
            
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1]['importance'], reverse=True)
            
            self.feature_importance = {
                'top_10': sorted_importance[:10],
                'all': importance_dict
            }
            
            return self.feature_importance
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def analyze_residuals(self) -> Dict:
        if not self.is_trained or self.X is None:
            return {}
        
        try:
            X_scaled = self.scaler.transform(self.X)
            predictions = self._ensemble_predict(X_scaled)
            residuals = self.y - predictions
            
            self.residuals = residuals
            self.predictions = predictions
            
            residual_stats = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals)),
                'q1': float(np.percentile(residuals, 25)),
                'q3': float(np.percentile(residuals, 75)),
                'iqr': float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
            }
            
            q1 = np.percentile(residuals, 25)
            q3 = np.percentile(residuals, 75)
            iqr = q3 - q1
            outliers = np.sum((residuals < q1 - 1.5 * iqr) | (residuals > q3 + 1.5 * iqr))
            
            residual_stats['outlier_count'] = int(outliers)
            residual_stats['outlier_ratio'] = float(outliers / len(residuals) if len(residuals) > 0 else 0)
            
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])
                residual_stats['shapiro_wilk_statistic'] = float(shapiro_stat)
                residual_stats['shapiro_wilk_p_value'] = float(shapiro_p)
                residual_stats['is_normal'] = shapiro_p > 0.05
            except:
                pass
            
            return residual_stats
            
        except Exception as e:
            logger.warning(f"Residual analysis failed: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        try:
            import joblib
            
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'ensemble_weights': self.ensemble_weights,
                'model_performance': self.model_performance,
                'best_model': self.best_model,
                'config_hash': self.config.get_xml_hash(),
                'is_enriched': self.is_enriched,
                'cv_scores': self.cv_scores,
                'anova_results': self.anova_results,
                'confidence_intervals': self.confidence_intervals,
                'prediction_intervals': self.prediction_intervals
            }
            
            joblib.dump(model_data, filepath)
            logger.success(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Save model error: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        try:
            import joblib
            
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.ensemble_weights = model_data['ensemble_weights']
            self.model_performance = model_data['model_performance']
            self.best_model = model_data['best_model']
            self.is_enriched = model_data.get('is_enriched', False)
            self.is_trained = True
            
            self.cv_scores = model_data.get('cv_scores')
            self.anova_results = model_data.get('anova_results')
            self.confidence_intervals = model_data.get('confidence_intervals')
            self.prediction_intervals = model_data.get('prediction_intervals')
            
            logger.success(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Load model error: {e}")
            return False

def create_result_images():
    """
    Creates 4 academic-level visualizations:
    1. DFT HOMO-LUMO Energy Diagram
    2. HSAB Hardness-Softness Compatibility Heatmap
    3. Mechanistic Barrier Analysis (OA, TM, RE)
    4. QSAR Drug-Likeness Radar Plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        if PREDICTOR is None or not PREDICTOR.is_trained or PREDICTOR.X is None:
            logger.warning("No trained model available for visualization")
            return []
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        images_dir = os.path.join('static/images', timestamp)
        os.makedirs(images_dir, exist_ok=True)
        logger.info(f"Created image directory: {images_dir}")
        
        X_scaled = PREDICTOR.scaler.transform(PREDICTOR.X)
        y_true = PREDICTOR.y
        ensemble_pred = PREDICTOR._ensemble_predict(X_scaled)
        
        if ensemble_pred is None:
            logger.warning("No predictions available for visualization")
            return []
        
        fig1, ax1 = plt.subplots(figsize=(10, 7))
        
        try:
            subs1_smiles = PREDICTOR.df['subs1'].iloc[0] if 'subs1' in PREDICTOR.df.columns else ''
            subs2_smiles = PREDICTOR.df['subs2'].iloc[0] if 'subs2' in PREDICTOR.df.columns else ''
            product_smiles = PREDICTOR.df['product'].iloc[0] if 'product' in PREDICTOR.df.columns else ''
            
            dft_subs1 = PREDICTOR.chemical.calculate_dft_descriptors(subs1_smiles) if subs1_smiles else {'homo_energy': -6.5, 'lumo_energy': -1.5}
            dft_subs2 = PREDICTOR.chemical.calculate_dft_descriptors(subs2_smiles) if subs2_smiles else {'homo_energy': -6.5, 'lumo_energy': -1.5}
            dft_product = PREDICTOR.chemical.calculate_dft_descriptors(product_smiles) if product_smiles else {'homo_energy': -6.5, 'lumo_energy': -1.5}
            
            molecules = ['Substrate 1', 'Substrate 2', 'Product']
            homo_energies = [dft_subs1.get('homo_energy', -6.5), dft_subs2.get('homo_energy', -6.5), dft_product.get('homo_energy', -6.5)]
            lumo_energies = [dft_subs1.get('lumo_energy', -1.5), dft_subs2.get('lumo_energy', -1.5), dft_product.get('lumo_energy', -1.5)]
            gaps = [lumo - homo for homo, lumo in zip(homo_energies, lumo_energies)]
            
            x = np.arange(len(molecules))
            width = 0.35
            
            bars1 = ax1.bar(x - width/2, homo_energies, width, label='HOMO', color='#EF4444', alpha=0.7, edgecolor='darkred')
            bars2 = ax1.bar(x + width/2, lumo_energies, width, label='LUMO', color='#3B82F6', alpha=0.7, edgecolor='darkblue')
            
            for i, (homo, lumo) in enumerate(zip(homo_energies, lumo_energies)):
                ax1.vlines(i, homo, lumo, color='black', linewidth=2, linestyle='--')
                ax1.text(i, (homo + lumo)/2, f'Gap: {gaps[i]:.2f} eV', ha='center', va='center', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            ax1.set_xticks(x)
            ax1.set_xticklabels(molecules)
            ax1.set_ylabel('Energy (eV)', fontsize=13)
            ax1.set_title('DFT HOMO-LUMO Energy Diagram', fontsize=15, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3, axis='y')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            
            ax1.set_ylim(min(homo_energies) - 1, max(lumo_energies) + 1)
            
            ax1.text(0.02, 0.98, f'Chemical Potential (μ): {dft_subs1.get("chemical_potential", 0):.2f} eV', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top')
            ax1.text(0.02, 0.92, f'Global Hardness (η): {dft_subs1.get("absolute_hardness", 0):.2f} eV', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top')
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'DFT Analysis Error: {str(e)}', ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        filepath1 = os.path.join(images_dir, 'resim1_dft_diagram.png')
        fig1.savefig(filepath1, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig1)
        logger.info(f"Saved DFT diagram: {filepath1}")
        
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        try:
            hsab_labels = ['Pd', 'Halide', 'Ligand', 'Base']
            softness_values = [
                PREDICTOR.chemical.pd_softness,
                PREDICTOR.chemical.halide_softness,
                PREDICTOR.chemical.ligand_softness,
                PREDICTOR.chemical.base_softness
            ]
            hardness_values = [
                PREDICTOR.chemical.absolute_hardness_pd,
                PREDICTOR.chemical.absolute_hardness_halide,
                PREDICTOR.chemical.absolute_hardness_ligand,
                4.0
            ]
            
            softness_matrix = np.array([[1 - abs(softness_values[i] - softness_values[j]) / 6 
                                         for j in range(len(hsab_labels))] 
                                        for i in range(len(hsab_labels))])
            
            hardness_matrix = np.array([[1 - abs(hardness_values[i] - hardness_values[j]) / 8 
                                         for j in range(len(hsab_labels))] 
                                        for i in range(len(hsab_labels))])
            
            combined_matrix = 0.6 * softness_matrix + 0.4 * hardness_matrix
            
            sns.heatmap(combined_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
                       xticklabels=hsab_labels, yticklabels=hsab_labels, 
                       cbar_kws={'label': 'HSAB Compatibility Score'}, ax=ax2)
            
            ax2.set_title('HSAB Pearson Compatibility Matrix\n(Soft-Soft / Hard-Hard Matching)', 
                         fontsize=15, fontweight='bold')
            ax2.set_xlabel('Component', fontsize=13)
            ax2.set_ylabel('Component', fontsize=13)
            
            avg_compatibility = np.mean(combined_matrix)
            ax2.text(0.02, 0.02, f'Overall Compatibility: {avg_compatibility:.3f}', 
                    transform=ax2.transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            classifications = ['Soft' if s > 3.0 else 'Hard' for s in softness_values]
            classification_text = ' | '.join([f'{label}: {cls}' for label, cls in zip(hsab_labels, classifications)])
            ax2.text(0.02, 0.08, classification_text, transform=ax2.transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
        except Exception as e:
            ax2.text(0.5, 0.5, f'HSAB Analysis Error: {str(e)}', ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        filepath2 = os.path.join(images_dir, 'resim2_hsab_heatmap.png')
        fig2.savefig(filepath2, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig2)
        logger.info(f"Saved HSAB heatmap: {filepath2}")
        
        fig3, ax3 = plt.subplots(figsize=(10, 7))
        
        try:
            barriers = [
                PREDICTOR.chemical.oa_barrier,
                PREDICTOR.chemical.tm_barrier,
                PREDICTOR.chemical.re_barrier
            ]
            rates = [
                PREDICTOR.chemical.oa_rate,
                PREDICTOR.chemical.tm_rate,
                PREDICTOR.chemical.re_rate
            ]
            sensitivities = [
                (PREDICTOR.chemical.oa_steric_sens + PREDICTOR.chemical.oa_electronic_sens) / 2,
                (PREDICTOR.chemical.tm_base_sens + PREDICTOR.chemical.tm_boronic_sens) / 2,
                (PREDICTOR.chemical.re_steric_sens + PREDICTOR.chemical.re_electronic_sens) / 2
            ]
            
            steps = ['Oxidative Addition\n(OA)', 'Transmetalation\n(TM)', 'Reductive Elimination\n(RE)']
            x = np.arange(len(steps))
            width = 0.25
            
            bars1 = ax3.bar(x - width, barriers, width, label='Activation Barrier (kJ/mol)', color='#EF4444', alpha=0.8)
            bars2 = ax3.bar(x, [r * 100 for r in rates], width, label='Rate Constant (x100)', color='#3B82F6', alpha=0.8)
            bars3 = ax3.bar(x + width, [s * 100 for s in sensitivities], width, label='Sensitivity (x100)', color='#10B981', alpha=0.8)
            
            ax3.set_xticks(x)
            ax3.set_xticklabels(steps, fontsize=11)
            ax3.set_ylabel('Value', fontsize=13)
            ax3.set_title('Mechanistic Analysis: OA, TM, RE Steps', fontsize=15, fontweight='bold')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3, axis='y')
            
            for i, (bar, rate, sens) in enumerate(zip(barriers, rates, sensitivities)):
                ax3.text(i - width, bar + 1, f'{bar:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax3.text(i, rate * 100 + 1, f'{rate:.3f}', ha='center', va='bottom', fontsize=9)
                ax3.text(i + width, sens * 100 + 1, f'{sens:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax3.text(0.02, 0.98, f'Reaction Coordinate: OA → TM → RE', 
                    transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            rds_index = np.argmax(barriers)
            ax3.text(0.02, 0.92, f'Rate-Determining Step: {steps[rds_index].replace("\\n", " ")}', 
                    transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF3C7', alpha=0.8))
            
        except Exception as e:
            ax3.text(0.5, 0.5, f'Mechanistic Analysis Error: {str(e)}', ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        filepath3 = os.path.join(images_dir, 'resim3_mechanistic_analysis.png')
        fig3.savefig(filepath3, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig3)
        logger.info(f"Saved mechanistic analysis: {filepath3}")
        
        fig4, ax4 = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        try:
            subs1_smiles = PREDICTOR.df['subs1'].iloc[0] if 'subs1' in PREDICTOR.df.columns else ''
            subs2_smiles = PREDICTOR.df['subs2'].iloc[0] if 'subs2' in PREDICTOR.df.columns else ''
            product_smiles = PREDICTOR.df['product'].iloc[0] if 'product' in PREDICTOR.df.columns else ''
            
            qsar_subs1 = PREDICTOR.chemical.calculate_qsar_descriptors(subs1_smiles) if subs1_smiles else {}
            qsar_subs2 = PREDICTOR.chemical.calculate_qsar_descriptors(subs2_smiles) if subs2_smiles else {}
            qsar_product = PREDICTOR.chemical.calculate_qsar_descriptors(product_smiles) if product_smiles else {}
            
            drug_likeness_subs1 = PREDICTOR.chemical.calculate_drug_likeness(subs1_smiles) if subs1_smiles else {'drug_likeness_score': 0, 'qed_score': 0}
            drug_likeness_subs2 = PREDICTOR.chemical.calculate_drug_likeness(subs2_smiles) if subs2_smiles else {'drug_likeness_score': 0, 'qed_score': 0}
            drug_likeness_product = PREDICTOR.chemical.calculate_drug_likeness(product_smiles) if product_smiles else {'drug_likeness_score': 0, 'qed_score': 0}
            
            categories = ['MW', 'LogP', 'HBA', 'HBD', 'RotBonds', 'TPSA']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            
            def normalize_value(value, min_val, max_val):
                if max_val == min_val:
                    return 0.5
                return np.clip((value - min_val) / (max_val - min_val), 0, 1)
            
            values_subs1 = [
                normalize_value(qsar_subs1.get('MW', 200), 50, 600),
                normalize_value(qsar_subs1.get('LogP', 2), -2, 6),
                normalize_value(qsar_subs1.get('HBA', 3), 0, 12),
                normalize_value(qsar_subs1.get('HBD', 1), 0, 6),
                normalize_value(qsar_subs1.get('RotBonds', 3), 0, 15),
                normalize_value(qsar_subs1.get('TPSA', 40), 0, 160)
            ]
            values_subs1 += values_subs1[:1]
            
            values_subs2 = [
                normalize_value(qsar_subs2.get('MW', 200), 50, 600),
                normalize_value(qsar_subs2.get('LogP', 2), -2, 6),
                normalize_value(qsar_subs2.get('HBA', 3), 0, 12),
                normalize_value(qsar_subs2.get('HBD', 1), 0, 6),
                normalize_value(qsar_subs2.get('RotBonds', 3), 0, 15),
                normalize_value(qsar_subs2.get('TPSA', 40), 0, 160)
            ]
            values_subs2 += values_subs2[:1]
            
            values_product = [
                normalize_value(qsar_product.get('MW', 200), 50, 600),
                normalize_value(qsar_product.get('LogP', 2), -2, 6),
                normalize_value(qsar_product.get('HBA', 3), 0, 12),
                normalize_value(qsar_product.get('HBD', 1), 0, 6),
                normalize_value(qsar_product.get('RotBonds', 3), 0, 15),
                normalize_value(qsar_product.get('TPSA', 40), 0, 160)
            ]
            values_product += values_product[:1]
            
            ax4.plot(angles, values_subs1, 'o-', linewidth=2, label='Substrate 1', color='#EF4444', alpha=0.8)
            ax4.fill(angles, values_subs1, alpha=0.1, color='#EF4444')
            
            ax4.plot(angles, values_subs2, 'o-', linewidth=2, label='Substrate 2', color='#3B82F6', alpha=0.8)
            ax4.fill(angles, values_subs2, alpha=0.1, color='#3B82F6')
            
            ax4.plot(angles, values_product, 'o-', linewidth=2, label='Product', color='#10B981', alpha=0.8)
            ax4.fill(angles, values_product, alpha=0.1, color='#10B981')
            
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories, fontsize=12)
            ax4.set_ylim(0, 1)
            ax4.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax4.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
            ax4.set_title('QSAR Drug-Likeness Radar Plot\n(Lipinski Rules Compliance)', 
                         fontsize=15, fontweight='bold', pad=20)
            ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax4.grid(True, alpha=0.3)
            
            qed_text = f"QED Scores: Sub1={drug_likeness_subs1.get('qed_score', 0):.3f} | Sub2={drug_likeness_subs2.get('qed_score', 0):.3f} | Product={drug_likeness_product.get('qed_score', 0):.3f}"
            ax4.text(0.5, -0.15, qed_text, transform=ax4.transAxes, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            dl_text = f"Drug-Likeness: Sub1={drug_likeness_subs1.get('drug_likeness_score', 0):.3f} | Sub2={drug_likeness_subs2.get('drug_likeness_score', 0):.3f} | Product={drug_likeness_product.get('drug_likeness_score', 0):.3f}"
            ax4.text(0.5, -0.20, dl_text, transform=ax4.transAxes, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
        except Exception as e:
            ax4.text(0.5, 0.5, f'QSAR Analysis Error: {str(e)}', ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        filepath4 = os.path.join(images_dir, 'resim4_qsar_radar.png')
        fig4.savefig(filepath4, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig4)
        logger.info(f"Saved QSAR radar plot: {filepath4}")
        
        info_file = os.path.join(images_dir, 'academic_visualizations_info.txt')
        with open(info_file, 'w') as f:
            f.write(f"Academic Visualizations Generated\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Model: {PREDICTOR.best_model if PREDICTOR.best_model else 'Ensemble'}\n")
            f.write(f"Dataset Size: {len(PREDICTOR.df) if PREDICTOR.df is not None else 0}\n")
            f.write(f"Features: {len(PREDICTOR.feature_columns)}\n")
            f.write(f"\nGenerated Images:\n")
            f.write(f"1. resim1_dft_diagram.png - DFT HOMO-LUMO Energy Diagram\n")
            f.write(f"2. resim2_hsab_heatmap.png - HSAB Pearson Compatibility Matrix\n")
            f.write(f"3. resim3_mechanistic_analysis.png - Mechanistic Barrier Analysis (OA/TM/RE)\n")
            f.write(f"4. resim4_qsar_radar.png - QSAR Drug-Likeness Radar Plot\n")
            if PREDICTOR.cv_scores:
                f.write(f"\nCross-Validation: R² = {PREDICTOR.cv_scores['mean']:.3f} ± {PREDICTOR.cv_scores['std']:.3f}\n")
            if PREDICTOR.confidence_intervals:
                f.write(f"Confidence Interval: [{PREDICTOR.confidence_intervals['lower']:.2f}, {PREDICTOR.confidence_intervals['upper']:.2f}]\n")
        
        plot_files = [filepath1, filepath2, filepath3, filepath4]
        logger.success(f"Created {len(plot_files)} academic visualization images in {images_dir}")
        return plot_files
        
    except Exception as e:
        logger.error(f"Error creating academic images: {str(e)}")
        logger.error(traceback.format_exc())
        return []

@predict_ml_bp.route('/')
@error_handler
def index():
    return render_template('predict_ml.html')

@predict_ml_bp.route('/api/get_csv_files', methods=['GET'])
@error_handler
@timing_decorator
def get_csv_files():
    dataset_dir = 'static/datasets'
    os.makedirs(dataset_dir, exist_ok=True)
    
    files = []
    for f in os.listdir(dataset_dir):
        if f.endswith('.csv'):
            path = os.path.join(dataset_dir, f)
            size = os.path.getsize(path)
            try:
                df_sample = pd.read_csv(path, nrows=5)
                is_enriched = is_enriched_dataset(df_sample)
            except:
                is_enriched = False
            
            files.append({
                'name': f,
                'size': format_size(size),
                'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
                'is_enriched': is_enriched
            })
    
    files.sort(key=lambda x: x['name'])
    logger.info(f"Found {len(files)} CSV files")
    
    return jsonify({
        'success': True,
        'files': files,
        'count': len(files)
    })

@predict_ml_bp.route('/api/upload_csv', methods=['POST'])
@error_handler
@timing_decorator
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if not file.filename.endswith('.csv'):
        return jsonify({'success': False, 'message': 'Only CSV files allowed'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join('static/datasets', filename)
    
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    
    max_size = 50 * 1024 * 1024
    if size > max_size:
        return jsonify({'success': False, 'message': f'File too large (max {max_size/1024/1024}MB)'})
    
    file.save(filepath)
    logger.info(f"Uploaded file: {filename} ({format_size(size)})")
    
    return jsonify({
        'success': True,
        'message': f'File uploaded: {filename}',
        'filename': filename,
        'size': format_size(size)
    })

@predict_ml_bp.route('/api/load_data', methods=['POST'])
@error_handler
@timing_decorator
def load_data():
    global PREDICTOR, CONFIG, DATA_INFO, CURRENT_FILE
    
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'success': False, 'message': 'Filename required'})
    
    filepath = os.path.join('static/datasets', filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': f'File not found: {filename}'})
    
    logger.info(f"Loading data from: {filename}")
    
    try:
        test_df = pd.read_csv(filepath, nrows=5)
        if not is_enriched_dataset(test_df):
            logger.error(f"{filename} is a BASIC dataset. ML training not allowed.")
            return jsonify({
                'success': False,
                'message': f'{filename} is a basic dataset without academic features.\n'
                          f'Please use dataset_routes.py to enrich your data first.',
                'is_enriched': False,
                'found_academic_features': [c for c in ACADEMIC_FEATURE_COLUMNS if c in test_df.columns]
            }), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'}), 400
    
    CONFIG = ConfigManager('config/info.xml')
    PREDICTOR = SuzukiPredictor(CONFIG)
    
    try:
        df = PREDICTOR.load_data(filepath)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e),
            'required_columns': REQUIRED_COLUMNS,
            'optional_columns': OPTIONAL_COLUMNS,
            'academic_features_required': ACADEMIC_FEATURE_COLUMNS[:10]
        }), 400
    
    CURRENT_FILE = filename
    
    DATA_INFO = {
        'rows': len(df),
        'columns': df.columns.tolist(),
        'catalysts': df['catalizor'].unique().tolist() if 'catalizor' in df.columns else [],
        'bases': df['base'].unique().tolist() if 'base' in df.columns else [],
        'solvents1': df['solv1'].unique().tolist() if 'solv1' in df.columns else [],
        'solvents2': df['solv2'].unique().tolist() if 'solv2' in df.columns else [],
        'yield_stats': {
            'mean': float(df['yield'].mean()),
            'std': float(df['yield'].std()),
            'min': float(df['yield'].min()),
            'max': float(df['yield'].max()),
            'median': float(df['yield'].median()),
            'q1': float(df['yield'].quantile(0.25)),
            'q3': float(df['yield'].quantile(0.75))
        },
        'feature_count': len(PREDICTOR.feature_columns),
        'feature_columns': PREDICTOR.feature_columns[:20],
        'is_enriched': PREDICTOR.is_enriched
    }
    
    result = PREDICTOR.train('Ensemble')
    
    images = []
    if result['success']:
        images = create_result_images()
    
    if result['success']:
        try:
            importance = PREDICTOR.get_feature_importance()
        except:
            importance = {}
        
        try:
            residuals = PREDICTOR.analyze_residuals()
        except:
            residuals = {}

        response_data = {
            'success': True,
            'message': f"Loaded {len(df)} enriched rows, {result.get('model_count', 0)} models trained",
            'data_info': convert_to_serializable(DATA_INFO),
            'performance': convert_to_serializable(result.get('performance', {})),
            'best_model': result.get('best_model', 'None'),
            'feature_importance': convert_to_serializable(importance.get('top_10', [])),
            'residual_stats': convert_to_serializable(residuals),
            'model_history': convert_to_serializable(PREDICTOR.model_history[-10:]),
            'visualizations': {
                'created': len(images) > 0,
                'image_count': len(images),
                'directory': os.path.dirname(images[0]) if images else None
            },
            'is_enriched': bool(PREDICTOR.is_enriched),
            'academic_analyses': {
                'cv_scores': convert_to_serializable(PREDICTOR.cv_scores),
                'anova_results': convert_to_serializable(PREDICTOR.anova_results),
                'confidence_intervals': convert_to_serializable(PREDICTOR.confidence_intervals),
                'prediction_intervals': convert_to_serializable(PREDICTOR.prediction_intervals)
            }
        }
        
        return jsonify(response_data)
    else:
        return jsonify({
            'success': False,
            'message': result.get('message', 'Training failed')
        })

@predict_ml_bp.route('/api/update_visualizations', methods=['POST'])
@error_handler
def update_visualizations():
    global PREDICTOR
    
    if PREDICTOR is None or not PREDICTOR.is_trained:
        return jsonify({'success': False, 'message': 'Model not trained yet'})
    
    try:
        images = create_result_images()
        return jsonify({
            'success': True,
            'message': f'Created {len(images)} visualizations',
            'images': [os.path.basename(img) for img in images],
            'directory': os.path.dirname(images[0]) if images else None
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@predict_ml_bp.route('/api/change_model', methods=['POST'])
@error_handler
@timing_decorator
def change_model():
    global PREDICTOR, CURRENT_MODEL
    
    data = request.get_json()
    model_name = data.get('model_name')
    
    if not model_name:
        return jsonify({'success': False, 'message': 'Model name required'})
    
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    
    if not PREDICTOR.is_enriched:
        return jsonify({
            'success': False,
            'message': 'Cannot change model on basic dataset. Please use enriched data first.'
        })
    
    model_map = {
        'Random Forest': 'Random_Forest',
        'Gradient Boosting': 'Gradient_Boosting',
        'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM',
        'CatBoost': 'CatBoost',
        'Extra Trees': 'Extra_Trees',
        'KNN': 'KNN',
        'Ridge': 'Ridge',
        'Lasso': 'Lasso',
        'ElasticNet': 'ElasticNet',
        'SVR': 'SVR',
        'Neural Network': 'Neural_Network',
        'Gaussian Process': 'Gaussian_Process'
    }
    
    key = model_map.get(model_name, model_name)
    logger.info(f"Changing model to: {key}")
    
    result = PREDICTOR.train(key)
    
    if result['success']:
        CURRENT_MODEL = model_name
        perf = result.get('performance', {})
        stats = list(perf.values())[0] if perf else {}
        
        images = create_result_images()
        
        return jsonify({
            'success': True,
            'message': f"Switched to {model_name}",
            'current_model': model_name,
            'stats': stats,
            'best_model': result.get('best_model'),
            'visualizations': {
                'created': len(images) > 0,
                'image_count': len(images),
                'directory': os.path.dirname(images[0]) if images else None
            }
        })
    else:
        return jsonify({
            'success': False,
            'message': result.get('message', 'Failed')
        })

@predict_ml_bp.route('/api/save_model', methods=['POST'])
@error_handler
def save_model():
    global PREDICTOR
    
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'No model to save'})
    
    data = request.get_json()
    filename = data.get('filename', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    filepath = os.path.join('static/models', filename)
    
    os.makedirs('static/models', exist_ok=True)
    
    success = PREDICTOR.save_model(filepath)
    
    return jsonify({
        'success': success,
        'message': f"Model saved to {filename}" if success else "Failed to save model",
        'filename': filename if success else None
    })

@predict_ml_bp.route('/api/load_model', methods=['POST'])
@error_handler
def load_model():
    global PREDICTOR, CONFIG
    
    data = request.get_json()
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'success': False, 'message': 'Filename required'})
    
    filepath = os.path.join('static/models', filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': f'Model not found: {filename}'})
    
    CONFIG = ConfigManager('config/info.xml')
    PREDICTOR = SuzukiPredictor(CONFIG)
    
    success = PREDICTOR.load_model(filepath)
    
    images = create_result_images() if success else []
    
    return jsonify({
        'success': success,
        'message': f"Model loaded from {filename}" if success else "Failed to load model",
        'best_model': PREDICTOR.best_model if success else None,
        'is_enriched': PREDICTOR.is_enriched if success else False,
        'visualizations': {
            'created': len(images) > 0,
            'image_count': len(images),
            'directory': os.path.dirname(images[0]) if images else None
        } if success else None
    })

@predict_ml_bp.route('/api/list_models', methods=['GET'])
@error_handler
def list_models():
    models_dir = 'static/models'
    os.makedirs(models_dir, exist_ok=True)
    
    models = []
    for f in os.listdir(models_dir):
        if f.endswith('.pkl'):
            path = os.path.join(models_dir, f)
            models.append({
                'name': f,
                'size': format_size(os.path.getsize(path)),
                'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
            })
    
    models.sort(key=lambda x: x['modified'], reverse=True)
    
    return jsonify({
        'success': True,
        'models': models
    })

@predict_ml_bp.route('/api/make_prediction', methods=['POST'])
@error_handler
@timing_decorator
def make_prediction():
    global PREDICTOR
    
    data = request.get_json()
    
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    
    required = ['temp', 'time', 'quantity', 'catalizor', 'base', 'solv1', 'subs1_smiles', 'subs2_smiles']
    for f in required:
        if f not in data or not data[f]:
            return jsonify({'success': False, 'message': f'Missing: {f}'})
    
    logger.info(f"Prediction request: temp={data['temp']}, time={data['time']}, catalyst={data['catalizor']}")
    
    result = PREDICTOR.predict({
        'temp': float(data['temp']),
        'time': float(data['time']),
        'quantity': float(data['quantity']),
        'catalizor': data['catalizor'],
        'base': data['base'],
        'solv1': data['solv1'],
        'solv2': data.get('solv2', ''),
        'subs1_smiles': data['subs1_smiles'],
        'subs2_smiles': data['subs2_smiles'],
        'sigma_m': float(data.get('sigma_m', 0)),
        'sigma_p': float(data.get('sigma_p', 0)),
        'sigma_plus': float(data.get('sigma_plus', 0)),
        'sigma_minus': float(data.get('sigma_minus', 0)),
        'taft_es': float(data.get('taft_es', 0))
    })
    
    if not result['success']:
        return jsonify({'success': False, 'message': result.get('message', 'Prediction failed')})
    
    mol_img = None
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mols = []
        labels = []
        for s, label in [(data.get('subs1_smiles'), 'Boronic Acid'), 
                        (data.get('subs2_smiles'), 'Aryl Halide')]:
            if s:
                m = Chem.MolFromSmiles(s)
                if m:
                    mols.append(m)
                    labels.append(label)
        if mols:
            img = Draw.MolsToGridImage(mols, molsPerRow=min(2, len(mols)), 
                                       subImgSize=(250, 250), legends=labels)
            buff = io.BytesIO()
            img.save(buff, format="PNG")
            mol_img = base64.b64encode(buff.getvalue()).decode()
    except Exception as e:
        pass
    
    return jsonify({
        'success': True,
        'prediction': result['prediction'],
        'ml_prediction': result.get('ml_prediction'),
        'ml_std': result.get('ml_std'),
        'chemical_prediction': result.get('chemical_prediction'),
        'model': result['model'],
        'yield_class': result.get('yield_class', 'Unknown'),
        'yield_class_color': result.get('yield_class_color', '#6B7280'),
        'confidence': result.get('confidence', 0.85),
        'best_model': result.get('best_model', 'None'),
        'model_count': result.get('model_count', 0),
        'is_enriched': result.get('is_enriched', False),
        'molecule_image': mol_img,
        'academic_details': result.get('academic_details', {}),
        'confidence_interval': result.get('confidence_interval'),
        'prediction_interval': result.get('prediction_interval')
    })

@predict_ml_bp.route('/api/optimize_catalyst', methods=['POST'])
@error_handler
@timing_decorator
def optimize_catalyst():
    global PREDICTOR
    data = request.get_json()
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    
    required = ['temp', 'time', 'quantity', 'base', 'solv1', 'subs1_smiles', 'subs2_smiles']
    for f in required:
        if f not in data or not data[f]:
            return jsonify({'success': False, 'message': f'Missing: {f}'})
    
    logger.info(f"Optimization request: temp={data['temp']}, time={data['time']}")
    
    results = PREDICTOR.optimize_catalyst({
        'temp': float(data['temp']),
        'time': float(data['time']),
        'quantity': float(data['quantity']),
        'base': data['base'],
        'solv1': data['solv1'],
        'solv2': data.get('solv2', ''),
        'subs1_smiles': data['subs1_smiles'],
        'subs2_smiles': data['subs2_smiles']
    })
    
    if not results:
        return jsonify({'success': False, 'message': 'Optimization failed'})
    
    return jsonify({
        'success': True,
        'results': results,
        'model': 'Ensemble'
    })

@predict_ml_bp.route('/api/model_performance', methods=['GET'])
@error_handler
def model_performance():
    global PREDICTOR, DATA_INFO
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    
    perf = PREDICTOR.model_performance or {}
    best_name, best_stats = PREDICTOR.get_best_model()
    residuals = PREDICTOR.analyze_residuals()
    
    return jsonify({
        'success': True,
        'stats': {
            'data_size': DATA_INFO['rows'] if DATA_INFO else 0,
            'current_model': CURRENT_MODEL,
            'yield_mean': DATA_INFO['yield_stats']['mean'] if DATA_INFO and 'yield_stats' in DATA_INFO else 0,
            'yield_std': DATA_INFO['yield_stats']['std'] if DATA_INFO and 'yield_stats' in DATA_INFO else 0,
            'best_model': best_name,
            'best_r2': best_stats.get('r2', 0) if best_stats else 0,
            'model_count': len(PREDICTOR.models),
            'feature_count': len(PREDICTOR.feature_columns),
            'is_trained': PREDICTOR.is_trained,
            'is_enriched': PREDICTOR.is_enriched,
            'performances': perf,
            'residuals': residuals,
            'model_history': PREDICTOR.model_history[-10:] if PREDICTOR.model_history else [],
            'cv_scores': PREDICTOR.cv_scores,
            'anova_results': PREDICTOR.anova_results,
            'confidence_intervals': PREDICTOR.confidence_intervals,
            'prediction_intervals': PREDICTOR.prediction_intervals
        }
    })

@predict_ml_bp.route('/api/model_comparison', methods=['POST'])
@error_handler
def model_comparison():
    global PREDICTOR
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': 'Filename required'})
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    
    perf = PREDICTOR.model_performance or {}
    results = {}
    for name, p in perf.items():
        results[name] = {
            'mean_score': p.get('r2', 0),
            'std_score': p.get('cv_std', 0.02),
            'scores': [p.get('r2', 0)],
            'mae': p.get('mae', 0),
            'rmse': p.get('rmse', 0),
            'mape': p.get('mape', 0),
            'cv_mean': p.get('cv_mean', 0),
            'cv_std': p.get('cv_std', 0)
        }
    
    return jsonify({'success': True, 'results': results})

@predict_ml_bp.route('/api/feature_importance', methods=['GET'])
@error_handler
def feature_importance():
    global PREDICTOR
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    
    importance = PREDICTOR.get_feature_importance()
    return jsonify({
        'success': True,
        'feature_importance': importance.get('top_10', []),
        'all_features': importance.get('all', {})
    })

@predict_ml_bp.route('/api/prediction_history', methods=['GET'])
@error_handler
def prediction_history():
    global PREDICTION_HISTORY
    return jsonify({
        'success': True,
        'history': PREDICTION_HISTORY[-50:],
        'count': len(PREDICTION_HISTORY)
    })

@predict_ml_bp.route('/api/get_xml_config', methods=['GET'])
@error_handler
def get_xml_config():
    path = 'config/info.xml'
    if not os.path.exists(path):
        return jsonify({'success': False, 'message': 'Config not found'})
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify({
        'success': True, 
        'content': content,
        'path': path,
        'size': len(content),
        'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat()
    })

@predict_ml_bp.route('/api/update_xml_config', methods=['POST'])
@error_handler
def update_xml_config():
    global CONFIG
    data = request.get_json()
    content = data.get('content')
    if not content:
        return jsonify({'success': False, 'message': 'Content required'})
    path = 'config/info.xml'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        ET.fromstring(content)
    except ET.ParseError as e:
        return jsonify({'success': False, 'message': f'Invalid XML: {str(e)}'})
    if os.path.exists(path):
        backup_path = f"{path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(path, backup_path)
        logger.info(f"Config backed up to {backup_path}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    CONFIG = ConfigManager(path)
    logger.success("Config updated successfully")
    return jsonify({
        'success': True, 
        'message': 'Config updated successfully',
        'hash': CONFIG.get_xml_hash()
    })

@predict_ml_bp.route('/api/reset_xml_config', methods=['POST'])
@error_handler
def reset_xml_config():
    global CONFIG
    path = 'config/info.xml'
    if os.path.exists(path):
        backup_path = f"{path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(path, backup_path)
        logger.info(f"Config backed up to {backup_path}")
    
    CONFIG = ConfigManager(path)
    logger.success("Config reset to default")
    
    return jsonify({
        'success': True,
        'message': 'Config reset to default',
        'content': CONFIG.get_raw_xml()
    })

@predict_ml_bp.route('/api/get_config_summary', methods=['GET'])
@error_handler
def get_config_summary():
    global CONFIG
    
    if CONFIG is None:
        return jsonify({'success': False, 'message': 'Config not loaded'})
    return jsonify({
        'success': True,
        'summary': CONFIG.get_summary(),
        'chemical_params': CONFIG.get_chemical_params(),
        'models': CONFIG.get_models_list()
    })

@predict_ml_bp.route('/api/validate_smiles', methods=['POST'])
@error_handler
def validate_smiles():
    data = request.get_json()
    smiles = data.get('smiles', '')
    if not smiles:
        return jsonify({'valid': False, 'message': 'Empty SMILES'})
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return jsonify({
                'valid': True,
                'message': 'Valid SMILES',
                'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                'mw': Descriptors.ExactMolWt(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms()
            })
        else:
            return jsonify({'valid': False, 'message': 'Invalid SMILES'})
    except Exception as e:
        return jsonify({'valid': False, 'message': f'Error: {str(e)}'})

@predict_ml_bp.route('/api/get_data_info', methods=['POST'])
@error_handler
def get_data_info():
    global DATA_INFO
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': 'Filename required'})
    if DATA_INFO is None:
        return jsonify({'success': False, 'message': 'No data loaded'})
    return jsonify({'success': True, 'info': DATA_INFO})

@predict_ml_bp.route('/api/export_results', methods=['POST'])
@error_handler
def export_results():
    data = request.get_json()
    results = data.get('results', [])
    fmt = data.get('format', 'csv')
    
    if not results:
        return jsonify({'success': False, 'message': 'No results'})
    
    df = pd.DataFrame(results)
    
    if fmt == 'csv':
        return jsonify({
            'success': True, 
            'content': df.to_csv(index=False), 
            'format': 'csv',
            'filename': f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        })
    elif fmt == 'json':
        return jsonify({
            'success': True, 
            'content': df.to_json(orient='records'), 
            'format': 'json',
            'filename': f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        })
    elif fmt == 'excel':
        buff = io.BytesIO()
        with pd.ExcelWriter(buff, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        buff.seek(0)
        content = base64.b64encode(buff.getvalue()).decode()
        return jsonify({
            'success': True,
            'content': content,
            'format': 'excel',
            'filename': f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        })
    else:
        return jsonify({'success': False, 'message': f'Unsupported format: {fmt}'})

@predict_ml_bp.route('/api/clear_cache', methods=['POST'])
@error_handler
def clear_cache():
    global CACHE, CACHE_HIT, CACHE_MISS
    cache_size = len(CACHE)
    CACHE.clear()
    CACHE_HIT = 0
    CACHE_MISS = 0
    logger.info(f"Cache cleared: {cache_size} entries")
    return jsonify({
        'success': True, 
        'message': f'Cache cleared ({cache_size} entries)',
        'cleared': cache_size
    })

@predict_ml_bp.route('/api/get_logs', methods=['GET'])
@error_handler
def get_logs():
    level = request.args.get('level')
    limit = request.args.get('limit', 100)
    
    logs = logger.get_logs(level)
    logs = logs[-int(limit):]
    
    return jsonify({
        'success': True,
        'logs': logs,
        'count': len(logs),
        'summary': logger.get_summary()
    })

@predict_ml_bp.route('/api/clear_logs', methods=['POST'])
@error_handler
def clear_logs():
    logger.clear()
    return jsonify({'success': True, 'message': 'Logs cleared'})

@predict_ml_bp.route('/api/get_yield_stats', methods=['GET'])
@error_handler
def get_yield_stats():
    global PREDICTOR

    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    
    stats = PREDICTOR.chemical.get_yield_stats()
    return jsonify({'success': True, 'stats': stats})

@predict_ml_bp.route('/api/get_model_list', methods=['GET'])
@error_handler
def get_model_list():
    global CONFIG
    
    if CONFIG is None:
        return jsonify({'success': False, 'message': 'Config not loaded'})
    
    models = CONFIG.get_models_list()
    return jsonify({'success': True, 'models': models})

@predict_ml_bp.route('/api/health', methods=['GET'])
@error_handler
def health_check():
    return jsonify({
        'success': True,
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'predictor_loaded': PREDICTOR is not None,
        'config_loaded': CONFIG is not None,
        'data_loaded': DATA_INFO is not None,
        'models_trained': PREDICTOR.is_trained if PREDICTOR else False,
        'model_count': len(PREDICTOR.models) if PREDICTOR else 0,
        'is_enriched': PREDICTOR.is_enriched if PREDICTOR else False,
        'cache_size': len(CACHE),
        'cache_hit': CACHE_HIT,
        'cache_miss': CACHE_MISS,
        'log_count': len(logger.logs),
        'academic_features': {
            'dft_available': GP_AVAILABLE,
            'shap_available': SHAP_AVAILABLE,
            'statsmodels_available': STATSMODELS_AVAILABLE,
            'cv_scores': PREDICTOR.cv_scores if PREDICTOR else None,
            'confidence_intervals': PREDICTOR.confidence_intervals if PREDICTOR else None,
            'prediction_intervals': PREDICTOR.prediction_intervals if PREDICTOR else None
        }
    })

def init_app():
    os.makedirs('static/datasets', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('static/models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    if not os.path.exists('config/info.xml'):
        ConfigManager('config/info.xml')
        logger.info("Default config created")
    
    logger.success("Application initialized successfully")

init_app()
