from flask import Blueprint, render_template, request, jsonify, session
import os
import json
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import io
import base64
from functools import wraps
import time
import re
import joblib
import warnings
import hashlib
from collections import defaultdict
from scipy import stats
from scipy.constants import R, h, k as boltzmann_k
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, explained_variance_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    PERM_IMP_AVAILABLE = True
except ImportError:
    PERM_IMP_AVAILABLE = False

try:
    from scipy.stats import shapiro
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

predict_ml_bp = Blueprint('predict_ml', __name__, url_prefix='/predict_ml')

PREDICTION_HISTORY_FILE = 'prediction_history.json'
PREDICTION_HISTORY = []

HAMMETT_SIGMA = {
    'H': {'sigma_m': 0.00, 'sigma_p': 0.00, 'taft_es': 0.00},
    'CH3': {'sigma_m': -0.07, 'sigma_p': -0.17, 'taft_es': 0.00},
    'OCH3': {'sigma_m': 0.12, 'sigma_p': -0.27, 'taft_es': -0.20},
    'OH': {'sigma_m': 0.12, 'sigma_p': -0.37, 'taft_es': -0.51},
    'F': {'sigma_m': 0.34, 'sigma_p': 0.06, 'taft_es': -0.46},
    'Cl': {'sigma_m': 0.37, 'sigma_p': 0.23, 'taft_es': -0.97},
    'Br': {'sigma_m': 0.39, 'sigma_p': 0.23, 'taft_es': -1.16},
    'I': {'sigma_m': 0.35, 'sigma_p': 0.18, 'taft_es': -1.40},
    'NO2': {'sigma_m': 0.71, 'sigma_p': 0.78, 'taft_es': -1.01},
    'CN': {'sigma_m': 0.56, 'sigma_p': 0.66, 'taft_es': -0.51},
    'CF3': {'sigma_m': 0.43, 'sigma_p': 0.54, 'taft_es': -2.40},
    'COOH': {'sigma_m': 0.37, 'sigma_p': 0.45, 'taft_es': -1.20},
    'COOCH3': {'sigma_m': 0.35, 'sigma_p': 0.39, 'taft_es': -1.10},
    'CHO': {'sigma_m': 0.36, 'sigma_p': 0.42, 'taft_es': -1.20},
    'NH2': {'sigma_m': -0.16, 'sigma_p': -0.66, 'taft_es': -0.20},
    'N(CH3)2': {'sigma_m': -0.15, 'sigma_p': -0.83, 'taft_es': -0.30},
    'SO2CH3': {'sigma_m': 0.60, 'sigma_p': 0.72, 'taft_es': -1.50},
    'B(OH)2': {'sigma_m': 0.04, 'sigma_p': -0.10, 'taft_es': 0.00},
    'Si(CH3)3': {'sigma_m': -0.04, 'sigma_p': -0.07, 'taft_es': -0.80},
    'C(CH3)3': {'sigma_m': -0.10, 'sigma_p': -0.20, 'taft_es': -1.54},
    'C6H5': {'sigma_m': 0.06, 'sigma_p': -0.01, 'taft_es': -1.20},
    'COCH3': {'sigma_m': 0.38, 'sigma_p': 0.50, 'taft_es': -1.20},
    'SO2NH2': {'sigma_m': 0.55, 'sigma_p': 0.62, 'taft_es': -1.30},
    'NHAc': {'sigma_m': 0.21, 'sigma_p': 0.00, 'taft_es': -0.50},
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
    'water': {'dielectric': 80.1, 'donor_number': 18.0, 'polarity_index': 10.2, 'alpha': 1.17, 'beta': 0.47, 'pi_star': 1.09, 'reichardt_et30': 63.1, 'hildebrand_delta': 47.8},
    'methanol': {'dielectric': 32.7, 'donor_number': 19.0, 'polarity_index': 5.1, 'alpha': 0.93, 'beta': 0.62, 'pi_star': 0.60, 'reichardt_et30': 55.5, 'hildebrand_delta': 29.7},
    'ethanol': {'dielectric': 24.6, 'donor_number': 19.2, 'polarity_index': 4.3, 'alpha': 0.83, 'beta': 0.77, 'pi_star': 0.54, 'reichardt_et30': 51.9, 'hildebrand_delta': 26.5},
    'isopropanol': {'dielectric': 19.9, 'donor_number': 18.5, 'polarity_index': 3.9, 'alpha': 0.76, 'beta': 0.84, 'pi_star': 0.48, 'reichardt_et30': 48.6, 'hildebrand_delta': 23.5},
    'acetone': {'dielectric': 20.7, 'donor_number': 17.0, 'polarity_index': 5.1, 'alpha': 0.08, 'beta': 0.48, 'pi_star': 0.71, 'reichardt_et30': 42.2, 'hildebrand_delta': 19.7},
    'acetonitrile': {'dielectric': 37.5, 'donor_number': 14.1, 'polarity_index': 5.8, 'alpha': 0.19, 'beta': 0.31, 'pi_star': 0.75, 'reichardt_et30': 46.0, 'hildebrand_delta': 24.3},
    'dmso': {'dielectric': 46.7, 'donor_number': 29.8, 'polarity_index': 7.2, 'alpha': 0.00, 'beta': 0.76, 'pi_star': 1.00, 'reichardt_et30': 45.1, 'hildebrand_delta': 26.7},
    'dmf': {'dielectric': 36.7, 'donor_number': 26.6, 'polarity_index': 6.4, 'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88, 'reichardt_et30': 43.8, 'hildebrand_delta': 24.9},
    'thf': {'dielectric': 7.5, 'donor_number': 20.0, 'polarity_index': 4.0, 'alpha': 0.00, 'beta': 0.55, 'pi_star': 0.58, 'reichardt_et30': 37.4, 'hildebrand_delta': 18.5},
    'dioxane': {'dielectric': 2.2, 'donor_number': 14.8, 'polarity_index': 4.8, 'alpha': 0.00, 'beta': 0.37, 'pi_star': 0.49, 'reichardt_et30': 36.0, 'hildebrand_delta': 19.9},
    'toluene': {'dielectric': 2.4, 'donor_number': 0.1, 'polarity_index': 2.4, 'alpha': 0.00, 'beta': 0.11, 'pi_star': 0.54, 'reichardt_et30': 33.9, 'hildebrand_delta': 18.2},
    'benzene': {'dielectric': 2.3, 'donor_number': 0.1, 'polarity_index': 2.7, 'alpha': 0.00, 'beta': 0.10, 'pi_star': 0.59, 'reichardt_et30': 34.5, 'hildebrand_delta': 18.6},
    'dichloromethane': {'dielectric': 8.9, 'donor_number': 0.0, 'polarity_index': 3.1, 'alpha': 0.13, 'beta': 0.10, 'pi_star': 0.82, 'reichardt_et30': 41.1, 'hildebrand_delta': 20.2},
    'chloroform': {'dielectric': 4.8, 'donor_number': 0.0, 'polarity_index': 4.1, 'alpha': 0.44, 'beta': 0.00, 'pi_star': 0.58, 'reichardt_et30': 39.1, 'hildebrand_delta': 19.0},
    'hexane': {'dielectric': 1.9, 'donor_number': 0.0, 'polarity_index': 0.1, 'alpha': 0.00, 'beta': 0.00, 'pi_star': 0.00, 'reichardt_et30': 31.0, 'hildebrand_delta': 14.9},
    'cyclohexane': {'dielectric': 2.0, 'donor_number': 0.0, 'polarity_index': 0.2, 'alpha': 0.00, 'beta': 0.00, 'pi_star': 0.00, 'reichardt_et30': 31.2, 'hildebrand_delta': 16.7},
    'ethyl acetate': {'dielectric': 6.0, 'donor_number': 14.0, 'polarity_index': 4.4, 'alpha': 0.00, 'beta': 0.45, 'pi_star': 0.55, 'reichardt_et30': 38.1, 'hildebrand_delta': 18.2},
    'diethyl ether': {'dielectric': 4.3, 'donor_number': 19.2, 'polarity_index': 2.8, 'alpha': 0.00, 'beta': 0.47, 'pi_star': 0.27, 'reichardt_et30': 34.6, 'hildebrand_delta': 15.4},
    'pyridine': {'dielectric': 12.3, 'donor_number': 33.1, 'polarity_index': 5.3, 'alpha': 0.00, 'beta': 0.64, 'pi_star': 0.87, 'reichardt_et30': 40.2, 'hildebrand_delta': 21.8},
    'nmp': {'dielectric': 32.2, 'donor_number': 27.3, 'polarity_index': 6.7, 'alpha': 0.00, 'beta': 0.77, 'pi_star': 0.92, 'reichardt_et30': 42.0, 'hildebrand_delta': 23.1},
    'dme': {'dielectric': 7.2, 'donor_number': 19.5, 'polarity_index': 3.5, 'alpha': 0.00, 'beta': 0.53, 'pi_star': 0.53, 'reichardt_et30': 36.5, 'hildebrand_delta': 17.6},
}

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
CACHE_HIT = 0
CACHE_MISS = 0

REQUIRED_COLUMNS = ['yield', 'temp', 'time', 'quantity', 'catalizor', 'base', 'solv1']
OPTIONAL_COLUMNS = ['solv2', 'subs1', 'subs2', 'product']
NULLABLE_COLUMNS = ['solv1', 'solv2']
FAILURE_COLUMN = 'yield'
CRITICAL_NON_NULLABLE_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in NULLABLE_COLUMNS and c != FAILURE_COLUMN]

ACADEMIC_FEATURE_COLUMNS = [
    'subs1_SMILES_logp', 'subs1_SMILES_sigma_p', 'subs1_SMILES_sigma_m',
    'subs1_SMILES_taft_es', 'subs1_SMILES_hba', 'subs1_SMILES_hbd',
    'subs1_SMILES_complexity', 'subs1_SMILES_kappa1',
    'subs2_SMILES_logp', 'subs2_SMILES_sigma_p', 'subs2_SMILES_sigma_m',
    'subs2_SMILES_taft_es', 'subs2_SMILES_hba', 'subs2_SMILES_hbd',
    'subs2_SMILES_complexity', 'subs2_SMILES_kappa1',
    'hsab_overall_compatibility', 'hsab_pd_halide_mismatch',
    'reaction_rate_indicator',
    'elecproxy_homo_energy', 'elecproxy_lumo_energy', 'elecproxy_gap_energy',
    'elecproxy_chemical_potential', 'elecproxy_absolute_hardness', 'elecproxy_electrophilicity',
    'elecproxy_fukui_plus', 'elecproxy_fukui_minus',
    'physchem_proxy_score'
]

MAX_SMILES_LENGTH = 2000
MAX_TEXT_FIELD_LENGTH = 500


def classify_and_filter_rows(df):
    working = df.copy()
    present_critical = [c for c in CRITICAL_NON_NULLABLE_COLUMNS if c in working.columns]
    missing_critical_cols = [c for c in CRITICAL_NON_NULLABLE_COLUMNS if c not in working.columns]
    if missing_critical_cols:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(missing_critical_cols)}")
    critical_ok_mask = working[present_critical].notnull().all(axis=1)
    rejected_df = working[~critical_ok_mask].copy()
    valid_structure_df = working[critical_ok_mask].copy()
    if FAILURE_COLUMN in valid_structure_df.columns:
        yield_present_mask = valid_structure_df[FAILURE_COLUMN].notnull()
    else:
        yield_present_mask = pd.Series(False, index=valid_structure_df.index)
    usable_df = valid_structure_df[yield_present_mask].copy()
    failed_df = valid_structure_df[~yield_present_mask].copy()
    return usable_df, failed_df, rejected_df


def save_prediction_history(prediction_data):
    global PREDICTION_HISTORY
    try:
        history_file = PREDICTION_HISTORY_FILE
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    PREDICTION_HISTORY = json.load(f)
            except:
                PREDICTION_HISTORY = []
        entry = {
            'timestamp': datetime.now().isoformat(),
            'conditions': {
                'temp': prediction_data.get('temp'),
                'time': prediction_data.get('time'),
                'quantity': prediction_data.get('quantity'),
                'catalizor': prediction_data.get('catalizor'),
                'base': prediction_data.get('base'),
                'solv1': prediction_data.get('solv1'),
                'solv2': prediction_data.get('solv2'),
                'subs1_smiles': prediction_data.get('subs1_smiles'),
                'subs2_smiles': prediction_data.get('subs2_smiles')
            },
            'result': {
                'yield': prediction_data.get('yield'),
                'yield_class': prediction_data.get('yield_class'),
                'model': prediction_data.get('model')
            }
        }
        PREDICTION_HISTORY.append(entry)
        if len(PREDICTION_HISTORY) > 1000:
            PREDICTION_HISTORY = PREDICTION_HISTORY[-1000:]
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(PREDICTION_HISTORY, f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass


def load_prediction_history():
    global PREDICTION_HISTORY
    try:
        if os.path.exists(PREDICTION_HISTORY_FILE):
            with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                PREDICTION_HISTORY = json.load(f)
        else:
            PREDICTION_HISTORY = []
    except Exception as e:
        PREDICTION_HISTORY = []


def get_last_prediction_for_conditions(conditions):
    global PREDICTION_HISTORY
    if not PREDICTION_HISTORY:
        return None
    key_fields = ['catalizor', 'base', 'solv1', 'solv2', 'subs1_smiles', 'subs2_smiles']
    for entry in reversed(PREDICTION_HISTORY):
        cond = entry.get('conditions', {})
        match = True
        for field in key_fields:
            if cond.get(field) != conditions.get(field):
                match = False
                break
        if match:
            return {
                'temp': cond.get('temp'),
                'time': cond.get('time'),
                'quantity': cond.get('quantity'),
                'yield': entry.get('result', {}).get('yield'),
                'timestamp': entry.get('timestamp')
            }
    return None


def get_dominated_history_max_yield(conditions):
    global PREDICTION_HISTORY
    if not PREDICTION_HISTORY:
        return None
    key_fields = ['catalizor', 'base', 'solv1', 'solv2', 'subs1_smiles', 'subs2_smiles']
    current_temp = conditions.get('temp')
    current_time = conditions.get('time')
    current_quantity = conditions.get('quantity')
    if current_temp is None or current_time is None or current_quantity is None:
        return None
    best = None
    for entry in PREDICTION_HISTORY:
        cond = entry.get('conditions', {})
        same_system = all(cond.get(field) == conditions.get(field) for field in key_fields)
        if not same_system:
            continue
        prev_temp = cond.get('temp')
        prev_time = cond.get('time')
        prev_quantity = cond.get('quantity')
        prev_yield = entry.get('result', {}).get('yield')
        if prev_temp is None or prev_time is None or prev_quantity is None or prev_yield is None:
            continue
        dominated = (current_temp >= prev_temp and current_time >= prev_time and current_quantity >= prev_quantity)
        if dominated and (best is None or prev_yield > best['yield']):
            best = {
                'yield': prev_yield,
                'temp': prev_temp,
                'time': prev_time,
                'quantity': prev_quantity,
                'timestamp': entry.get('timestamp')
            }
    return best


def calculate_logarithmic_increase(base_value, new_value, base_yield, max_increase=2.0):
    if new_value <= base_value:
        return 0.0
    diff = new_value - base_value
    log_factor = np.log1p(diff / max(base_value, 1.0))
    max_log_increase = np.log1p(max_increase / max(base_value, 1.0))
    normalized = log_factor / max(max_log_increase, 0.001)
    increase = normalized * max_increase * 0.35
    return min(increase, max_increase)


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


def clean_feature_name(name):
    tr_map = {'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c'}
    for old, new in tr_map.items():
        name = name.replace(old, new)
    name = re.sub(r'[^a-zA-Z0-9_.]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if len(name) > 50:
        digest = hashlib.md5(name.encode('utf-8')).hexdigest()[:10]
        name = name[:38].rstrip('_') + '_' + digest
    return name if name else 'feature'


def _dedupe_columns(df):
    seen = {}
    new_cols = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_dup{seen[col]}")
    df = df.copy()
    df.columns = new_cols
    return df


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
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)


def is_enriched_dataset(df):
    cols = df.columns.tolist()
    academic_count = sum(1 for col in ACADEMIC_FEATURE_COLUMNS if col in cols)
    if academic_count >= 3:
        return True
    smiles_cols = [c for c in cols if '_SMILES_' in c]
    if len(smiles_cols) >= 5:
        return True
    sigma_cols = [c for c in cols if '_sigma_' in c]
    if len(sigma_cols) >= 2:
        return True
    return False


def validate_reaction_conditions(conditions):
    if 'yield' in conditions and conditions['yield'] is not None and conditions['yield'] != '':
        try:
            yield_val = float(conditions['yield'])
            if yield_val < 0 or yield_val > 100:
                return False, "Yield must be between 0 and 100"
        except:
            return False, "Invalid yield value"
    else:
        return True, "Yield is null - reaction recorded as failed"
    for field in CRITICAL_NON_NULLABLE_COLUMNS:
        if field == 'yield':
            continue
        value = conditions.get(field)
        if value is None or value == '':
            return False, f"Required field '{field}' is missing"
    if conditions.get('solv2'):
        return True, "Two-solvent system detected"
    if conditions.get('solv1'):
        return True, "Single-solvent system"
    return True, "No solvent recorded"


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

    def set_level(self, level):
        if level in self.levels:
            self.current_level = level

    def log(self, msg, level='INFO'):
        if self.levels.get(level, 1) < self.levels.get(self.current_level, 1):
            return
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        elapsed = (datetime.now() - self.start_time).total_seconds()
        log_entry = {'timestamp': timestamp, 'elapsed': elapsed, 'level': level, 'message': msg}
        self.logs.append(log_entry)
        color = self.colors.get(level, '')
        reset = self.reset
        print(f"{color}[{timestamp}] [{level}] [{elapsed:.1f}s] {msg}{reset}")
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]

    def info(self, msg): self.log(msg, 'INFO')
    def success(self, msg): self.log(msg, 'SUCCESS')
    def debug(self, msg): self.log(msg, 'DEBUG')
    def warning(self, msg): self.log(msg, 'WARNING')
    def error(self, msg): self.log(msg, 'ERROR')

    def get_logs(self, level=None):
        if level:
            return [l for l in self.logs if l['level'] == level]
        return self.logs

    def clear(self):
        self.logs = []

    def get_summary(self):
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
        except (TypeError, KeyError, ValueError) as e:
            logger.error(f"Bad request in {func.__name__}: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'Invalid or incomplete request data',
                'timestamp': datetime.now().isoformat()
            }), 400
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            debug_on = os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes')
            return jsonify({
                'success': False,
                'message': str(e) if debug_on else 'An internal error occurred. Please try again.',
                'traceback': traceback.format_exc() if debug_on else None,
                'timestamp': datetime.now().isoformat()
            }), 500
    return wrapper


def get_json_body():
    try:
        data = request.get_json(silent=True, force=False)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {}
    return data


def to_float(value, default=None, field_name='value'):
    if value is None or value == '':
        if default is not None:
            return default
        raise ValueError(f'{field_name} is required and must be a number')
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field_name} must be a valid number')
    if not np.isfinite(f):
        raise ValueError(f'{field_name} must be a finite number')
    return f


def clean_text(value, max_length=MAX_TEXT_FIELD_LENGTH):
    if value is None:
        return ''
    text = str(value).strip()
    return text[:max_length]


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def cache_result(ttl=300, max_size=100):
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
                    return entry['data']
                else:
                    del CACHE[cache_key]
            CACHE_MISS += 1
            result = func(*args, **kwargs)
            CACHE[cache_key] = {'data': result, 'time': time.time(), 'hits': 0}
            if len(CACHE) > max_size:
                oldest = min(CACHE.keys(), key=lambda k: CACHE[k]['time'])
                del CACHE[oldest]
            return result
        return wrapper
    return decorator


class ConfigManager:
    def __init__(self, config_path='config/info.xml'):
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
            else:
                self._create_default()
        except:
            self._create_default()

    def _create_default(self):
        default = """<?xml version="1.0" encoding="UTF-8"?>
<suzuki_config version="7.4.0">
    <metadata>
        <version>7.4.0</version>
        <last_updated>2026-07-19</last_updated>
    </metadata>
    <chemical_intuition>
        <temperature>
            <baseline_temp>25</baseline_temp>
            <min_temp>25</min_temp>
            <max_temp>250</max_temp>
            <low_temp_penalty>0.20</low_temp_penalty>
            <high_temp_penalty>0.85</high_temp_penalty>
            <optimal_temp>110</optimal_temp>
            <temp_coefficient>0.85</temp_coefficient>
            <arrhenius_prefactor>1.2e13</arrhenius_prefactor>
            <activation_energy>63.0</activation_energy>
            <gas_constant>8.314</gas_constant>
            <temperature_effect_power>0.65</temperature_effect_power>
            <temperature_scaling>0.5</temperature_scaling>
            <logarithmic_gain>0.55</logarithmic_gain>
            <linear_gain>0.30</linear_gain>
            <sigmoidal_gain>0.15</sigmoidal_gain>
            <sigmoidal_steepness>0.06</sigmoidal_steepness>
            <sigmoidal_midpoint>70</sigmoidal_midpoint>
            <enthalpy_activation>35.2</enthalpy_activation>
            <entropy_activation>-16.5</entropy_activation>
            <pre_exponential_factor>1.0e13</pre_exponential_factor>
            <degradation_threshold>120</degradation_threshold>
            <degradation_penalty>0.35</degradation_penalty>
            <degradation_rate>0.045</degradation_rate>
            <degradation_activation_energy>58.6</degradation_activation_energy>
            <solvent_bp_margin>20</solvent_bp_margin>
            <solvent_bp_penalty>0.70</solvent_bp_penalty>
            <solvent_decomposition_threshold>15</solvent_decomposition_threshold>
            <optimal_temp_bonus>1.18</optimal_temp_bonus>
            <temp_range>40</temp_range>
            <curve_steepness>0.08</curve_steepness>
            <curve_asymmetry>1.1</curve_asymmetry>
            <eyring_prefactor>1.0e13</eyring_prefactor>
            <eyring_weight>0.25</eyring_weight>
            <temp_increment_factor>0.55</temp_increment_factor>
            <base_temp_yield>40</base_temp_yield>
        </temperature>
        <time>
            <baseline_time>1</baseline_time>
            <min_time>1</min_time>
            <max_time>72</max_time>
            <short_time_penalty>0.10</short_time_penalty>
            <long_time_penalty>0.92</long_time_penalty>
            <optimal_time>24</optimal_time>
            <time_coefficient>0.75</time_coefficient>
            <reaction_half_life>6.9</reaction_half_life>
            <rate_constant>0.10</rate_constant>
            <time_effect_power>0.40</time_effect_power>
            <time_scaling>0.45</time_scaling>
            <logarithmic_factor>0.65</logarithmic_factor>
            <linear_factor>0.35</linear_factor>
            <saturation_point>24</saturation_point>
            <plateau_factor>0.93</plateau_factor>
            <diminishing_returns>0.20</diminishing_returns>
            <early_time_multiplier>2.0</early_time_multiplier>
            <late_time_multiplier>0.5</late_time_multiplier>
            <long_time_bonus>1.08</long_time_bonus>
            <optimal_time_bonus>1.12</optimal_time_bonus>
            <reaction_order>1.1</reaction_order>
            <half_life_temperature_dependence>-0.06</half_life_temperature_dependence>
            <diffusion_limited_rate>0.70</diffusion_limited_rate>
            <time_range>12</time_range>
            <time_increment_factor>0.22</time_increment_factor>
            <base_time_yield>35</base_time_yield>
        </time>
        <catalyst>
            <k_m>0.0008</k_m>
            <v_max>22</v_max>
            <baseline_quantity>0.0001</baseline_quantity>
            <min_quantity>0.0001</min_quantity>
            <max_quantity>0.50</max_quantity>
            <low_quantity_penalty>0.10</low_quantity_penalty>
            <catalyst_effect_power>0.55</catalyst_effect_power>
            <catalyst_scaling>0.60</catalyst_scaling>
            <mm_weight>0.50</mm_weight>
            <linear_weight>0.30</linear_weight>
            <logarithmic_weight>0.20</logarithmic_weight>
            <turnover_number>4500</turnover_number>
            <turnover_frequency>110.2</turnover_frequency>
            <catalyst_efficiency>0.90</catalyst_efficiency>
            <early_catalyst_multiplier>2.2</early_catalyst_multiplier>
            <late_catalyst_multiplier>0.4</late_catalyst_multiplier>
            <ligand_pd_ratio>2.5</ligand_pd_ratio>
            <ligand_bite_angle>92</ligand_bite_angle>
            <ligand_electron_donating>0.50</ligand_electron_donating>
            <ligand_steric_bulk>1.4</ligand_steric_bulk>
            <high_quantity_bonus>1.12</high_quantity_bonus>
            <optimal_quantity_bonus>1.20</optimal_quantity_bonus>
            <monodentate_ligand_factor>0.90</monodentate_ligand_factor>
            <bidentate_ligand_factor>1.10</bidentate_ligand_factor>
            <bulky_ligand_factor>0.85</bulky_ligand_factor>
            <electron_rich_ligand_factor>1.15</electron_rich_ligand_factor>
            <pd_oxidation_state>2</pd_oxidation_state>
            <ligand_coordination_number>4</ligand_coordination_number>
            <degradation_threshold>0.50</degradation_threshold>
            <quality_coefficient>0.75</quality_coefficient>
            <catalyst_increment_factor>0.090</catalyst_increment_factor>
            <base_catalyst_yield>30</base_catalyst_yield>
        </catalyst>
        <electronic_proxy>
            <base_homo_energy>-6.50</base_homo_energy>
            <base_lumo_energy>-1.50</base_lumo_energy>
            <homo_shift_factor>0.80</homo_shift_factor>
            <lumo_shift_factor>1.20</lumo_shift_factor>
            <chemical_potential_weight>0.28</chemical_potential_weight>
            <hardness_weight>0.22</hardness_weight>
            <electrophilicity_weight>0.18</electrophilicity_weight>
            <fukui_weight>0.32</fukui_weight>
            <fukui_plus_threshold>0.12</fukui_plus_threshold>
            <fukui_minus_threshold>0.12</fukui_minus_threshold>
            <fukui_zero_threshold>0.10</fukui_zero_threshold>
            <electrophilicity_threshold>1.3</electrophilicity_threshold>
            <nucleophilicity_threshold>1.8</nucleophilicity_threshold>
            <homo_energy_factor>0.35</homo_energy_factor>
            <lumo_energy_factor>0.25</lumo_energy_factor>
            <gap_energy_factor>0.40</gap_energy_factor>
            <homo_lumo_correlation>0.15</homo_lumo_correlation>
        </electronic_proxy>
        <mechanistic>
            <oxidative_addition_barrier>26.5</oxidative_addition_barrier>
            <oxidative_addition_barrier_sigma_effect>4.2</oxidative_addition_barrier_sigma_effect>
            <oxidative_addition_barrier_steric_effect>3.8</oxidative_addition_barrier_steric_effect>
            <oxidative_addition_rate>0.048</oxidative_addition_rate>
            <oxidative_addition_steric_sensitivity>0.60</oxidative_addition_steric_sensitivity>
            <oxidative_addition_electronic_sensitivity>0.80</oxidative_addition_electronic_sensitivity>
            <oxidative_addition_pd_effect>1.08</oxidative_addition_pd_effect>
            <transmetalation_barrier>20.8</transmetalation_barrier>
            <transmetalation_barrier_base_effect>-3.2</transmetalation_barrier_base_effect>
            <transmetalation_barrier_boronic_effect>-2.8</transmetalation_barrier_boronic_effect>
            <transmetalation_rate>0.082</transmetalation_rate>
            <transmetalation_base_sensitivity>0.70</transmetalation_base_sensitivity>
            <transmetalation_boronic_sensitivity>0.65</transmetalation_boronic_sensitivity>
            <transmetalation_halide_sensitivity>0.55</transmetalation_halide_sensitivity>
            <reductive_elimination_barrier>17.2</reductive_elimination_barrier>
            <reductive_elimination_barrier_steric_effect>5.6</reductive_elimination_barrier_steric_effect>
            <reductive_elimination_rate>0.125</reductive_elimination_rate>
            <reductive_elimination_steric_sensitivity>0.85</reductive_elimination_steric_sensitivity>
            <reductive_elimination_electronic_sensitivity>0.55</reductive_elimination_electronic_sensitivity>
            <reductive_elimination_ring_closure_bonus>1.12</reductive_elimination_ring_closure_bonus>
            <transition_state_asymmetry>1.1</transition_state_asymmetry>
            <reaction_coordinate_step>0.08</reaction_coordinate_step>
            <intermediate_stability_factor>0.75</intermediate_stability_factor>
            <pre_exponential_factor>1.0e13</pre_exponential_factor>
            <oa_weight>0.35</oa_weight>
            <tm_weight>0.35</tm_weight>
            <re_weight>0.30</re_weight>
        </mechanistic>
        <steric>
            <threshold>0.20</threshold>
            <penalty_coefficient>7.5</penalty_coefficient>
            <ring_penalty_factor>0.30</ring_penalty_factor>
            <bulky_group_penalty>2.0</bulky_group_penalty>
            <ortho_substituent_penalty>2.8</ortho_substituent_penalty>
            <meta_substituent_penalty>1.3</meta_substituent_penalty>
            <para_substituent_penalty>0.8</para_substituent_penalty>
            <molecular_volume_threshold>180</molecular_volume_threshold>
            <volume_penalty_factor>0.020</volume_penalty_factor>
            <rotatable_bond_penalty>0.15</rotatable_bond_penalty>
            <rotatable_bond_threshold>8</rotatable_bond_threshold>
            <fused_ring_penalty>0.45</fused_ring_penalty>
            <bridged_ring_penalty>0.65</bridged_ring_penalty>
            <spiro_ring_penalty>0.55</spiro_ring_penalty>
            <heavy_atom_steric_factor>0.08</heavy_atom_steric_factor>
            <halogen_steric_factor>0.15</halogen_steric_factor>
            <a_value_methyl>1.74</a_value_methyl>
            <a_value_ethyl>1.75</a_value_ethyl>
            <a_value_isopropyl>2.21</a_value_isopropyl>
            <a_value_tertbutyl>4.9</a_value_tertbutyl>
            <a_value_cyclohexyl>2.15</a_value_cyclohexyl>
            <a_value_phenyl>2.8</a_value_phenyl>
        </steric>
        <electronic>
            <logp_coefficient>0.25</logp_coefficient>
            <hbd_penalty>2.2</hbd_penalty>
            <hba_bonus>1.6</hba_bonus>
            <hammett_coefficient>3.0</hammett_coefficient>
            <taft_coefficient>1.6</taft_coefficient>
            <sigma_m_electron_withdrawing>0.70</sigma_m_electron_withdrawing>
            <sigma_p_electron_withdrawing>0.80</sigma_p_electron_withdrawing>
            <sigma_m_electron_donating>-0.28</sigma_m_electron_donating>
            <sigma_p_electron_donating>-0.40</sigma_p_electron_donating>
            <sigma_plus_coefficient>3.2</sigma_plus_coefficient>
            <sigma_minus_coefficient>2.6</sigma_minus_coefficient>
            <brown_sigma_plus_factor>1.2</brown_sigma_plus_factor>
            <hammett_reaction_constant>1.1</hammett_reaction_constant>
            <lfer_weight>0.12</lfer_weight>
            <electron_donating_bonus>1.25</electron_donating_bonus>
            <electron_withdrawing_penalty>0.72</electron_withdrawing_penalty>
            <conjugation_effect>1.12</conjugation_effect>
            <inductive_effect>0.90</inductive_effect>
            <resonance_effect>1.18</resonance_effect>
            <polarity_factor>0.12</polarity_factor>
            <solubility_threshold>-2.2</solubility_threshold>
            <solubility_penalty>0.58</solubility_penalty>
            <taft_es_methyl>0.00</taft_es_methyl>
            <taft_es_ethyl>-0.07</taft_es_ethyl>
            <taft_es_isopropyl>-0.47</taft_es_isopropyl>
            <taft_es_tertbutyl>-1.54</taft_es_tertbutyl>
            <taft_es_phenyl>-1.20</taft_es_phenyl>
            <taft_es_benzyl>-0.45</taft_es_benzyl>
        </electronic>
        <hsab>
            <pd_softness>2.8</pd_softness>
            <halide_softness>3.2</halide_softness>
            <ligand_softness>2.5</ligand_softness>
            <base_softness>3.0</base_softness>
            <absolute_hardness_pd>3.8</absolute_hardness_pd>
            <absolute_hardness_halide>4.2</absolute_hardness_halide>
            <absolute_hardness_ligand>3.5</absolute_hardness_ligand>
            <chemical_potential_pd>-5.2</chemical_potential_pd>
            <electronegativity_pd>5.2</electronegativity_pd>
            <pearson_softness_threshold>6.0</pearson_softness_threshold>
            <pd_halide_match>0.85</pd_halide_match>
            <pd_ligand_match>0.90</pd_ligand_match>
            <ligand_halide_match>0.75</ligand_halide_match>
            <overall_compatibility>0.80</overall_compatibility>
            <soft_soft_bonus>1.22</soft_soft_bonus>
            <hard_hard_bonus>1.12</hard_hard_bonus>
            <soft_hard_penalty>0.68</soft_hard_penalty>
            <mismatch_penalty>0.48</mismatch_penalty>
            <soft_soft_weight>0.40</soft_soft_weight>
            <hard_hard_weight>0.35</hard_hard_weight>
            <mismatch_penalty_weight>0.25</mismatch_penalty_weight>
        </hsab>
        <solvent>
            <dielectric_optimal>28.0</dielectric_optimal>
            <dielectric_range>18.0</dielectric_range>
            <dielectric_weight>0.10</dielectric_weight>
            <donor_optimal>20.0</donor_optimal>
            <donor_range>16.0</donor_range>
            <donor_weight>0.08</donor_weight>
            <polarity_optimal>4.5</polarity_optimal>
            <polarity_range>3.5</polarity_range>
            <polarity_weight>0.06</polarity_weight>
            <alpha_weight>0.08</alpha_weight>
            <beta_weight>0.08</beta_weight>
            <pi_star_weight>0.06</pi_star_weight>
            <reichardt_weight>0.05</reichardt_weight>
            <hildebrand_weight>0.04</hildebrand_weight>
            <aprotic_solvent_bonus>1.10</aprotic_solvent_bonus>
            <protic_solvent_penalty>0.88</protic_solvent_penalty>
            <polar_solvent_bonus>1.06</polar_solvent_bonus>
            <nonpolar_solvent_penalty>0.92</nonpolar_solvent_penalty>
            <solvent_mixtures>
                <toluene_ethanol>1.18</toluene_ethanol>
                <dioxane_water>1.12</dioxane_water>
                <thf_water>1.08</thf_water>
                <dme_water>1.10</dme_water>
                <acetonitrile_water>1.06</acetonitrile_water>
                <dmso_water>1.15</dmso_water>
            </solvent_mixtures>
        </solvent>
        <base>
            <pka_threshold>18.0</pka_threshold>
            <strong_base_bonus>1.18</strong_base_bonus>
            <weak_base_penalty>0.82</weak_base_penalty>
            <inorganic_base_factor>1.00</inorganic_base_factor>
            <organic_base_factor>0.94</organic_base_factor>
            <carbonate_base_factor>1.10</carbonate_base_factor>
            <phosphate_base_factor>1.06</phosphate_base_factor>
            <acetate_base_factor>0.88</acetate_base_factor>
            <fluoride_base_factor>0.82</fluoride_base_factor>
            <soluble_base_bonus>1.08</soluble_base_bonus>
            <insoluble_base_penalty>0.78</insoluble_base_penalty>
            <hygroscopic_base_penalty>0.86</hygroscopic_base_penalty>
            <cation_radius_effect>1.02</cation_radius_effect>
            <pka_effect>0.030</pka_effect>
            <pkb_effect>0.035</pkb_effect>
            <pka_weight>0.30</pka_weight>
            <solubility_weight>0.25</solubility_weight>
            <class_weight>0.20</class_weight>
            <cation_radius_weight>0.25</cation_radius_weight>
        </base>
        <physicochemical_proxy>
            <mw_weight>0.30</mw_weight>
            <logp_weight>0.25</logp_weight>
            <tpsa_weight>0.25</tpsa_weight>
            <rotbonds_weight>0.20</rotbonds_weight>
            <mw_center>300</mw_center>
            <mw_scale>200</mw_scale>
            <logp_center>2.5</logp_center>
            <logp_scale>3.0</logp_scale>
            <tpsa_center>60</tpsa_center>
            <tpsa_scale>60</tpsa_scale>
            <rotbonds_center>4</rotbonds_center>
            <rotbonds_scale>6</rotbonds_scale>
        </physicochemical_proxy>
        <yield_parameters>
            <max_yield>100</max_yield>
            <min_yield>0</min_yield>
            <base_yield_offset>25</base_yield_offset>
            <reproducibility_factor>0.93</reproducibility_factor>
            <scale_up_factor>0.90</scale_up_factor>
            <yield_mean>70.5</yield_mean>
            <yield_std>17.5</yield_std>
            <excellent_threshold>85</excellent_threshold>
            <good_threshold>70</good_threshold>
            <moderate_threshold>50</moderate_threshold>
            <poor_threshold>30</poor_threshold>
            <very_poor_threshold>15</very_poor_threshold>
            <confidence_interval_alpha>0.05</confidence_interval_alpha>
            <prediction_interval_alpha>0.10</prediction_interval_alpha>
            <yield_skewness>-0.40</yield_skewness>
            <yield_kurtosis>0.28</yield_kurtosis>
        </yield_parameters>
    </chemical_intuition>
    <data_integrity>
        <forbid_random_or_synthetic_data>true</forbid_random_or_synthetic_data>
        <nullable_columns>solv1,solv2</nullable_columns>
        <failure_column>yield</failure_column>
        <null_yield_meaning>failed_reaction</null_yield_meaning>
        <keep_failed_reactions_in_audit_trail>true</keep_failed_reactions_in_audit_trail>
        <exclude_failed_reactions_from_training>true</exclude_failed_reactions_from_training>
        <critical_non_nullable_columns>temp,time,quantity,catalizor,base</critical_non_nullable_columns>
        <reject_row_on_missing_critical_field>true</reject_row_on_missing_critical_field>
        <minimum_usable_rows>5</minimum_usable_rows>
        <log_row_classification_summary>true</log_row_classification_summary>
    </data_integrity>
    <monotonicity>
        <enabled>true</enabled>
        <system_identity_fields>catalizor,base,solv1,solv2,subs1_smiles,subs2_smiles</system_identity_fields>
        <incremental_gain>
            <curve_shape>logarithmic</curve_shape>
            <curve_multiplier>0.35</curve_multiplier>
            <temp_max_increase_pct>2.0</temp_max_increase_pct>
            <time_max_increase_pct>1.5</time_max_increase_pct>
            <catalyst_max_increase_pct>1.5</catalyst_max_increase_pct>
        </incremental_gain>
        <hard_floor>
            <enabled>true</enabled>
            <enforce_against_full_history>true</enforce_against_full_history>
            <require_all_conditions_dominated>true</require_all_conditions_dominated>
        </hard_floor>
    </monotonicity>
    <ml_training_safeguards>
        <min_samples_per_feature>10</min_samples_per_feature>
        <min_samples_for_full_ensemble>30</min_samples_for_full_ensemble>
        <honest_cv_max_folds>5</honest_cv_max_folds>
    </ml_training_safeguards>
    <model_parameters>
        <Random_Forest>
            <n_estimators>300</n_estimators>
            <max_depth>15</max_depth>
            <min_samples_split>3</min_samples_split>
            <min_samples_leaf>1</min_samples_leaf>
            <max_features>sqrt</max_features>
            <bootstrap>true</bootstrap>
            <oob_score>true</oob_score>
            <random_state>42</random_state>
            <n_jobs>1</n_jobs>
            <ccp_alpha>0.001</ccp_alpha>
        </Random_Forest>
        <Gradient_Boosting>
            <n_estimators>350</n_estimators>
            <max_depth>7</max_depth>
            <min_samples_split>4</min_samples_split>
            <min_samples_leaf>2</min_samples_leaf>
            <learning_rate>0.07</learning_rate>
            <subsample>0.8</subsample>
            <max_features>sqrt</max_features>
            <validation_fraction>0.15</validation_fraction>
            <n_iter_no_change>15</n_iter_no_change>
            <tol>0.001</tol>
            <random_state>42</random_state>
            <loss>squared_error</loss>
            <criterion>friedman_mse</criterion>
        </Gradient_Boosting>
        <Hist_Gradient_Boosting>
            <max_iter>350</max_iter>
            <max_depth>8</max_depth>
            <min_samples_leaf>2</min_samples_leaf>
            <learning_rate>0.09</learning_rate>
            <max_bins>255</max_bins>
            <l2_regularization>0.01</l2_regularization>
            <early_stopping>true</early_stopping>
            <scoring>neg_mean_squared_error</scoring>
            <validation_fraction>0.15</validation_fraction>
            <n_iter_no_change>15</n_iter_no_change>
            <random_state>42</random_state>
            <loss>squared_error</loss>
            <max_leaf_nodes>31</max_leaf_nodes>
        </Hist_Gradient_Boosting>
        <XGBoost>
            <n_estimators>350</n_estimators>
            <max_depth>7</max_depth>
            <learning_rate>0.08</learning_rate>
            <subsample>0.85</subsample>
            <colsample_bytree>0.9</colsample_bytree>
            <colsample_bylevel>0.8</colsample_bylevel>
            <reg_alpha>0.1</reg_alpha>
            <reg_lambda>1.0</reg_lambda>
            <min_child_weight>2</min_child_weight>
            <gamma>0.1</gamma>
            <random_state>42</random_state>
            <n_jobs>1</n_jobs>
            <objective>reg:squarederror</objective>
            <eval_metric>rmse</eval_metric>
            <booster>gbtree</booster>
            <tree_method>hist</tree_method>
            <grow_policy>lossguide</grow_policy>
            <max_leaves>31</max_leaves>
        </XGBoost>
        <LightGBM>
            <n_estimators>400</n_estimators>
            <max_depth>10</max_depth>
            <num_leaves>31</num_leaves>
            <learning_rate>0.06</learning_rate>
            <subsample>0.8</subsample>
            <colsample_bytree>0.85</colsample_bytree>
            <min_child_samples>5</min_child_samples>
            <reg_alpha>0.1</reg_alpha>
            <reg_lambda>0.1</reg_lambda>
            <min_split_gain>0.01</min_split_gain>
            <random_state>42</random_state>
            <n_jobs>1</n_jobs>
            <boosting_type>gbdt</boosting_type>
            <objective>regression</objective>
            <metric>rmse</metric>
            <verbose>-1</verbose>
            <bagging_freq>0</bagging_freq>
            <cat_smooth>10.0</cat_smooth>
            <cat_l2>10.0</cat_l2>
        </LightGBM>
        <CatBoost>
            <iterations>400</iterations>
            <depth>7</depth>
            <learning_rate>0.07</learning_rate>
            <l2_leaf_reg>3</l2_leaf_reg>
            <border_count>128</border_count>
            <random_seed>42</random_seed>
            <verbose>false</verbose>
            <loss_function>RMSE</loss_function>
            <eval_metric>RMSE</eval_metric>
            <od_type>Iter</od_type>
            <od_wait>20</od_wait>
        </CatBoost>
        <Extra_Trees>
            <n_estimators>250</n_estimators>
            <max_depth>12</max_depth>
            <min_samples_split>3</min_samples_split>
            <min_samples_leaf>1</min_samples_leaf>
            <max_features>sqrt</max_features>
            <bootstrap>true</bootstrap>
            <random_state>42</random_state>
            <n_jobs>1</n_jobs>
            <ccp_alpha>0.001</ccp_alpha>
        </Extra_Trees>
        <Gaussian_Process>
            <kernel>1.0 * RBF(1.0) + WhiteKernel(0.1)</kernel>
            <alpha>1e-10</alpha>
            <optimizer>fmin_l_bfgs_b</optimizer>
            <n_restarts_optimizer>10</n_restarts_optimizer>
            <normalize_y>true</normalize_y>
            <random_state>42</random_state>
        </Gaussian_Process>
        <SVR>
            <kernel>rbf</kernel>
            <C>1.5</C>
            <epsilon>0.06</epsilon>
            <gamma>scale</gamma>
            <degree>3</degree>
            <coef0>0.0</coef0>
            <shrinking>true</shrinking>
            <tol>0.001</tol>
            <max_iter>-1</max_iter>
            <cache_size>200</cache_size>
        </SVR>
        <Neural_Network>
            <hidden_layer_sizes>256,128,64,32</hidden_layer_sizes>
            <activation>relu</activation>
            <solver>adam</solver>
            <alpha>0.0005</alpha>
            <learning_rate_init>0.001</learning_rate_init>
            <max_iter>1500</max_iter>
            <tol>0.0001</tol>
            <momentum>0.9</momentum>
            <nesterovs_momentum>true</nesterovs_momentum>
            <early_stopping>true</early_stopping>
            <validation_fraction>0.15</validation_fraction>
            <beta_1>0.9</beta_1>
            <beta_2>0.999</beta_2>
            <epsilon>1e-08</epsilon>
            <n_iter_no_change>15</n_iter_no_change>
            <random_state>42</random_state>
            <warm_start>false</warm_start>
        </Neural_Network>
        <Ridge>
            <alpha>0.8</alpha>
            <fit_intercept>true</fit_intercept>
            <copy_X>true</copy_X>
            <max_iter>None</max_iter>
            <tol>0.001</tol>
            <solver>auto</solver>
            <random_state>42</random_state>
        </Ridge>
        <Lasso>
            <alpha>0.8</alpha>
            <fit_intercept>true</fit_intercept>
            <max_iter>1000</max_iter>
            <tol>0.0001</tol>
            <selection>cyclic</selection>
            <random_state>42</random_state>
        </Lasso>
        <ElasticNet>
            <alpha>0.8</alpha>
            <l1_ratio>0.5</l1_ratio>
            <fit_intercept>true</fit_intercept>
            <max_iter>1000</max_iter>
            <tol>0.0001</tol>
            <selection>cyclic</selection>
            <random_state>42</random_state>
        </ElasticNet>
        <KNN>
            <n_neighbors>5</n_neighbors>
            <weights>distance</weights>
            <algorithm>auto</algorithm>
            <leaf_size>30</leaf_size>
            <p>2</p>
            <metric>minkowski</metric>
        </KNN>
        <Ensemble>
            <weights>
                <Random_Forest>0.16</Random_Forest>
                <Gradient_Boosting>0.10</Gradient_Boosting>
                <Hist_Gradient_Boosting>0.16</Hist_Gradient_Boosting>
                <XGBoost>0.10</XGBoost>
                <LightGBM>0.07</LightGBM>
                <CatBoost>0.07</CatBoost>
                <Extra_Trees>0.04</Extra_Trees>
                <Gaussian_Process>0.08</Gaussian_Process>
                <SVR>0.02</SVR>
                <Neural_Network>0.03</Neural_Network>
                <Ridge>0.02</Ridge>
                <ElasticNet>0.02</ElasticNet>
                <KNN>0.01</KNN>
            </weights>
            <stacking>true</stacking>
            <stacking_meta_model>Random_Forest</stacking_meta_model>
            <voting>soft</voting>
            <ensemble_validation>true</ensemble_validation>
            <ensemble_cv_folds>3</ensemble_cv_folds>
        </Ensemble>
    </model_parameters>
    <feature_importance>
        <temperature>0.18</temperature>
        <time>0.14</time>
        <catalyst_quantity>0.12</catalyst_quantity>
        <substrate1_steric>0.08</substrate1_steric>
        <substrate2_steric>0.08</substrate2_steric>
        <solvent_effect>0.07</solvent_effect>
        <base_effect>0.06</base_effect>
        <electronic_effects>0.05</electronic_effects>
        <hammett_effects>0.04</hammett_effects>
        <taft_effects>0.03</taft_effects>
        <hsab_effects>0.03</hsab_effects>
        <mechanistic_effects>0.03</mechanistic_effects>
        <elecproxy_effects>0.02</elecproxy_effects>
        <physchem_effects>0.03</physchem_effects>
    </feature_importance>
    <data_processing>
        <missing_values>
            <strategy>median_imputation</strategy>
            <categorical_strategy>mode_imputation</categorical_strategy>
            <threshold>0.30</threshold>
            <numeric_method>median</numeric_method>
            <knn_neighbors>5</knn_neighbors>
        </missing_values>
        <normalization>
            <numeric_method>standard_scaler</numeric_method>
            <categorical_method>one_hot_encoding</categorical_method>
            <target_scaling>minmax</target_scaling>
            <robust_scaling>true</robust_scaling>
            <quantile_transform>true</quantile_transform>
        </normalization>
        <feature_selection>
            <method>mutual_information</method>
            <k_best>50</k_best>
            <variance_threshold>0.01</variance_threshold>
            <correlation_threshold>0.85</correlation_threshold>
            <select_from_model>true</select_from_model>
            <rfe_n_features>30</rfe_n_features>
        </feature_selection>
        <augmentation>
            <enabled>true</enabled>
            <method>gaussian_noise</method>
            <noise_level>0.05</noise_level>
            <n_augmentations>100</n_augmentations>
            <bootstrap_samples>2000</bootstrap_samples>
            <smote_enabled>false</smote_enabled>
        </augmentation>
        <split>
            <test_size>0.20</test_size>
            <validation_size>0.15</validation_size>
            <stratify>true</stratify>
            <random_state>42</random_state>
            <shuffle>true</shuffle>
        </split>
        <outlier_detection>
            <method>iqr</method>
            <threshold>1.5</threshold>
            <handle_method>clip</handle_method>
            <zscore_threshold>3.0</zscore_threshold>
        </outlier_detection>
    </data_processing>
    <optimization>
        <top_candidates>15</top_candidates>
        <catalyst_search>
            <min_quantity>0.0001</min_quantity>
            <max_quantity>0.50</max_quantity>
            <step_size>0.0005</step_size>
            <n_candidates>20</n_candidates>
            <log_scale>true</log_scale>
        </catalyst_search>
    </optimization>
    <performance_metrics>
        <metrics>
            <r2>true</r2>
            <mae>true</mae>
            <rmse>true</rmse>
            <mape>true</mape>
            <explained_variance>true</explained_variance>
        </metrics>
        <cross_validation>
            <enabled>true</enabled>
            <folds>5</folds>
            <shuffle>true</shuffle>
            <random_state>42</random_state>
            <stratified>true</stratified>
            <n_jobs>1</n_jobs>
            <repeated_cv>true</repeated_cv>
            <n_repeats>3</n_repeats>
        </cross_validation>
        <statistical_tests>
            <shapiro_wilk>true</shapiro_wilk>
        </statistical_tests>
        <uncertainty>
            <method>bootstrap</method>
            <n_bootstrap>100</n_bootstrap>
            <confidence_level>0.95</confidence_level>
            <prediction_interval_level>0.90</prediction_interval_level>
        </uncertainty>
    </performance_metrics>
    <visualization>
        <molecule_images>
            <enabled>true</enabled>
            <image_size>400</image_size>
            <format>png</format>
            <dpi>200</dpi>
            <show_atoms>true</show_atoms>
            <show_bonds>true</show_bonds>
            <show_hydrogens>false</show_hydrogens>
            <highlight_atoms>true</highlight_atoms>
        </molecule_images>
        <plots>
            <feature_importance>true</feature_importance>
            <actual_vs_predicted>true</actual_vs_predicted>
            <residuals>true</residuals>
            <parity_plot>true</parity_plot>
            <prediction_distribution>true</prediction_distribution>
            <residual_qq>true</residual_qq>
            <coefficient_plot>true</coefficient_plot>
        </plots>
        <colors>
            <primary>#2563EB</primary>
            <secondary>#10B981</secondary>
            <warning>#F59E0B</warning>
            <danger>#EF4444</danger>
            <background>#F8FAFC</background>
            <text>#1E293B</text>
            <grid>#E2E8F0</grid>
        </colors>
    </visualization>
    <logging>
        <log_level>INFO</log_level>
        <log_file>logs/predict_ml.log</log_file>
        <max_log_size>10MB</max_log_size>
        <backup_count>5</backup_count>
        <console_output>true</console_output>
        <json_format>true</json_format>
        <error_handling>
            <retry_attempts>3</retry_attempts>
            <retry_delay>1.0</retry_delay>
            <fallback_model>Random_Forest</fallback_model>
            <log_traceback>true</log_traceback>
            <email_alerts>false</email_alerts>
        </error_handling>
        <monitoring>
            <enabled>true</enabled>
            <metrics_interval>60</metrics_interval>
            <alert_threshold>0.05</alert_threshold>
        </monitoring>
    </logging>
    <security>
        <file_upload>
            <allowed_extensions>csv</allowed_extensions>
            <max_file_size>50MB</max_file_size>
            <max_files>10</max_files>
            <allowed_mime_types>text/csv,application/csv</allowed_mime_types>
            <virus_scan>false</virus_scan>
        </file_upload>
        <api>
            <rate_limit>100</rate_limit>
            <rate_limit_period>60</rate_limit_period>
            <max_payload_size>1MB</max_payload_size>
            <cors_enabled>true</cors_enabled>
            <allowed_origins>*</allowed_origins>
            <api_key_required>false</api_key_required>
        </api>
        <sanitization>
            <strip_xss>true</strip_xss>
            <strip_sql_injection>true</strip_sql_injection>
            <validate_smiles>true</validate_smiles>
            <max_smiles_length>500</max_smiles_length>
            <allowed_smiles_patterns>all</allowed_smiles_patterns>
            <sanitize_inputs>true</sanitize_inputs>
        </sanitization>
        <encryption>
            <model_encryption>false</model_encryption>
            <data_encryption>false</data_encryption>
            <ssl_enabled>false</ssl_enabled>
        </encryption>
    </security>
    <experimental_mode>
        <enabled>false</enabled>
        <temperature>
            <base_offset>0.30</base_offset>
            <increment_factor>3.50</increment_factor>
            <reference_temp>25</reference_temp>
            <min_factor>0.10</min_factor>
            <no_degradation_ceiling>true</no_degradation_ceiling>
            <no_arrhenius_dropoff>true</no_arrhenius_dropoff>
        </temperature>
        <time>
            <base_offset>0.55</base_offset>
            <increment_factor>1.80</increment_factor>
            <reference_time>1</reference_time>
            <min_factor>0.10</min_factor>
            <no_saturation_ceiling>true</no_saturation_ceiling>
            <no_plateau_factor>true</no_plateau_factor>
        </time>
        <catalyst>
            <base_offset>0.50</base_offset>
            <increment_factor>2.20</increment_factor>
            <reference_quantity>0.0001</reference_quantity>
            <min_factor>0.10</min_factor>
            <no_degradation_threshold>true</no_degradation_threshold>
            <no_mm_saturation>true</no_mm_saturation>
        </catalyst>
        <validation>
            <temp_min>25</temp_min>
            <temp_max>250</temp_max>
            <time_min>1</time_min>
            <time_max>72</time_max>
            <quantity_min>0.0001</quantity_min>
            <quantity_max>0.50</quantity_max>
            <yield_ceiling>100</yield_ceiling>
        </validation>
        <ui>
            <badge_color>#8B5CF6</badge_color>
            <badge_label>EXPERIMENTAL</badge_label>
            <banner_color>#f5f3ff</banner_color>
            <banner_color_dark>#2a1a3a</banner_color_dark>
        </ui>
    </experimental_mode>
</suzuki_config>"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(default)
        self.tree = ET.parse(self.config_path)
        self.root = self.tree.getroot()
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.raw_xml = f.read()

    def _parse_all(self):
        self.params = self._parse_element(self.root)

    def _parse_element(self, element):
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
        required = ['chemical_intuition/temperature/baseline_temp', 'chemical_intuition/time/baseline_time', 'chemical_intuition/catalyst/baseline_quantity']
        missing = []
        for path in required:
            val = self.get(path)
            if val is None:
                missing.append(path)
        if missing:
            logger.warning(f"Required parameters missing: {', '.join(missing)}")

    def _compute_xml_hash(self):
        self._xml_hash = hashlib.md5(self.raw_xml.encode()).hexdigest()

    def get(self, path, default=None):
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

    def get_float(self, path, default=0.0):
        try:
            val = self.get(path)
            return float(val) if val is not None else default
        except:
            return default

    def get_int(self, path, default=0):
        try:
            val = self.get(path)
            return int(val) if val is not None else default
        except:
            return default

    def get_bool(self, path, default=False):
        try:
            val = self.get(path)
            return bool(val) if val is not None else default
        except:
            return default

    def get_dict(self, path):
        try:
            val = self.get(path)
            return val if isinstance(val, dict) else {}
        except:
            return {}

    def get_model_params(self, model_name):
        params = self.get_dict(f'model_parameters/{model_name}')
        if 'n_jobs' in params:
            try:
                if int(params['n_jobs']) < 1:
                    params['n_jobs'] = 1
            except:
                params['n_jobs'] = 1
        return params

    def get_chemical_params(self):
        return self.get_dict('chemical_intuition')

    def get_feature_importance(self):
        return self.get_dict('feature_importance')

    def get_models_list(self):
        models = self.get_dict('model_parameters')
        return list(models.keys()) if models else []

    def reload(self):
        try:
            self._cache.clear()
            self._load_or_create()
            self._parse_all()
            self._validate_config()
            self._compute_xml_hash()
            return True
        except Exception as e:
            logger.error(f"Reload error: {e}")
            return False

    def save(self):
        try:
            self.tree.write(self.config_path, encoding='UTF-8', xml_declaration=True)
            self._cache.clear()
            self._parse_all()
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.raw_xml = f.read()
            self._compute_xml_hash()
            return True
        except Exception as e:
            logger.error(f"Save error: {e}")
            return False

    def get_raw_xml(self):
        return self.raw_xml

    def get_all_params(self):
        return self.params

    def get_xml_hash(self):
        return self._xml_hash

    def get_summary(self):
        return {
            'config_path': self.config_path,
            'xml_size': len(self.raw_xml),
            'xml_hash': self._xml_hash[:8] if self._xml_hash else None,
            'top_level_sections': len(self.params),
            'sections': list(self.params.keys()),
            'cache_size': len(self._cache)
        }


class ChemicalCalculator:
    def __init__(self, config):
        self.config = config
        self._load_all_params()
        self._load_feature_importance()
        self._load_experimental_bounds()
        self._validate_params()

    def _load_experimental_bounds(self):
        self.exp_temp_min = self.config.get_float('experimental_mode/validation/temp_min', 25)
        self.exp_temp_max = self.config.get_float('experimental_mode/validation/temp_max', 250)
        self.exp_time_min = self.config.get_float('experimental_mode/validation/time_min', 1)
        self.exp_time_max = self.config.get_float('experimental_mode/validation/time_max', 72)
        self.exp_quantity_min = self.config.get_float('experimental_mode/validation/quantity_min', 0.0001)
        self.exp_quantity_max = self.config.get_float('experimental_mode/validation/quantity_max', 0.50)
        self.exp_temp_base_offset = self.config.get_float('experimental_mode/temperature/base_offset', 0.30)
        self.exp_time_base_offset = self.config.get_float('experimental_mode/time/base_offset', 0.55)
        self.exp_catalyst_base_offset = self.config.get_float('experimental_mode/catalyst/base_offset', 0.50)
        self.exp_temp_increment_factor = self.config.get_float('experimental_mode/temperature/increment_factor', 1.70)
        self.exp_time_increment_factor = self.config.get_float('experimental_mode/time/increment_factor', 1.50)
        self.exp_catalyst_increment_factor = self.config.get_float('experimental_mode/catalyst/increment_factor', 1.75)

    def _load_all_params(self):
        chem = self.config.get_chemical_params()
        temp = chem.get('temperature', {})
        self.optimal_temp = temp.get('optimal_temp', 85)
        self.temp_range = temp.get('temp_range', 35)
        self.baseline_temp = temp.get('baseline_temp', temp.get('min_temp', 40))
        self.min_temp = temp.get('min_temp', self.baseline_temp)
        self.max_temp = temp.get('max_temp', 150)
        self.low_temp_penalty = temp.get('low_temp_penalty', 0.65)
        self.high_temp_penalty = temp.get('high_temp_penalty', 0.40)
        self.degradation_threshold = temp.get('degradation_threshold', 130)
        self.temp_coefficient = temp.get('temp_coefficient', 0.8)
        self.activation_energy = temp.get('activation_energy', 45.2)
        self.arrhenius_prefactor = temp.get('arrhenius_prefactor', 1.2e12)
        self.gas_constant = temp.get('gas_constant', 8.314)
        self.curve_steepness = temp.get('curve_steepness', 0.15)
        self.curve_asymmetry = temp.get('curve_asymmetry', 1.2)
        self.optimal_temp_bonus = temp.get('optimal_temp_bonus', 1.15)
        self.solvent_bp_margin = temp.get('solvent_bp_margin', 15)
        self.solvent_bp_penalty = temp.get('solvent_bp_penalty', 0.85)
        self.eyring_prefactor = temp.get('eyring_prefactor', 1.0e13)
        self.entropy_activation = temp.get('entropy_activation', -20.5)
        self.enthalpy_activation = temp.get('enthalpy_activation', 42.8)
        self.degradation_rate = temp.get('degradation_rate', 0.045)
        self.degradation_activation_energy = temp.get('degradation_activation_energy', 58.6)
        self.eyring_weight = temp.get('eyring_weight', 0.25)
        self.temp_increment_factor = temp.get('temp_increment_factor', 0.022)
        self.base_temp_yield = temp.get('base_temp_yield', 40)

        time_p = chem.get('time', {})
        self.optimal_time = time_p.get('optimal_time', 18)
        self.time_range = time_p.get('time_range', 12)
        self.baseline_time = time_p.get('baseline_time', time_p.get('min_time', 1))
        self.min_time = time_p.get('min_time', self.baseline_time)
        self.max_time = time_p.get('max_time', 48)
        self.short_time_penalty = time_p.get('short_time_penalty', 0.40)
        self.long_time_penalty = time_p.get('long_time_penalty', 0.70)
        self.time_coefficient = time_p.get('time_coefficient', 1.2)
        self.reaction_half_life = time_p.get('reaction_half_life', 6.5)
        self.rate_constant = time_p.get('rate_constant', 0.107)
        self.diffusion_limit = time_p.get('diffusion_limit', 0.85)
        self.saturation_point = time_p.get('saturation_point', 24)
        self.plateau_factor = time_p.get('plateau_factor', 0.92)
        self.diminishing_returns = time_p.get('diminishing_returns', 0.35)
        self.optimal_time_bonus = time_p.get('optimal_time_bonus', 1.10)
        self.reaction_order = time_p.get('reaction_order', 1.5)
        self.half_life_temperature_dependence = time_p.get('half_life_temperature_dependence', -0.12)
        self.time_increment_factor = time_p.get('time_increment_factor', 0.018)
        self.base_time_yield = time_p.get('base_time_yield', 35)

        cat = chem.get('catalyst', {})
        self.k_m = cat.get('k_m', 0.003)
        self.v_max = cat.get('v_max', 18)
        self.baseline_quantity = cat.get('baseline_quantity', cat.get('min_quantity', 0.0005))
        self.min_quantity = cat.get('min_quantity', self.baseline_quantity)
        self.max_quantity = cat.get('max_quantity', 0.08)
        self.low_quantity_penalty = cat.get('low_quantity_penalty', 0.30)
        self.high_quantity_penalty = cat.get('high_quantity_penalty', 0.50)
        self.degradation_threshold_cat = cat.get('degradation_threshold', 0.04)
        self.quality_coefficient = cat.get('quality_coefficient', 0.6)
        self.turnover_number = cat.get('turnover_number', 1200)
        self.turnover_frequency = cat.get('turnover_frequency', 45.6)
        self.catalyst_efficiency_xml = cat.get('catalyst_efficiency', 0.78)
        self.ligand_pd_ratio = cat.get('ligand_pd_ratio', 4.0)
        self.ligand_bite_angle = cat.get('ligand_bite_angle', 102)
        self.ligand_electron_donating = cat.get('ligand_electron_donating', 0.45)
        self.ligand_steric_bulk = cat.get('ligand_steric_bulk', 1.8)
        self.optimal_quantity_bonus = cat.get('optimal_quantity_bonus', 1.20)
        self.monodentate_ligand_factor = cat.get('monodentate_ligand_factor', 0.90)
        self.bidentate_ligand_factor = cat.get('bidentate_ligand_factor', 1.10)
        self.bulky_ligand_factor = cat.get('bulky_ligand_factor', 0.85)
        self.electron_rich_ligand_factor = cat.get('electron_rich_ligand_factor', 1.15)
        self.pd_oxidation_state = cat.get('pd_oxidation_state', 2)
        self.ligand_coordination_number = cat.get('ligand_coordination_number', 4)
        self.catalyst_increment_factor = cat.get('catalyst_increment_factor', 0.020)
        self.base_catalyst_yield = cat.get('base_catalyst_yield', 30)

        elec = chem.get('electronic_proxy', {})
        self.base_homo_energy = elec.get('base_homo_energy', -6.5)
        self.base_lumo_energy = elec.get('base_lumo_energy', -1.5)
        self.homo_shift_factor = elec.get('homo_shift_factor', 0.8)
        self.lumo_shift_factor = elec.get('lumo_shift_factor', 1.2)
        self.chemical_potential_weight = elec.get('chemical_potential_weight', 0.28)
        self.hardness_weight = elec.get('hardness_weight', 0.22)
        self.electrophilicity_weight = elec.get('electrophilicity_weight', 0.18)
        self.fukui_weight = elec.get('fukui_weight', 0.32)
        self.fukui_plus_threshold = elec.get('fukui_plus_threshold', 0.12)
        self.fukui_minus_threshold = elec.get('fukui_minus_threshold', 0.12)
        self.fukui_zero_threshold = elec.get('fukui_zero_threshold', 0.10)
        self.electrophilicity_threshold = elec.get('electrophilicity_threshold', 1.3)
        self.nucleophilicity_threshold = elec.get('nucleophilicity_threshold', 1.8)
        self.homo_energy_factor = elec.get('homo_energy_factor', 0.35)
        self.lumo_energy_factor = elec.get('lumo_energy_factor', 0.25)
        self.gap_energy_factor = elec.get('gap_energy_factor', 0.40)
        self.homo_lumo_correlation = elec.get('homo_lumo_correlation', 0.15)

        mech = chem.get('mechanistic', {})
        self.oa_barrier = mech.get('oxidative_addition_barrier', 28.5)
        self.oa_barrier_sigma_effect = mech.get('oxidative_addition_barrier_sigma_effect', 4.2)
        self.oa_barrier_steric_effect = mech.get('oxidative_addition_barrier_steric_effect', 3.8)
        self.oa_rate = mech.get('oxidative_addition_rate', 0.045)
        self.oa_steric_sens = mech.get('oxidative_addition_steric_sensitivity', 0.65)
        self.oa_electronic_sens = mech.get('oxidative_addition_electronic_sensitivity', 0.85)
        self.tm_barrier = mech.get('transmetalation_barrier', 22.3)
        self.tm_barrier_base_effect = mech.get('transmetalation_barrier_base_effect', -3.2)
        self.tm_barrier_boronic_effect = mech.get('transmetalation_barrier_boronic_effect', -2.8)
        self.tm_rate = mech.get('transmetalation_rate', 0.078)
        self.tm_base_sens = mech.get('transmetalation_base_sensitivity', 0.75)
        self.tm_boronic_sens = mech.get('transmetalation_boronic_sensitivity', 0.70)
        self.re_barrier = mech.get('reductive_elimination_barrier', 18.7)
        self.re_barrier_steric_effect = mech.get('reductive_elimination_barrier_steric_effect', 5.6)
        self.re_rate = mech.get('reductive_elimination_rate', 0.120)
        self.re_steric_sens = mech.get('reductive_elimination_steric_sensitivity', 0.90)
        self.re_electronic_sens = mech.get('reductive_elimination_electronic_sensitivity', 0.60)
        self.oa_weight = mech.get('oa_weight', 0.35)
        self.tm_weight = mech.get('tm_weight', 0.35)
        self.re_weight = mech.get('re_weight', 0.30)
        self.transition_state_asymmetry = mech.get('transition_state_asymmetry', 1.2)
        self.intermediate_stability_factor = mech.get('intermediate_stability_factor', 0.8)

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

        pcp = chem.get('physicochemical_proxy', {})
        self.pcp_mw_weight = pcp.get('mw_weight', 0.30)
        self.pcp_logp_weight = pcp.get('logp_weight', 0.25)
        self.pcp_tpsa_weight = pcp.get('tpsa_weight', 0.25)
        self.pcp_rotbonds_weight = pcp.get('rotbonds_weight', 0.20)
        self.pcp_mw_center = pcp.get('mw_center', 300)
        self.pcp_mw_scale = pcp.get('mw_scale', 200)
        self.pcp_logp_center = pcp.get('logp_center', 2.5)
        self.pcp_logp_scale = pcp.get('logp_scale', 3.0)
        self.pcp_tpsa_center = pcp.get('tpsa_center', 60)
        self.pcp_tpsa_scale = pcp.get('tpsa_scale', 60)
        self.pcp_rotbonds_center = pcp.get('rotbonds_center', 4)
        self.pcp_rotbonds_scale = pcp.get('rotbonds_scale', 6)

        yield_p = chem.get('yield_parameters', {})
        self.max_yield = yield_p.get('max_yield', 98)
        self.min_yield = yield_p.get('min_yield', 5)
        self.base_yield_offset = yield_p.get('base_yield_offset', 45)
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
        self.elecproxy_weight = fi.get('elecproxy_effects', 0.02)
        self.physchem_weight = fi.get('physchem_effects', 0.03)

    def _validate_params(self):
        if self.optimal_temp <= 0:
            logger.warning(f"Optimal temp ({self.optimal_temp}) must be positive")
        if self.optimal_time <= 0:
            logger.warning(f"Optimal time ({self.optimal_time}) must be positive")
        if self.k_m <= 0:
            logger.warning(f"Km must be positive: {self.k_m}")
        if self.min_yield >= self.max_yield:
            logger.warning(f"Min yield ({self.min_yield}) >= max yield ({self.max_yield})")
        total_weight = (self.temp_weight + self.time_weight + self.catalyst_weight +
                       self.substrate1_steric_weight + self.substrate2_steric_weight +
                       self.solvent_weight + self.base_weight + self.electronic_weight +
                       self.hsab_weight + self.mechanistic_weight + self.hammett_weight +
                       self.taft_weight + self.elecproxy_weight + self.physchem_weight)
        if abs(total_weight - 1.0) > 0.05:
            logger.warning(f"Feature importance weights sum to {total_weight:.2f}, not 1.0")

    def calculate_electronic_proxy_parameters(self, smiles):
        result = {}
        try:
            if not RDKIT_AVAILABLE:
                return result
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return result
            sigma_p = 0.0
            for group, vals in HAMMETT_SIGMA.items():
                if group in smiles:
                    sigma_p += vals.get('sigma_p', 0)
            homo_energy = self.base_homo_energy - (sigma_p * self.homo_shift_factor)
            lumo_energy = self.base_lumo_energy - (sigma_p * self.lumo_shift_factor)
            gap_energy = abs(lumo_energy - homo_energy)
            result['homo_energy'] = round(homo_energy, 4)
            result['lumo_energy'] = round(lumo_energy, 4)
            result['gap_energy'] = round(gap_energy, 4)
            chemical_potential = (homo_energy + lumo_energy) / 2
            result['chemical_potential'] = round(chemical_potential, 4)
            hardness = (lumo_energy - homo_energy) / 2
            result['absolute_hardness'] = round(hardness, 4)
            electrophilicity = (chemical_potential ** 2) / (2 * hardness) if hardness > 0 else 0
            result['electrophilicity'] = round(electrophilicity, 4)
            fukui_plus = max(0, -sigma_p * 0.3 + 0.1)
            fukui_minus = max(0, sigma_p * 0.3 + 0.1)
            result['fukui_plus'] = round(fukui_plus, 4)
            result['fukui_minus'] = round(fukui_minus, 4)
            homo_tce = -8.0
            nucleophilicity = max(0, (homo_energy - homo_tce) / 1.0)
            result['nucleophilicity'] = round(nucleophilicity, 4)
        except Exception as e:
            pass
        return result

    def calculate_physicochemical_proxy_parameters(self, smiles):
        result = {}
        try:
            if not RDKIT_AVAILABLE:
                return result
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return result
            mw = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hba = Lipinski.NumHAcceptors(mol)
            hbd = Lipinski.NumHDonors(mol)
            rot_bonds = Lipinski.NumRotatableBonds(mol)
            result['mw'] = round(mw, 4)
            result['logp'] = round(logp, 4)
            result['tpsa'] = round(tpsa, 4)
            result['hba'] = hba
            result['hbd'] = hbd
            result['rotatable_bonds'] = rot_bonds
            mw_term = 1.0 - min(1.0, abs(mw - self.pcp_mw_center) / self.pcp_mw_scale)
            logp_term = 1.0 - min(1.0, abs(logp - self.pcp_logp_center) / self.pcp_logp_scale)
            tpsa_term = 1.0 - min(1.0, abs(tpsa - self.pcp_tpsa_center) / self.pcp_tpsa_scale)
            rotbonds_term = 1.0 - min(1.0, abs(rot_bonds - self.pcp_rotbonds_center) / self.pcp_rotbonds_scale)
            proxy_score = (self.pcp_mw_weight * mw_term + self.pcp_logp_weight * logp_term +
                          self.pcp_tpsa_weight * tpsa_term + self.pcp_rotbonds_weight * rotbonds_term)
            result['proxy_score'] = round(max(0.0, proxy_score), 4)
        except Exception as e:
            pass
        return result

    def calculate_eyring_rate(self, temp, delta_h, delta_s):
        T = temp + 273.15
        return (boltzmann_k / h) * T * np.exp(-(delta_h * 1000) / (R * T)) * np.exp(delta_s / R)

    def calculate_gibbs_energy(self, temp, delta_h, delta_s):
        T = temp + 273.15
        return delta_h - T * delta_s / 1000

    def calculate_equilibrium_constant(self, temp, delta_g):
        T = temp + 273.15
        return np.exp(-delta_g * 1000 / (R * T))

    def calculate_lfer(self, sigma, rho):
        return np.exp(rho * sigma)

    def calculate_hsab_absolute_hardness(self, ip, ea):
        return ip - ea

    def calculate_hsab_chemical_potential(self, ip, ea):
        return -(ip + ea) / 2

    def calculate_hsab_electronegativity(self, ip, ea):
        return (ip + ea) / 2

    def calculate_solvent_kamlet_taft(self, alpha, beta, pi_star):
        return alpha * 0.3 + beta * 0.3 + pi_star * 0.4

    def calculate_steric_a_value(self, substituent):
        a_values = {
            'methyl': self.a_value_methyl,
            'ethyl': self.a_value_ethyl,
            'isopropyl': self.a_value_isopropyl,
            'tertbutyl': self.a_value_tertbutyl
        }
        return a_values.get(substituent.lower(), 1.74)

    def calculate_hammett_sigma_plus(self, substituent):
        sigma_plus = HAMMETT_SIGMA.get(substituent, {}).get('sigma_plus', 0)
        return sigma_plus * self.sigma_plus_coeff

    def calculate_hammett_sigma_minus(self, substituent):
        sigma_minus = HAMMETT_SIGMA.get(substituent, {}).get('sigma_minus', 0)
        return sigma_minus * self.sigma_minus_coeff

    def calculate_taft_steric_parameter(self, substituent):
        return HAMMETT_SIGMA.get(substituent, {}).get('taft_es', 0)

    def calculate_oxidative_addition_barrier(self, sigma_p, steric_bulk):
        barrier = self.oa_barrier + (sigma_p * self.oa_barrier_sigma_effect) + (steric_bulk * self.oa_barrier_steric_effect)
        return max(0, barrier)

    def calculate_transmetalation_barrier(self, base_strength, boronic_bulk):
        barrier = self.tm_barrier + (base_strength * self.tm_barrier_base_effect) + (boronic_bulk * self.tm_barrier_boronic_effect)
        return max(0, barrier)

    def calculate_reductive_elimination_barrier(self, steric_bulk):
        barrier = self.re_barrier + (steric_bulk * self.re_barrier_steric_effect)
        return max(0, barrier)

    def temperature_factor(self, temp, experimental=False):
        if experimental:
            T_ref_K = max(self.baseline_temp, 25.0) + 273.15
            T_K = max(temp, -273.0) + 273.15
            R_val = self.gas_constant
            Ea = self.activation_energy * 1000
            rate_ratio = np.exp(-Ea / R_val * (1 / T_K - 1 / T_ref_K))
            T_ceiling_K = self.exp_temp_max + 273.15
            rate_ratio_ceiling = np.exp(-Ea / R_val * (1 / T_ceiling_K - 1 / T_ref_K))
            normalized = rate_ratio / rate_ratio_ceiling if rate_ratio_ceiling > 0 else 1.0
            factor = self.exp_temp_base_offset + self.exp_temp_increment_factor * normalized
            return max(0.10, factor)
        if temp > self.degradation_threshold:
            degradation_penalty = np.exp(-self.degradation_rate * (temp - self.degradation_threshold))
            factor = max(0.05, degradation_penalty)
        else:
            if temp < self.optimal_temp:
                sigma = self.temp_range / 3 / self.curve_asymmetry
            else:
                sigma = self.temp_range / 3 * self.curve_asymmetry
            deviation = abs(temp - self.optimal_temp)
            factor = np.exp(-(deviation ** 2) / (2 * sigma ** 2))
            R_val = self.gas_constant
            T_opt = self.optimal_temp + 273.15
            T_curr = max(temp, -273.0) + 273.15
            arrhenius = np.exp(-self.activation_energy * 1000 / R_val * (1 / T_curr - 1 / T_opt))
            factor = factor * arrhenius
            if abs(temp - self.optimal_temp) < 5:
                factor = factor * self.optimal_temp_bonus
            eyring_rate = self.calculate_eyring_rate(temp, self.enthalpy_activation, self.entropy_activation)
            eyring_rate_opt = self.calculate_eyring_rate(self.optimal_temp, self.enthalpy_activation, self.entropy_activation)
            eyring_factor = eyring_rate / eyring_rate_opt if eyring_rate_opt > 0 else 1.0
            factor = factor * (0.7 + 0.3 * eyring_factor)
        log_factor = 1 + self.temp_increment_factor * np.log1p(max(temp, 0.0) / self.baseline_temp)
        factor = factor * log_factor
        return max(0.05, factor)

    def time_factor(self, time_hours, experimental=False):
        if experimental:
            conversion = 1 - np.exp(-self.rate_constant * max(time_hours, 0.0))
            factor = self.exp_time_base_offset + self.exp_time_increment_factor * conversion
            return max(0.10, factor)
        factor = 1 - np.exp(-self.rate_constant * time_hours)
        if factor > self.diffusion_limit:
            factor = factor * (1 - self.diminishing_returns * (factor - self.diffusion_limit))
        if time_hours > self.saturation_point:
            factor = factor * self.plateau_factor
        if abs(time_hours - self.optimal_time) < 2:
            factor = factor * self.optimal_time_bonus
        reaction_order_factor = max(time_hours, 0.0) ** (1 / self.reaction_order)
        factor = factor * (0.7 + 0.3 * reaction_order_factor / (self.optimal_time ** (1 / self.reaction_order)))
        half_life = self.reaction_half_life * np.exp(self.half_life_temperature_dependence * 0)
        half_life_factor = 1 - np.exp(-np.log(2) * time_hours / half_life)
        factor = factor * (0.85 + 0.15 * half_life_factor)
        log_factor = 1 + self.time_increment_factor * np.log1p(max(time_hours, 0.0) / self.baseline_time)
        factor = factor * log_factor
        return max(0.05, factor)

    def catalyst_factor(self, quantity, experimental=False):
        if experimental:
            mm_saturation = max(quantity, 0.0) / (self.k_m + max(quantity, 0.0))
            factor = self.exp_catalyst_base_offset + self.exp_catalyst_increment_factor * mm_saturation
            return max(0.10, factor)
        if quantity > self.degradation_threshold_cat:
            rate = self.v_max * quantity / (self.k_m + quantity)
            factor = (rate / self.v_max) * self.quality_coefficient * 0.5
        else:
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
        log_factor = 1 + self.catalyst_increment_factor * np.log1p(max(quantity, 0.0) / self.baseline_quantity)
        factor = factor * log_factor
        return max(0.05, factor)

    def steric_factor(self, conditions):
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

    def electronic_factor(self, conditions):
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

    def hsab_factor(self, conditions):
        pd_soft = getattr(self, 'pd_softness', 2.8)
        halide_soft = getattr(self, 'halide_softness', 3.2)
        ligand_soft = getattr(self, 'ligand_softness', 2.5)
        base_soft = getattr(self, 'base_softness', 3.0)
        pd_halide_match_soft = 1 - abs(pd_soft - halide_soft) / 6
        pd_ligand_match_soft = 1 - abs(pd_soft - ligand_soft) / 6
        ligand_halide_match_soft = 1 - abs(ligand_soft - halide_soft) / 6
        pd_hard = getattr(self, 'absolute_hardness_pd', 3.8)
        halide_hard = getattr(self, 'absolute_hardness_halide', 4.2)
        ligand_hard = getattr(self, 'absolute_hardness_ligand', 3.5)
        pd_halide_match_hard = 1 - abs(pd_hard - halide_hard) / 8
        pd_ligand_match_hard = 1 - abs(pd_hard - ligand_hard) / 8
        ligand_halide_match_hard = 1 - abs(ligand_hard - halide_hard) / 8
        w1 = getattr(self, 'pd_halide_match_xml', 0.85)
        w2 = getattr(self, 'pd_ligand_match_xml', 0.90)
        w3 = getattr(self, 'ligand_halide_match_xml', 0.75)
        overall_soft = (pd_halide_match_soft * w1 + pd_ligand_match_soft * w2 + ligand_halide_match_soft * w3) / (w1 + w2 + w3)
        overall_hard = (pd_halide_match_hard * w1 + pd_ligand_match_hard * w2 + ligand_halide_match_hard * w3) / (w1 + w2 + w3)
        overall = overall_soft * 0.6 + overall_hard * 0.4
        chem_pot_effect = np.exp((getattr(self, 'chemical_potential_pd', -5.2) - getattr(self, 'electronegativity_pd', 5.2)) / 4)
        overall_compatibility = getattr(self, 'overall_compatibility_xml', 0.80)
        soft_soft_bonus = getattr(self, 'soft_soft_bonus', 1.20)
        soft_hard_penalty = getattr(self, 'soft_hard_penalty', 0.70)
        mismatch_penalty = getattr(self, 'mismatch_penalty', 0.50)
        if overall > overall_compatibility:
            factor = soft_soft_bonus * chem_pot_effect
        elif overall > 0.5:
            factor = 1.0 * chem_pot_effect
        else:
            factor = soft_hard_penalty * chem_pot_effect
        if overall < 0.3:
            factor *= mismatch_penalty
        result = np.clip(factor, 0.3, 1.3)
        return result

    def solvent_factor(self, solv1, solv2=''):
        if not solv1 or solv1 == '':
            return 0.5
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
        if solv2 and solv2 != '' and solv2 != 'O':
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

    def base_factor(self, base):
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

    def mechanistic_factor(self, conditions):
        temp = conditions.get('temp', 80)
        time_hours = conditions.get('time', 24)
        steric_factor = conditions.get('steric_bulk', 0.5)
        electronic_factor = conditions.get('electronic_sensitivity', 1.0)
        base_strength = conditions.get('base_strength', 1.0)
        sigma_p = conditions.get('sigma_p', 0)
        R_val = self.gas_constant
        T = temp + 273.15
        oa_barrier = self.calculate_oxidative_addition_barrier(sigma_p, steric_factor)
        k_oa = self.oa_rate * np.exp(-oa_barrier * 1000 / (R_val * T))
        k_oa = k_oa * (1 - self.oa_steric_sens * steric_factor)
        k_oa = k_oa * (1 + self.oa_electronic_sens * electronic_factor)
        tm_barrier = self.calculate_transmetalation_barrier(base_strength, steric_factor)
        k_tm = self.tm_rate * np.exp(-tm_barrier * 1000 / (R_val * T))
        k_tm = k_tm * (1 + self.tm_base_sens * base_strength)
        k_tm = k_tm * (1 + self.tm_boronic_sens * 0.5)
        re_barrier = self.calculate_reductive_elimination_barrier(steric_factor)
        k_re = self.re_rate * np.exp(-re_barrier * 1000 / (R_val * T))
        k_re = k_re * (1 - self.re_steric_sens * steric_factor)
        k_re = k_re * (1 + self.re_electronic_sens * electronic_factor)
        intermediate_stability = self.intermediate_stability_factor * np.exp(-(oa_barrier + tm_barrier) * 1000 / (2 * R_val * T))
        rate = (self.oa_weight * k_oa + self.tm_weight * k_tm + self.re_weight * k_re) * intermediate_stability
        time_factor = 1 - np.exp(-rate * time_hours * 60)
        mechanistic_efficiency = (k_oa * k_tm * k_re) / (max(k_oa, 0.001) * max(k_tm, 0.001) * max(k_re, 0.001) + 0.001)
        transition_state_asymmetry_factor = np.exp(-self.transition_state_asymmetry * abs(k_oa - k_re) / (k_oa + k_re + 0.001))
        factor = time_factor * (0.8 + 0.2 * mechanistic_efficiency) * transition_state_asymmetry_factor
        result = {
            'factor': np.clip(factor * 1.5, 0.1, 1.3),
            'oa_rate': k_oa,
            'tm_rate': k_tm,
            're_rate': k_re,
            'efficiency': mechanistic_efficiency,
            'rate_indicator': rate,
            'oa_barrier_calculated': oa_barrier,
            'tm_barrier_calculated': tm_barrier,
            're_barrier_calculated': re_barrier
        }
        return result

    def elecproxy_factor(self, conditions):
        sigma_p = conditions.get('sigma_p', 0)
        sigma_m = conditions.get('sigma_m', 0)
        homo_energy = self.base_homo_energy - (sigma_p * self.homo_shift_factor)
        lumo_energy = self.base_lumo_energy - (sigma_p * self.lumo_shift_factor)
        gap_energy = abs(lumo_energy - homo_energy)
        homo_lumo_correlation = getattr(self, 'homo_lumo_correlation', 0.15)
        chemical_potential_weight = getattr(self, 'chemical_potential_weight', 0.28)
        hardness_weight = getattr(self, 'hardness_weight', 0.22)
        electrophilicity_weight = getattr(self, 'electrophilicity_weight', 0.18)
        fukui_weight = getattr(self, 'fukui_weight', 0.32)
        homo_lumo_effect = sigma_p * homo_lumo_correlation
        chemical_potential_effect = -sigma_p * chemical_potential_weight
        hardness_effect = (1 - abs(sigma_p)) * hardness_weight
        electrophilicity_effect = max(0, sigma_p) * electrophilicity_weight
        fukui_effect = abs(sigma_p) * fukui_weight
        factor = 1 + homo_lumo_effect + chemical_potential_effect + hardness_effect + electrophilicity_effect + fukui_effect
        factor = factor * (0.8 + 0.2 * (gap_energy / 5.0))
        return np.clip(factor, 0.7, 1.3)

    def physchem_factor(self, conditions):
        mw = conditions.get('mw', 200)
        logp = conditions.get('logp', 2)
        rot_bonds = conditions.get('rotatable_bonds', 4)
        tpsa = conditions.get('tpsa', 60)
        mw_term = 1.0 - min(1.0, abs(mw - self.pcp_mw_center) / self.pcp_mw_scale)
        logp_term = 1.0 - min(1.0, abs(logp - self.pcp_logp_center) / self.pcp_logp_scale)
        tpsa_term = 1.0 - min(1.0, abs(tpsa - self.pcp_tpsa_center) / self.pcp_tpsa_scale)
        rotbonds_term = 1.0 - min(1.0, abs(rot_bonds - self.pcp_rotbonds_center) / self.pcp_rotbonds_scale)
        proxy_score = (self.pcp_mw_weight * mw_term + self.pcp_logp_weight * logp_term +
                      self.pcp_tpsa_weight * tpsa_term + self.pcp_rotbonds_weight * rotbonds_term)
        proxy_score = max(0.0, proxy_score)
        return np.clip(0.7 + 0.3 * proxy_score, 0.7, 1.3)

    def calculate_yield(self, conditions):
        temp = conditions.get('temp', 80)
        time_hours = conditions.get('time', 24)
        quantity = conditions.get('quantity', 0.0025)
        experimental = bool(conditions.get('experimental_mode', False))
        temp_factor = self.temperature_factor(temp, experimental=experimental)
        time_factor = self.time_factor(time_hours, experimental=experimental)
        cat_factor = self.catalyst_factor(quantity, experimental=experimental)
        steric_factor_1 = self.steric_factor({**conditions, **{'substrate': 1}})
        steric_factor_2 = self.steric_factor({**conditions, **{'substrate': 2}})
        solvent_factor = self.solvent_factor(conditions.get('solv1', ''), conditions.get('solv2', ''))
        base_factor = self.base_factor(conditions.get('base', ''))
        electronic_factor = self.electronic_factor(conditions)
        hsab_factor = self.hsab_factor(conditions)
        mechanistic_result = self.mechanistic_factor(conditions)
        mechanistic_factor = mechanistic_result['factor']
        elecproxy_factor = self.elecproxy_factor(conditions)
        physchem_factor = self.physchem_factor(conditions)
        total_weight = (self.temp_weight + self.time_weight + self.catalyst_weight +
                       self.substrate1_steric_weight + self.substrate2_steric_weight +
                       self.solvent_weight + self.base_weight + self.electronic_weight +
                       self.hsab_weight + self.mechanistic_weight + self.hammett_weight +
                       self.taft_weight + self.elecproxy_weight + self.physchem_weight)
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
            (self.taft_weight / total_weight) * electronic_factor * 0.3 +
            (self.elecproxy_weight / total_weight) * elecproxy_factor +
            (self.physchem_weight / total_weight) * physchem_factor
        )
        raw_yield = self.base_yield_offset + (self.max_yield - self.base_yield_offset) * combined_factor
        final_yield = raw_yield * self.reproducibility * self.scale_up_factor
        result = {
            'yield': np.clip(final_yield, self.min_yield, self.max_yield),
            'temp_factor': float(temp_factor),
            'time_factor': float(time_factor),
            'catalyst_factor': float(cat_factor),
            'steric_factor_1': float(steric_factor_1),
            'steric_factor_2': float(steric_factor_2),
            'solvent_factor': float(solvent_factor),
            'base_factor': float(base_factor),
            'electronic_factor': float(electronic_factor),
            'hsab_factor': float(hsab_factor),
            'mechanistic_factor': float(mechanistic_factor),
            'elecproxy_factor': float(elecproxy_factor),
            'physchem_factor': float(physchem_factor),
            'mechanistic_details': mechanistic_result,
            'combined_factor': float(combined_factor)
        }
        return result

    def get_yield_class(self, yield_val):
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

    def get_yield_stats(self):
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
    def __init__(self, config):
        self.config = config
        self.feature_columns = []
        self.scaler = None
        self.encoders = {}
        self.imputer = None
        self.selected_features = []
        self.pca = None
        self.feature_names = []
        self._load_params()

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

    def extract_smiles_features(self, smiles):
        features = {}
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen
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
        except Exception as e:
            pass
        return features

    def engineer_features(self, df):
        df = df.copy()
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
        return df

    def select_features(self, df, target='yield'):
        try:
            from sklearn.feature_selection import mutual_info_regression
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)
            if len(numeric_cols) <= 5:
                return df[numeric_cols] if numeric_cols else df
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
            return df[selected] if selected else df[numeric_cols]
        except Exception as e:
            logger.error(f"Feature selection error: {e}")
            return df.select_dtypes(include=[np.number])


class SuzukiPredictor:
    def __init__(self, config):
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
        self.fallback_model = None
        self.training_data_warning = None
        self.honest_cv_performance = None
        self.feature_columns_used = []
        self.min_samples_per_feature = self.config.get_int('ml_training_safeguards/min_samples_per_feature', 10)
        self.min_samples_for_full_ensemble = self.config.get_int('ml_training_safeguards/min_samples_for_full_ensemble', 30)
        self.honest_cv_max_folds = self.config.get_int('ml_training_safeguards/honest_cv_max_folds', 5)

    def _load_weights(self):
        try:
            w = self.config.get_dict('model_parameters/Ensemble/weights')
            if w:
                return {k: float(v) for k, v in w.items() if float(v) > 0}
        except Exception as e:
            pass
        return {
            'Random_Forest': 0.16,
            'Gradient_Boosting': 0.10,
            'Hist_Gradient_Boosting': 0.16,
            'XGBoost': 0.10,
            'LightGBM': 0.07,
            'CatBoost': 0.07,
            'Extra_Trees': 0.04,
            'Gaussian_Process': 0.08,
            'SVR': 0.02,
            'Neural_Network': 0.03,
            'Ridge': 0.02,
            'ElasticNet': 0.02
        }

    def validate_csv(self, filepath):
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

    def load_data(self, filepath):
        valid, missing = self.validate_csv(filepath)
        if not valid:
            missing_required = [m for m in missing if 'optional' not in m]
            if missing_required:
                raise ValueError(f"CSV validation failed. Missing required columns: {', '.join(missing_required)}")
        try:
            self.df = pd.read_csv(filepath)
            if 'yield' not in self.df.columns:
                raise ValueError("'yield' column not found")
            if self.df['yield'].isnull().all():
                raise ValueError("All yield values are missing")
            self.is_enriched = is_enriched_dataset(self.df)
            if not self.is_enriched:
                raise ValueError(
                    "This is a basic dataset without academic features.\n"
                    "Please use dataset_routes.py to enrich your data first."
                )
            usable_df, failed_df, rejected_df = classify_and_filter_rows(self.df)
            self.failed_reactions_df = failed_df
            self.rejected_rows_df = rejected_df
            self.df = usable_df
            if len(self.df) < 5:
                raise ValueError(f"Dataset must have at least 5 rows with valid yield data. Current: {len(self.df)}")
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
            'reaction_rate_indicator',
            'elecproxy_homo_energy', 'elecproxy_lumo_energy', 'elecproxy_gap_energy',
            'elecproxy_chemical_potential', 'elecproxy_absolute_hardness', 'elecproxy_electrophilicity',
            'elecproxy_fukui_plus', 'elecproxy_fukui_minus',
            'physchem_proxy_score'
        ]
        available_features = [c for c in important_features if c in self.df.columns]
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
        X_categorical = _dedupe_columns(X_categorical)
        self.X = pd.concat([X_numeric, X_categorical], axis=1) if not X_categorical.empty else X_numeric
        self.y = self.df['yield'].values
        self.feature_columns = [clean_feature_name(c) for c in self.X.columns]
        self.X.columns = self.feature_columns
        self.X = _dedupe_columns(self.X)
        self.feature_columns = list(self.X.columns)
        if self.X.isnull().any().any():
            self.X = self.X.fillna(0)

    def _create_model(self, name):
        try:
            params = self.config.get_model_params(name)
            invalid_params = ['feature_importance_type', 'early_stopping_rounds', 'early_stopping']
            for ip in invalid_params:
                if ip in params:
                    del params[ip]
            params = {k: self._parse_param(v) for k, v in params.items()}
            if name == 'Random_Forest':
                return RandomForestRegressor(**params)
            elif name == 'Gradient_Boosting':
                return GradientBoostingRegressor(**params)
            elif name == 'Hist_Gradient_Boosting':
                return HistGradientBoostingRegressor(**params)
            elif name == 'XGBoost' and XGB_AVAILABLE:
                if 'early_stopping_rounds' in params:
                    del params['early_stopping_rounds']
                return XGBRegressor(**params)
            elif name == 'LightGBM' and LGBM_AVAILABLE:
                if 'early_stopping_rounds' in params:
                    del params['early_stopping_rounds']
                return LGBMRegressor(**params)
            elif name == 'CatBoost' and CATBOOST_AVAILABLE:
                if 'early_stopping_rounds' in params:
                    del params['early_stopping_rounds']
                return CatBoostRegressor(**params)
            elif name == 'Extra_Trees':
                return ExtraTreesRegressor(**params)
            elif name == 'Gaussian_Process' and GP_AVAILABLE:
                if 'kernel' in params and isinstance(params['kernel'], str):
                    try:
                        params['kernel'] = eval(params['kernel'])
                    except:
                        params['kernel'] = RBF(1.0) + WhiteKernel(0.1)
                return GaussianProcessRegressor(**params)
            elif name == 'SVR':
                return SVR(**params)
            elif name == 'Neural_Network':
                return MLPRegressor(**params)
            elif name == 'Ridge':
                return Ridge(**params)
            elif name == 'Lasso':
                return Lasso(**params)
            elif name == 'ElasticNet':
                return ElasticNet(**params)
            elif name == 'KNN':
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor(**params)
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

    def train(self, model_type='Ensemble'):
        try:
            if self.X is None or len(self.X) == 0:
                raise ValueError("Data must be loaded first")
            if not self.is_enriched:
                raise ValueError("Cannot train ML model on basic dataset.")
            if len(self.y) == 0 or np.all(np.isnan(self.y)):
                raise ValueError("No valid yield data available for training")
            n_samples = len(self.X)
            n_features_available = len(self.feature_columns)
            if n_samples < self.min_samples_per_feature:
                raise ValueError(f"Refusing to train: only {n_samples} labeled reactions available.")
            self.training_data_warning = None
            max_features_for_sample_size = max(1, n_samples // self.min_samples_per_feature)
            if n_samples < self.min_samples_for_full_ensemble or n_features_available > max_features_for_sample_size:
                self.training_data_warning = (
                    f"LOW-DATA REGIME: {n_samples} samples for {n_features_available} candidate features."
                )
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
            imputer = SimpleImputer(strategy='mean')
            X_scaled = pd.DataFrame(imputer.fit_transform(X_scaled), columns=self.feature_columns)
            if n_features_available > max_features_for_sample_size:
                selector = SelectKBest(score_func=f_regression, k=max_features_for_sample_size)
                X_selected = selector.fit_transform(X_scaled.fillna(0), self.y)
                selected_cols = [c for c, keep in zip(self.feature_columns, selector.get_support()) if keep]
                X_scaled = pd.DataFrame(X_selected, columns=selected_cols)
                self.feature_columns_used = selected_cols
            else:
                self.feature_columns_used = list(self.feature_columns)
            test_size = min(0.2, max(0.1, 3.0 / n_samples)) if n_samples > 3 else 0.1
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, self.y, test_size=test_size, random_state=42)
            models = {}
            performances = {}
            if model_type == 'Ensemble' or model_type == 'all':
                if n_samples < self.min_samples_for_full_ensemble:
                    model_names = ['Ridge', 'Lasso', 'ElasticNet']
                else:
                    model_names = [
                        'Random_Forest', 'Gradient_Boosting', 'Hist_Gradient_Boosting',
                        'XGBoost', 'LightGBM', 'CatBoost', 'Extra_Trees',
                        'Gaussian_Process', 'SVR', 'Neural_Network',
                        'Ridge', 'Lasso', 'ElasticNet', 'KNN'
                    ]
            else:
                model_names = [model_type]
            for name in model_names:
                try:
                    model = self._create_model(name)
                    if model is not None:
                        model.fit(X_train, y_train)
                        models[name] = model
                        y_pred = model.predict(X_test)
                        if len(y_pred) > 0 and not np.isnan(y_pred).all():
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                            ev = explained_variance_score(y_test, y_pred)
                            performances[name] = {
                                'r2': float(r2),
                                'mae': float(mae),
                                'rmse': float(rmse),
                                'mape': float(mape),
                                'explained_variance': float(ev)
                            }
                except Exception as e:
                    pass
            if not models:
                try:
                    model = Ridge(alpha=1.0)
                    model.fit(X_train, y_train)
                    models['Ridge_Fallback'] = model
                    y_pred = model.predict(X_test)
                    if len(y_pred) > 0 and not np.isnan(y_pred).all():
                        performances['Ridge_Fallback'] = {
                            'r2': float(r2_score(y_test, y_pred)),
                            'mae': float(mean_absolute_error(y_test, y_pred)),
                            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                            'mape': float(mean_absolute_percentage_error(y_test, y_pred) * 100),
                            'explained_variance': float(explained_variance_score(y_test, y_pred))
                        }
                        self.fallback_model = 'Ridge_Fallback'
                except Exception as e:
                    return {'success': False, 'message': 'No models could be trained'}
            if not models:
                return {'success': False, 'message': 'No models could be trained'}
            self.models = models
            self.is_trained = True
            self.model_performance = performances
            if performances:
                best_name = max(performances.items(), key=lambda x: x[1].get('r2', 0))[0]
                self.best_model = best_name
            self.cv_results = self._perform_cross_validation()
            self.feature_importance = self._calculate_feature_importance()
            self.honest_cv_performance = None
            if self.training_data_warning is not None:
                try:
                    X_raw = self.X.copy()
                    X_raw = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X_raw), columns=self.feature_columns)
                    y_arr = np.asarray(self.y)
                    n_splits = max(2, min(self.honest_cv_max_folds, n_samples // 2))
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                    honest_results = {}
                    for name in models.keys():
                        oof_pred = np.full(n_samples, np.nan)
                        for train_idx, test_idx in kf.split(X_raw):
                            pipe = Pipeline([
                                ('scaler', StandardScaler()),
                                ('select', SelectKBest(score_func=f_regression, k=max_features_for_sample_size)),
                                ('model', self._create_model(name))
                            ])
                            try:
                                pipe.fit(X_raw.iloc[train_idx], y_arr[train_idx])
                                oof_pred[test_idx] = pipe.predict(X_raw.iloc[test_idx])
                            except Exception as fold_err:
                                pass
                        valid = ~np.isnan(oof_pred)
                        if valid.sum() >= 2:
                            honest_results[name] = {
                                'r2_cv_honest': float(r2_score(y_arr[valid], oof_pred[valid])),
                                'mae_cv_honest': float(mean_absolute_error(y_arr[valid], oof_pred[valid])),
                                'n_folds': n_splits,
                                'n_evaluated': int(valid.sum())
                            }
                    self.honest_cv_performance = honest_results
                except Exception as e:
                    pass
            return {
                'success': True,
                'message': f"Trained {len(models)} models",
                'performance': convert_to_serializable(performances),
                'best_model': self.best_model,
                'model_count': len(models),
                'cv_results': convert_to_serializable(self.cv_results),
                'fallback_used': self.fallback_model is not None,
                'training_data_warning': self.training_data_warning,
                'honest_cv_performance': convert_to_serializable(self.honest_cv_performance) if self.honest_cv_performance else None,
                'features_used': getattr(self, 'feature_columns_used', self.feature_columns)
            }
        except Exception as e:
            logger.error(f"Train error: {str(e)}")
            raise

    def _perform_cross_validation(self):
        try:
            if not self.is_trained or not self.models:
                return {}
            X_scaled = self.scaler.transform(self.X)
            cv_results = {}
            kf = KFold(n_splits=min(5, len(self.X)), shuffle=True, random_state=42)
            for name, model in self.models.items():
                try:
                    scores = cross_val_score(model, X_scaled, self.y, cv=kf, scoring='r2')
                    cv_results[name] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'scores': [float(s) for s in scores]
                    }
                except Exception as e:
                    pass
            return cv_results
        except Exception as e:
            return {}

    def _calculate_feature_importance(self):
        try:
            if not self.is_trained or not self.models:
                return {}
            model = list(self.models.values())[0]
            cols_used = getattr(self, 'feature_columns_used', None) or self.feature_columns
            X_scaled_full = self.scaler.transform(self.X)
            X_scaled_full = pd.DataFrame(X_scaled_full, columns=self.feature_columns)
            X_scaled = X_scaled_full[cols_used].values
            if PERM_IMP_AVAILABLE:
                result = permutation_importance(model, X_scaled, self.y, n_repeats=10, random_state=42)
                importance_dict = {}
                for i, col in enumerate(cols_used):
                    importance_dict[col] = {
                        'importance': float(result.importances_mean[i]),
                        'std': float(result.importances_std[i])
                    }
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1]['importance'], reverse=True)
                return {'top_10': sorted_importance[:10], 'all': importance_dict}
            return {}
        except Exception as e:
            return {}

    def _ensemble_predict(self, X):
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

    def _bootstrap_uncertainty(self, X, n_bootstrap=100):
        try:
            if not self.is_trained or not self.models:
                return {}
            n_samples = X.shape[0]
            all_predictions = []
            for _ in range(min(n_bootstrap, 100)):
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot = X[indices]
                pred = self._ensemble_predict(X_boot)
                all_predictions.append(pred)
            all_predictions = np.array(all_predictions)
            mean_pred = np.mean(all_predictions, axis=0)
            std_pred = np.std(all_predictions, axis=0)
            lower = np.percentile(all_predictions, 2.5, axis=0)
            upper = np.percentile(all_predictions, 97.5, axis=0)
            return {'mean': mean_pred, 'std': std_pred, 'lower_ci': lower, 'upper_ci': upper}
        except Exception as e:
            return {}

    def predict(self, conditions):
        try:
            experimental = bool(conditions.get('experimental_mode', False))
            temp_in = conditions.get('temp', 80)
            time_in = conditions.get('time', 24)
            qty_in = conditions.get('quantity', 0.0025)
            bounds_violations = []
            if temp_in < self.chemical.exp_temp_min or temp_in > self.chemical.exp_temp_max:
                bounds_violations.append(f"temp={temp_in} (allowed: {self.chemical.exp_temp_min}-{self.chemical.exp_temp_max} C)")
            if time_in < self.chemical.exp_time_min or time_in > self.chemical.exp_time_max:
                bounds_violations.append(f"time={time_in} (allowed: {self.chemical.exp_time_min}-{self.chemical.exp_time_max} h)")
            if qty_in < self.chemical.exp_quantity_min or qty_in > self.chemical.exp_quantity_max:
                bounds_violations.append(f"quantity={qty_in} (allowed: {self.chemical.exp_quantity_min}-{self.chemical.exp_quantity_max})")
            if bounds_violations:
                return {'success': False, 'message': "Input values outside physical bounds: " + "; ".join(bounds_violations)}
            last_pred = get_last_prediction_for_conditions(conditions)
            elecproxy_params = self.chemical.calculate_electronic_proxy_parameters(conditions.get('subs1_smiles', ''))
            elecproxy_params2 = self.chemical.calculate_electronic_proxy_parameters(conditions.get('subs2_smiles', ''))
            physchem_params = self.chemical.calculate_physicochemical_proxy_parameters(conditions.get('subs1_smiles', ''))
            chemical_result = self.chemical.calculate_yield(conditions)
            chemical_yield = chemical_result['yield']
            ml_yield = None
            uncertainty = {}
            if self.is_enriched and self.is_trained and self.models:
                try:
                    feature_vector = self._create_feature_vector(conditions)
                    if feature_vector is not None:
                        if len(self.models) > 1:
                            ml_pred = self._ensemble_predict([feature_vector])
                            ml_yield = float(ml_pred[0]) if len(ml_pred) > 0 else None
                            uncertainty = self._bootstrap_uncertainty(np.array([feature_vector]), n_bootstrap=50)
                        else:
                            model = list(self.models.values())[0]
                            ml_pred = model.predict([feature_vector])
                            ml_yield = float(ml_pred[0]) if len(ml_pred) > 0 else None
                except Exception as e:
                    pass
            if ml_yield is not None and not np.isnan(ml_yield) and self.is_enriched:
                base_yield = 0.6 * ml_yield + 0.4 * chemical_yield
                model_name = 'Ensemble'
            else:
                base_yield = chemical_yield
                model_name = 'Chemical Intuition'
            final_yield = base_yield
            ENABLE_HISTORY_DEPENDENT_BONUS = False
            if last_pred and not experimental and ENABLE_HISTORY_DEPENDENT_BONUS:
                prev_temp = last_pred.get('temp')
                prev_time = last_pred.get('time')
                prev_quantity = last_pred.get('quantity')
                prev_yield = last_pred.get('yield')
                current_temp = conditions.get('temp', 80)
                current_time = conditions.get('time', 24)
                current_quantity = conditions.get('quantity', 0.0025)
                if prev_temp is not None and current_temp > prev_temp:
                    temp_increase = calculate_logarithmic_increase(prev_temp, current_temp, prev_yield, max_increase=2.0)
                    if temp_increase > 0:
                        final_yield += temp_increase
                if prev_time is not None and current_time > prev_time:
                    time_increase = calculate_logarithmic_increase(prev_time, current_time, prev_yield, max_increase=1.5)
                    if time_increase > 0:
                        final_yield += time_increase
                if prev_quantity is not None and current_quantity > prev_quantity:
                    qty_increase = calculate_logarithmic_increase(prev_quantity, current_quantity, prev_yield, max_increase=1.5)
                    if qty_increase > 0:
                        final_yield += qty_increase
            dominated = get_dominated_history_max_yield(conditions)
            if dominated is not None and final_yield < dominated['yield']:
                final_yield = dominated['yield']
            final_yield = np.clip(final_yield, 0, 100)
            confidence_interval = None
            prediction_interval = None
            if uncertainty:
                ci_lower = final_yield - 1.96 * uncertainty.get('std', [0])[0]
                ci_upper = final_yield + 1.96 * uncertainty.get('std', [0])[0]
                confidence_interval = {'lower': max(0, ci_lower), 'upper': min(100, ci_upper)}
                pi_lower = final_yield - 1.645 * uncertainty.get('std', [0])[0]
                pi_upper = final_yield + 1.645 * uncertainty.get('std', [0])[0]
                prediction_interval = {'lower': max(0, pi_lower), 'upper': min(100, pi_upper)}
            yield_class, color = self.chemical.get_yield_class(final_yield)
            confidence = self._calculate_confidence(ml_yield, chemical_yield, final_yield)
            solvent_status = "single"
            if conditions.get('solv2') and conditions.get('solv2') != '' and conditions.get('solv2') != 'O':
                solvent_status = "binary"
            elif not conditions.get('solv1') or conditions.get('solv1') == '':
                solvent_status = "none"
            prediction_data = {
                'temp': conditions.get('temp'),
                'time': conditions.get('time'),
                'quantity': conditions.get('quantity'),
                'catalizor': conditions.get('catalizor'),
                'base': conditions.get('base'),
                'solv1': conditions.get('solv1'),
                'solv2': conditions.get('solv2'),
                'subs1_smiles': conditions.get('subs1_smiles'),
                'subs2_smiles': conditions.get('subs2_smiles'),
                'yield': float(final_yield),
                'yield_class': yield_class,
                'model': model_name,
                'experimental_mode': experimental
            }
            save_prediction_history(prediction_data)
            exp_label = " [EXPERIMENTAL MODE]" if experimental else ""
            experimental_details = None
            if experimental:
                experimental_details = {
                    'mode': 'experimental',
                    'temp_factor_raw': float(chemical_result.get('temp_factor', 1.0)),
                    'time_factor_raw': float(chemical_result.get('time_factor', 1.0)),
                    'catalyst_factor_raw': float(chemical_result.get('catalyst_factor', 1.0)),
                    'monotonicity_guaranteed': True,
                    'upper_bound': '100 %'
                }
            return {
                'success': True,
                'prediction': float(final_yield),
                'prediction_display': f"~ {final_yield:.4f} % (est.){exp_label}",
                'ml_prediction': float(ml_yield) if ml_yield is not None else None,
                'chemical_prediction': float(chemical_yield),
                'model': model_name,
                'yield_class': yield_class,
                'yield_class_color': color,
                'confidence': float(confidence),
                'confidence_interval': confidence_interval,
                'prediction_interval': prediction_interval,
                'best_model': self.best_model,
                'model_count': len(self.models) if self.models else 0,
                'is_enriched': self.is_enriched,
                'experimental_mode': experimental,
                'experimental_details': experimental_details,
                'academic_details': {
                    'subs1_elecproxy': elecproxy_params,
                    'subs2_elecproxy': elecproxy_params2,
                    'subs1_physicochemical_proxy': physchem_params,
                    'hsab': {
                        'compatibility': chemical_result.get('hsab_factor', 0.8),
                        'pd_halide_match': self.chemical.pd_halide_match_xml,
                        'pd_ligand_match': self.chemical.pd_ligand_match_xml
                    },
                    'mechanistic': {
                        'rate_indicator': chemical_result.get('mechanistic_details', {}).get('rate_indicator', 0),
                        'oa_rate': chemical_result.get('mechanistic_details', {}).get('oa_rate', 0),
                        'tm_rate': chemical_result.get('mechanistic_details', {}).get('tm_rate', 0),
                        're_rate': chemical_result.get('mechanistic_details', {}).get('re_rate', 0),
                        'oa_barrier_calculated': chemical_result.get('mechanistic_details', {}).get('oa_barrier_calculated', 0),
                        'tm_barrier_calculated': chemical_result.get('mechanistic_details', {}).get('tm_barrier_calculated', 0),
                        're_barrier_calculated': chemical_result.get('mechanistic_details', {}).get('re_barrier_calculated', 0)
                    },
                    'solvent_status': solvent_status,
                    'factor_breakdown': {
                        'temperature': chemical_result.get('temp_factor', 1.0),
                        'time': chemical_result.get('time_factor', 1.0),
                        'catalyst': chemical_result.get('catalyst_factor', 1.0),
                        'steric1': chemical_result.get('steric_factor_1', 1.0),
                        'steric2': chemical_result.get('steric_factor_2', 1.0),
                        'solvent': chemical_result.get('solvent_factor', 1.0),
                        'base': chemical_result.get('base_factor', 1.0),
                        'electronic': chemical_result.get('electronic_factor', 1.0),
                        'hsab': chemical_result.get('hsab_factor', 1.0),
                        'mechanistic': chemical_result.get('mechanistic_factor', 1.0),
                        'electronic_proxy': chemical_result.get('elecproxy_factor', 1.0),
                        'physicochemical_proxy': chemical_result.get('physchem_factor', 1.0)
                    }
                },
                'cv_results': self.cv_results,
                'fallback_used': self.fallback_model is not None,
                'solvent_status': solvent_status,
                'history_count': len(PREDICTION_HISTORY)
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    def _create_feature_vector(self, conditions):
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
        else:
            f['subs1_length'] = 0
            f['subs1_logp'] = 0
            f['subs1_mw'] = 0
            f['subs1_rings'] = 0
            f['subs1_aromatic_rings'] = 0
            f['subs1_hba'] = 0
            f['subs1_hbd'] = 0
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
        if subs2:
            mf = self.fe.extract_smiles_features(subs2)
            f['subs2_length'] = len(subs2)
            f['subs2_logp'] = mf.get('logp', 0)
            f['subs2_mw'] = mf.get('mw', 0)
            f['subs2_rings'] = mf.get('rings', 0)
            f['subs2_aromatic_rings'] = mf.get('aromatic_rings', 0)
            f['subs2_hba'] = mf.get('hba', 0)
            f['subs2_hbd'] = mf.get('hbd', 0)
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
        else:
            f['subs2_length'] = 0
            f['subs2_logp'] = 0
            f['subs2_mw'] = 0
            f['subs2_rings'] = 0
            f['subs2_aromatic_rings'] = 0
            f['subs2_hba'] = 0
            f['subs2_hbd'] = 0
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
        elec = self.chemical.calculate_electronic_proxy_parameters(subs1)
        f['elecproxy_homo_energy'] = elec.get('homo_energy', -6.5)
        f['elecproxy_lumo_energy'] = elec.get('lumo_energy', -1.5)
        f['elecproxy_gap_energy'] = elec.get('gap_energy', 5.0)
        f['elecproxy_chemical_potential'] = elec.get('chemical_potential', -4.0)
        f['elecproxy_absolute_hardness'] = elec.get('absolute_hardness', 2.5)
        f['elecproxy_electrophilicity'] = elec.get('electrophilicity', 3.0)
        f['elecproxy_fukui_plus'] = elec.get('fukui_plus', 0.1)
        f['elecproxy_fukui_minus'] = elec.get('fukui_minus', 0.1)
        pcp = self.chemical.calculate_physicochemical_proxy_parameters(subs1)
        f['physchem_proxy_score'] = pcp.get('proxy_score', 0.7)
        vector = []
        for col in self.feature_columns:
            vector.append(f.get(col, 0))
        if self.scaler is not None:
            try:
                vector = self.scaler.transform([vector])[0]
            except Exception as e:
                pass
        return np.array(vector).reshape(1, -1)

    def _calculate_confidence(self, ml_yield, chemical_yield, final_yield):
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
        return np.clip(confidence, 0.3, 0.98)

    def optimize_catalyst(self, conditions):
        try:
            results = []
            catalysts = []
            if self.df is not None and 'catalizor' in self.df.columns:
                catalysts = self.df['catalizor'].unique().tolist()
            else:
                catalysts = ['Pd(PPh3)4', 'PdCl2(dppf)', 'Pd(OAc)2', 'Pd2(dba)3',
                             'PdCl2(PPh3)2', 'Pd(PPh3)2Cl2', 'PdCl2', 'Pd(acac)2',
                             'Pd(PhCN)2Cl2', 'PdCl2(MeCN)2', 'PdCl2(COD)', 'Pd(TFA)2', 'Pd(OPiv)2']
            for idx, catalyst in enumerate(catalysts[:20]):
                test_conditions = conditions.copy()
                test_conditions['catalizor'] = catalyst
                best_yield = 0
                best_qty = conditions.get('quantity', 0.0025)
                quantities = [0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]
                for qty in quantities:
                    test_conditions['quantity'] = qty
                    result = self.predict(test_conditions)
                    if result['success'] and result['prediction'] > best_yield:
                        best_yield = result['prediction']
                        best_qty = qty
                results.append((catalyst, best_yield, best_qty))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return []

    def get_best_model(self):
        if not self.model_performance:
            return None, {}
        best = max(self.model_performance.items(), key=lambda x: x[1].get('r2', 0))
        return best[0], best[1]

    def get_feature_importance(self):
        return self.feature_importance

    def analyze_residuals(self):
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
            if len(residuals) >= 3 and len(residuals) <= 5000:
                try:
                    shapiro_stat, shapiro_p = shapiro(residuals)
                    residual_stats['shapiro_wilk_stat'] = float(shapiro_stat)
                    residual_stats['shapiro_wilk_p'] = float(shapiro_p)
                    residual_stats['normality'] = shapiro_p > 0.05
                except:
                    pass
            return residual_stats
        except Exception as e:
            return {}

    def save_model(self, filepath):
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'ensemble_weights': self.ensemble_weights,
                'model_performance': self.model_performance,
                'best_model': self.best_model,
                'config_hash': self.config.get_xml_hash(),
                'is_enriched': self.is_enriched,
                'cv_results': self.cv_results,
                'feature_importance': self.feature_importance,
                'fallback_model': self.fallback_model
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            logger.error(f"Save model error: {e}")
            return False

    def load_model(self, filepath):
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.ensemble_weights = model_data['ensemble_weights']
            self.model_performance = model_data['model_performance']
            self.best_model = model_data['best_model']
            self.is_enriched = model_data.get('is_enriched', False)
            self.is_trained = True
            self.cv_results = model_data.get('cv_results', {})
            self.feature_importance = model_data.get('feature_importance', {})
            self.fallback_model = model_data.get('fallback_model', None)
            return True
        except Exception as e:
            logger.error(f"Load model error: {e}")
            return False


@predict_ml_bp.route('/api/prediction_history', methods=['GET'])
@error_handler
def get_prediction_history():
    load_prediction_history()
    return jsonify({'success': True, 'history': PREDICTION_HISTORY, 'count': len(PREDICTION_HISTORY)})


@predict_ml_bp.route('/api/clear_history', methods=['POST'])
@error_handler
def clear_prediction_history():
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = []
    if os.path.exists(PREDICTION_HISTORY_FILE):
        os.remove(PREDICTION_HISTORY_FILE)
    return jsonify({'success': True, 'message': 'Prediction history cleared'})


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
    return jsonify({'success': True, 'files': files, 'count': len(files)})


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
    return jsonify({'success': True, 'message': f'File uploaded: {filename}', 'filename': filename, 'size': format_size(size)})


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
    try:
        test_df = pd.read_csv(filepath, nrows=5)
        if not is_enriched_dataset(test_df):
            return jsonify({'success': False, 'message': 'Dataset is not enriched. Please use dataset_routes.py first.', 'is_enriched': False}), 400
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'}), 400
    CONFIG = ConfigManager('config/info.xml')
    PREDICTOR = SuzukiPredictor(CONFIG)
    try:
        df = PREDICTOR.load_data(filepath)
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e), 'required_columns': REQUIRED_COLUMNS}), 400
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
            'valid_count': int(df['yield'].count())
        },
        'feature_count': len(PREDICTOR.feature_columns),
        'is_enriched': PREDICTOR.is_enriched
    }
    result = PREDICTOR.train('Ensemble')
    images = []
    if result['success']:
        images = create_result_images()
    if result['success']:
        importance = PREDICTOR.get_feature_importance()
        residuals = PREDICTOR.analyze_residuals()
        return jsonify({
            'success': True,
            'message': f"Loaded {len(df)} enriched rows, {result.get('model_count', 0)} models trained",
            'data_info': convert_to_serializable(DATA_INFO),
            'performance': convert_to_serializable(result.get('performance', {})),
            'best_model': result.get('best_model', 'None'),
            'feature_importance': convert_to_serializable(importance.get('top_10', [])),
            'residual_stats': convert_to_serializable(residuals),
            'cv_results': convert_to_serializable(result.get('cv_results', {})),
            'visualizations': {'created': len(images) > 0, 'image_count': len(images), 'directory': os.path.dirname(images[0]) if images else None},
            'is_enriched': bool(PREDICTOR.is_enriched),
            'fallback_used': result.get('fallback_used', False)
        })
    else:
        return jsonify({'success': False, 'message': result.get('message', 'Training failed')})


@predict_ml_bp.route('/api/update_visualizations', methods=['POST'])
@error_handler
def update_visualizations():
    global PREDICTOR
    if PREDICTOR is None or not PREDICTOR.is_trained:
        return jsonify({'success': False, 'message': 'Model not trained yet'})
    try:
        images = create_result_images()
        return jsonify({'success': True, 'message': f'Created {len(images)} visualizations',
                       'images': [os.path.basename(img) for img in images],
                       'directory': os.path.dirname(images[0]) if images else None})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@predict_ml_bp.route('/api/change_model', methods=['POST'])
@error_handler
@timing_decorator
def change_model():
    global PREDICTOR, CURRENT_MODEL
    data = get_json_body()
    model_name = clean_text(data.get('model_name'), max_length=64)
    if not model_name:
        return jsonify({'success': False, 'message': 'Model name required'})
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    if not PREDICTOR.is_enriched:
        return jsonify({'success': False, 'message': 'Cannot change model on basic dataset.'})
    model_map = {
        'Random Forest': 'Random_Forest',
        'Gradient Boosting': 'Gradient_Boosting',
        'Hist Gradient Boosting': 'Hist_Gradient_Boosting',
        'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM',
        'CatBoost': 'CatBoost',
        'Extra Trees': 'Extra_Trees',
        'Gaussian Process': 'Gaussian_Process',
        'Neural Network': 'Neural_Network',
        'SVR': 'SVR',
        'Ridge': 'Ridge',
        'Lasso': 'Lasso',
        'ElasticNet': 'ElasticNet'
    }
    allowed_keys = set(model_map.values())
    key = model_map.get(model_name, model_name)
    if key not in allowed_keys:
        return jsonify({'success': False, 'message': f'Unknown model: {model_name}'})
    if PREDICTOR.df is not None:
        PREDICTOR._prepare_features()
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
            'cv_results': result.get('cv_results', {}),
            'visualizations': {'created': len(images) > 0, 'image_count': len(images), 'directory': os.path.dirname(images[0]) if images else None},
            'fallback_used': result.get('fallback_used', False)
        })
    else:
        return jsonify({'success': False, 'message': result.get('message', 'Failed to train model')})


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
    return jsonify({'success': success, 'message': f"Model saved to {filename}" if success else "Failed to save model",
                   'filename': filename if success else None})


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
        'visualizations': {'created': len(images) > 0, 'image_count': len(images), 'directory': os.path.dirname(images[0]) if images else None} if success else None
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
            models.append({'name': f, 'size': format_size(os.path.getsize(path)),
                          'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat()})
    models.sort(key=lambda x: x['modified'], reverse=True)
    return jsonify({'success': True, 'models': models})


@predict_ml_bp.route('/api/make_prediction', methods=['POST'])
@error_handler
@timing_decorator
def make_prediction():
    global PREDICTOR
    data = get_json_body()
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    required = ['temp', 'time', 'quantity', 'catalizor', 'base', 'solv1', 'subs1_smiles', 'subs2_smiles']
    for f in required:
        if f not in data or data[f] in (None, ''):
            return jsonify({'success': False, 'message': f'Missing: {f}'})
    if len(str(data.get('subs1_smiles', ''))) > MAX_SMILES_LENGTH or len(str(data.get('subs2_smiles', ''))) > MAX_SMILES_LENGTH:
        return jsonify({'success': False, 'message': 'SMILES input too long'})
    if not data['solv1'] or data['solv1'] == '':
        return jsonify({'success': False, 'message': 'Solvent 1 is required'})
    try:
        temp = to_float(data['temp'], field_name='temp')
        time_h = to_float(data['time'], field_name='time')
        quantity = to_float(data['quantity'], field_name='quantity')
        sigma_m = to_float(data.get('sigma_m', 0), default=0.0, field_name='sigma_m')
        sigma_p = to_float(data.get('sigma_p', 0), default=0.0, field_name='sigma_p')
        taft_es = to_float(data.get('taft_es', 0), default=0.0, field_name='taft_es')
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    experimental_mode = bool(data.get('experimental_mode', False))
    result = PREDICTOR.predict({
        'temp': temp,
        'time': time_h,
        'quantity': quantity,
        'catalizor': clean_text(data['catalizor']),
        'base': clean_text(data['base']),
        'solv1': clean_text(data['solv1']),
        'solv2': clean_text(data.get('solv2', '')),
        'subs1_smiles': clean_text(data['subs1_smiles'], MAX_SMILES_LENGTH),
        'subs2_smiles': clean_text(data['subs2_smiles'], MAX_SMILES_LENGTH),
        'sigma_m': sigma_m,
        'sigma_p': sigma_p,
        'taft_es': taft_es,
        'experimental_mode': experimental_mode
    })
    if not result['success']:
        return jsonify({'success': False, 'message': result.get('message', 'Prediction failed')})
    mol_img = None
    try:
        if RDKIT_AVAILABLE:
            mols = []
            labels = []
            for s, label in [(data.get('subs1_smiles'), 'Boronic Acid'), (data.get('subs2_smiles'), 'Aryl Halide')]:
                if s:
                    m = Chem.MolFromSmiles(s)
                    if m:
                        mols.append(m)
                        labels.append(label)
            if mols:
                img = Draw.MolsToGridImage(mols, molsPerRow=min(2, len(mols)), subImgSize=(250, 250), legends=labels)
                buff = io.BytesIO()
                img.save(buff, format="PNG")
                mol_img = base64.b64encode(buff.getvalue()).decode()
    except Exception as e:
        pass
    return jsonify({
        'success': True,
        'prediction': result['prediction'],
        'prediction_display': result.get('prediction_display', f"~ {result['prediction']:.4f} % (est.)"),
        'ml_prediction': result.get('ml_prediction'),
        'chemical_prediction': result.get('chemical_prediction'),
        'model': result['model'],
        'yield_class': result.get('yield_class', 'Unknown'),
        'yield_class_color': result.get('yield_class_color', '#6B7280'),
        'confidence': result.get('confidence', 0.85),
        'confidence_interval': result.get('confidence_interval'),
        'prediction_interval': result.get('prediction_interval'),
        'best_model': result.get('best_model', 'None'),
        'model_count': result.get('model_count', 0),
        'is_enriched': result.get('is_enriched', False),
        'academic_details': convert_to_serializable(result.get('academic_details', {})),
        'molecule_image': mol_img,
        'fallback_used': result.get('fallback_used', False),
        'solvent_status': result.get('solvent_status', 'single'),
        'history_count': result.get('history_count', 0),
        'experimental_mode': experimental_mode,
        'experimental_details': result.get('experimental_details', None)
    })


@predict_ml_bp.route('/api/optimize_catalyst', methods=['POST'])
@error_handler
@timing_decorator
def optimize_catalyst():
    global PREDICTOR
    data = get_json_body()
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    required = ['temp', 'time', 'quantity', 'base', 'solv1', 'subs1_smiles', 'subs2_smiles']
    for f in required:
        if f not in data or data[f] in (None, ''):
            return jsonify({'success': False, 'message': f'Missing: {f}'})
    if len(str(data.get('subs1_smiles', ''))) > MAX_SMILES_LENGTH or len(str(data.get('subs2_smiles', ''))) > MAX_SMILES_LENGTH:
        return jsonify({'success': False, 'message': 'SMILES input too long'})
    if not data['solv1'] or data['solv1'] == '':
        return jsonify({'success': False, 'message': 'Solvent 1 is required'})
    try:
        temp = to_float(data['temp'], field_name='temp')
        time_h = to_float(data['time'], field_name='time')
        quantity = to_float(data['quantity'], field_name='quantity')
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    results = PREDICTOR.optimize_catalyst({
        'temp': temp,
        'time': time_h,
        'quantity': quantity,
        'base': clean_text(data['base']),
        'solv1': clean_text(data['solv1']),
        'solv2': clean_text(data.get('solv2', '')),
        'subs1_smiles': clean_text(data['subs1_smiles'], MAX_SMILES_LENGTH),
        'subs2_smiles': clean_text(data['subs2_smiles'], MAX_SMILES_LENGTH)
    })
    if not results:
        return jsonify({'success': False, 'message': 'Optimization failed'})
    return jsonify({'success': True, 'results': results, 'model': 'Ensemble'})


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
            'yield_valid_count': DATA_INFO['yield_stats']['valid_count'] if DATA_INFO and 'yield_stats' in DATA_INFO else 0,
            'best_model': best_name,
            'best_r2': best_stats.get('r2', 0) if best_stats else 0,
            'model_count': len(PREDICTOR.models),
            'feature_count': len(PREDICTOR.feature_columns),
            'is_trained': PREDICTOR.is_trained,
            'is_enriched': PREDICTOR.is_enriched,
            'performances': perf,
            'residuals': residuals,
            'cv_results': PREDICTOR.cv_results,
            'fallback_used': PREDICTOR.fallback_model is not None
        }
    })


@predict_ml_bp.route('/api/feature_importance', methods=['GET'])
@error_handler
def feature_importance():
    global PREDICTOR
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    importance = PREDICTOR.get_feature_importance()
    return jsonify({'success': True, 'feature_importance': importance.get('top_10', []), 'all_features': importance.get('all', {})})


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
        'fallback_used': PREDICTOR.fallback_model is not None if PREDICTOR else False,
        'history_count': len(PREDICTION_HISTORY)
    })


def create_result_images():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        if PREDICTOR is None or not PREDICTOR.is_trained or PREDICTOR.X is None:
            return []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        images_dir = os.path.join('static/images', timestamp)
        os.makedirs(images_dir, exist_ok=True)
        X_scaled = PREDICTOR.scaler.transform(PREDICTOR.X)
        y_true = PREDICTOR.y
        all_predictions = {}
        for name, model in PREDICTOR.models.items():
            try:
                all_predictions[name] = model.predict(X_scaled)
            except:
                pass
        if len(all_predictions) > 1:
            ensemble_pred = PREDICTOR._ensemble_predict(X_scaled)
        else:
            ensemble_pred = list(all_predictions.values())[0] if all_predictions else None
        if ensemble_pred is None:
            return []
        r2 = r2_score(y_true, ensemble_pred)
        mae = mean_absolute_error(y_true, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        ax1 = axes[0, 0]
        ax1.scatter(y_true, ensemble_pred, alpha=0.6, s=50, color='#2563EB')
        min_val = min(y_true.min(), ensemble_pred.min())
        max_val = max(y_true.max(), ensemble_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Actual Yield (%)')
        ax1.set_ylabel('Predicted Yield (%)')
        ax1.set_title(f'Parity Plot\nR² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}')
        ax1.grid(True, alpha=0.3)
        ax2 = axes[0, 1]
        residuals = y_true - ensemble_pred
        ax2.hist(residuals, bins=20, color='#8B5CF6', edgecolor='white', alpha=0.7, density=True)
        mu, std = np.mean(residuals), np.std(residuals)
        x_normal = np.linspace(residuals.min(), residuals.max(), 100)
        y_normal = (1/(std * np.sqrt(2*np.pi))) * np.exp(-(x_normal - mu)**2 / (2*std**2))
        ax2.plot(x_normal, y_normal, 'r-', linewidth=2)
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Residual Distribution\nMean = {mu:.4f}, Std = {std:.4f}')
        ax2.grid(True, alpha=0.3)
        ax3 = axes[1, 0]
        if all_predictions:
            model_names = []
            r2_scores = []
            mae_scores = []
            rmse_scores = []
            for name, pred in all_predictions.items():
                if len(pred) == len(y_true):
                    model_names.append(name.replace('_', ' '))
                    r2_scores.append(r2_score(y_true, pred))
                    mae_scores.append(mean_absolute_error(y_true, pred))
                    rmse_scores.append(np.sqrt(mean_squared_error(y_true, pred)))
            if model_names:
                x = np.arange(len(model_names))
                width = 0.25
                ax3.bar(x - width, r2_scores, width, label='R² Score', color='#2563EB', alpha=0.8)
                ax3.bar(x, mae_scores, width, label='MAE', color='#10B981', alpha=0.8)
                ax3.bar(x + width, rmse_scores, width, label='RMSE', color='#F59E0B', alpha=0.8)
                ax3.set_xlabel('Models')
                ax3.set_ylabel('Score / Error')
                ax3.set_title('Model Performance Comparison')
                ax3.set_xticks(x)
                ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        ax4 = axes[1, 1]
        if PREDICTOR.feature_importance and PREDICTOR.feature_importance.get('top_10'):
            importance_data = PREDICTOR.feature_importance['top_10']
            if importance_data:
                features = [item[0][:20] + '...' if len(item[0]) > 20 else item[0] for item in importance_data]
                importance = [item[1]['importance'] for item in importance_data]
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))[::-1]
                ax4.barh(features, importance, color=colors, edgecolor='white', linewidth=1.5)
                ax4.set_xlabel('Importance')
                ax4.set_title('Top 10 Feature Importance')
                ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_files = []
        for idx, (name, axes_group) in enumerate([('parity', axes[0,0]), ('residuals', axes[0,1]),
                                                   ('comparison', axes[1,0]), ('importance', axes[1,1])]):
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            for child in axes_group.get_children():
                if hasattr(child, 'get_data'):
                    try:
                        if isinstance(child, plt.Line2D):
                            x_data, y_data = child.get_data()
                            ax2.plot(x_data, y_data, color=child.get_color(),
                                    linewidth=child.get_linewidth(),
                                    linestyle=child.get_linestyle(),
                                    label=child.get_label())
                    except:
                        pass
                if isinstance(child, plt.Rectangle):
                    rect = plt.Rectangle((child.get_x(), child.get_y()),
                        child.get_width(), child.get_height(),
                        facecolor=child.get_facecolor(),
                        edgecolor='white', linewidth=1.5)
                    ax2.add_patch(rect)
                if isinstance(child, plt.Polygon):
                    if hasattr(child, 'get_xy'):
                        xy = child.get_xy()
                        if len(xy) > 0:
                            polygon = plt.Polygon(xy, facecolor=child.get_facecolor(),
                                                edgecolor=child.get_edgecolor(),
                                                linewidth=child.get_linewidth())
                            ax2.add_patch(polygon)
                if isinstance(child, plt.Text):
                    ax2.text(child.get_position()[0], child.get_position()[1],
                           child.get_text(), fontsize=child.get_fontsize())
            ax2.set_title(axes_group.get_title(), fontsize=14)
            ax2.set_xlabel(axes_group.get_xlabel(), fontsize=12)
            ax2.set_ylabel(axes_group.get_ylabel(), fontsize=12)
            ax2.grid(True, alpha=0.3)
            if axes_group.get_legend():
                ax2.legend(loc='best')
            filepath = os.path.join(images_dir, f'{name}_plot.png')
            fig2.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig2)
            plot_files.append(filepath)
        plt.close('all')
        return plot_files
    except Exception as e:
        logger.error(f"Error creating images: {str(e)}")
        return []


def init_app():
    os.makedirs('static/datasets', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('static/models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    if not os.path.exists('config/info.xml'):
        ConfigManager('config/info.xml')
    load_prediction_history()


init_app()
