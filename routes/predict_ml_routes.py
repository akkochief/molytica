from flask import Blueprint, render_template, request, jsonify, send_from_directory
import os
import json
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
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
import threading
import logging
from logging.handlers import RotatingFileHandler
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.stats import shapiro, pearsonr, skew, kurtosis
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, RepeatedKFold, 
    LeaveOneOut, learning_curve, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, 
    mean_absolute_percentage_error, explained_variance_score, 
    max_error, mean_squared_log_error
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    VarianceThreshold, RFE, SelectFromModel
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    ExtraTreesRegressor, HistGradientBoostingRegressor,
    StackingRegressor, VotingRegressor
)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Draw, AllChem
    from rdkit.Chem.Draw import IPythonConsole
    from rdkit.Chem import PandasTools
    RDKIT_AVAILABLE = True
    RDKIT_3D_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    RDKIT_3D_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    PERM_IMP_AVAILABLE = True
except ImportError:
    PERM_IMP_AVAILABLE = False

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

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from mapie.regression import MapieRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False

try:
    from xtb.interface import Calculator
    XTB_AVAILABLE = True
except ImportError:
    XTB_AVAILABLE = False

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONWARNINGS'] = 'ignore'

predict_ml_bp = Blueprint('predict_ml', __name__, url_prefix='/predict_ml')

PREDICTION_HISTORY_FILE = 'prediction_history.json'
MODEL_REGISTRY_FILE = 'model_registry.json'
PREDICTION_HISTORY = []
MODEL_REGISTRY = {}
PREDICTOR = None
CONFIG = None
DATA_INFO = None
CURRENT_FILE = None
CURRENT_MODEL = 'Ensemble'
MODEL_PERFORMANCE = {}
CACHE = {}
FEATURE_IMPORTANCE = {}
CACHE_HIT = 0
CACHE_MISS = 0
MAX_SMILES_LENGTH = 500
MAX_TEXT_FIELD_LENGTH = 500
MAX_FILE_SIZE = 50 * 1024 * 1024
UPLOAD_FOLDER = 'static/datasets'
MODEL_FOLDER = 'static/models'
LOG_FOLDER = 'logs'
CONFIG_FOLDER = 'config'

for folder in [UPLOAD_FOLDER, MODEL_FOLDER, LOG_FOLDER, CONFIG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

logger = logging.getLogger('MolyticaPredictor')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(
        os.path.join(LOG_FOLDER, 'predict_ml.log'),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

REQUIRED_COLUMNS = ['yield', 'temp', 'time', 'quantity', 'catalizor', 'base', 'solv1']
OPTIONAL_COLUMNS = ['solv2', 'subs1', 'subs2', 'product']
NULLABLE_COLUMNS = ['solv1', 'solv2']
FAILURE_COLUMN = 'yield'
CRITICAL_NON_NULLABLE_COLUMNS = ['temp', 'time', 'quantity', 'catalizor', 'base']

ACADEMIC_FEATURE_COLUMNS = [
    'subs1_SMILES_logp', 'subs1_SMILES_sigma_p', 'subs1_SMILES_sigma_m',
    'subs1_SMILES_taft_es', 'subs1_SMILES_hba', 'subs1_SMILES_hbd',
    'subs1_SMILES_complexity', 'subs1_SMILES_kappa1',
    'subs2_SMILES_logp', 'subs2_SMILES_sigma_p', 'subs2_SMILES_sigma_m',
    'subs2_SMILES_taft_es', 'subs2_SMILES_hba', 'subs2_SMILES_hbd',
    'subs2_SMILES_complexity', 'subs2_SMILES_kappa1',
]

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

GROUP_SMARTS = {
    'CH3': '[CH3;!$(C=O)]',
    'OCH3': '[OX2][CH3]',
    'NO2': '[NX3](=O)=O',
    'Cl': '[Cl]',
    'F': '[F]',
    'CN': '[C]#[N]',
    'CF3': '[C](F)(F)F',
    'COOH': 'C(=O)[OH]',
    'COOCH3': 'C(=O)[O][CH3]',
    'CHO': '[CH1](=O)',
    'NH2': '[NX3;H2;!$(N-C=O)]',
    'N(CH3)2': '[NX3]([CH3])([CH3])',
    'SO2CH3': 'S(=O)(=O)[CH3]',
    'B(OH)2': '[B]([OH])([OH])',
    'Si(CH3)3': '[Si]([CH3])([CH3])([CH3])',
    'C(CH3)3': '[C]([CH3])([CH3])([CH3])',
    'C6H5': '[c]1[c][c][c][c][c]1',
    'COCH3': 'C(=O)[CH3]',
    'OH': '[OX2H]',
    'I': '[I]',
    'Br': '[Br]',
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

SOLVENT_PHYSICS = {
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
    'ethyl_acetate': {'dielectric': 6.0, 'donor_number': 14.0, 'polarity_index': 4.4, 'alpha': 0.00, 'beta': 0.45, 'pi_star': 0.55, 'reichardt_et30': 38.1, 'hildebrand_delta': 18.2},
    'diethyl_ether': {'dielectric': 4.3, 'donor_number': 19.2, 'polarity_index': 2.8, 'alpha': 0.00, 'beta': 0.47, 'pi_star': 0.27, 'reichardt_et30': 34.6, 'hildebrand_delta': 15.4},
    'pyridine': {'dielectric': 12.3, 'donor_number': 33.1, 'polarity_index': 5.3, 'alpha': 0.00, 'beta': 0.64, 'pi_star': 0.87, 'reichardt_et30': 40.2, 'hildebrand_delta': 21.8},
    'nmp': {'dielectric': 32.2, 'donor_number': 27.3, 'polarity_index': 6.7, 'alpha': 0.00, 'beta': 0.77, 'pi_star': 0.92, 'reichardt_et30': 42.0, 'hildebrand_delta': 23.1},
    'dme': {'dielectric': 7.2, 'donor_number': 19.5, 'polarity_index': 3.5, 'alpha': 0.00, 'beta': 0.53, 'pi_star': 0.53, 'reichardt_et30': 36.5, 'hildebrand_delta': 17.6},
}

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
            },
            'experimental_mode': prediction_data.get('experimental_mode', False)
        }
        PREDICTION_HISTORY.append(entry)
        if len(PREDICTION_HISTORY) > 1000:
            PREDICTION_HISTORY = PREDICTION_HISTORY[-1000:]
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(PREDICTION_HISTORY, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save prediction history: {str(e)}")

def load_prediction_history():
    global PREDICTION_HISTORY
    try:
        if os.path.exists(PREDICTION_HISTORY_FILE):
            with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                PREDICTION_HISTORY = json.load(f)
        else:
            PREDICTION_HISTORY = []
    except Exception as e:
        logger.error(f"Failed to load prediction history: {str(e)}")
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
    else:
        return str(obj)

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

def format_size(bytes_val):
    if bytes_val == 0:
        return '0 B'
    k = 1024
    sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while bytes_val >= k and i < len(sizes) - 1:
        bytes_val /= k
        i += 1
    return f"{bytes_val:.1f} {sizes[i]}"

def secure_filename_custom(filename):
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

def clean_text(value, max_length=MAX_TEXT_FIELD_LENGTH):
    if value is None:
        return ''
    text = str(value).strip()
    return text[:max_length]

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

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"Function {func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, KeyError, ValueError) as e:
            logger.warning(f"Validation error in {func.__name__}: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 400
        except RequestEntityTooLarge as e:
            logger.warning(f"File too large: {str(e)}")
            return jsonify({'success': False, 'message': 'File too large'}), 413
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
            debug_on = os.getenv('DEBUG', '').lower() in ('1', 'true', 'yes')
            return jsonify({
                'success': False,
                'message': str(e) if debug_on else 'An internal error occurred',
                'traceback': traceback.format_exc() if debug_on else None
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
        self.max_smiles_length = self.get_int('security/sanitization/max_smiles_length', 500)

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
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            self._create_default()

    def _create_default(self):
        default_xml = """<?xml version="1.0" encoding="UTF-8"?>
<suzuki_config>
    <metadata>
        <last_updated>2026-09-07</last_updated>
        <description>Academic Suzuki ML predictor</description>
    </metadata>
    <data_integrity>
        <critical_non_nullable_columns>temp,time,quantity,catalizor,base</critical_non_nullable_columns>
        <minimum_usable_rows>5</minimum_usable_rows>
        <nullable_columns>solv1,solv2</nullable_columns>
        <failure_column>yield</failure_column>
    </data_integrity>
    <ml_training_safeguards>
        <min_samples_per_feature>10</min_samples_per_feature>
        <min_samples_for_full_ensemble>30</min_samples_for_full_ensemble>
        <bootstrap_iterations>1000</bootstrap_iterations>
        <confidence_level>0.95</confidence_level>
    </ml_training_safeguards>
    <model_parameters>
        <Random_Forest>
            <n_estimators>300</n_estimators>
            <max_depth>15</max_depth>
            <random_state>42</random_state>
        </Random_Forest>
        <Gradient_Boosting>
            <n_estimators>350</n_estimators>
            <max_depth>7</max_depth>
            <learning_rate>0.07</learning_rate>
            <random_state>42</random_state>
        </Gradient_Boosting>
        <XGBoost>
            <n_estimators>350</n_estimators>
            <max_depth>7</max_depth>
            <learning_rate>0.08</learning_rate>
            <random_state>42</random_state>
        </XGBoost>
        <LightGBM>
            <n_estimators>400</n_estimators>
            <max_depth>10</max_depth>
            <num_leaves>31</num_leaves>
            <learning_rate>0.06</learning_rate>
            <random_state>42</random_state>
        </LightGBM>
        <CatBoost>
            <iterations>400</iterations>
            <depth>7</depth>
            <learning_rate>0.07</learning_rate>
            <random_seed>42</random_seed>
        </CatBoost>
        <Ridge>
            <alpha>1.0</alpha>
            <random_state>42</random_state>
        </Ridge>
        <Lasso>
            <alpha>1.0</alpha>
            <random_state>42</random_state>
        </Lasso>
        <Ensemble>
            <weights>
                <Random_Forest>0.22</Random_Forest>
                <Gradient_Boosting>0.18</Gradient_Boosting>
                <XGBoost>0.14</XGBoost>
                <LightGBM>0.14</LightGBM>
                <CatBoost>0.12</CatBoost>
                <Ridge>0.10</Ridge>
                <Lasso>0.10</Lasso>
            </weights>
        </Ensemble>
    </model_parameters>
    <feature_importance>
        <temperature>0.16</temperature>
        <time>0.13</time>
        <catalyst_quantity>0.11</catalyst_quantity>
        <electronegativity>0.03</electronegativity>
        <flexibility>0.03</flexibility>
        <molecular_volume>0.02</molecular_volume>
    </feature_importance>
    <data_processing>
        <normalization>
            <numeric_method>standard_scaler</numeric_method>
        </normalization>
        <split>
            <test_size>0.20</test_size>
            <random_state>42</random_state>
        </split>
        <outlier_detection>
            <mahalanobis_threshold>3.5</mahalanobis_threshold>
        </outlier_detection>
    </data_processing>
    <performance_metrics>
        <metrics>
            <r2>true</r2>
            <mae>true</mae>
            <rmse>true</rmse>
            <aic>true</aic>
            <bic>true</bic>
            <shapiro_wilk>true</shapiro_wilk>
        </metrics>
        <uncertainty>
            <n_bootstrap>1000</n_bootstrap>
            <confidence_level>0.95</confidence_level>
        </uncertainty>
        <learning_curve>
            <enabled>true</enabled>
            <cv_folds>5</cv_folds>
        </learning_curve>
    </performance_metrics>
    <security>
        <sanitization>
            <max_smiles_length>500</max_smiles_length>
        </sanitization>
    </security>
</suzuki_config>"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            f.write(default_xml)
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
                else:
                    try:
                        if '.' in val or 'e' in val.lower():
                            val = float(val)
                        else:
                            val = int(val)
                    except:
                        pass
                result[child.tag] = val
        return result

    def _validate_config(self):
        required = ['data_integrity/critical_non_nullable_columns']
        for path in required:
            if self.get(path) is None:
                logger.warning(f"Required parameter missing: {path}")

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

    def get_feature_importance(self):
        return self.get_dict('feature_importance')

    def reload(self):
        try:
            self._cache.clear()
            self._load_or_create()
            self._parse_all()
            self._validate_config()
            self._compute_xml_hash()
            return True
        except Exception as e:
            logger.error(f"Failed to reload config: {str(e)}")
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

class DFTEngine:
    def __init__(self):
        self.xtb_available = XTB_AVAILABLE
        self.rdkit_3d = RDKIT_3D_AVAILABLE

    def calculate_homo_lumo(self, smiles):
        if not RDKIT_AVAILABLE:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            if self.rdkit_3d:
                mol3d = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol3d, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol3d)
                if self.xtb_available:
                    try:
                        calc = Calculator(method="GFN2-xTB", charge=0, mult=1)
                        calc.singlepoint(mol3d)
                        return {
                            'homo_energy': float(calc.get_homo()),
                            'lumo_energy': float(calc.get_lumo()),
                            'gap_energy': float(calc.get_gap()),
                            'method': 'GFN2-xTB',
                            'is_dft': True,
                            'conformer_generated': True
                        }
                    except Exception as e:
                        logger.error(f"xtb calculation failed: {str(e)}")
            return {
                'homo_energy': None,
                'lumo_energy': None,
                'gap_energy': None,
                'method': 'DFT not available',
                'is_dft': False,
                'conformer_generated': False
            }
        except Exception as e:
            logger.error(f"DFT calculation failed: {str(e)}")
            return None

class SHAPAnalyzer:
    def __init__(self):
        self.shap_available = SHAP_AVAILABLE

    def analyze(self, model, X, feature_names):
        if not self.shap_available:
            return None
        try:
            if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor,
                                  XGBRegressor, LGBMRegressor, CatBoostRegressor)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                return {
                    'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                    'base_value': float(explainer.expected_value) if hasattr(explainer.expected_value, '__float__') else explainer.expected_value,
                    'feature_names': feature_names,
                    'method': 'SHAP TreeExplainer'
                }
            else:
                explainer = shap.KernelExplainer(model.predict, X[:100])
                shap_values = explainer.shap_values(X[:100])
                return {
                    'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                    'base_value': float(explainer.expected_value),
                    'feature_names': feature_names,
                    'method': 'SHAP KernelExplainer'
                }
        except Exception as e:
            logger.error(f"SHAP analysis failed: {str(e)}")
            return None

class ConformalPredictor:
    def __init__(self):
        self.mapie_available = MAPIE_AVAILABLE

    def predict(self, model, X_train, y_train, X_test, alpha=0.05):
        if not self.mapie_available:
            return None
        try:
            mapie = MapieRegressor(model, method="plus", cv=5)
            mapie.fit(X_train, y_train)
            y_pred, y_std = mapie.predict(X_test, alpha=alpha)
            return {
                'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
                'std': y_std.tolist() if hasattr(y_std, 'tolist') else y_std,
                'alpha': alpha,
                'method': 'Conformal Prediction (MAPIE)'
            }
        except Exception as e:
            logger.error(f"Conformal prediction failed: {str(e)}")
            return None

class HyperparameterOptimizer:
    def __init__(self):
        self.grid_search_available = True

    def optimize(self, model_type, X, y):
        param_grids = {
            'Random_Forest': {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [15, 31, 63],
                'max_depth': [3, 5, 7, 10]
            },
            'Gradient_Boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 3, 5]
            },
            'Ridge': {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
            'Lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
            'ElasticNet': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
            'SVR': {'C': [0.1, 1.0, 10.0, 100.0], 'epsilon': [0.01, 0.05, 0.1, 0.2], 'gamma': ['scale', 'auto']},
            'Neural_Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01]
            }
        }
        if model_type not in param_grids:
            return None
        try:
            model_creators = {
                'Random_Forest': RandomForestRegressor,
                'XGBoost': XGBRegressor,
                'LightGBM': LGBMRegressor,
                'Gradient_Boosting': GradientBoostingRegressor,
                'Ridge': Ridge,
                'Lasso': Lasso,
                'ElasticNet': ElasticNet,
                'SVR': SVR,
                'Neural_Network': MLPRegressor
            }
            if model_type not in model_creators:
                return None
            base_model = model_creators[model_type]()
            search = RandomizedSearchCV(
                base_model, param_grids[model_type],
                n_iter=20, cv=3, scoring='r2',
                random_state=42, n_jobs=1
            )
            search.fit(X, y)
            return {
                'best_params': search.best_params_,
                'best_score': float(search.best_score_),
                'method': 'RandomizedSearchCV',
                'n_iter': 20
            }
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            return None

class ValidationReporter:
    def __init__(self):
        pass

    def generate_report(self, y_true, y_pred, model, X_train, y_train, X_test):
        try:
            report = {
                'test_metrics': {
                    'r2': float(r2_score(y_true, y_pred)),
                    'mae': float(mean_absolute_error(y_true, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    'mape': float(mean_absolute_percentage_error(y_true, y_pred) * 100),
                    'max_error': float(max_error(y_true, y_pred)),
                    'explained_variance': float(explained_variance_score(y_true, y_pred))
                }
            }
            if len(y_true) >= 3:
                try:
                    corr, p_val = pearsonr(y_true, y_pred)
                    report['test_metrics']['pearson_correlation'] = float(corr)
                    report['test_metrics']['pearson_p_value'] = float(p_val)
                except:
                    pass
            if len(y_true) >= 3 and len(y_true) <= 5000:
                try:
                    residuals = np.array(y_true) - np.array(y_pred)
                    shapiro_stat, shapiro_p = shapiro(residuals)
                    report['residual_normality'] = {
                        'shapiro_stat': float(shapiro_stat),
                        'shapiro_p': float(shapiro_p),
                        'is_normal': bool(shapiro_p > 0.05)
                    }
                except:
                    pass
            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X_train, y_train, cv=3,
                    train_sizes=np.linspace(0.1, 1.0, 5),
                    scoring='r2', n_jobs=1
                )
                report['learning_curve'] = {
                    'train_sizes': train_sizes.tolist(),
                    'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                    'test_scores_mean': np.mean(test_scores, axis=1).tolist()
                }
            except:
                pass
            return report
        except Exception as e:
            logger.error(f"Validation report generation failed: {str(e)}")
            return None

class AdvancedFeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.selected_features = []
        self.anomaly_results = {}
        self.EN = {
            'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
            'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
            'B': 2.04, 'Si': 1.90, 'Pd': 2.20, 'Pt': 2.28, 'Li': 0.98,
            'Na': 0.93, 'K': 0.82, 'Cs': 0.79, 'Mg': 1.31, 'Ca': 1.00,
            'Zn': 1.65, 'Cu': 1.90, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91
        }

    def extract_electronegativity_features(self, smiles):
        if not RDKIT_AVAILABLE or not smiles:
            return {}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            en_values = [self.EN.get(atom.GetSymbol(), 2.5) for atom in mol.GetAtoms()]
            if not en_values:
                return {}
            return {
                'avg_electronegativity': float(np.mean(en_values)),
                'max_electronegativity': float(np.max(en_values)),
                'min_electronegativity': float(np.min(en_values)),
                'electronegativity_range': float(np.max(en_values) - np.min(en_values)),
                'electronegativity_std': float(np.std(en_values))
            }
        except:
            return {}

    def extract_flexibility_features(self, smiles):
        if not RDKIT_AVAILABLE or not smiles:
            return {}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            rot_bonds = Lipinski.NumRotatableBonds(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            if heavy_atoms == 0:
                return {}
            return {
                'flexibility_ratio': float(rot_bonds / (heavy_atoms + 1)),
                'conformational_entropy': float(rot_bonds * 0.5 * np.log(heavy_atoms + 1)),
                'rot_bonds_per_heavy_atom': float(rot_bonds / (heavy_atoms + 1)),
                'heavy_atom_rot_bond_ratio': float(heavy_atoms / (rot_bonds + 1))
            }
        except:
            return {}

    def extract_tpsa_distribution(self, smiles):
        if not RDKIT_AVAILABLE or not smiles:
            return {}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            total_tpsa = Descriptors.TPSA(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            n_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'N')
            o_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O')
            s_count = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'S')
            n_contribution = n_count * 12.0
            o_contribution = o_count * 20.0
            s_contribution = s_count * 5.0
            return {
                'n_tpsa_contribution': float(n_contribution),
                'o_tpsa_contribution': float(o_contribution),
                's_tpsa_contribution': float(s_contribution),
                'tpsa_per_heavy_atom': float(total_tpsa / (heavy_atoms + 1)),
                'tpsa_hetero_ratio': float(total_tpsa / ((n_count + o_count + s_count) + 1))
            }
        except:
            return {}

    def extract_molecular_volume(self, smiles):
        if not RDKIT_AVAILABLE or not smiles:
            return {}
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            mw = Descriptors.ExactMolWt(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            volume = mw / 1.2
            kappa1 = Descriptors.Kappa1(mol)
            spherocity = min(1.0, heavy_atoms / (kappa1 + 1) if kappa1 > 0 else 0.5)
            return {
                'molecular_volume': float(volume),
                'spherocity': float(spherocity),
                'volume_per_atom': float(volume / (heavy_atoms + 1)),
                'density_estimate': float(mw / (volume + 1))
            }
        except:
            return {}

    def extract_all_features(self, smiles):
        features = {}
        features.update(self.extract_electronegativity_features(smiles))
        features.update(self.extract_flexibility_features(smiles))
        features.update(self.extract_tpsa_distribution(smiles))
        features.update(self.extract_molecular_volume(smiles))
        return features

    def detect_anomalies(self, X, threshold=3.5):
        try:
            X = np.array(X)
            if X.shape[0] < 3 or X.shape[1] < 2:
                return {'anomaly_indices': [], 'anomaly_scores': [], 'anomaly_ratio': 0.0}
            mean = np.mean(X, axis=0)
            cov = np.cov(X.T)
            inv_cov = np.linalg.pinv(cov)
            mahalanobis_distances = []
            for x in X:
                d = mahalanobis(x, mean, inv_cov)
                mahalanobis_distances.append(float(d))
            mahalanobis_distances = np.array(mahalanobis_distances)
            std = np.std(mahalanobis_distances) if np.std(mahalanobis_distances) > 0 else 1
            z_scores = (mahalanobis_distances - np.mean(mahalanobis_distances)) / std
            anomaly_indices = np.where(np.abs(z_scores) > threshold)[0].tolist()
            self.anomaly_results = {
                'anomaly_indices': anomaly_indices,
                'anomaly_scores': mahalanobis_distances.tolist(),
                'anomaly_ratio': float(len(anomaly_indices) / len(X) if len(X) > 0 else 0),
                'threshold': threshold
            }
            return self.anomaly_results
        except:
            return {'anomaly_indices': [], 'anomaly_scores': [], 'anomaly_ratio': 0.0}

    def correlation_analysis(self, df, target='yield', threshold=0.2):
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in numeric_cols:
                numeric_cols.remove(target)
            if not numeric_cols:
                return {'selected_features': [], 'correlations': {}}
            correlations = {}
            for col in numeric_cols:
                try:
                    corr = df[col].corr(df[target])
                    if not np.isnan(corr) and abs(corr) > threshold:
                        correlations[col] = float(abs(corr))
                except:
                    continue
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            return {
                'selected_features': [f[0] for f in sorted_features],
                'correlations': dict(sorted_features),
                'feature_count': len(sorted_features),
                'top_features': sorted_features[:10]
            }
        except:
            return {'selected_features': [], 'correlations': {}}

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.selected_features = []
        self.advanced = AdvancedFeatureEngineer(config)

    def extract_smiles_features(self, smiles):
        features = {}
        if not RDKIT_AVAILABLE or not smiles:
            return features
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return features
            features['mw'] = Descriptors.ExactMolWt(mol)
            features['logp'] = Descriptors.MolLogP(mol)
            features['tpsa'] = Descriptors.TPSA(mol)
            features['refractivity'] = Descriptors.MolMR(mol)
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
            features['halogen_count'] = features['f_count'] + features['cl_count'] + features['br_count'] + features['i_count']
            features['hetero_count'] = features['n_count'] + features['o_count'] + features['s_count'] + features['p_count'] + features['halogen_count']
            features['aromatic_bonds'] = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
            features['single_bonds'] = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.SINGLE)
            features['double_bonds'] = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE)
            features['triple_bonds'] = sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE)
            features['total_bonds'] = mol.GetNumBonds()
            features['chiral_centers'] = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
            features['spiro_atoms'] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            features['bridgehead_atoms'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            features['branch_nodes'] = sum(1 for atom in mol.GetAtoms() if atom.GetDegree() > 2)
            features['smiles_length'] = len(smiles)
            features.update(self.advanced.extract_all_features(smiles))
        except Exception as e:
            logger.debug(f"SMILES feature extraction failed for {smiles}: {str(e)}")
        return features

    def engineer_features(self, df):
        df = df.copy()
        if 'temp' in df.columns and 'time' in df.columns:
            df['temp_time_product'] = df['temp'] * df['time']
            df['temp_time_ratio'] = df['temp'] / (df['time'] + 1)
            df['temp_time_interaction'] = df['temp'] * df['time'] / 100
            df['temp_log_time'] = df['temp'] * np.log1p(df['time'])
            df['time_log_temp'] = df['time'] * np.log1p(df['temp'])
        if 'quantity' in df.columns:
            df['quantity_log1p'] = np.log1p(df['quantity'])
            df['quantity_sqrt'] = np.sqrt(df['quantity'])
            df['quantity_squared'] = df['quantity'] ** 2
        if 'temp' in df.columns and 'quantity' in df.columns:
            df['temp_quantity_product'] = df['temp'] * df['quantity']
            df['catalyst_loading'] = df['quantity'] / (df['temp'] + 1)
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
            df['substrate_steric_product'] = df['subs1_length'] * df['subs2_length']
        if 'subs1_logp' in df.columns and 'subs2_logp' in df.columns:
            df['substrate_logp_avg'] = (df['subs1_logp'] + df['subs2_logp']) / 2
            df['substrate_logp_diff'] = abs(df['subs1_logp'] - df['subs2_logp'])
            df['substrate_logp_sum'] = df['subs1_logp'] + df['subs2_logp']
        if 'subs1_mw' in df.columns and 'subs2_mw' in df.columns:
            df['substrate_mw_avg'] = (df['subs1_mw'] + df['subs2_mw']) / 2
            df['substrate_mw_diff'] = abs(df['subs1_mw'] - df['subs2_mw'])
            df['substrate_mw_sum'] = df['subs1_mw'] + df['subs2_mw']
        if 'subs1_rings' in df.columns and 'subs2_rings' in df.columns:
            df['total_rings'] = df['subs1_rings'] + df['subs2_rings']
            df['ring_diff'] = abs(df['subs1_rings'] - df['subs2_rings'])
            df['aromatic_sum'] = df.get('subs1_aromatic_rings', 0) + df.get('subs2_aromatic_rings', 0)
            df['halogen_total'] = df.get('subs1_halogen_count', 0) + df.get('subs2_halogen_count', 0)
        return df

    def select_features(self, df, target='yield'):
        try:
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
            k_best = min(40, len(numeric_cols) // 2)
            k_best = max(5, k_best)
            selected = [f for f, _ in feature_scores[:k_best]]
            self.selected_features = selected
            return df[selected] if selected else df[numeric_cols]
        except Exception:
            return df.select_dtypes(include=[np.number])

class SuzukiPredictor:
    def __init__(self, config):
        self.config = config
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
        self.best_model = None
        self.cv_results = {}
        self.feature_importance = {}
        self.is_enriched = False
        self.fallback_model = None
        self.X_test_ = None
        self.y_test_ = None
        self.anomaly_results = {}
        self.learning_curve_data = {}
        self.bootstrap_results = {}
        self.aic_bic_results = {}
        self.correlation_analysis = {}
        self.uncertainty_results = {}
        self.normality_test = None
        self.shap_analysis = None
        self.conformal_prediction = None
        self.dft_properties = {}
        self.validation_report = None
        self.min_samples_per_feature = self.config.get_int('ml_training_safeguards/min_samples_per_feature', 10)
        self.min_samples_for_full_ensemble = self.config.get_int('ml_training_safeguards/min_samples_for_full_ensemble', 30)
        self.bootstrap_iterations = self.config.get_int('ml_training_safeguards/bootstrap_iterations', 1000)
        self.confidence_level = self.config.get_float('ml_training_safeguards/confidence_level', 0.95)

    def _load_weights(self):
        try:
            w = self.config.get_dict('model_parameters/Ensemble/weights')
            if w:
                return {k: float(v) for k, v in w.items() if float(v) > 0}
        except:
            pass
        return {
            'Random_Forest': 0.22,
            'Gradient_Boosting': 0.18,
            'XGBoost': 0.14,
            'LightGBM': 0.14,
            'CatBoost': 0.12,
            'Ridge': 0.10,
            'Lasso': 0.10
        }

    def validate_csv(self, filepath):
        try:
            df = pd.read_csv(filepath, nrows=1)
            columns = df.columns.tolist()
            missing = [col for col in REQUIRED_COLUMNS if col not in columns]
            if missing:
                return False, missing
            return True, []
        except:
            return False, ['Could not read CSV']

    def load_data(self, filepath):
        valid, missing = self.validate_csv(filepath)
        if not valid:
            raise ValueError(f"CSV validation failed. Missing columns: {', '.join(missing)}")
        try:
            self.df = pd.read_csv(filepath)
            if 'yield' not in self.df.columns:
                raise ValueError("'yield' column not found")
            if self.df['yield'].isnull().all():
                raise ValueError("All yield values are missing")
            self.is_enriched = is_enriched_dataset(self.df)
            if not self.is_enriched:
                raise ValueError("Dataset not enriched. Please use dataset_routes.py first.")
            usable_df, failed_df, rejected_df = classify_and_filter_rows(self.df)
            self.df = usable_df
            if len(self.df) < 5:
                raise ValueError(f"At least 5 valid rows needed. Current: {len(self.df)}")
            self.df = self.fe.engineer_features(self.df)
            self._prepare_features()
            return self.df
        except Exception as e:
            raise

    def _prepare_features(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'yield' in numeric_cols:
            numeric_cols.remove('yield')
        important_features = [
            'temp', 'time', 'quantity',
            'temp_time_product', 'temp_time_ratio', 'temp_time_interaction',
            'temp_log_time', 'time_log_temp',
            'quantity_log1p', 'quantity_sqrt', 'quantity_squared',
            'catalyst_loading', 'temp_quantity_product',
            'subs1_length', 'subs2_length',
            'substrate_steric_sum', 'substrate_steric_diff', 'substrate_steric_product',
            'subs1_logp', 'subs2_logp',
            'substrate_logp_avg', 'substrate_logp_diff', 'substrate_logp_sum',
            'subs1_mw', 'subs2_mw',
            'substrate_mw_avg', 'substrate_mw_diff', 'substrate_mw_sum',
            'subs1_rings', 'subs2_rings',
            'total_rings', 'ring_diff', 'aromatic_sum',
            'subs1_hba', 'subs2_hba',
            'subs1_hbd', 'subs2_hbd',
            'subs1_complexity', 'subs2_complexity',
            'subs1_tpsa', 'subs2_tpsa',
            'halogen_total',
            'subs1_rotatable_bonds', 'subs2_rotatable_bonds',
            'subs1_fraction_csp3', 'subs2_fraction_csp3',
            'subs1_avg_electronegativity', 'subs2_avg_electronegativity',
            'subs1_flexibility_ratio', 'subs2_flexibility_ratio',
            'subs1_tpsa_per_heavy_atom', 'subs2_tpsa_per_heavy_atom',
            'subs1_molecular_volume', 'subs2_molecular_volume',
            'subs1_spherocity', 'subs2_spherocity'
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
        self.X = pd.concat([X_numeric, X_categorical], axis=1) if not X_categorical.empty else X_numeric
        self.y = self.df['yield'].values
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
            elif name == 'XGBoost' and XGB_AVAILABLE:
                return XGBRegressor(**params)
            elif name == 'LightGBM' and LGBM_AVAILABLE:
                return LGBMRegressor(**params)
            elif name == 'CatBoost' and CATBOOST_AVAILABLE:
                return CatBoostRegressor(**params)
            elif name == 'Ridge':
                return Ridge(**params)
            elif name == 'Lasso':
                return Lasso(**params)
            else:
                return None
        except Exception as e:
            logger.error(f"Model creation error for {name}: {str(e)}")
            return None

    def _parse_param(self, val):
        if val is None:
            return None
        if isinstance(val, str):
            if val.lower() in ['true', 'false']:
                return val.lower() == 'true'
            if val.lower() == 'none':
                return None
            try:
                return float(val) if '.' in val else int(val)
            except:
                return val
        return val

    def _perform_bootstrap(self, y_true, y_pred, n_iterations=1000):
        try:
            errors = y_true - y_pred
            n = len(errors)
            bootstrap_means = []
            for _ in range(min(n_iterations, 1000)):
                indices = np.random.choice(n, n, replace=True)
                bootstrap_errors = errors[indices]
                bootstrap_means.append(np.mean(bootstrap_errors))
            bootstrap_means = np.array(bootstrap_means)
            alpha = 1 - self.confidence_level
            lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
            self.bootstrap_results = {
                'mean_error': float(np.mean(errors)),
                'ci_lower': float(lower),
                'ci_upper': float(upper),
                'ci_width': float(upper - lower),
                'n_bootstrap': len(bootstrap_means)
            }
            return self.bootstrap_results
        except:
            return {}

    def _calculate_aic_bic(self, model, X, y):
        try:
            n = len(y)
            k = X.shape[1] if hasattr(X, 'shape') else 1
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            if mse <= 0:
                return {}
            aic = n * np.log(mse) + 2 * k
            bic = n * np.log(mse) + k * np.log(n)
            self.aic_bic_results = {
                'AIC': float(aic),
                'BIC': float(bic),
                'log_likelihood': float(-n/2 * np.log(2 * np.pi * mse) - n/2)
            }
            return self.aic_bic_results
        except:
            return {}

    def _calculate_learning_curve(self, model, X, y, cv=5):
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=min(cv, len(y)//2),
                train_sizes=np.linspace(0.1, 1.0, min(10, len(y))),
                scoring='r2'
            )
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            overfitting_ratio = (train_mean - test_mean) / (train_mean + 1e-6)
            self.learning_curve_data = {
                'train_sizes': train_sizes.tolist(),
                'train_scores_mean': train_mean.tolist(),
                'train_scores_std': train_std.tolist(),
                'test_scores_mean': test_mean.tolist(),
                'test_scores_std': test_std.tolist(),
                'overfitting_ratio': float(np.mean(overfitting_ratio)),
                'is_overfitting': bool(np.mean(overfitting_ratio) > 0.1)
            }
            return self.learning_curve_data
        except:
            return {}

    def _calculate_ensemble_uncertainty(self, X):
        try:
            predictions = []
            for name, model in self.models.items():
                if name in self.ensemble_weights and self.ensemble_weights[name] > 0:
                    try:
                        pred = model.predict(X)
                        if not np.isnan(pred).all():
                            predictions.append(pred)
                    except:
                        continue
            if not predictions:
                return {}
            predictions = np.array(predictions)
            mean = np.mean(predictions, axis=0)
            std = np.std(predictions, axis=0)
            ci_lower = np.percentile(predictions, 2.5, axis=0)
            ci_upper = np.percentile(predictions, 97.5, axis=0)
            self.uncertainty_results = {
                'mean': mean.tolist() if len(mean) > 1 else float(mean[0]),
                'std': std.tolist() if len(std) > 1 else float(std[0]),
                'ci_lower': ci_lower.tolist() if len(ci_lower) > 1 else float(ci_lower[0]),
                'ci_upper': ci_upper.tolist() if len(ci_upper) > 1 else float(ci_upper[0]),
                'relative_uncertainty': (std / (np.abs(mean) + 1e-6)).tolist() if len(std) > 1 else float(std[0] / (abs(mean[0]) + 1e-6))
            }
            return self.uncertainty_results
        except:
            return {}

    def _test_residual_normality(self, y_true, y_pred):
        try:
            residuals = y_true - y_pred
            if len(residuals) >= 3 and len(residuals) <= 5000:
                statistic, p_value = shapiro(residuals)
                return {
                    'shapiro_statistic': float(statistic),
                    'shapiro_p_value': float(p_value),
                    'is_normal': bool(p_value > 0.05),
                    'interpretation': 'Normal distribution' if p_value > 0.05 else 'Non-normal distribution'
                }
        except:
            pass
        return None

    def optimize_bayesian(self, conditions, n_calls=50):
        if not SKOPT_AVAILABLE:
            return None, None, None
        try:
            space = [
                Real(0.0001, 0.50, name='quantity', prior='log-uniform'),
                Real(25, 250, name='temp', prior='uniform'),
                Real(1, 72, name='time', prior='uniform')
            ]
            @use_named_args(space)
            def objective(**params):
                test_conditions = conditions.copy()
                test_conditions['quantity'] = params['quantity']
                test_conditions['temp'] = params['temp']
                test_conditions['time'] = params['time']
                try:
                    result = self.predict(test_conditions)
                    if result['success']:
                        return -result['prediction']
                    else:
                        return 100.0
                except:
                    return 100.0
            result = gp_minimize(
                func=objective,
                dimensions=space,
                n_calls=n_calls,
                n_initial_points=10,
                initial_point_generator='random',
                acq_func='EI',
                acq_optimizer='sampling',
                random_state=42,
                verbose=False
            )
            best_params = {
                'quantity': float(result.x[0]),
                'temp': float(result.x[1]),
                'time': float(result.x[2])
            }
            best_yield = -float(result.fun)
            history = {
                'method': 'Bayesian',
                'n_calls': n_calls,
                'func_vals': [-f for f in result.func_vals],
                'x_iters': result.x_iters,
                'best_yield': best_yield,
                'best_params': best_params
            }
            return best_params, best_yield, history
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {str(e)}")
            return None, None, None

    def optimize_genetic(self, conditions, population_size=50, generations=30):
        if not DEAP_AVAILABLE:
            return None, None, None
        try:
            try:
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                creator.create("Individual", list, fitness=creator.FitnessMax)
            except:
                pass
            toolbox = base.Toolbox()
            toolbox.register("attr_qty", np.random.uniform, 0.0001, 0.50)
            toolbox.register("attr_temp", np.random.uniform, 25, 250)
            toolbox.register("attr_time", np.random.uniform, 1, 72)
            toolbox.register("individual", tools.initCycle, creator.Individual,
                            (toolbox.attr_qty, toolbox.attr_temp, toolbox.attr_time), n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            def evaluate(individual):
                qty, temp, time = individual
                test_conditions = conditions.copy()
                test_conditions['quantity'] = qty
                test_conditions['temp'] = temp
                test_conditions['time'] = time
                try:
                    result = self.predict(test_conditions)
                    if result['success']:
                        return (result['prediction'],)
                    else:
                        return (0.0,)
                except:
                    return (0.0,)
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutPolynomialBounded,
                            low=[0.0001, 25, 1],
                            up=[0.50, 250, 72],
                            eta=20.0,
                            indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            population = toolbox.population(n=population_size)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
            population, logbook = algorithms.eaSimple(
                population, toolbox,
                cxpb=0.7, mutpb=0.2, ngen=generations,
                stats=stats, verbose=False
            )
            best_ind = tools.selBest(population, 1)[0]
            best_params = {
                'quantity': float(best_ind[0]),
                'temp': float(best_ind[1]),
                'time': float(best_ind[2])
            }
            best_yield = float(best_ind.fitness.values[0])
            history = {
                'method': 'Genetic',
                'population_size': population_size,
                'generations': generations,
                'best_yield': best_yield,
                'best_params': best_params
            }
            return best_params, best_yield, history
        except Exception as e:
            logger.error(f"Genetic optimization failed: {str(e)}")
            return None, None, None

    def train(self, model_type='Ensemble'):
        try:
            if self.X is None or len(self.X) == 0:
                raise ValueError("Data must be loaded first")
            if not self.is_enriched:
                raise ValueError("Cannot train ML model on basic dataset.")
            if len(self.y) == 0 or np.all(np.isnan(self.y)):
                raise ValueError("No valid yield data available")
            n_samples = len(self.X)
            n_features = len(self.feature_columns)
            if n_samples < self.min_samples_per_feature:
                raise ValueError(f"Refusing to train: only {n_samples} labeled reactions.")
            X_scaled_orig = StandardScaler().fit_transform(self.X)
            self.anomaly_results = self.fe.advanced.detect_anomalies(
                X_scaled_orig,
                threshold=self.config.get_float('data_processing/outlier_detection/mahalanobis_threshold', 3.5)
            )
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
            test_size = min(0.2, max(0.1, 3.0 / n_samples))
            y_binned = pd.qcut(self.y, q=4, labels=False, duplicates='drop')
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.y, test_size=test_size, random_state=42,
                stratify=y_binned if len(np.unique(y_binned)) > 1 else None
            )
            if model_type == 'Ensemble' or model_type == 'all':
                if n_samples < self.min_samples_for_full_ensemble:
                    model_names = ['Ridge', 'Lasso']
                else:
                    model_names = ['Random_Forest', 'Gradient_Boosting', 'Ridge', 'Lasso']
                    if XGB_AVAILABLE:
                        model_names.append('XGBoost')
                    if LGBM_AVAILABLE:
                        model_names.append('LightGBM')
                    if CATBOOST_AVAILABLE and n_samples > 50:
                        model_names.append('CatBoost')
            else:
                model_names = [model_type]
            models = {}
            performances = {}
            for name in model_names:
                try:
                    model = self._create_model(name)
                    if model is not None:
                        model.fit(X_train, y_train)
                        models[name] = model
                        y_pred = model.predict(X_test)
                        if len(y_pred) > 0 and not np.isnan(y_pred).all():
                            performances[name] = {
                                'r2': float(r2_score(y_test, y_pred)),
                                'mae': float(mean_absolute_error(y_test, y_pred)),
                                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                                'explained_variance': float(explained_variance_score(y_test, y_pred))
                            }
                except Exception as e:
                    logger.error(f"Model training failed for {name}: {str(e)}")
                    continue
            if not models:
                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                models['Ridge_Fallback'] = model
                y_pred = model.predict(X_test)
                performances['Ridge_Fallback'] = {
                    'r2': float(r2_score(y_test, y_pred)),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'explained_variance': float(explained_variance_score(y_test, y_pred))
                }
                self.fallback_model = 'Ridge_Fallback'
            if not models:
                return {'success': False, 'message': 'No models could be trained'}
            self.models = models
            self.is_trained = True
            self.model_performance = performances
            if performances:
                best_name = max(performances.items(), key=lambda x: x[1].get('r2', 0))[0]
                self.best_model = best_name
            if n_samples < 15:
                cv = LeaveOneOut()
            else:
                cv = RepeatedKFold(n_splits=min(3, n_samples//2), n_repeats=2, random_state=42)
            self.cv_results = self._perform_cross_validation(cv)
            self.feature_importance = self._calculate_feature_importance()
            self.X_test_, self.y_test_ = X_test, y_test
            best_model = models[self.best_model]
            y_pred_final = best_model.predict(X_test)
            self.bootstrap_results = self._perform_bootstrap(y_test, y_pred_final, self.bootstrap_iterations)
            self.aic_bic_results = self._calculate_aic_bic(best_model, X_train, y_train)
            self.learning_curve_data = self._calculate_learning_curve(best_model, X_train, y_train, cv=min(3, len(y_train)//2))
            self.uncertainty_results = self._calculate_ensemble_uncertainty(X_test)
            self.correlation_analysis = self.fe.advanced.correlation_analysis(self.df, target='yield')
            self.normality_test = self._test_residual_normality(y_test, y_pred_final)
            try:
                self.shap_analysis = SHAPAnalyzer().analyze(
                    best_model, X_test[:min(100, len(X_test))],
                    self.feature_columns[:min(50, len(self.feature_columns))]
                )
            except:
                pass
            try:
                self.conformal_prediction = ConformalPredictor().predict(
                    best_model, X_train, y_train, X_test, alpha=0.05
                )
            except:
                pass
            return {
                'success': True,
                'message': f"Trained {len(models)} models",
                'performance': convert_to_serializable(performances),
                'best_model': self.best_model,
                'model_count': len(models),
                'cv_results': convert_to_serializable(self.cv_results),
                'fallback_used': self.fallback_model is not None,
                'features_used': self.feature_columns,
                'anomaly_results': convert_to_serializable(self.anomaly_results),
                'bootstrap_results': convert_to_serializable(self.bootstrap_results),
                'aic_bic_results': convert_to_serializable(self.aic_bic_results),
                'learning_curve': convert_to_serializable(self.learning_curve_data),
                'uncertainty_results': convert_to_serializable(self.uncertainty_results),
                'correlation_analysis': convert_to_serializable(self.correlation_analysis),
                'normality_test': convert_to_serializable(self.normality_test),
                'shap_analysis': convert_to_serializable(self.shap_analysis),
                'conformal_prediction': convert_to_serializable(self.conformal_prediction)
            }
        except Exception as e:
            logger.error(f"Training error: {str(e)}\n{traceback.format_exc()}")
            raise

    def _perform_cross_validation(self, cv):
        try:
            if not self.is_trained or not self.models:
                return {}
            X_scaled = self.scaler.transform(self.X)
            cv_results = {}
            for name, model in self.models.items():
                try:
                    scores = cross_val_score(model, X_scaled, self.y, cv=cv, scoring='r2')
                    cv_results[name] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'scores': [float(s) for s in scores]
                    }
                except:
                    pass
            return cv_results
        except:
            return {}

    def _calculate_feature_importance(self):
        try:
            if not self.is_trained or not self.models or self.X_test_ is None:
                return {}
            model = list(self.models.values())[0]
            if PERM_IMP_AVAILABLE:
                result = permutation_importance(model, self.X_test_, self.y_test_, n_repeats=10, random_state=42)
                importance_dict = {}
                for i, col in enumerate(self.feature_columns[:len(result.importances_mean)]):
                    importance_dict[col] = {
                        'importance': float(result.importances_mean[i]),
                        'std': float(result.importances_std[i])
                    }
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1]['importance'], reverse=True)
                return {'top_10': sorted_importance[:10], 'all': importance_dict}
            return {}
        except:
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
                    except:
                        continue
        if not predictions:
            return np.zeros(len(X))
        weights = np.array(weights) / np.sum(weights)
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += weight * pred
        return ensemble_pred

    def _create_feature_vector(self, conditions):
        if not self.feature_columns:
            return None
        f = {}
        f['temp'] = conditions.get('temp', 80)
        f['time'] = conditions.get('time', 24)
        f['quantity'] = conditions.get('quantity', 0.0025)
        f['temp_time_product'] = f['temp'] * f['time']
        f['temp_time_ratio'] = f['temp'] / (f['time'] + 1)
        f['temp_time_interaction'] = f['temp'] * f['time'] / 100
        f['temp_log_time'] = f['temp'] * np.log1p(f['time'])
        f['time_log_temp'] = f['time'] * np.log1p(f['temp'])
        f['quantity_log1p'] = np.log1p(f['quantity'])
        f['quantity_sqrt'] = np.sqrt(f['quantity'])
        f['quantity_squared'] = f['quantity'] ** 2
        f['catalyst_loading'] = f['quantity'] / (f['temp'] + 1)
        f['temp_quantity_product'] = f['temp'] * f['quantity']
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
            f['subs1_tpsa'] = mf.get('tpsa', 0)
            f['subs1_halogen_count'] = mf.get('halogen_count', 0)
            f['subs1_rotatable_bonds'] = mf.get('rotatable_bonds', 0)
            f['subs1_fraction_csp3'] = mf.get('fraction_csp3', 0)
            f['subs1_avg_electronegativity'] = mf.get('avg_electronegativity', 0)
            f['subs1_flexibility_ratio'] = mf.get('flexibility_ratio', 0)
            f['subs1_tpsa_per_heavy_atom'] = mf.get('tpsa_per_heavy_atom', 0)
            f['subs1_molecular_volume'] = mf.get('molecular_volume', 0)
            f['subs1_spherocity'] = mf.get('spherocity', 0)
        else:
            f['subs1_length'] = 0
            f['subs1_logp'] = 0
            f['subs1_mw'] = 0
            f['subs1_rings'] = 0
            f['subs1_aromatic_rings'] = 0
            f['subs1_hba'] = 0
            f['subs1_hbd'] = 0
            f['subs1_complexity'] = 0
            f['subs1_tpsa'] = 0
            f['subs1_halogen_count'] = 0
            f['subs1_rotatable_bonds'] = 0
            f['subs1_fraction_csp3'] = 0
            f['subs1_avg_electronegativity'] = 0
            f['subs1_flexibility_ratio'] = 0
            f['subs1_tpsa_per_heavy_atom'] = 0
            f['subs1_molecular_volume'] = 0
            f['subs1_spherocity'] = 0
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
            f['subs2_tpsa'] = mf.get('tpsa', 0)
            f['subs2_halogen_count'] = mf.get('halogen_count', 0)
            f['subs2_rotatable_bonds'] = mf.get('rotatable_bonds', 0)
            f['subs2_fraction_csp3'] = mf.get('fraction_csp3', 0)
            f['subs2_avg_electronegativity'] = mf.get('avg_electronegativity', 0)
            f['subs2_flexibility_ratio'] = mf.get('flexibility_ratio', 0)
            f['subs2_tpsa_per_heavy_atom'] = mf.get('tpsa_per_heavy_atom', 0)
            f['subs2_molecular_volume'] = mf.get('molecular_volume', 0)
            f['subs2_spherocity'] = mf.get('spherocity', 0)
        else:
            f['subs2_length'] = 0
            f['subs2_logp'] = 0
            f['subs2_mw'] = 0
            f['subs2_rings'] = 0
            f['subs2_aromatic_rings'] = 0
            f['subs2_hba'] = 0
            f['subs2_hbd'] = 0
            f['subs2_complexity'] = 0
            f['subs2_tpsa'] = 0
            f['subs2_halogen_count'] = 0
            f['subs2_rotatable_bonds'] = 0
            f['subs2_fraction_csp3'] = 0
            f['subs2_avg_electronegativity'] = 0
            f['subs2_flexibility_ratio'] = 0
            f['subs2_tpsa_per_heavy_atom'] = 0
            f['subs2_molecular_volume'] = 0
            f['subs2_spherocity'] = 0
        f['substrate_steric_sum'] = f['subs1_length'] + f['subs2_length']
        f['substrate_steric_diff'] = abs(f['subs1_length'] - f['subs2_length'])
        f['substrate_steric_product'] = f['subs1_length'] * f['subs2_length']
        f['substrate_logp_avg'] = (f['subs1_logp'] + f['subs2_logp']) / 2
        f['substrate_logp_diff'] = abs(f['subs1_logp'] - f['subs2_logp'])
        f['substrate_logp_sum'] = f['subs1_logp'] + f['subs2_logp']
        f['substrate_mw_avg'] = (f['subs1_mw'] + f['subs2_mw']) / 2
        f['substrate_mw_diff'] = abs(f['subs1_mw'] - f['subs2_mw'])
        f['substrate_mw_sum'] = f['subs1_mw'] + f['subs2_mw']
        f['total_rings'] = f['subs1_rings'] + f['subs2_rings']
        f['ring_diff'] = abs(f['subs1_rings'] - f['subs2_rings'])
        f['aromatic_sum'] = f['subs1_aromatic_rings'] + f['subs2_aromatic_rings']
        f['halogen_total'] = f['subs1_halogen_count'] + f['subs2_halogen_count']
        vector = []
        for col in self.feature_columns:
            vector.append(f.get(col, 0))
        if self.scaler is not None:
            try:
                vector = self.scaler.transform([vector])[0]
            except:
                pass
        return np.array(vector).reshape(1, -1)

    def predict(self, conditions):
        try:
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
                'experimental_mode': conditions.get('experimental_mode', False)
            }
            ml_yield = None
            if self.is_enriched and self.is_trained and self.models:
                try:
                    feature_vector = self._create_feature_vector(conditions)
                    if feature_vector is not None:
                        if len(self.models) > 1:
                            ml_pred = self._ensemble_predict(feature_vector)
                            ml_yield = float(ml_pred[0]) if len(ml_pred) > 0 else None
                        else:
                            model = list(self.models.values())[0]
                            ml_pred = model.predict(feature_vector)
                            ml_yield = float(ml_pred[0]) if len(ml_pred) > 0 else None
                except Exception as e:
                    logger.error(f"ML prediction failed: {str(e)}")
            confidence_interval = None
            if ml_yield is not None and not np.isnan(ml_yield):
                if self.bootstrap_results:
                    std_est = self.bootstrap_results.get('std', 5.0)
                else:
                    std_est = 5.0
                confidence_interval = {
                    'lower': max(0, ml_yield - 1.96 * std_est),
                    'upper': min(100, ml_yield + 1.96 * std_est)
                }
            final_yield = np.clip(ml_yield if ml_yield is not None else 50, 0, 100)
            model_name = 'Ensemble' if self.is_trained else 'Basic'
            if final_yield >= 85:
                yield_class, color = 'Excellent', '#10B981'
            elif final_yield >= 70:
                yield_class, color = 'Good', '#3B82F6'
            elif final_yield >= 50:
                yield_class, color = 'Moderate', '#F59E0B'
            elif final_yield >= 30:
                yield_class, color = 'Poor', '#EF4444'
            else:
                yield_class, color = 'Very Poor', '#DC2626'
            prediction_data['yield'] = float(final_yield)
            prediction_data['yield_class'] = yield_class
            prediction_data['model'] = model_name
            save_prediction_history(prediction_data)
            return {
                'success': True,
                'prediction': float(final_yield),
                'prediction_display': f"~ {final_yield:.4f} % (est.)",
                'ml_prediction': float(ml_yield) if ml_yield is not None else None,
                'model': model_name,
                'yield_class': yield_class,
                'yield_class_color': color,
                'confidence': 0.85 if ml_yield is not None else 0.5,
                'confidence_interval': confidence_interval,
                'best_model': self.best_model,
                'model_count': len(self.models) if self.models else 0,
                'is_enriched': self.is_enriched,
                'fallback_used': self.fallback_model is not None,
                'history_count': len(PREDICTION_HISTORY),
                'academic_details': {
                    'anomaly_score': self.anomaly_results.get('anomaly_scores', [])[:1] if self.anomaly_results else [],
                    'learning_curve': self.learning_curve_data,
                    'bootstrap_ci': self.bootstrap_results,
                    'aic_bic': self.aic_bic_results,
                    'uncertainty': self.uncertainty_results,
                    'normality_test': self.normality_test,
                    'correlation_analysis': self.correlation_analysis,
                    'shap_analysis': self.shap_analysis,
                    'conformal_prediction': self.conformal_prediction
                }
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}\n{traceback.format_exc()}")
            raise

    def optimize_catalyst(self, conditions, method='grid'):
        try:
            catalysts = []
            if self.df is not None and 'catalizor' in self.df.columns:
                catalysts = self.df['catalizor'].unique().tolist()
            else:
                catalysts = ['Pd(PPh3)4', 'PdCl2(dppf)', 'Pd(OAc)2', 'Pd2(dba)3']
            results = []
            for catalyst in catalysts[:10]:
                test_conditions = conditions.copy()
                test_conditions['catalizor'] = catalyst
                if method == 'bayesian':
                    best_params, best_yield, history = self.optimize_bayesian(test_conditions, n_calls=30)
                    if best_params is None:
                        best_params = {'quantity': conditions.get('quantity', 0.0025), 'temp': conditions.get('temp', 80), 'time': conditions.get('time', 24)}
                        best_yield = 0
                    results.append((catalyst, best_yield, best_params['quantity'], best_params['temp'], best_params['time'], history))
                elif method == 'genetic':
                    best_params, best_yield, history = self.optimize_genetic(test_conditions, population_size=30, generations=20)
                    if best_params is None:
                        best_params = {'quantity': conditions.get('quantity', 0.0025), 'temp': conditions.get('temp', 80), 'time': conditions.get('time', 24)}
                        best_yield = 0
                    results.append((catalyst, best_yield, best_params['quantity'], best_params['temp'], best_params['time'], history))
                else:
                    best_yield = 0
                    best_qty = conditions.get('quantity', 0.0025)
                    best_temp = conditions.get('temp', 80)
                    best_time = conditions.get('time', 24)
                    quantities = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
                    for qty in quantities:
                        for temp in [60, 80, 100]:
                            for time in [12, 24, 48]:
                                test_conditions['quantity'] = qty
                                test_conditions['temp'] = temp
                                test_conditions['time'] = time
                                result = self.predict(test_conditions)
                                if result['success'] and result['prediction'] > best_yield:
                                    best_yield = result['prediction']
                                    best_qty = qty
                                    best_temp = temp
                                    best_time = time
                    results.append((catalyst, best_yield, best_qty, best_temp, best_time, None))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return []

    def get_feature_importance(self):
        return self.feature_importance

    def get_best_model(self):
        if not self.model_performance:
            return None, {}
        best = max(self.model_performance.items(), key=lambda x: x[1].get('r2', 0))
        return best[0], best[1]

    def analyze_residuals(self):
        if not self.is_trained or self.X_test_ is None:
            return {}
        try:
            predictions = self._ensemble_predict(self.X_test_)
            residuals = self.y_test_ - predictions
            residual_stats = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals)),
                'skewness': float(skew(residuals)),
                'kurtosis': float(kurtosis(residuals)),
                'normality_test': self.normality_test
            }
            return residual_stats
        except:
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
                'is_enriched': self.is_enriched,
                'cv_results': self.cv_results,
                'feature_importance': self.feature_importance,
                'anomaly_results': self.anomaly_results,
                'bootstrap_results': self.bootstrap_results,
                'aic_bic_results': self.aic_bic_results,
                'learning_curve_data': self.learning_curve_data,
                'uncertainty_results': self.uncertainty_results,
                'correlation_analysis': self.correlation_analysis,
                'normality_test': self.normality_test,
                'shap_analysis': self.shap_analysis,
                'conformal_prediction': self.conformal_prediction
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
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
            self.anomaly_results = model_data.get('anomaly_results', {})
            self.bootstrap_results = model_data.get('bootstrap_results', {})
            self.aic_bic_results = model_data.get('aic_bic_results', {})
            self.learning_curve_data = model_data.get('learning_curve_data', {})
            self.uncertainty_results = model_data.get('uncertainty_results', {})
            self.correlation_analysis = model_data.get('correlation_analysis', {})
            self.normality_test = model_data.get('normality_test', {})
            self.shap_analysis = model_data.get('shap_analysis', {})
            self.conformal_prediction = model_data.get('conformal_prediction', {})
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False

predict_ml_bp.route('/')
@error_handler
def index():
    return render_template('predict_ml.html')

predict_ml_bp.route('/api/get_csv_files', methods=['GET'])
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

predict_ml_bp.route('/api/upload_csv', methods=['POST'])
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
    filename = secure_filename_custom(file.filename)
    filepath = os.path.join('static/datasets', filename)
    file.save(filepath)
    logger.info(f"File uploaded: {filename}")
    return jsonify({'success': True, 'message': f'File uploaded: {filename}', 'filename': filename})

predict_ml_bp.route('/api/load_data', methods=['POST'])
@error_handler
@timing_decorator
def load_data():
    global PREDICTOR, CONFIG, DATA_INFO, CURRENT_FILE
    data = get_json_body()
    filename = data.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': 'Filename required'})
    filename = secure_filename_custom(filename)
    filepath = os.path.join('static/datasets', filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': f'File not found: {filename}'})
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
        'is_enriched': PREDICTOR.is_enriched,
        'anomaly_results': PREDICTOR.anomaly_results,
        'correlation_analysis': PREDICTOR.correlation_analysis
    }
    result = PREDICTOR.train('Ensemble')
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
            'is_enriched': bool(PREDICTOR.is_enriched),
            'fallback_used': result.get('fallback_used', False),
            'anomaly_results': convert_to_serializable(result.get('anomaly_results', {})),
            'bootstrap_results': convert_to_serializable(result.get('bootstrap_results', {})),
            'aic_bic_results': convert_to_serializable(result.get('aic_bic_results', {})),
            'learning_curve': convert_to_serializable(result.get('learning_curve', {})),
            'uncertainty_results': convert_to_serializable(result.get('uncertainty_results', {})),
            'correlation_analysis': convert_to_serializable(result.get('correlation_analysis', {})),
            'normality_test': convert_to_serializable(result.get('normality_test', {})),
            'shap_analysis': convert_to_serializable(result.get('shap_analysis', {})),
            'conformal_prediction': convert_to_serializable(result.get('conformal_prediction', {}))
        })
    else:
        return jsonify({'success': False, 'message': result.get('message', 'Training failed')})

predict_ml_bp.route('/api/change_model', methods=['POST'])
@error_handler
@timing_decorator
def change_model():
    global PREDICTOR
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
        'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM',
        'CatBoost': 'CatBoost',
        'Ridge': 'Ridge',
        'Lasso': 'Lasso'
    }
    allowed_keys = set(model_map.values())
    key = model_map.get(model_name, model_name)
    if key not in allowed_keys:
        return jsonify({'success': False, 'message': f'Unknown model: {model_name}'})
    if PREDICTOR.df is not None:
        PREDICTOR._prepare_features()
    result = PREDICTOR.train(key)
    if result['success']:
        perf = result.get('performance', {})
        stats = list(perf.values())[0] if perf else {}
        return jsonify({
            'success': True,
            'message': f"Switched to {model_name}",
            'current_model': model_name,
            'stats': stats,
            'best_model': result.get('best_model'),
            'cv_results': result.get('cv_results', {}),
            'fallback_used': result.get('fallback_used', False)
        })
    else:
        return jsonify({'success': False, 'message': result.get('message', 'Failed to train model')})

predict_ml_bp.route('/api/save_model', methods=['POST'])
@error_handler
def save_model():
    global PREDICTOR
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'No model to save'})
    data = get_json_body()
    filename = secure_filename_custom(data.get('filename', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'))
    if not filename:
        return jsonify({'success': False, 'message': 'Invalid filename'}), 400
    filepath = os.path.join('static/models', filename)
    success = PREDICTOR.save_model(filepath)
    return jsonify({'success': success, 'message': f"Model saved to {filename}" if success else "Failed to save model", 'filename': filename if success else None})

predict_ml_bp.route('/api/load_model', methods=['POST'])
@error_handler
def load_model():
    global PREDICTOR, CONFIG
    data = get_json_body()
    filename = secure_filename_custom(data.get('filename', ''))
    if not filename:
        return jsonify({'success': False, 'message': 'Filename required'}), 400
    filepath = os.path.join('static/models', filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': f'Model not found: {filename}'})
    CONFIG = ConfigManager('config/info.xml')
    PREDICTOR = SuzukiPredictor(CONFIG)
    success = PREDICTOR.load_model(filepath)
    return jsonify({
        'success': success,
        'message': f"Model loaded from {filename}" if success else "Failed to load model",
        'best_model': PREDICTOR.best_model if success else None,
        'is_enriched': PREDICTOR.is_enriched if success else False
    })

predict_ml_bp.route('/api/list_models', methods=['GET'])
@error_handler
def list_models():
    models_dir = 'static/models'
    os.makedirs(models_dir, exist_ok=True)
    models = []
    for f in os.listdir(models_dir):
        if f.endswith('.pkl'):
            path = os.path.join(models_dir, f)
            models.append({'name': f, 'size': format_size(os.path.getsize(path)), 'modified': datetime.fromtimestamp(os.path.getmtime(path)).isoformat()})
    models.sort(key=lambda x: x['modified'], reverse=True)
    return jsonify({'success': True, 'models': models})

predict_ml_bp.route('/api/make_prediction', methods=['POST'])
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
    max_smiles_len = CONFIG.max_smiles_length if CONFIG else 500
    if len(str(data.get('subs1_smiles', ''))) > max_smiles_len or len(str(data.get('subs2_smiles', ''))) > max_smiles_len:
        return jsonify({'success': False, 'message': f'SMILES too long (max {max_smiles_len})'})
    if not data['solv1'] or data['solv1'] == '':
        return jsonify({'success': False, 'message': 'Solvent 1 is required'})
    try:
        temp = to_float(data['temp'], field_name='temp')
        time_h = to_float(data['time'], field_name='time')
        quantity = to_float(data['quantity'], field_name='quantity')
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    exp_mode = bool(data.get('experimental_mode', False))
    result = PREDICTOR.predict({
        'temp': temp,
        'time': time_h,
        'quantity': quantity,
        'catalizor': clean_text(data['catalizor']),
        'base': clean_text(data['base']),
        'solv1': clean_text(data['solv1']),
        'solv2': clean_text(data.get('solv2', '')),
        'subs1_smiles': clean_text(data['subs1_smiles'], max_smiles_len),
        'subs2_smiles': clean_text(data['subs2_smiles'], max_smiles_len),
        'experimental_mode': exp_mode
    })
    if not result['success']:
        return jsonify({'success': False, 'message': result.get('message', 'Prediction failed')})
    return jsonify({
        'success': True,
        'prediction': result['prediction'],
        'prediction_display': result.get('prediction_display', f"~ {result['prediction']:.4f} % (est.)"),
        'ml_prediction': result.get('ml_prediction'),
        'model': result['model'],
        'yield_class': result.get('yield_class', 'Unknown'),
        'yield_class_color': result.get('yield_class_color', '#6B7280'),
        'confidence': result.get('confidence', 0.85),
        'confidence_interval': result.get('confidence_interval'),
        'best_model': result.get('best_model', 'None'),
        'model_count': result.get('model_count', 0),
        'is_enriched': result.get('is_enriched', False),
        'fallback_used': result.get('fallback_used', False),
        'history_count': result.get('history_count', 0),
        'academic_details': convert_to_serializable(result.get('academic_details', {}))
    })

predict_ml_bp.route('/api/optimize_catalyst', methods=['POST'])
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
    max_smiles_len = CONFIG.max_smiles_length if CONFIG else 500
    if len(str(data.get('subs1_smiles', ''))) > max_smiles_len or len(str(data.get('subs2_smiles', ''))) > max_smiles_len:
        return jsonify({'success': False, 'message': f'SMILES too long (max {max_smiles_len})'})
    if not data['solv1'] or data['solv1'] == '':
        return jsonify({'success': False, 'message': 'Solvent 1 is required'})
    try:
        temp = to_float(data['temp'], field_name='temp')
        time_h = to_float(data['time'], field_name='time')
        quantity = to_float(data['quantity'], field_name='quantity')
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400
    method = data.get('method', 'grid')
    exp_mode = bool(data.get('experimental_mode', False))
    results = PREDICTOR.optimize_catalyst({
        'temp': temp,
        'time': time_h,
        'quantity': quantity,
        'base': clean_text(data['base']),
        'solv1': clean_text(data['solv1']),
        'solv2': clean_text(data.get('solv2', '')),
        'subs1_smiles': clean_text(data['subs1_smiles'], max_smiles_len),
        'subs2_smiles': clean_text(data['subs2_smiles'], max_smiles_len),
        'experimental_mode': exp_mode
    }, method=method)
    if not results:
        return jsonify({'success': False, 'message': 'Optimization failed'})
    return jsonify({'success': True, 'results': results, 'method': method})

predict_ml_bp.route('/api/model_performance', methods=['GET'])
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
            'cv_results': PREDICTOR.cv_results,
            'anomaly_results': PREDICTOR.anomaly_results,
            'bootstrap_results': PREDICTOR.bootstrap_results,
            'aic_bic_results': PREDICTOR.aic_bic_results,
            'learning_curve': PREDICTOR.learning_curve_data,
            'uncertainty_results': PREDICTOR.uncertainty_results,
            'correlation_analysis': PREDICTOR.correlation_analysis,
            'normality_test': PREDICTOR.normality_test
        }
    })

predict_ml_bp.route('/api/feature_importance', methods=['GET'])
@error_handler
def feature_importance():
    global PREDICTOR
    if PREDICTOR is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    importance = PREDICTOR.get_feature_importance()
    return jsonify({'success': True, 'feature_importance': importance.get('top_10', []), 'all_features': importance.get('all', {})})

predict_ml_bp.route('/api/prediction_history', methods=['GET'])
@error_handler
def get_prediction_history():
    load_prediction_history()
    return jsonify({'success': True, 'history': PREDICTION_HISTORY, 'count': len(PREDICTION_HISTORY)})

predict_ml_bp.route('/api/clear_history', methods=['POST'])
@error_handler
def clear_prediction_history():
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = []
    if os.path.exists(PREDICTION_HISTORY_FILE):
        os.remove(PREDICTION_HISTORY_FILE)
    return jsonify({'success': True, 'message': 'History cleared'})

predict_ml_bp.route('/api/health', methods=['GET'])
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
        'rdkit_available': RDKIT_AVAILABLE,
        'xgb_available': XGB_AVAILABLE,
        'lgbm_available': LGBM_AVAILABLE,
        'catboost_available': CATBOOST_AVAILABLE,
        'skopt_available': SKOPT_AVAILABLE,
        'deap_available': DEAP_AVAILABLE,
        'shap_available': SHAP_AVAILABLE,
        'mapie_available': MAPIE_AVAILABLE,
        'xtb_available': XTB_AVAILABLE,
        'history_count': len(PREDICTION_HISTORY),
        'features': len(PREDICTOR.feature_columns) if PREDICTOR else 0
    })

DEFAULT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<suzuki_config>
    <metadata>
        <last_updated>2026-09-07</last_updated>
        <description>Production-ready Suzuki ML predictor</description>
    </metadata>
    <data_integrity>
        <critical_non_nullable_columns>temp,time,quantity,catalizor,base</critical_non_nullable_columns>
        <minimum_usable_rows>5</minimum_usable_rows>
    </data_integrity>
    <ml_training_safeguards>
        <min_samples_per_feature>10</min_samples_per_feature>
        <min_samples_for_full_ensemble>30</min_samples_for_full_ensemble>
        <bootstrap_iterations>1000</bootstrap_iterations>
        <confidence_level>0.95</confidence_level>
    </ml_training_safeguards>
    <model_parameters>
        <Random_Forest>
            <n_estimators>300</n_estimators>
            <max_depth>15</max_depth>
            <random_state>42</random_state>
            <n_jobs>1</n_jobs>
        </Random_Forest>
        <Gradient_Boosting>
            <n_estimators>350</n_estimators>
            <max_depth>7</max_depth>
            <learning_rate>0.07</learning_rate>
            <random_state>42</random_state>
        </Gradient_Boosting>
        <XGBoost>
            <n_estimators>350</n_estimators>
            <max_depth>7</max_depth>
            <learning_rate>0.08</learning_rate>
            <random_state>42</random_state>
            <n_jobs>1</n_jobs>
        </XGBoost>
        <LightGBM>
            <n_estimators>400</n_estimators>
            <max_depth>10</max_depth>
            <num_leaves>31</num_leaves>
            <learning_rate>0.06</learning_rate>
            <random_state>42</random_state>
            <n_jobs>1</n_jobs>
        </LightGBM>
        <CatBoost>
            <iterations>400</iterations>
            <depth>7</depth>
            <learning_rate>0.07</learning_rate>
            <random_seed>42</random_seed>
        </CatBoost>
        <Ridge>
            <alpha>1.0</alpha>
            <random_state>42</random_state>
        </Ridge>
        <Lasso>
            <alpha>1.0</alpha>
            <random_state>42</random_state>
        </Lasso>
        <Ensemble>
            <weights>
                <Random_Forest>0.22</Random_Forest>
                <Gradient_Boosting>0.18</Gradient_Boosting>
                <XGBoost>0.14</XGBoost>
                <LightGBM>0.14</LightGBM>
                <CatBoost>0.12</CatBoost>
                <Ridge>0.10</Ridge>
                <Lasso>0.10</Lasso>
            </weights>
        </Ensemble>
    </model_parameters>
    <feature_importance>
        <temperature>0.16</temperature>
        <time>0.13</time>
        <catalyst_quantity>0.11</catalyst_quantity>
        <electronegativity>0.03</electronegativity>
        <flexibility>0.03</flexibility>
        <molecular_volume>0.02</molecular_volume>
    </feature_importance>
    <data_processing>
        <normalization>
            <numeric_method>standard_scaler</numeric_method>
        </normalization>
        <split>
            <test_size>0.20</test_size>
            <random_state>42</random_state>
        </split>
        <outlier_detection>
            <mahalanobis_threshold>3.5</mahalanobis_threshold>
        </outlier_detection>
    </data_processing>
    <performance_metrics>
        <metrics>
            <r2>true</r2>
            <mae>true</mae>
            <rmse>true</rmse>
            <aic>true</aic>
            <bic>true</bic>
            <shapiro_wilk>true</shapiro_wilk>
        </metrics>
        <uncertainty>
            <n_bootstrap>1000</n_bootstrap>
            <confidence_level>0.95</confidence_level>
        </uncertainty>
        <learning_curve>
            <enabled>true</enabled>
            <cv_folds>5</cv_folds>
        </learning_curve>
    </performance_metrics>
    <security>
        <sanitization>
            <max_smiles_length>500</max_smiles_length>
        </sanitization>
    </security>
</suzuki_config>"""

def init_app():
    os.makedirs('static/datasets', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs('static/models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    load_prediction_history()

init_app()
