from flask import Blueprint, render_template, request, jsonify
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
from scipy import stats
from scipy.spatial.distance import mahalanobis
from scipy.stats import shapiro, pearsonr, skew, kurtosis
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, RepeatedKFold,
    LeaveOneOut, learning_curve
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, explained_variance_score,
    max_error
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from werkzeug.utils import secure_filename

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
PREDICTOR = None
CONFIG = None
DATA_INFO = None
CURRENT_FILE = None

REQUIRED_COLUMNS = ['yield', 'temp', 'time', 'quantity', 'catalizor', 'base', 'solv1']
OPTIONAL_COLUMNS = ['solv2', 'subs1', 'subs2', 'product']
CRITICAL_NON_NULLABLE_COLUMNS = ['temp', 'time', 'quantity', 'catalizor', 'base']
MAX_SMILES_LENGTH = 500
MAX_TEXT_FIELD_LENGTH = 500


def classify_and_filter_rows(df):
    working = df.copy()
    present_critical = [c for c in CRITICAL_NON_NULLABLE_COLUMNS if c in working.columns]
    missing_critical_cols = [c for c in CRITICAL_NON_NULLABLE_COLUMNS if c not in working.columns]
    if missing_critical_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_critical_cols)}")
    critical_ok_mask = working[present_critical].notnull().all(axis=1)
    rejected_df = working[~critical_ok_mask].copy()
    valid_structure_df = working[critical_ok_mask].copy()
    if 'yield' in valid_structure_df.columns:
        yield_present_mask = valid_structure_df['yield'].notnull()
    else:
        yield_present_mask = pd.Series(False, index=valid_structure_df.index)
    usable_df = valid_structure_df[yield_present_mask].copy()
    failed_df = valid_structure_df[~yield_present_mask].copy()
    return usable_df, failed_df, rejected_df


def convert_to_serializable(obj):
    if obj is None:
        return None
    elif isinstance(obj, np.ndarray):
        return [convert_to_serializable(x) for x in obj.tolist()]
    elif isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, str)):
        return obj
    elif isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return str(obj)


def clean_text(value, max_length=MAX_TEXT_FIELD_LENGTH):
    if value is None:
        return ''
    text = str(value).strip()
    return text[:max_length]


def to_float(value, default=None, field_name='value'):
    if value is None or value == '':
        if default is not None:
            return default
        raise ValueError(f'{field_name} must be a number')
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field_name} must be a valid number')
    if not np.isfinite(f):
        raise ValueError(f'{field_name} must be a finite number')
    return f


def secure_filename_custom(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)


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


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result
    return wrapper


def error_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, KeyError, ValueError) as e:
            return jsonify({'success': False, 'message': str(e)}), 400
        except Exception as e:
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


def is_enriched_dataset(df):
    cols = df.columns.tolist()
    smiles_cols = [c for c in cols if '_SMILES_' in c and not c.endswith('_status')]
    return len(smiles_cols) >= 3


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
    except Exception:
        pass


def load_prediction_history():
    global PREDICTION_HISTORY
    try:
        if os.path.exists(PREDICTION_HISTORY_FILE):
            with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                PREDICTION_HISTORY = json.load(f)
        else:
            PREDICTION_HISTORY = []
    except Exception:
        PREDICTION_HISTORY = []


class ConfigManager:
    def __init__(self, config_path='config/info.xml'):
        self.config_path = config_path
        self.params = {}
        self.raw_xml = ""
        self._ensure_dir()
        self._load_or_create()
        self._parse_all()
        self.max_smiles_length = self.get_int('security/sanitization/max_smiles_length', 500)

    def _ensure_dir(self):
        dir_path = os.path.dirname(self.config_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def _load_or_create(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.raw_xml = f.read()
            self.tree = ET.parse(self.config_path)
            self.root = self.tree.getroot()
        else:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(DEFAULT_XML)
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

    def get(self, path, default=None):
        try:
            keys = path.split('/')
            current = self.params
            for key in keys:
                if key in current:
                    current = current[key]
                else:
                    return default
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

    def get_raw_xml(self):
        return self.raw_xml


class AdvancedFeatureEngineer:
    def __init__(self):
        self.selected_features = []
        self.anomaly_results = {}
        self.EN = {
            'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
            'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
            'B': 2.04, 'Si': 1.90, 'Pd': 2.20, 'Pt': 2.28
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
            return {
                'n_tpsa_contribution': float(n_count * 12.0),
                'o_tpsa_contribution': float(o_count * 20.0),
                's_tpsa_contribution': float(s_count * 5.0),
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
    def __init__(self):
        self.selected_features = []
        self.advanced = AdvancedFeatureEngineer()

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
        except:
            pass
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
        except:
            return df.select_dtypes(include=[np.number])


class SuzukiPredictor:
    def __init__(self, config):
        self.config = config
        self.fe = FeatureEngineer()
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
        self.min_samples_per_feature = self.config.get_int('ml_training_safeguards/min_samples_per_feature', 10)
        self.min_samples_for_full_ensemble = self.config.get_int('ml_training_safeguards/min_samples_for_full_ensemble', 30)
        self.bootstrap_iterations = self.config.get_int('ml_training_safeguards/bootstrap_iterations', 1000)
        self.confidence_level = self.config.get_float('ml_training_safeguards/confidence_level', 0.95)

        self.experimental_mode_enabled = self.config.get_bool('experimental_mode/enabled', True)
        self.min_samples_for_ml_confidence = self.config.get_int('experimental_mode/min_samples_for_ml_confidence', 20)
        self.raw_extrapolation_guard = self.config.get_float('experimental_mode/raw_extrapolation_guard', 60.0)
        self.heuristic_temp_weight = self.config.get_float('experimental_mode/temp_weight', 0.5)
        self.heuristic_time_weight = self.config.get_float('experimental_mode/time_weight', 0.3)
        self.heuristic_quantity_weight = self.config.get_float('experimental_mode/quantity_weight', 0.2)
        self.heuristic_temp_sensitivity = self.config.get_float('experimental_mode/temp_sensitivity', 20.0)

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
        except Exception:
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
                'std': float(np.std(errors)),
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
                    'interpretation': 'Normal' if p_value > 0.05 else 'Non-normal'
                }
        except:
            pass
        return None

    def _safe_stratified_split(self, X_scaled, y, test_size, random_state=42):
        n_samples = len(y)
        n_test = int(round(n_samples * test_size))
        n_test = max(1, n_test)
        max_bins = min(4, n_test, n_samples // 2)

        stratify_arg = None
        if max_bins >= 2:
            try:
                y_binned = pd.qcut(y, q=max_bins, labels=False, duplicates='drop')
                n_classes = len(np.unique(y_binned))
                if n_classes >= 2 and n_test >= n_classes:
                    counts = pd.Series(y_binned).value_counts()
                    if counts.min() >= 2:
                        stratify_arg = y_binned
            except Exception:
                stratify_arg = None

        try:
            return train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state,
                stratify=stratify_arg
            )
        except ValueError:
            return train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state,
                stratify=None
            )

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

            self.anomaly_results = self.fe.advanced.detect_anomalies(
                self.X.values, threshold=3.5
            )

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.X)
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)

            test_size = min(0.2, max(0.1, 3.0 / n_samples))

            X_train, X_test, y_train, y_test = self._safe_stratified_split(
                X_scaled, self.y, test_size=test_size, random_state=42
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
                except Exception:
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
                'normality_test': convert_to_serializable(self.normality_test)
            }
        except Exception as e:
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

    def _experimental_heuristic_predict(self, conditions):
        temp = float(conditions.get('temp', 80) or 80)
        time_h = float(conditions.get('time', 12) or 12)
        qty = float(conditions.get('quantity', 0.0025) or 0.0025)

        if self.df is not None and len(self.df) > 0 and 'yield' in self.df.columns:
            base_yield = float(self.df['yield'].mean())
            temp_ref = float(self.df['temp'].mean()) if 'temp' in self.df.columns and self.df['temp'].notnull().any() else 80.0
            time_ref = float(self.df['time'].mean()) if 'time' in self.df.columns and self.df['time'].notnull().any() else 12.0
            qty_ref = float(self.df['quantity'].mean()) if 'quantity' in self.df.columns and self.df['quantity'].notnull().any() else 0.0025
        else:
            base_yield, temp_ref, time_ref, qty_ref = 50.0, 80.0, 12.0, 0.0025

        temp_ref = temp_ref if temp_ref > 0 else 80.0
        time_ref = time_ref if time_ref > 0 else 12.0
        qty_ref = qty_ref if qty_ref > 0 else 0.0025
        sensitivity = self.heuristic_temp_sensitivity if self.heuristic_temp_sensitivity > 0 else 20.0
        temp_factor = 1.0 / (1.0 + np.exp(-(temp - temp_ref) / sensitivity))

        time_factor = 1.0 - np.exp(-max(time_h, 0.0) / time_ref)
        qty_factor = 1.0 - np.exp(-max(qty, 0.0) / qty_ref)

        w_t, w_h, w_q = self.heuristic_temp_weight, self.heuristic_time_weight, self.heuristic_quantity_weight
        weight_sum = (w_t + w_h + w_q) or 1.0
        combined = (w_t * temp_factor + w_h * time_factor + w_q * qty_factor) / weight_sum

        heuristic_yield = base_yield * (0.5 + combined)
        heuristic_yield = float(np.clip(heuristic_yield, 1.0, 99.0))
        return heuristic_yield

    def predict(self, conditions, force_experimental=False):
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
                'subs2_smiles': conditions.get('subs2_smiles')
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
                except:
                    pass

            used_heuristic = False
            heuristic_reason = None
            n_train_samples = len(self.X) if self.X is not None else 0

            if force_experimental:
                used_heuristic = True
                heuristic_reason = 'user_requested'
            elif not (self.is_enriched and self.is_trained and self.models):
                used_heuristic = True
                heuristic_reason = 'no_trained_model'
            elif ml_yield is None or not np.isfinite(ml_yield):
                used_heuristic = True
                heuristic_reason = 'invalid_ml_output'
            elif self.experimental_mode_enabled and n_train_samples < self.min_samples_for_ml_confidence:
                used_heuristic = True
                heuristic_reason = f'small_dataset (n={n_train_samples} < {self.min_samples_for_ml_confidence})'
            elif self.experimental_mode_enabled and (
                ml_yield < -self.raw_extrapolation_guard or ml_yield > 100 + self.raw_extrapolation_guard
            ):
                used_heuristic = True
                heuristic_reason = f'extreme_extrapolation (raw ml output={ml_yield:.2f})'

            if used_heuristic:
                final_yield = self._experimental_heuristic_predict(conditions)
                model_name = 'Experimental Heuristic (non-scientific fallback)'
            else:
                final_yield = float(np.clip(ml_yield if ml_yield is not None else 50, 0, 100))
                model_name = 'Ensemble' if self.is_trained else 'Basic'

            confidence_interval = None
            if not used_heuristic and ml_yield is not None and not np.isnan(ml_yield):
                if self.bootstrap_results:
                    std_est = self.bootstrap_results.get('std', 5.0)
                else:
                    std_est = 5.0
                confidence_interval = {
                    'lower': max(0, ml_yield - 1.96 * std_est),
                    'upper': min(100, ml_yield + 1.96 * std_est)
                }

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

            academic_details = {
                'anomaly_score': self.anomaly_results.get('anomaly_scores', [])[:1] if self.anomaly_results else [],
                'learning_curve': self.learning_curve_data,
                'bootstrap_ci': self.bootstrap_results,
                'aic_bic': self.aic_bic_results,
                'uncertainty': self.uncertainty_results,
                'normality_test': self.normality_test,
                'correlation_analysis': self.correlation_analysis,
                'cv_results': self.cv_results
            }
            if used_heuristic:
                academic_details['note'] = (
                    'Deneysel (heuristic) mod aktif oldugu icin bu tek '
                    'tahmin ML modelinden degil, kural tabanli bir '
                    'yaklasimdan geldi. Asagidaki istatistikler ise bu '
                    'tahmine degil, egitilmis modelin genel performansina '
                    'aittir.'
                )

            return {
                'success': True,
                'prediction': float(final_yield),
                'prediction_display': f"~ {final_yield:.4f} % (est.)",
                'ml_prediction': float(ml_yield) if ml_yield is not None else None,
                'model': model_name,
                'mode': 'experimental_heuristic' if used_heuristic else 'ml_ensemble',
                'mode_reason': heuristic_reason,
                'mode_disclaimer': (
                    "Bu tahmin, egitilmis ML modelinin guvenilir olmadigi "
                    "durumlarda (kucuk veri seti veya asiri ekstrapolasyon) "
                    "ya da kullanicinin bu modu elle secmesi durumunda "
                    "devreye giren, bilimsel/akademik olarak dogrulanmamis "
                    "basit bir kural tabanli (deneysel) tahmindir: sicaklik, "
                    "sure ve katalizor miktari arttikca verim mantikli "
                    "yonde degisir, ancak bu sayi bir literatur veya "
                    "mekanistik modele dayanmaz."
                ) if used_heuristic else None,
                'yield_class': yield_class,
                'yield_class_color': color,
                'confidence': (0.85 if ml_yield is not None else 0.5) if not used_heuristic else 0.3,
                'confidence_interval': confidence_interval,
                'best_model': self.best_model,
                'model_count': len(self.models) if self.models else 0,
                'is_enriched': self.is_enriched,
                'fallback_used': self.fallback_model is not None,
                'history_count': len(PREDICTION_HISTORY),
                'academic_details': convert_to_serializable(academic_details)
            }
        except Exception as e:
            raise

    def optimize_catalyst(self, conditions):
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
                best_yield = 0
                best_qty = conditions.get('quantity', 0.0025)
                quantities = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
                for qty in quantities:
                    test_conditions['quantity'] = qty
                    result = self.predict(test_conditions)
                    if result['success'] and result['prediction'] > best_yield:
                        best_yield = result['prediction']
                        best_qty = qty
                results.append((catalyst, best_yield, best_qty))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
        except Exception:
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
                'normality_test': self.normality_test
            }
            joblib.dump(model_data, filepath)
            return True
        except:
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
            return True
        except:
            return False

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
    filename = secure_filename_custom(file.filename)
    filepath = os.path.join('static/datasets', filename)
    file.save(filepath)
    return jsonify({'success': True, 'message': f'File uploaded: {filename}', 'filename': filename})


@predict_ml_bp.route('/api/load_data', methods=['POST'])
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
            'normality_test': convert_to_serializable(result.get('normality_test', {}))
        })
    else:
        return jsonify({'success': False, 'message': result.get('message', 'Training failed')})


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

    force_experimental = bool(data.get('force_experimental', False))
    result = PREDICTOR.predict({
        'temp': temp,
        'time': time_h,
        'quantity': quantity,
        'catalizor': clean_text(data['catalizor']),
        'base': clean_text(data['base']),
        'solv1': clean_text(data['solv1']),
        'solv2': clean_text(data.get('solv2', '')),
        'subs1_smiles': clean_text(data['subs1_smiles'], max_smiles_len),
        'subs2_smiles': clean_text(data['subs2_smiles'], max_smiles_len)
    }, force_experimental=force_experimental)
    if not result['success']:
        return jsonify({'success': False, 'message': result.get('message', 'Prediction failed')})
    return jsonify({
        'success': True,
        'prediction': result['prediction'],
        'prediction_display': result.get('prediction_display', f"~ {result['prediction']:.4f} % (est.)"),
        'ml_prediction': result.get('ml_prediction'),
        'model': result['model'],
        'mode': result.get('mode', 'ml_ensemble'),
        'mode_reason': result.get('mode_reason'),
        'mode_disclaimer': result.get('mode_disclaimer'),
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
    results = PREDICTOR.optimize_catalyst({
        'temp': temp,
        'time': time_h,
        'quantity': quantity,
        'base': clean_text(data['base']),
        'solv1': clean_text(data['solv1']),
        'solv2': clean_text(data.get('solv2', '')),
        'subs1_smiles': clean_text(data['subs1_smiles'], max_smiles_len),
        'subs2_smiles': clean_text(data['subs2_smiles'], max_smiles_len)
    })
    if not results:
        return jsonify({'success': False, 'message': 'Optimization failed'})
    return jsonify({'success': True, 'results': results})


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
            'yield_mean': DATA_INFO['yield_stats']['mean'] if DATA_INFO and 'yield_stats' in DATA_INFO else 0,
            'yield_std': DATA_INFO['yield_stats']['std'] if DATA_INFO and 'yield_stats' in DATA_INFO else 0,
            'best_model': best_name,
            'best_r2': best_stats.get('r2', 0) if best_stats else 0,
            'model_count': len(PREDICTOR.models),
            'feature_count': len(PREDICTOR.feature_columns),
            'is_trained': PREDICTOR.is_trained,
            'is_enriched': PREDICTOR.is_enriched,
            'performances': convert_to_serializable(perf),
            'residuals': convert_to_serializable(residuals),
            'cv_results': convert_to_serializable(PREDICTOR.cv_results),
            'anomaly_results': convert_to_serializable(PREDICTOR.anomaly_results),
            'bootstrap_results': convert_to_serializable(PREDICTOR.bootstrap_results),
            'aic_bic_results': convert_to_serializable(PREDICTOR.aic_bic_results),
            'learning_curve': convert_to_serializable(PREDICTOR.learning_curve_data),
            'uncertainty_results': convert_to_serializable(PREDICTOR.uncertainty_results),
            'correlation_analysis': convert_to_serializable(PREDICTOR.correlation_analysis),
            'normality_test': convert_to_serializable(PREDICTOR.normality_test)
        }
    })


@predict_ml_bp.route('/api/prediction_history', methods=['GET'])
@error_handler
def get_prediction_history():
    load_prediction_history()
    return jsonify({
        'success': True,
        'history': PREDICTION_HISTORY,
        'count': len(PREDICTION_HISTORY)
    })


@predict_ml_bp.route('/api/clear_history', methods=['POST'])
@error_handler
def clear_prediction_history():
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = []
    if os.path.exists(PREDICTION_HISTORY_FILE):
        os.remove(PREDICTION_HISTORY_FILE)
    return jsonify({'success': True, 'message': 'History cleared'})


@predict_ml_bp.route('/api/change_model', methods=['POST'])
@error_handler
def change_model():
    global PREDICTOR
    data = get_json_body()
    model_name = data.get('model_name', 'Ensemble')
    if PREDICTOR is None or PREDICTOR.X is None:
        return jsonify({'success': False, 'message': 'Load data first'})
    result = PREDICTOR.train(model_name)
    if not result['success']:
        return jsonify({'success': False, 'message': result.get('message', 'Training failed')})
    perf = result.get('performance', {})
    stats = perf.get(model_name)
    if stats is None and perf:
        stats = list(perf.values())[0]
    cv = result.get('cv_results', {})
    return jsonify({
        'success': True,
        'message': f"Switched to {model_name}",
        'current_model': model_name,
        'stats': stats or {},
        'cv_results': convert_to_serializable(cv),
        'best_model': result.get('best_model')
    })


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
        'rdkit_available': RDKIT_AVAILABLE,
        'xgb_available': XGB_AVAILABLE,
        'lgbm_available': LGBM_AVAILABLE,
        'catboost_available': CATBOOST_AVAILABLE,
        'history_count': len(PREDICTION_HISTORY),
        'features': len(PREDICTOR.feature_columns) if PREDICTOR else 0
    })


DEFAULT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<suzuki_config>
    <metadata>
        <last_updated>2026-09-07</last_updated>
        <description>Academic Suzuki ML predictor</description>
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
    <experimental_mode>
        <enabled>true</enabled>
        <min_samples_for_ml_confidence>20</min_samples_for_ml_confidence>
        <raw_extrapolation_guard>60</raw_extrapolation_guard>
        <temp_weight>0.5</temp_weight>
        <time_weight>0.3</time_weight>
        <quantity_weight>0.2</quantity_weight>
        <temp_sensitivity>20.0</temp_sensitivity>
    </experimental_mode>
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
