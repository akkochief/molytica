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
    GridSearchCV, RandomizedSearchCV, ParameterGrid,
    RepeatedKFold, cross_validate, StratifiedShuffleSplit
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
    RFE, SelectFromModel, SequentialFeatureSelector, f_regression
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel, DotProduct
GP_AVAILABLE = True
from sklearn.isotonic import IsotonicRegression
import warnings

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen, AllChem, Draw
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcNumHBD, CalcNumHBA
    from rdkit.Chem.Draw import IPythonConsole, MolToImage
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    PERM_IMP_AVAILABLE = True
except ImportError:
    PERM_IMP_AVAILABLE = False

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
    from scipy.stats import shapiro, levene, kruskal, f_oneway, mannwhitneyu, chi2_contingency, ks_2samp
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
    'H': {'sigma_m': 0.00, 'sigma_p': 0.00, 'taft_es': 0.00, 'sigma_plus': 0.00, 'sigma_minus': 0.00, 'sigma_i': 0.00, 'sigma_r': 0.00},
    'CH3': {'sigma_m': -0.07, 'sigma_p': -0.17, 'taft_es': 0.00, 'sigma_plus': -0.31, 'sigma_minus': -0.17, 'sigma_i': -0.05, 'sigma_r': -0.12},
    'OCH3': {'sigma_m': 0.12, 'sigma_p': -0.27, 'taft_es': -0.20, 'sigma_plus': -0.78, 'sigma_minus': -0.27, 'sigma_i': 0.30, 'sigma_r': -0.57},
    'OH': {'sigma_m': 0.12, 'sigma_p': -0.37, 'taft_es': -0.51, 'sigma_plus': -0.92, 'sigma_minus': -0.37, 'sigma_i': 0.30, 'sigma_r': -0.67},
    'F': {'sigma_m': 0.34, 'sigma_p': 0.06, 'taft_es': -0.46, 'sigma_plus': -0.07, 'sigma_minus': 0.06, 'sigma_i': 0.52, 'sigma_r': -0.46},
    'Cl': {'sigma_m': 0.37, 'sigma_p': 0.23, 'taft_es': -0.97, 'sigma_plus': 0.11, 'sigma_minus': 0.23, 'sigma_i': 0.47, 'sigma_r': -0.24},
    'Br': {'sigma_m': 0.39, 'sigma_p': 0.23, 'taft_es': -1.16, 'sigma_plus': 0.15, 'sigma_minus': 0.23, 'sigma_i': 0.45, 'sigma_r': -0.22},
    'I': {'sigma_m': 0.35, 'sigma_p': 0.18, 'taft_es': -1.40, 'sigma_plus': 0.14, 'sigma_minus': 0.18, 'sigma_i': 0.40, 'sigma_r': -0.22},
    'NO2': {'sigma_m': 0.71, 'sigma_p': 0.78, 'taft_es': -1.01, 'sigma_plus': 0.79, 'sigma_minus': 1.27, 'sigma_i': 0.65, 'sigma_r': 0.13},
    'CN': {'sigma_m': 0.56, 'sigma_p': 0.66, 'taft_es': -0.51, 'sigma_plus': 0.66, 'sigma_minus': 1.00, 'sigma_i': 0.58, 'sigma_r': 0.08},
    'CF3': {'sigma_m': 0.43, 'sigma_p': 0.54, 'taft_es': -2.40, 'sigma_plus': 0.61, 'sigma_minus': 0.54, 'sigma_i': 0.45, 'sigma_r': 0.09},
    'COOH': {'sigma_m': 0.37, 'sigma_p': 0.45, 'taft_es': -1.20, 'sigma_plus': 0.42, 'sigma_minus': 0.45, 'sigma_i': 0.32, 'sigma_r': 0.13},
    'COOCH3': {'sigma_m': 0.35, 'sigma_p': 0.39, 'taft_es': -1.10, 'sigma_plus': 0.32, 'sigma_minus': 0.39, 'sigma_i': 0.30, 'sigma_r': 0.09},
    'CHO': {'sigma_m': 0.36, 'sigma_p': 0.42, 'taft_es': -1.20, 'sigma_plus': 0.42, 'sigma_minus': 0.42, 'sigma_i': 0.32, 'sigma_r': 0.10},
    'NH2': {'sigma_m': -0.16, 'sigma_p': -0.66, 'taft_es': -0.20, 'sigma_plus': -1.30, 'sigma_minus': -0.66, 'sigma_i': 0.12, 'sigma_r': -0.78},
    'N(CH3)2': {'sigma_m': -0.15, 'sigma_p': -0.83, 'taft_es': -0.30, 'sigma_plus': -1.70, 'sigma_minus': -0.83, 'sigma_i': 0.10, 'sigma_r': -0.93},
    'SO2CH3': {'sigma_m': 0.60, 'sigma_p': 0.72, 'taft_es': -1.50, 'sigma_plus': 0.73, 'sigma_minus': 0.72, 'sigma_i': 0.55, 'sigma_r': 0.17},
    'B(OH)2': {'sigma_m': 0.04, 'sigma_p': -0.10, 'taft_es': 0.00, 'sigma_plus': -0.10, 'sigma_minus': -0.10, 'sigma_i': 0.10, 'sigma_r': -0.20},
    'Si(CH3)3': {'sigma_m': -0.04, 'sigma_p': -0.07, 'taft_es': -0.80, 'sigma_plus': -0.07, 'sigma_minus': -0.07, 'sigma_i': -0.02, 'sigma_r': -0.05},
    'C(CH3)3': {'sigma_m': -0.10, 'sigma_p': -0.20, 'taft_es': -1.54, 'sigma_plus': -0.26, 'sigma_minus': -0.20, 'sigma_i': -0.05, 'sigma_r': -0.15},
    'C6H5': {'sigma_m': 0.06, 'sigma_p': -0.01, 'taft_es': -1.20, 'sigma_plus': -0.18, 'sigma_minus': -0.01, 'sigma_i': 0.08, 'sigma_r': -0.09},
    'COCH3': {'sigma_m': 0.38, 'sigma_p': 0.50, 'taft_es': -1.20, 'sigma_plus': 0.50, 'sigma_minus': 0.50, 'sigma_i': 0.32, 'sigma_r': 0.18},
    'SO2NH2': {'sigma_m': 0.55, 'sigma_p': 0.62, 'taft_es': -1.30, 'sigma_plus': 0.62, 'sigma_minus': 0.62, 'sigma_i': 0.50, 'sigma_r': 0.12},
    'NHAc': {'sigma_m': 0.21, 'sigma_p': 0.00, 'taft_es': -0.50, 'sigma_plus': -0.20, 'sigma_minus': 0.00, 'sigma_i': 0.22, 'sigma_r': -0.22},
}

LIGAND_PROPERTIES_EXTENDED = {
    'triphenylphosphine': {
        'cone_angle': 145.0, 'tep': 2068.9, 'denticity': 1, 
        'class': 'triarylphosphine', 'pka_bh': 2.73, 'softness': 6.2,
        'electronic_donating': 0.45, 'steric_bulk': 1.8, 'tolman_angle': 145.0,
        'bite_angle': 0.0, 'nucleophilicity': 3.8
    },
    'pph3': {
        'cone_angle': 145.0, 'tep': 2068.9, 'denticity': 1,
        'class': 'triarylphosphine', 'pka_bh': 2.73, 'softness': 6.2,
        'electronic_donating': 0.45, 'steric_bulk': 1.8, 'tolman_angle': 145.0,
        'bite_angle': 0.0, 'nucleophilicity': 3.8
    },
    'sphos': {
        'cone_angle': 163.0, 'tep': 2064.0, 'denticity': 1,
        'class': 'dialkylbiarylphosphine', 'pka_bh': 7.70, 'softness': 7.8,
        'electronic_donating': 0.65, 'steric_bulk': 2.2, 'tolman_angle': 163.0,
        'bite_angle': 0.0, 'nucleophilicity': 5.2
    },
    'xphos': {
        'cone_angle': 180.0, 'tep': 2062.4, 'denticity': 1,
        'class': 'dialkylbiarylphosphine', 'pka_bh': 7.90, 'softness': 8.0,
        'electronic_donating': 0.70, 'steric_bulk': 2.5, 'tolman_angle': 180.0,
        'bite_angle': 0.0, 'nucleophilicity': 5.5
    },
    'dppf': {
        'cone_angle': 150.0, 'tep': 2065.3, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 4.50, 'softness': 7.1,
        'electronic_donating': 0.55, 'steric_bulk': 1.9, 'bite_angle': 99.07,
        'tolman_angle': 150.0, 'nucleophilicity': 4.5
    },
    'xantphos': {
        'cone_angle': 120.0, 'tep': 2060.0, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 5.20, 'softness': 7.3,
        'electronic_donating': 0.50, 'steric_bulk': 2.0, 'bite_angle': 110.0,
        'tolman_angle': 120.0, 'nucleophilicity': 4.8
    },
    'ipr': {
        'cone_angle': 170.0, 'tep': 2050.0, 'denticity': 1,
        'class': 'nhc', 'pka_bh': 8.50, 'softness': 8.5,
        'electronic_donating': 0.85, 'steric_bulk': 2.8, 'tolman_angle': 170.0,
        'bite_angle': 0.0, 'nucleophilicity': 6.5
    },
    'imes': {
        'cone_angle': 160.0, 'tep': 2055.0, 'denticity': 1,
        'class': 'nhc', 'pka_bh': 8.20, 'softness': 8.3,
        'electronic_donating': 0.80, 'steric_bulk': 2.5, 'tolman_angle': 160.0,
        'bite_angle': 0.0, 'nucleophilicity': 6.2
    },
    'sipr': {
        'cone_angle': 165.0, 'tep': 2052.0, 'denticity': 1,
        'class': 'nhc', 'pka_bh': 8.40, 'softness': 8.4,
        'electronic_donating': 0.82, 'steric_bulk': 2.6, 'tolman_angle': 165.0,
        'bite_angle': 0.0, 'nucleophilicity': 6.3
    },
    'binap': {
        'cone_angle': 130.0, 'tep': 2065.0, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 4.80, 'softness': 6.8,
        'electronic_donating': 0.48, 'steric_bulk': 2.3, 'bite_angle': 90.0,
        'tolman_angle': 130.0, 'nucleophilicity': 4.2
    },
    'dppp': {
        'cone_angle': 140.0, 'tep': 2067.0, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 4.60, 'softness': 6.9,
        'electronic_donating': 0.50, 'steric_bulk': 1.7, 'bite_angle': 90.0,
        'tolman_angle': 140.0, 'nucleophilicity': 4.3
    },
    'dppb': {
        'cone_angle': 135.0, 'tep': 2066.0, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 4.70, 'softness': 6.9,
        'electronic_donating': 0.50, 'steric_bulk': 1.8, 'bite_angle': 85.0,
        'tolman_angle': 135.0, 'nucleophilicity': 4.3
    },
    'dppe': {
        'cone_angle': 130.0, 'tep': 2067.0, 'denticity': 2,
        'class': 'bidentate_phosphine', 'pka_bh': 4.80, 'softness': 6.8,
        'electronic_donating': 0.48, 'steric_bulk': 1.8, 'bite_angle': 85.0,
        'tolman_angle': 130.0, 'nucleophilicity': 4.2
    },
    'pcy3': {
        'cone_angle': 170.0, 'tep': 2068.0, 'denticity': 1,
        'class': 'trialkylphosphine', 'pka_bh': 9.70, 'softness': 8.0,
        'electronic_donating': 0.75, 'steric_bulk': 2.6, 'tolman_angle': 170.0,
        'bite_angle': 0.0, 'nucleophilicity': 6.0
    },
    'ptbu3': {
        'cone_angle': 182.0, 'tep': 2058.0, 'denticity': 1,
        'class': 'trialkylphosphine', 'pka_bh': 11.40, 'softness': 8.2,
        'electronic_donating': 0.80, 'steric_bulk': 3.0, 'tolman_angle': 182.0,
        'bite_angle': 0.0, 'nucleophilicity': 6.5
    },
}

BASE_PROPERTIES = {
    'k2co3': {'pka': 10.3, 'solubility': 0.1, 'cation_radius': 1.38, 'hygroscopic': False, 'class': 'carbonate', 'pkb': 3.7, 'mw': 138.21, 'density': 2.43},
    'cs2co3': {'pka': 10.3, 'solubility': 2.6, 'cation_radius': 1.67, 'hygroscopic': True, 'class': 'carbonate', 'pkb': 3.7, 'mw': 325.82, 'density': 4.07},
    'na2co3': {'pka': 10.3, 'solubility': 0.2, 'cation_radius': 1.02, 'hygroscopic': False, 'class': 'carbonate', 'pkb': 3.7, 'mw': 105.99, 'density': 2.54},
    'k3po4': {'pka': 12.3, 'solubility': 0.5, 'cation_radius': 1.38, 'hygroscopic': True, 'class': 'phosphate', 'pkb': 1.7, 'mw': 212.27, 'density': 2.56},
    'naoh': {'pka': 15.7, 'solubility': 1.1, 'cation_radius': 1.02, 'hygroscopic': True, 'class': 'hydroxide', 'pkb': -1.7, 'mw': 40.00, 'density': 2.13},
    'koh': {'pka': 15.7, 'solubility': 1.2, 'cation_radius': 1.38, 'hygroscopic': True, 'class': 'hydroxide', 'pkb': -1.7, 'mw': 56.11, 'density': 2.04},
    'tea': {'pka': 10.7, 'solubility': 0.8, 'cation_radius': None, 'hygroscopic': False, 'class': 'amine', 'pkb': 3.3, 'mw': 101.19, 'density': 0.73},
    'dipea': {'pka': 11.4, 'solubility': 0.6, 'cation_radius': None, 'hygroscopic': False, 'class': 'amine', 'pkb': 2.6, 'mw': 129.25, 'density': 0.74},
    'koac': {'pka': 4.8, 'solubility': 0.1, 'cation_radius': 1.38, 'hygroscopic': False, 'class': 'acetate', 'pkb': 9.2, 'mw': 98.14, 'density': 1.57},
    'csf': {'pka': 3.2, 'solubility': 0.3, 'cation_radius': 1.67, 'hygroscopic': True, 'class': 'fluoride', 'pkb': 10.8, 'mw': 151.90, 'density': 4.11},
    'kf': {'pka': 3.2, 'solubility': 0.2, 'cation_radius': 1.38, 'hygroscopic': True, 'class': 'fluoride', 'pkb': 10.8, 'mw': 58.10, 'density': 2.48},
    'k2hpo4': {'pka': 12.3, 'solubility': 0.4, 'cation_radius': 1.38, 'hygroscopic': False, 'class': 'phosphate', 'pkb': 1.7, 'mw': 174.18, 'density': 2.44},
    'nahco3': {'pka': 6.4, 'solubility': 0.1, 'cation_radius': 1.02, 'hygroscopic': False, 'class': 'bicarbonate', 'pkb': 7.6, 'mw': 84.01, 'density': 2.20},
    'dbu': {'pka': 12.0, 'solubility': 0.5, 'cation_radius': None, 'hygroscopic': False, 'class': 'amidine', 'pkb': 2.0, 'mw': 152.24, 'density': 0.90},
    'dabco': {'pka': 8.8, 'solubility': 0.4, 'cation_radius': None, 'hygroscopic': False, 'class': 'amine', 'pkb': 5.2, 'mw': 112.18, 'density': 1.02},
    'pyridine': {'pka': 5.2, 'solubility': 0.3, 'cation_radius': None, 'hygroscopic': False, 'class': 'amine', 'pkb': 8.8, 'mw': 79.10, 'density': 0.98},
}

SOLVENT_PHYSICS_ADVANCED = {
    'water': {
        'dielectric': 80.1, 'bp_c': 100.0, 'polarity_index': 10.2, 
        'donor_number': 18.0, 'reichardt_et30': 63.1, 
        'hildebrand_delta': 47.8, 'viscosity_cp': 0.89, 
        'acceptor_number': 54.8, 'class': 'protic',
        'alpha': 1.17, 'beta': 0.47, 'pi_star': 1.09,
        'density': 0.997, 'mw': 18.02, 'surface_tension': 71.99
    },
    'methanol': {
        'dielectric': 32.7, 'bp_c': 64.7, 'polarity_index': 5.1,
        'donor_number': 19.0, 'reichardt_et30': 55.5,
        'hildebrand_delta': 29.7, 'viscosity_cp': 0.54,
        'acceptor_number': 41.5, 'class': 'protic',
        'alpha': 0.93, 'beta': 0.62, 'pi_star': 0.60,
        'density': 0.792, 'mw': 32.04, 'surface_tension': 22.60
    },
    'ethanol': {
        'dielectric': 24.6, 'bp_c': 78.4, 'polarity_index': 4.3,
        'donor_number': 19.2, 'reichardt_et30': 51.9,
        'hildebrand_delta': 26.5, 'viscosity_cp': 1.08,
        'acceptor_number': 37.9, 'class': 'protic',
        'alpha': 0.83, 'beta': 0.77, 'pi_star': 0.54,
        'density': 0.789, 'mw': 46.07, 'surface_tension': 22.39
    },
    'isopropanol': {
        'dielectric': 19.9, 'bp_c': 82.3, 'polarity_index': 3.9,
        'donor_number': 18.5, 'reichardt_et30': 48.6,
        'hildebrand_delta': 23.5, 'viscosity_cp': 2.04,
        'acceptor_number': 33.5, 'class': 'protic',
        'alpha': 0.76, 'beta': 0.84, 'pi_star': 0.48,
        'density': 0.786, 'mw': 60.10, 'surface_tension': 21.70
    },
    'acetone': {
        'dielectric': 20.7, 'bp_c': 56.1, 'polarity_index': 5.1,
        'donor_number': 17.0, 'reichardt_et30': 42.2,
        'hildebrand_delta': 19.7, 'viscosity_cp': 0.32,
        'acceptor_number': 12.5, 'class': 'aprotic',
        'alpha': 0.08, 'beta': 0.48, 'pi_star': 0.71,
        'density': 0.791, 'mw': 58.08, 'surface_tension': 23.70
    },
    'acetonitrile': {
        'dielectric': 37.5, 'bp_c': 82.0, 'polarity_index': 5.8,
        'donor_number': 14.1, 'reichardt_et30': 46.0,
        'hildebrand_delta': 24.3, 'viscosity_cp': 0.37,
        'acceptor_number': 18.9, 'class': 'aprotic',
        'alpha': 0.19, 'beta': 0.31, 'pi_star': 0.75,
        'density': 0.786, 'mw': 41.05, 'surface_tension': 29.29
    },
    'dmso': {
        'dielectric': 46.7, 'bp_c': 189.0, 'polarity_index': 7.2,
        'donor_number': 29.8, 'reichardt_et30': 45.1,
        'hildebrand_delta': 26.7, 'viscosity_cp': 1.99,
        'acceptor_number': 19.3, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.76, 'pi_star': 1.00,
        'density': 1.100, 'mw': 78.13, 'surface_tension': 43.54
    },
    'dmf': {
        'dielectric': 36.7, 'bp_c': 153.0, 'polarity_index': 6.4,
        'donor_number': 26.6, 'reichardt_et30': 43.8,
        'hildebrand_delta': 24.9, 'viscosity_cp': 0.82,
        'acceptor_number': 16.0, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.69, 'pi_star': 0.88,
        'density': 0.948, 'mw': 73.09, 'surface_tension': 37.10
    },
    'thf': {
        'dielectric': 7.5, 'bp_c': 66.0, 'polarity_index': 4.0,
        'donor_number': 20.0, 'reichardt_et30': 37.4,
        'hildebrand_delta': 18.5, 'viscosity_cp': 0.46,
        'acceptor_number': 8.0, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.55, 'pi_star': 0.58,
        'density': 0.889, 'mw': 72.11, 'surface_tension': 26.40
    },
    'dioxane': {
        'dielectric': 2.2, 'bp_c': 101.0, 'polarity_index': 4.8,
        'donor_number': 14.8, 'reichardt_et30': 36.0,
        'hildebrand_delta': 19.9, 'viscosity_cp': 1.37,
        'acceptor_number': 10.8, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.37, 'pi_star': 0.49,
        'density': 1.033, 'mw': 88.11, 'surface_tension': 33.60
    },
    'toluene': {
        'dielectric': 2.4, 'bp_c': 110.6, 'polarity_index': 2.4,
        'donor_number': 0.1, 'reichardt_et30': 33.9,
        'hildebrand_delta': 18.2, 'viscosity_cp': 0.59,
        'acceptor_number': 3.3, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.11, 'pi_star': 0.54,
        'density': 0.867, 'mw': 92.14, 'surface_tension': 28.52
    },
    'benzene': {
        'dielectric': 2.3, 'bp_c': 80.1, 'polarity_index': 2.7,
        'donor_number': 0.1, 'reichardt_et30': 34.5,
        'hildebrand_delta': 18.6, 'viscosity_cp': 0.60,
        'acceptor_number': 8.2, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.10, 'pi_star': 0.59,
        'density': 0.879, 'mw': 78.11, 'surface_tension': 28.88
    },
    'dichloromethane': {
        'dielectric': 8.9, 'bp_c': 39.6, 'polarity_index': 3.1,
        'donor_number': 0.0, 'reichardt_et30': 41.1,
        'hildebrand_delta': 20.2, 'viscosity_cp': 0.44,
        'acceptor_number': 20.4, 'class': 'aprotic',
        'alpha': 0.13, 'beta': 0.10, 'pi_star': 0.82,
        'density': 1.326, 'mw': 84.93, 'surface_tension': 28.12
    },
    'chloroform': {
        'dielectric': 4.8, 'bp_c': 61.2, 'polarity_index': 4.1,
        'donor_number': 0.0, 'reichardt_et30': 39.1,
        'hildebrand_delta': 19.0, 'viscosity_cp': 0.54,
        'acceptor_number': 23.1, 'class': 'aprotic',
        'alpha': 0.44, 'beta': 0.00, 'pi_star': 0.58,
        'density': 1.489, 'mw': 119.38, 'surface_tension': 27.16
    },
    'hexane': {
        'dielectric': 1.9, 'bp_c': 68.7, 'polarity_index': 0.1,
        'donor_number': 0.0, 'reichardt_et30': 31.0,
        'hildebrand_delta': 14.9, 'viscosity_cp': 0.29,
        'acceptor_number': 0.0, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.00, 'pi_star': 0.00,
        'density': 0.659, 'mw': 86.18, 'surface_tension': 18.40
    },
    'cyclohexane': {
        'dielectric': 2.0, 'bp_c': 80.7, 'polarity_index': 0.2,
        'donor_number': 0.0, 'reichardt_et30': 31.2,
        'hildebrand_delta': 16.7, 'viscosity_cp': 0.89,
        'acceptor_number': 0.0, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.00, 'pi_star': 0.00,
        'density': 0.779, 'mw': 84.16, 'surface_tension': 25.00
    },
    'ethyl acetate': {
        'dielectric': 6.0, 'bp_c': 77.1, 'polarity_index': 4.4,
        'donor_number': 14.0, 'reichardt_et30': 38.1,
        'hildebrand_delta': 18.2, 'viscosity_cp': 0.43,
        'acceptor_number': 9.3, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.45, 'pi_star': 0.55,
        'density': 0.902, 'mw': 88.11, 'surface_tension': 23.90
    },
    'diethyl ether': {
        'dielectric': 4.3, 'bp_c': 34.6, 'polarity_index': 2.8,
        'donor_number': 19.2, 'reichardt_et30': 34.6,
        'hildebrand_delta': 15.4, 'viscosity_cp': 0.22,
        'acceptor_number': 3.9, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.47, 'pi_star': 0.27,
        'density': 0.713, 'mw': 74.12, 'surface_tension': 17.60
    },
    'pyridine': {
        'dielectric': 12.3, 'bp_c': 115.2, 'polarity_index': 5.3,
        'donor_number': 33.1, 'reichardt_et30': 40.2,
        'hildebrand_delta': 21.8, 'viscosity_cp': 0.88,
        'acceptor_number': 14.2, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.64, 'pi_star': 0.87,
        'density': 0.982, 'mw': 79.10, 'surface_tension': 38.00
    },
    'nmp': {
        'dielectric': 32.2, 'bp_c': 202.0, 'polarity_index': 6.7,
        'donor_number': 27.3, 'reichardt_et30': 42.0,
        'hildebrand_delta': 23.1, 'viscosity_cp': 1.67,
        'acceptor_number': 13.6, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.77, 'pi_star': 0.92,
        'density': 1.028, 'mw': 99.13, 'surface_tension': 40.70
    },
    'dme': {
        'dielectric': 7.2, 'bp_c': 85.0, 'polarity_index': 3.5,
        'donor_number': 19.5, 'reichardt_et30': 36.5,
        'hildebrand_delta': 17.6, 'viscosity_cp': 0.46,
        'acceptor_number': 8.5, 'class': 'aprotic',
        'alpha': 0.00, 'beta': 0.53, 'pi_star': 0.53,
        'density': 0.868, 'mw': 90.12, 'surface_tension': 24.30
    },
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

CRITICAL_NON_NULLABLE_COLUMNS = [
    c for c in REQUIRED_COLUMNS if c not in NULLABLE_COLUMNS and c != FAILURE_COLUMN
]


def classify_and_filter_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Applies the academic data-integrity rules to a raw dataframe.

    Rules (no exceptions, no synthetic/random substitution):
      1. solv1 / solv2 may be null -> row is kept regardless.
      2. yield may be null -> row is kept in the audit trail as a FAILED reaction,
         but is excluded from the numeric training target.
      3. Any other required column (temp, time, quantity, catalizor, base) missing
         on a row -> the row is corrupt and is dropped entirely (never used anywhere).

    Returns:
        usable_df   -> rows with a valid, non-null yield (used to fit the regressor)
        failed_df   -> rows with all other required fields present but yield is null
                       (real, recorded failed reactions - kept for reporting only)
        rejected_df -> rows missing a critical non-nullable field (never used at all)
    """
    working = df.copy()

    present_critical = [c for c in CRITICAL_NON_NULLABLE_COLUMNS if c in working.columns]
    missing_critical_cols = [c for c in CRITICAL_NON_NULLABLE_COLUMNS if c not in working.columns]
    if missing_critical_cols:
        raise ValueError(
            f"Dataset is missing required column(s): {', '.join(missing_critical_cols)}. "
            f"Cannot proceed - academic data must be complete for these fields."
        )

    critical_ok_mask = working[present_critical].notnull().all(axis=1)
    rejected_df = working[~critical_ok_mask].copy()
    valid_structure_df = working[critical_ok_mask].copy()

    if FAILURE_COLUMN in valid_structure_df.columns:
        yield_present_mask = valid_structure_df[FAILURE_COLUMN].notnull()
    else:
        yield_present_mask = pd.Series(False, index=valid_structure_df.index)

    usable_df = valid_structure_df[yield_present_mask].copy()
    failed_df = valid_structure_df[~yield_present_mask].copy()

    logger.info(
        f"Data integrity check: {len(usable_df)} usable rows, "
        f"{len(failed_df)} failed reactions (null yield, kept for audit), "
        f"{len(rejected_df)} rejected rows (missing critical fields, discarded)"
    )

    return usable_df, failed_df, rejected_df

ACADEMIC_FEATURE_COLUMNS = [
    'subs1_SMILES_logp', 'subs1_SMILES_sigma_p', 'subs1_SMILES_sigma_m', 
    'subs1_SMILES_taft_es', 'subs1_SMILES_hba', 'subs1_SMILES_hbd',
    'subs1_SMILES_qed', 'subs1_SMILES_complexity', 'subs1_SMILES_kappa1',
    'subs2_SMILES_logp', 'subs2_SMILES_sigma_p', 'subs2_SMILES_sigma_m',
    'subs2_SMILES_taft_es', 'subs2_SMILES_hba', 'subs2_SMILES_hbd',
    'subs2_SMILES_qed', 'subs2_SMILES_complexity', 'subs2_SMILES_kappa1',
    'hsab_overall_compatibility', 'hsab_pd_halide_mismatch',
    'mechanistic_predictor_electronic_softness', 'reaction_rate_indicator',
    'elecproxy_homo_energy', 'elecproxy_lumo_energy', 'elecproxy_gap_energy',
    'elecproxy_chemical_potential', 'elecproxy_absolute_hardness', 'elecproxy_electrophilicity',
    'elecproxy_fukui_plus', 'elecproxy_fukui_minus',
    'physchem_proxy_score'
]


def save_prediction_history(prediction_data: Dict):
    """Save prediction to history"""
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
        logger.warning(f"Could not save prediction history: {e}")

def load_prediction_history():
    """Load prediction history from file"""
    global PREDICTION_HISTORY
    
    try:
        if os.path.exists(PREDICTION_HISTORY_FILE):
            with open(PREDICTION_HISTORY_FILE, 'r', encoding='utf-8') as f:
                PREDICTION_HISTORY = json.load(f)
            logger.info(f"Loaded {len(PREDICTION_HISTORY)} prediction history entries")
        else:
            PREDICTION_HISTORY = []
    except Exception as e:
        logger.warning(f"Could not load prediction history: {e}")
        PREDICTION_HISTORY = []

def get_last_prediction_for_conditions(conditions: Dict) -> Optional[Dict]:
    """Find the last prediction for the same conditions"""
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

def get_dominated_history_max_yield(conditions: Dict) -> Optional[Dict]:
    """
    Monotonicity floor: for a fixed catalyst/base/solvent/substrate system,
    increasing temperature, time, or catalyst quantity must never decrease
    yield relative to an earlier recorded run with equal-or-lower values.

    [PARAM]/REPRODUCIBILITY NOTE: this makes predict()'s output depend on
    PREDICTION_HISTORY (global, session/process state), not on `conditions`
    alone - the SAME input can return a different (floored) yield depending
    on what has been predicted earlier in the session. This is a much
    milder issue than the additive bonus flagged elsewhere in this file
    (it only raises a floor to fix an internal inconsistency, it does not
    compound), but calling it a "hard academic guarantee" overstates what
    it is: a runtime patch for the fact that the underlying scoring
    function is not guaranteed monotonic on its own. If publishing results
    from this code, report predictions from a fixed, reset history (or
    disable this mechanism) so reported numbers are a pure function of the
    stated conditions.

    This scans the FULL prediction history (not just the most recent entry) for every
    past record with matching categorical conditions (catalizor, base, solv1, solv2,
    subs1_smiles, subs2_smiles) whose temp, time and quantity are all <= the current
    request's values. Among those "dominated" records it returns the maximum recorded
    yield. That maximum becomes a floor for the new prediction.
    """
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

        dominated = (
            current_temp >= prev_temp and
            current_time >= prev_time and
            current_quantity >= prev_quantity
        )

        if dominated and (best is None or prev_yield > best['yield']):
            best = {
                'yield': prev_yield,
                'temp': prev_temp,
                'time': prev_time,
                'quantity': prev_quantity,
                'timestamp': entry.get('timestamp')
            }

    return best


def calculate_logarithmic_increase(base_value: float, new_value: float, 
                                   base_yield: float, max_increase: float = 2.0) -> float:
    """
    Calculate a logarithmic bonus with diminishing returns.

    [PARAM] WARNING: this function is a history-dependent additive bonus
    (its result depends on the PREVIOUS prediction, not just the current
    conditions). It is the same mechanism that a prior changelog entry
    (v7.2.2) says was removed from experimental mode for exactly this
    reason: applying it means identical input conditions can yield
    DIFFERENT results depending on what was predicted earlier in the
    session, which is not scientifically reproducible. See the
    "REPRODUCIBILITY WARNING" note where this function is called in
    predict() - it is disabled by default in v7.2.7 for this reason.

    [PARAM] the 0.35 scaling factor here is an engineering constant. An
    earlier version of this docstring attributed it to "empirical
    Suzuki-Miyaura data" in Billingsley, K. L.; Buchwald, S. L. J. Org.
    Chem. 2008, 73, 5589-5591 - that paper is real (it concerns
    Pd-catalyzed amination/coupling ligand scope), but no verification
    was done that it reports this specific 0.35 figure for this specific
    purpose, so the attribution has been removed as unverifiable rather
    than repeated. Do not re-add a citation for this constant unless it
    has actually been checked against the source.
    """
    if new_value <= base_value:
        return 0.0
    
    diff = new_value - base_value
    log_factor = np.log1p(diff / max(base_value, 1.0))
    
    max_log_increase = np.log1p(max_increase / max(base_value, 1.0))
    
    normalized = log_factor / max(max_log_increase, 0.001)
    increase = normalized * max_increase * 0.35
    
    return min(increase, max_increase)

def convert_to_serializable(obj):
    """Convert numpy/pandas objects to JSON-serializable format"""
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
    """Clean feature name for use in models"""
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
        digest = hashlib.md5(name.encode('utf-8')).hexdigest()[:10]
        name = name[:38].rstrip('_') + '_' + digest
    
    return name if name else 'feature'

def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee unique column names.

    Belt-and-suspenders safety net for clean_feature_name(): if two source
    columns ever still map to the same cleaned name (e.g. a freak hash
    collision, or any future change to clean_feature_name that reintroduces
    truncation collisions), suffix the repeats with _dup1, _dup2, ... instead
    of letting sklearn/narwhals blow up with DuplicateError at fit time.
    """
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
    """Format file size for display"""
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
    """Sanitize filename"""
    filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)
    return filename

def is_enriched_dataset(df: pd.DataFrame) -> bool:
    """Check if dataset contains academic features"""
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

def validate_reaction_conditions(conditions: Dict) -> Tuple[bool, str]:
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
            return False, f"Required field '{field}' is missing - row cannot be used"
    
    if conditions.get('solv2'):
        return True, "Two-solvent system detected"
    if conditions.get('solv1'):
        return True, "Single-solvent system"
    return True, "No solvent recorded (solv1/solv2 null) - permitted"


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
        
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]
    
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
        except (TypeError, KeyError, ValueError) as e:
            # Most commonly caused by malformed/missing request bodies -> treat as client error
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
                # Never leak raw exception text / stack traces to the client in production
                'message': str(e) if debug_on else 'An internal error occurred. Please try again.',
                'traceback': traceback.format_exc() if debug_on else None,
                'timestamp': datetime.now().isoformat()
            }), 500
    return wrapper


def get_json_body():
    """Safely parse the JSON request body. Never raises; always returns a dict."""
    try:
        data = request.get_json(silent=True, force=False)
    except Exception:
        data = None
    if not isinstance(data, dict):
        return {}
    return data


def to_float(value, default=None, field_name='value'):
    """Coerce a request field to float, raising a clean ValueError with a useful message."""
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


MAX_SMILES_LENGTH = 2000
MAX_TEXT_FIELD_LENGTH = 500


def clean_text(value, max_length=MAX_TEXT_FIELD_LENGTH):
    """Trim and length-cap a user supplied text field to avoid oversized/abusive payloads."""
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
        logger.success("ConfigManager initialized with FULL XML parse")
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
<suzuki_config version="7.2.6">
    <metadata>
        <version>7.2.6</version>
        <last_updated>2026-07-18</last_updated>
        <author>Molytica AI Team - Full Academic Edition</author>
        <description>Suzuki-Miyaura coupling yield predictor. The model is a heuristic, physically-INSPIRED scoring function: a subset of its inputs (activation energy, Taft/Hammett-type substituent constants, A-values, rate law FORMS) are taken from or consistent with published literature and are labelled [LIT] below; the remaining combination weights, bonuses, and thresholds are engineering design parameters chosen by the authors to produce chemically-plausible monotonic trends and are labelled [PARAM] - they are NOT fitted to, or measured in, any cited publication, and must not be presented as such. See PARAMETER_PROVENANCE.md for the full audit. v7.2.7: corrected a mis-cited reference (Bourouina et al. was Catalysts 2020, 10, 989 - not 2019, 9, 1070 as previously stated); removed an unverifiable specific patent citation for the pseudo-first-order rate constant and replaced it with an honest general statement; added explicit [LIT]/[PARAM] provenance tags throughout. v7.2.4-7.2.6 (retained): experimental mode uses the real named kinetic equation FORMS (Arrhenius / pseudo-first-order / Michaelis-Menten) with ceiling terms removed; increment_factor values are [PARAM] engineering constants calibrated so the combined-yield feature weighting still produces a visible trend, not literature-measured coefficients.</description>
        <dataset_size>15</dataset_size>
        <target_variable>yield</target_variable>
        <target_range_min>0</target_range_min>
        <target_range_max>100</target_range_max>
    </metadata>

    <!-- ============================================================ -->
    <!-- CHEMICAL INTUITION - PHYSICALLY-INSPIRED SCORING PARAMETERS   -->
    <!--                                                                -->
    <!-- PROVENANCE LEGEND (applies to every value below):             -->
    <!--   [LIT]   = the specific number (or the functional form) is   -->
    <!--             taken from, or directly consistent with, a named  -->
    <!--             published source cited in that block.             -->
    <!--   [PARAM] = an engineering design constant (weight, bonus,     -->
    <!--             threshold, penalty) chosen by the authors to shape -->
    <!--             the scoring function. It is inspired by the named  -->
    <!--             theory/framework but is NOT a number measured or   -->
    <!--             reported in the cited publication. Presenting a    -->
    <!--             [PARAM] value as an experimentally-measured or      -->
    <!--             literature-fitted constant would be a misrepresen- -->
    <!--             tation of the model and must be avoided in any      -->
    <!--             academic write-up.                                 -->
    <!-- Full parameter-by-parameter audit: PARAMETER_PROVENANCE.md    -->
    <!-- ============================================================ -->
    <chemical_intuition>

        <!-- ========================================================== -->
        <!-- TEMPERATURE                                                -->
        <!-- [LIT] activation_energy = 63.0 kJ/mol (std dev 11 kJ/mol,  -->
        <!--   confidence coefficient 0.88): the measured global         -->
        <!-- activation energy for the Herrmann-Beller palladacycle-    -->
        <!-- catalyzed Suzuki-Miyaura coupling of 4-iodoacetophenone +  -->
        <!-- phenylboronic acid (oxidative addition rate-determining).  -->
        <!-- CORRECTED CITATION: Bourouina, A.; Oswald, A.; Lido, V.;   -->
        <!-- Dong, L.; Rataboul, F.; Djakovitch, L.; de Bellefon, C.;   -->
        <!-- Meille, V. "Kinetic Study of the Herrmann-Beller           -->
        <!-- Palladacycle-Catalyzed Suzuki-Miyaura Coupling of          -->
        <!-- 4-Iodoacetophenone and Phenylboronic Acid." Catalysts      -->
        <!-- 2020, 10(9), 989. https://doi.org/10.3390/catal10090989   -->
        <!-- (the previous "Catalysts 2019, 9, 1070" citation in this   -->
        <!-- file was an incorrect year/volume/page and has been fixed  -->
        <!-- in v7.2.7 - the article does not exist at that reference). -->
        <!-- This is ONE measured system, not a universal Suzuki-Miyaura -->
        <!-- constant - reported Ea values for other Pd/ligand/substrate -->
        <!-- combinations range roughly 50-113 kJ/mol in the literature. -->
        <!-- [LIT] the general form "rate increases with temperature     -->
        <!-- following an Arrhenius/Eyring relationship" - Arrhenius, S. -->
        <!-- Z. Phys. Chem. 1889, 4, 226-248; Eyring, H. J. Chem. Phys.  -->
        <!-- 1935, 3, 107-115.                                          -->
        <!-- [PARAM] every weight/bonus/threshold below (temp_coefficient, -->
        <!-- logarithmic_gain, linear_gain, sigmoidal_gain/steepness/    -->
        <!-- midpoint, degradation_threshold/penalty/rate, solvent_bp_*, -->
        <!-- optimal_temp_bonus, curve_steepness/asymmetry, eyring_weight, -->
        <!-- temp_increment_factor) is an authors' engineering constant, -->
        <!-- not a literature-reported value.                            -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- TIME                                                       -->
        <!-- [LIT] the functional FORM - integrated pseudo-first-order  -->
        <!-- kinetics, conversion = 1 - exp(-k*t) - is standard chemical -->
        <!-- kinetics (see e.g. Fersht, A. Structure and Mechanism in    -->
        <!-- Protein Science, 1999, for the general treatment).          -->
        <!-- [PARAM] rate_constant = 0.10/h (implying ~91% conversion by -->
        <!-- 24h) is NOT taken from a specific measured Suzuki-Miyaura    -->
        <!-- system. It was chosen because published Suzuki-Miyaura       -->
        <!-- protocols commonly report near-complete conversion within    -->
        <!-- roughly 1-24 h under standard batch conditions, but the      -->
        <!-- true rate constant is strongly substrate/catalyst/ligand/    -->
        <!-- base-dependent and no single universal value exists.         -->
        <!-- v7.2.7: removed a specific patent citation ("US Patent        -->
        <!-- 7687640") that was attached to this constant in a prior       -->
        <!-- version but could not be independently verified as            -->
        <!-- supporting this exact number - citing it would have been a    -->
        <!-- fabricated/unverifiable attribution, so it has been dropped   -->
        <!-- in favour of this honest, uncited engineering justification.  -->
        <!-- If exact literature grounding is required, this constant      -->
        <!-- should be re-fit against a specific published time-course      -->
        <!-- dataset for the target substrate/catalyst pair before any     -->
        <!-- academic claim is made about it.                              -->
        <!-- [PARAM] every other weight/bonus/threshold below              -->
        <!-- (time_coefficient, logarithmic_factor, linear_factor,          -->
        <!-- plateau_factor, diminishing_returns, early/late_time_          -->
        <!-- multiplier, long_time_bonus, optimal_time_bonus, reaction_     -->
        <!-- order, half_life_temperature_dependence, diffusion_limited_    -->
        <!-- rate, time_increment_factor) is an engineering constant, not   -->
        <!-- a literature-reported value.                                   -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- CATALYST                                                   -->
        <!-- [LIT] the functional FORM - saturation kinetics             -->
        <!-- q/(k_m+q) - is borrowed BY ANALOGY from enzyme kinetics:    -->
        <!-- Michaelis, L.; Menten, M. L. Biochem. Z. 1913, 49, 333-369. -->
        <!-- Using Michaelis-Menten for a homogeneous Pd catalyst        -->
        <!-- loading (rather than an enzyme/substrate system) is a       -->
        <!-- modelling CHOICE, not a literature-validated equivalence -   -->
        <!-- it is a reasonable saturation-curve analogy, not a proven    -->
        <!-- mechanism for catalyst-loading-vs-yield in Suzuki-Miyaura.   -->
        <!-- [PARAM] k_m = 0.0008 mol is a plausible order-of-magnitude   -->
        <!-- estimate (modern XPhos/SPhos-type ligand systems are widely  -->
        <!-- reported to reach useful turnover at loadings as low as      -->
        <!-- ~0.05-0.1 mol%), not a value measured or fitted for a         -->
        <!-- specific published dataset.                                  -->
        <!-- [PARAM] all remaining weights/bonuses/factors below           -->
        <!-- (v_max, turnover_number, turnover_frequency,                  -->
        <!-- catalyst_efficiency, early/late_catalyst_multiplier,          -->
        <!-- ligand_pd_ratio, ligand_bite_angle, ligand_* factors,         -->
        <!-- high/optimal_quantity_bonus, mono/bidentate/bulky/electron_   -->
        <!-- rich_ligand_factor, catalyst_increment_factor) are engineering-->
        <!-- constants, not literature-reported values.                    -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- HAMMETT-DRIVEN ELECTRONIC PROXY (renamed from "DFT" - this  -->
        <!-- is NOT a DFT/quantum-chemistry module; see rationale below) -->
        <!-- [LIT] the DEFINITIONS of chemical potential, hardness, and -->
        <!-- electrophilicity from HOMO/LUMO energies are the standard   -->
        <!-- conceptual-DFT framework of Parr, R. G.; Pearson, R. G.     -->
        <!-- J. Am. Chem. Soc. 1983, 105, 7512-7516. That citation is    -->
        <!-- genuine and stays - only the misleading "dft_" naming was    -->
        <!-- removed from the code and this file.                        -->
        <!-- [PARAM] IMPORTANT: base_homo_energy/base_lumo_energy below  -->
        <!-- are NOT outputs of an actual DFT calculation on the input   -->
        <!-- molecule (no quantum-chemical calculation is performed by   -->
        <!-- this code) - they are fixed placeholder values shifted by   -->
        <!-- the Hammett sigma_p constant as a cheap proxy. All weights/  -->
        <!-- thresholds (chemical_potential_weight, hardness_weight,     -->
        <!-- fukui_weight, fukui_*_threshold, electrophilicity/           -->
        <!-- nucleophilicity_threshold, homo/lumo/gap_energy_factor,      -->
        <!-- homo_lumo_correlation) are engineering constants. If real     -->
        <!-- HOMO/LUMO values are required for an academic claim, they     -->
        <!-- must be computed per-molecule with an actual DFT package      -->
        <!-- (e.g. Gaussian, ORCA, PySCF) rather than read from this block. -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- MECHANISTIC STEP BARRIERS (illustrative, not calculated)   -->
        <!-- [PARAM] IMPORTANT CORRECTION: no B3LYP/6-31G* (or any other) -->
        <!-- DFT calculation is actually run by this codebase. The        -->
        <!-- barrier values below (oxidative_addition_barrier = 26.5      -->
        <!-- kJ/mol, transmetalation_barrier = 20.8, reductive_           -->
        <!-- elimination_barrier = 17.2) are illustrative numbers in the   -->
        <!-- range commonly reported for Pd(0)/Pd(II) cross-coupling       -->
        <!-- cycles in the computational literature, chosen to be          -->
        <!-- chemically plausible and consistent with oxidative addition   -->
        <!-- typically being rate-limiting - they are NOT the output of     -->
        <!-- a calculation performed on the user's specific input           -->
        <!-- molecule and must not be described as "real DFT-calculated."   -->
        <!-- For an academic claim, these barriers should be replaced by     -->
        <!-- literature values for the specific catalyst/substrate system    -->
        <!-- being modelled, with individual citations, or by an actual      -->
        <!-- DFT calculation.                                                -->
        <!-- [PARAM] all sensitivity/weight/rate/bonus terms below are        -->
        <!-- engineering constants, not measured values.                      -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- A "docking" block used to live here (hydrophobic/electro-  -->
        <!-- static/H-bond/entropic terms styled after AutoDock Vina).   -->
        <!-- It has been REMOVED, not renamed: docking scores a small    -->
        <!-- molecule against a protein binding pocket, and this model   -->
        <!-- has no protein target - a homogeneous Pd-catalyzed Suzuki   -->
        <!-- coupling between two small molecules has no meaningful      -->
        <!-- "docking score" to approximate. No AutoDock Vina run (or    -->
        <!-- any other docking software) ever executed in this codebase. -->
        <!-- ========================================================== -->

        <!-- ========================================================== -->
        <!-- STERIC                                                     -->
        <!-- [LIT] a_value_methyl/ethyl/isopropyl/tertbutyl/cyclohexyl/  -->
        <!-- phenyl are standard textbook A-values (cyclohexane           -->
        <!-- conformational free-energy values), broadly consistent with -->
        <!-- Eliel, E. L.; Wilen, S. H. Stereochemistry of Organic        -->
        <!-- Compounds, 1994, and are genuinely literature-grounded.       -->
        <!-- [PARAM] every other threshold/penalty/factor below           -->
        <!-- (penalty_coefficient, ring/bulky/ortho/meta/para              -->
        <!-- substituent penalties, molecular_volume_threshold,            -->
        <!-- volume_penalty_factor, rotatable_bond_penalty/threshold,      -->
        <!-- fused/bridged/spiro_ring_penalty, heavy_atom/halogen_steric_  -->
        <!-- factor) is an engineering constant chosen to give plausible   -->
        <!-- monotonic penalties, not a value reported in the cited text.  -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- ELECTRONIC                                                 -->
        <!-- [LIT] sigma_m/p_electron_withdrawing/donating and the       -->
        <!-- taft_es_* values are broadly consistent with the compiled    -->
        <!-- Hammett sigma and Taft steric (Es) constants in Hansch, C.;  -->
        <!-- Leo, A.; Taft, R. W. Chem. Rev. 1991, 91, 165-195, and are    -->
        <!-- genuinely literature-grounded (treat as representative        -->
        <!-- literature values, not necessarily the exact per-substituent  -->
        <!-- entry - verify the specific substituent against the compiled  -->
        <!-- table before citing an exact figure in a paper).               -->
        <!-- [PARAM] every weight/bonus/penalty below (hammett/taft_        -->
        <!-- coefficient, sigma_plus/minus_coefficient, brown_sigma_plus_   -->
        <!-- factor, hammett_reaction_constant, lfer_weight, electron_      -->
        <!-- donating_bonus/withdrawing_penalty, conjugation/inductive/     -->
        <!-- resonance_effect, polarity_factor, solubility_threshold/       -->
        <!-- penalty, logp_coefficient, hbd_penalty, hba_bonus) is an        -->
        <!-- engineering constant, not a literature-reported value.          -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- HSAB                                                       -->
        <!-- [LIT] the qualitative Hard/Soft Acid-Base FRAMEWORK (soft   -->
        <!-- species prefer soft partners) is Pearson, R. G. J. Am.       -->
        <!-- Chem. Soc. 1963, 85, 3533-3539.                              -->
        <!-- [PARAM] IMPORTANT: the specific numeric softness/hardness/   -->
        <!-- chemical-potential/electronegativity values below (pd_       -->
        <!-- softness=2.8, halide_softness=3.2, etc.) are illustrative     -->
        <!-- placeholders on a self-defined scale, not values tabulated    -->
        <!-- in the cited paper or any absolute-hardness reference table   -->
        <!-- for these specific species - do not present them as measured  -->
        <!-- or literature-tabulated numbers. All match/bonus/penalty       -->
        <!-- weights below are engineering constants.                       -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- SOLVENT                                                    -->
        <!-- [LIT] the descriptor CATEGORIES (dielectric constant,       -->
        <!-- donor number, Kamlet-Taft alpha/beta/pi*, Reichardt's ET(30) -->
        <!-- polarity, Hildebrand solubility parameter) are real,          -->
        <!-- established solvent-polarity scales: Reichardt, C. Chem.      -->
        <!-- Rev. 1994, 94, 2319-2358; Kamlet, M. J.; Taft, R. W. J. Am.    -->
        <!-- Chem. Soc. 1976, 98, 377-383 (beta scale; the pi* scale        -->
        <!-- follows in Kamlet, Abboud, Taft, J. Am. Chem. Soc. 1977, 99,   -->
        <!-- 6027-6038, and the alpha scale in J. Am. Chem. Soc. 1976, 98,  -->
        <!-- 2886-2894 - this file's original single citation was correct  -->
        <!-- for the beta scale but the full alpha/beta/pi* triad spans     -->
        <!-- these three papers, not one).                                  -->
        <!-- [PARAM] the specific optimal/range/weight values and all        -->
        <!-- solvent-mixture bonus factors below are engineering choices,     -->
        <!-- not values measured for this reaction or reported in the         -->
        <!-- cited papers.                                                     -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- BASE                                                       -->
        <!-- [LIT] pKa as a general predictor of base strength in         -->
        <!-- proton-transfer equilibria is standard physical organic       -->
        <!-- chemistry; Bordwell, F. G. Acc. Chem. Res. 1988, 21, 456-463  -->
        <!-- is a real, relevant compilation of pKa values, though it       -->
        <!-- covers acidity in DMSO generally rather than being a study     -->
        <!-- specific to Suzuki-Miyaura base selection.                     -->
        <!-- [PARAM] pka_threshold and every bonus/penalty/weight below     -->
        <!-- (strong/weak_base_bonus/penalty, inorganic/organic/carbonate/  -->
        <!-- phosphate/acetate/fluoride_base_factor, soluble/insoluble/      -->
        <!-- hygroscopic_base_bonus/penalty, cation_radius_effect, pka/pkb_ -->
        <!-- effect, and the four *_weight terms) are engineering            -->
        <!-- constants, not values reported in the cited source.             -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- PHYSICOCHEMICAL PROXY (renamed/redesigned from "QSAR" -     -->
        <!-- see predict_ml_routes.py CHANGELOG v7.4.0)                  -->
        <!-- [LIT] mw/logp/tpsa/rotatable_bonds are genuine RDKit-        -->
        <!-- computed descriptors for the actual input molecule.          -->
        <!-- [REMOVED] the previous version scored these against          -->
        <!-- Lipinski/Ghose/Veber/QED drug-likeness pass-fail cutoffs -     -->
        <!-- real, correctly-cited rules, but derived for oral drug         -->
        <!-- bioavailability, not Suzuki-Miyaura yield. Deleted rather       -->
        <!-- than kept-with-a-caveat.                                        -->
        <!-- [PARAM] What remains: a plain continuous symmetric penalty       -->
        <!-- for each descriptor's distance from a configurable "typical      -->
        <!-- substrate" center (mw_center/logp_center/tpsa_center/             -->
        <!-- rotbonds_center), scaled by *_scale and combined by *_weight -     -->
        <!-- openly an engineering-constant proxy, no borrowed rule names.      -->
        <!-- ========================================================== -->
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

        <!-- ========================================================== -->
        <!-- YIELD PARAMETERS                                           -->
        <!-- [DATA] yield_mean/yield_std/yield_skewness/yield_kurtosis   -->
        <!-- should be computed FROM the actual uploaded training         -->
        <!-- dataset (currently: 15 rows, smiles_..._suzuki_dataset.csv) -->
        <!-- and re-generated whenever that dataset changes - with only   -->
        <!-- 15 rows these are rough sample statistics, not population     -->
        <!-- parameters, and should be reported with that caveat in any    -->
        <!-- write-up. excellent/good/moderate/poor/very_poor_threshold    -->
        <!-- and base_yield_offset/reproducibility_factor/scale_up_factor  -->
        <!-- are [PARAM] engineering choices for labelling/UI purposes,     -->
        <!-- not statistically derived cutoffs.                             -->
        <!-- ========================================================== -->
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

    <!-- ============================================================ -->
    <!-- DATA INTEGRITY - Academic rules, no random/synthetic data   -->
    <!-- ============================================================ -->
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

    <!-- ============================================================ -->
    <!-- MONOTONICITY - Academic guarantee for reaction conditions   -->
    <!-- Note: Only enforced within the same catalyst/base/solvent/  -->
    <!-- substrate system. Does not assume monotonicity across       -->
    <!-- different chemical systems.                                 -->
    <!-- ============================================================ -->
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

    <!-- ============================================================ -->
    <!-- MODEL PARAMETERS - Full ML ensemble                         -->
    <!-- ============================================================ -->
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

    <!-- ============================================================ -->
    <!-- FEATURE IMPORTANCE - Academic weights                       -->
    <!-- ============================================================ -->
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
        <!-- docking_effects removed (see CHANGELOG); its 0.01 default is
             folded into physchem_effects below so the total still sums to ~1.0 -->
        <physchem_effects>0.03</physchem_effects>
    </feature_importance>

    <!-- ============================================================ -->
    <!-- DATA PROCESSING - Academic standards                        -->
    <!-- ============================================================ -->
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

    <!-- ============================================================ -->
    <!-- OPTIMIZATION - Full academic                                -->
    <!-- ============================================================ -->
    <optimization>

        <top_candidates>15</top_candidates>

        <catalyst_search>
            <min_quantity>0.0001</min_quantity>
            <max_quantity>0.50</max_quantity>
            <step_size>0.0005</step_size>
            <n_candidates>20</n_candidates>
            <log_scale>true</log_scale>
        </catalyst_search>

        <grid_search>
            <enabled>true</enabled>
            <n_candidates>50</n_candidates>
            <n_jobs>1</n_jobs>
            <scoring>neg_mean_squared_error</scoring>
            <refit>true</refit>
        </grid_search>

        <bayesian>
            <enabled>true</enabled>
            <n_iterations>30</n_iterations>
            <n_initial_points>10</n_initial_points>
            <acquisition_function>ei</acquisition_function>
            <acq_optimizer>auto</acq_optimizer>
            <random_state>42</random_state>
        </bayesian>

        <genetic_algorithm>
            <enabled>true</enabled>
            <population_size>30</population_size>
            <generations>40</generations>
            <mutation_rate>0.1</mutation_rate>
            <crossover_rate>0.8</crossover_rate>
            <tournament_size>3</tournament_size>
        </genetic_algorithm>

    </optimization>

    <!-- ============================================================ -->
    <!-- PERFORMANCE METRICS - Full academic                         -->
    <!-- ============================================================ -->
    <performance_metrics>

        <metrics>
            <r2>true</r2>
            <mae>true</mae>
            <rmse>true</rmse>
            <mape>true</mape>
            <max_error>true</max_error>
            <explained_variance>true</explained_variance>
            <median_absolute_error>true</median_absolute_error>
            <mean_squared_log_error>true</mean_squared_log_error>
            <r2_adjusted>true</r2_adjusted>
            <aic>true</aic>
            <bic>true</bic>
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

        <learning_curve>
            <enabled>true</enabled>
            <train_sizes>0.1,0.3,0.5,0.7,0.9</train_sizes>
            <n_jobs>1</n_jobs>
            <cv_folds>3</cv_folds>
        </learning_curve>

        <statistical_tests>
            <anova>true</anova>
            <tukey_hsd>true</tukey_hsd>
            <levene>true</levene>
            <shapiro_wilk>true</shapiro_wilk>
            <kruskal_wallis>true</kruskal_wallis>
            <mann_whitney>true</mann_whitney>
            <chi_square>true</chi_square>
            <kolmogorov_smirnov>true</kolmogorov_smirnov>
        </statistical_tests>

        <calibration>
            <enabled>true</enabled>
            <method>isotonic</method>
            <cv_folds>3</cv_folds>
            <spline_degree>3</spline_degree>
        </calibration>

        <uncertainty>
            <method>bootstrap</method>
            <n_bootstrap>100</n_bootstrap>
            <confidence_level>0.95</confidence_level>
            <prediction_interval_level>0.90</prediction_interval_level>
            <mc_dropout>true</mc_dropout>
            <n_dropout_samples>50</n_dropout_samples>
        </uncertainty>

    </performance_metrics>

    <!-- ============================================================ -->
    <!-- VISUALIZATION - Full academic                               -->
    <!-- ============================================================ -->
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
            <highlight_reaction_centers>true</highlight_reaction_centers>
        </molecule_images>

        <plots>
            <feature_importance>true</feature_importance>
            <actual_vs_predicted>true</actual_vs_predicted>
            <residuals>true</residuals>
            <learning_curve>true</learning_curve>
            <parity_plot>true</parity_plot>
            <shap_values>true</shap_values>
            <partial_dependence>true</partial_dependence>
            <prediction_distribution>true</prediction_distribution>
            <residual_qq>true</residual_qq>
            <coefficient_plot>true</coefficient_plot>
            <calibration_plot>true</calibration_plot>
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

    <!-- ============================================================ -->
    <!-- LOGGING - Full academic                                     -->
    <!-- ============================================================ -->
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

    <!-- ============================================================ -->
    <!-- SECURITY - Full academic                                    -->
    <!-- ============================================================ -->
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

    <!-- ============================================================ -->
    <!-- EXPERIMENTAL MODE                                             -->
    <!-- [LIT] the FUNCTIONAL FORMS used here (Arrhenius temperature     -->
    <!-- dependence, integrated pseudo-first-order time dependence,      -->
    <!-- Michaelis-Menten-style saturation for catalyst loading) are      -->
    <!-- real named kinetic equations, used with their artificial          -->
    <!-- ceiling/plateau/degradation penalty terms switched off, instead    -->
    <!-- of an ad-hoc growth curve.                                          -->
    <!-- [PARAM] base_offset and increment_factor for each variable are      -->
    <!-- NOT literature-fitted coefficients - they are engineering            -->
    <!-- constants, kept deliberately separate from the standard-mode         -->
    <!-- coefficients of the same physical variable, and calibrated so         -->
    <!-- that the combined-yield feature weighting (temperature/time/           -->
    <!-- catalyst each contribute roughly 13-21% of the total weight in           -->
    <!-- the standard-mode model) still produces a clearly visible,                -->
    <!-- monotonic yield change across the full valid input range                   -->
    <!-- (including sub-ranges such as 150->200 C). This is a UI/                     -->
    <!-- pedagogical design choice, not a claim that these are measured               -->
    <!-- rate constants for a specific reaction.                                       -->
    <!-- "Experimental" in this codebase means "ceiling terms disabled",                 -->
    <!-- not "empirically measured" - do not describe this mode's output                -->
    <!-- as experimentally validated in any academic material.                           -->
    <!-- ============================================================ -->
    <experimental_mode>
        <enabled>false</enabled>
        <description>Theoretical maximum catalytic scenario within physically realistic operating bounds: no substrate decomposition, no catalyst deactivation, no product inhibition modelled. Temperature, time, and catalyst quantity each drive yield via their real named kinetic law (Arrhenius, pseudo-first-order, Michaelis-Menten respectively), just with the decomposition/plateau/diffusion ceiling terms removed. Inputs remain constrained to the physically valid range for solution-phase Suzuki-Miyaura chemistry (see validation below) - "experimental" describes which ceiling terms are switched off, not permission to submit chemically impossible conditions.</description>

        <temperature>
            <!-- factor = base_offset + increment_factor * normalized, where -->
            <!-- normalized = ln[k(T)/k(T_ref)] / ln[k(T_max)/k(T_ref)] and    -->
            <!-- ln[k(T)/k(T_ref)] = Ea/R * (1/T_ref - 1/T). [LIT] this is the -->
            <!-- real, log-space Arrhenius equation (Arrhenius, S. Z. Phys.    -->
            <!-- Chem. 1889, 4, 226-248), using Ea = 63.0 kJ/mol [LIT, see      -->
            <!-- corrected citation in the <temperature> block above:            -->
            <!-- Bourouina et al., Catalysts 2020, 10(9), 989]. Log-space          -->
            <!-- normalization (rather than the raw rate ratio against the          -->
            <!-- ceiling temperature) spreads sensitivity more evenly across         -->
            <!-- the 25-250 C window instead of compressing it toward 250 C.         -->
            <!-- [PARAM] increment_factor = 3.50 and base_offset = 0.30 are           -->
            <!-- engineering constants (see EXPERIMENTAL MODE header note              -->
            <!-- above) - not measured or literature-fitted values. -->
            <base_offset>0.30</base_offset>
            <increment_factor>3.50</increment_factor>
            <reference_temp>25</reference_temp>
            <min_factor>0.10</min_factor>
            <no_degradation_ceiling>true</no_degradation_ceiling>
            <no_arrhenius_dropoff>true</no_arrhenius_dropoff>
        </temperature>

        <time>
            <!-- factor = base_offset + increment_factor * (1 - exp(-k*t)). [LIT] -->
            <!-- this is the real integrated pseudo-first-order rate law. k        -->
            <!-- (rate_constant = 0.10/h, see <time> block above) is a [PARAM]      -->
            <!-- engineering estimate, not a value measured for a specific           -->
            <!-- published system (see honest justification and caveat above).       -->
            <!-- Saturates toward base_offset + increment_factor as t -> infinity;    -->
            <!-- this is complete conversion, not an artificial cap.                   -->
            <!-- [PARAM] increment_factor = 1.80, base_offset = 0.55 are engineering    -->
            <!-- constants, not literature-fitted values. -->
            <base_offset>0.55</base_offset>
            <increment_factor>1.80</increment_factor>
            <reference_time>1</reference_time>
            <min_factor>0.10</min_factor>
            <no_saturation_ceiling>true</no_saturation_ceiling>
            <no_plateau_factor>true</no_plateau_factor>
        </time>

        <catalyst>
            <!-- factor = base_offset + increment_factor * (q / (k_m + q)). [LIT] -->
            <!-- this is the real Michaelis-Menten saturation FORM, used here BY  -->
            <!-- ANALOGY (see honesty note in <catalyst> block above - this is a   -->
            <!-- modelling choice, not a proven mechanism for Pd loading). k_m       -->
            <!-- (0.0008 mol) is a [PARAM] order-of-magnitude estimate loosely        -->
            <!-- informed by reports that modern XPhos/SPhos-type ligand systems       -->
            <!-- reach useful turnover at loadings as low as ~0.05-0.1 mol%, not a      -->
            <!-- value fitted to a specific dataset. Saturates toward base_offset +      -->
            <!-- increment_factor as q -> infinity; this is the assumed v_max               -->
            <!-- turnover limit, not an artificial cap.                                       -->
            <!-- [PARAM] increment_factor = 2.20, base_offset = 0.50 are engineering           -->
            <!-- constants, not literature-fitted values. -->
            <base_offset>0.50</base_offset>
            <increment_factor>2.20</increment_factor>
            <reference_quantity>0.0001</reference_quantity>
            <min_factor>0.10</min_factor>
            <no_degradation_threshold>true</no_degradation_threshold>
            <no_mm_saturation>true</no_mm_saturation>
        </catalyst>

        <validation>
            <!-- v7.2.2: experimental mode removes the artificial KINETIC ceiling -->
            <!-- (degradation/plateau/saturation penalties) but NOT the physical -->
            <!-- operating limits of solution-phase chemistry. Values beyond these -->
            <!-- are not "theoretical maximum yield" scenarios, they are chemically -->
            <!-- meaningless (solvents boil/decompose, ligands and Pd complexes -->
            <!-- thermally degrade well below 300 C). Enforced server-side in -->
            <!-- predict_ml_routes.py - do not rely on the client to enforce this. -->
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
            <warning>Results represent theoretical upper bounds. Real reactions will be limited by substrate decomposition, catalyst deactivation, and side reactions not modelled here.</warning>
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
        logger.debug(f"Parsed {len(self.params)} top-level sections")
    
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
            'chemical_intuition/temperature/baseline_temp',
            'chemical_intuition/time/baseline_time',
            'chemical_intuition/catalyst/baseline_quantity'
        ]
        missing = []
        for path in required:
            val = self.get(path)
            if val is None:
                missing.append(path)
        if missing:
            logger.warning(f"Required parameters missing: {', '.join(missing)}")
    
    def _compute_xml_hash(self):
        self._xml_hash = hashlib.md5(self.raw_xml.encode()).hexdigest()
    
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
            'cache_size': len(self._cache)
        }


class ChemicalCalculator:
    """
    Core chemical intuition engine for Suzuki-Miyaura coupling prediction.
    
    All parameters are sourced from the XML configuration file and referenced
    to peer-reviewed literature where applicable.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self._load_all_params()
        self._load_feature_importance()
        self._load_experimental_bounds()
        self._validate_params()
        logger.success("ChemicalCalculator initialized with FULL XML params")
        logger.info(f"   Temperature optimal: {self.optimal_temp}C")
        logger.info(f"   Time optimal: {self.optimal_time}h")
        logger.info(f"   Catalyst Km: {self.k_m} mmol")
        logger.info(f"   Max yield: {self.max_yield}%")
    
    def _load_experimental_bounds(self):
        """
        Load the physically real operating bounds for experimental mode.

        v7.2.2 fix: experimental mode disables the artificial KINETIC ceiling
        (degradation penalty / plateau / diffusion limit) so that higher temp,
        time or catalyst quantity always increases yield monotonically. That is
        a modelling choice. It is NOT permission to submit chemically impossible
        conditions (e.g. 1000-1200 C) - solvents boil away, ligands and Pd
        complexes decompose, and glassware fails well below that. These bounds
        are the real physical operating envelope for solution-phase
        Suzuki-Miyaura chemistry and are enforced in SuzukiPredictor.predict(),
        regardless of experimental_mode.
        """
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
        """Load all chemical parameters from XML configuration"""
        chem = self.config.get_chemical_params()
        logger.debug("Loading all chemical parameters from XML...")
        
        temp = chem.get('temperature', {})
        self.optimal_temp = temp.get('optimal_temp', 85)
        self.temp_range = temp.get('temp_range', 35)
        self.baseline_temp = temp.get('baseline_temp', temp.get('min_temp', 40))
        self.min_temp = temp.get('min_temp', self.baseline_temp)
        self.max_temp = temp.get('max_temp', 150)
        self.low_temp_penalty = temp.get('low_temp_penalty', 0.65)
        self.high_temp_penalty = temp.get('high_temp_penalty', 0.40)
        self.degradation_threshold = temp.get('degradation_threshold', 130)
        self.too_low_threshold = temp.get('too_low_threshold', 50)
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
        logger.debug(f"   Temperature params loaded: optimal={self.optimal_temp}, range={self.temp_range}")
        
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
        logger.debug(f"   Time params loaded: optimal={self.optimal_time}, range={self.time_range}")
        
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
        logger.debug(f"   Catalyst params loaded: Km={self.k_m}, Vmax={self.v_max}")
        
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
        logger.debug(f"   Electronic-proxy params loaded (Hammett-based, NOT DFT): base_homo={self.base_homo_energy}, base_lumo={self.base_lumo_energy}")
        
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
        logger.debug(f"   Mechanistic params loaded: OA={self.oa_barrier}, TM={self.tm_barrier}, RE={self.re_barrier}")
        
        
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
        logger.debug(f"   Physicochemical proxy params loaded (NOT drug-likeness rules): mw_center={self.pcp_mw_center}")
        

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
        logger.debug(f"   Yield params loaded: max={self.max_yield}, min={self.min_yield}")
        
        logger.info("All 200+ academic parameters loaded from XML")
    
    def _load_feature_importance(self):
        """Load feature importance weights from XML"""
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
        logger.debug(f"Feature importance weights: temp={self.temp_weight}, time={self.time_weight}, catalyst={self.catalyst_weight}")
    
    def _validate_params(self):
        """Validate that critical parameters are reasonable"""
        if self.optimal_temp <= 0:
            logger.warning(f"Optimal temp ({self.optimal_temp}) must be positive")
        if self.optimal_time <= 0:
            logger.warning(f"Optimal time ({self.optimal_time}) must be positive")
        if self.optimal_temp < self.min_temp or self.optimal_temp > self.max_temp:
            logger.warning(f"Optimal temp ({self.optimal_temp}) outside legacy min-max reference range")
        if self.optimal_time < self.min_time or self.optimal_time > self.max_time:
            logger.warning(f"Optimal time ({self.optimal_time}) outside legacy min-max reference range")
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
    
    
    def calculate_electronic_proxy_parameters(self, smiles: str) -> Dict:
        """
        Calculate a Hammett-sigma-based PROXY for electronic descriptors
        from SMILES (chemical_potential, hardness, electrophilicity, etc.).

        [PARAM] This is NOT a DFT calculation and does not approximate one
        for a specific molecule - it is a linear function of the Hammett
        sigma_p constant for the substituent, dressed up in conceptual-DFT
        terminology. If per-molecule electronic descriptors are needed for
        an academic claim, run an actual DFT package (ORCA, Gaussian,
        PySCF, ...) on the specific input molecule; do not cite this
        function's output as DFT-derived.

        Source for the DEFINITIONS only (not the numbers): Parr, R. G.;
        Pearson, R. G. J. Am. Chem. Soc. 1983, 105, 7512-7516.
        """
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
    
    
    def calculate_physicochemical_proxy_parameters(self, smiles: str) -> Dict:
        """
        Calculate raw physicochemical descriptors from SMILES, plus a
        continuous "typical substrate" proxy score.

        [LIT] mw/logp/tpsa/hba/hbd/rotatable_bonds are genuine RDKit-
        computed values for the actual input molecule - real data, not
        placeholders.
        [REMOVED in v7.4.0] this function used to also score these
        descriptors against Lipinski/Ghose/Veber Rule-of-Five-style
        pass/fail cutoffs and a QED composite score. Those rules are
        genuine and were correctly cited (Lipinski, C. A. et al. Adv.
        Drug Deliv. Rev. 1997, 23, 3-25; Ghose, A. K. et al. J. Comb.
        Chem. 1999, 1, 55-68; Veber, D. F. et al. J. Med. Chem. 2002, 45,
        2615-2623; Bickerton, G. R. et al. Nat. Chem. 2012, 4, 90-98) -
        but they were derived for oral drug bioavailability, not
        Suzuki-Miyaura coupling yield, so applying an "MW > 500 fails"
        style cutoff here had no chemical justification even with a
        disclaimer. They have been deleted, not kept-with-a-caveat.
        [PARAM] proxy_score below is a plain, symmetric, continuous
        penalty for each descriptor's distance from a configurable
        "typical substrate" center (see <physicochemical_proxy> in
        info.xml) - an openly engineering-constant proxy with no
        borrowed rule names or thresholds.
        """
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
    
    
    
    def calculate_eyring_rate(self, temp: float, delta_h: float, delta_s: float) -> float:
        """Calculate reaction rate using Eyring equation."""
        T = temp + 273.15
        return (boltzmann_k / h) * T * np.exp(-(delta_h * 1000) / (R * T)) * np.exp(delta_s / R)
    
    def calculate_gibbs_energy(self, temp: float, delta_h: float, delta_s: float) -> float:
        """Calculate Gibbs free energy."""
        T = temp + 273.15
        return delta_h - T * delta_s / 1000
    
    def calculate_equilibrium_constant(self, temp: float, delta_g: float) -> float:
        """Calculate equilibrium constant from Gibbs energy."""
        T = temp + 273.15
        return np.exp(-delta_g * 1000 / (R * T))
    
    def calculate_lfer(self, sigma: float, rho: float) -> float:
        """Calculate Linear Free Energy Relationship factor."""
        return np.exp(rho * sigma)
    
    
    def calculate_hsab_absolute_hardness(self, ip: float, ea: float) -> float:
        """Calculate absolute hardness (η = IP - EA)."""
        return ip - ea
    
    def calculate_hsab_chemical_potential(self, ip: float, ea: float) -> float:
        """Calculate chemical potential (μ = -(IP + EA)/2)."""
        return -(ip + ea) / 2
    
    def calculate_hsab_electronegativity(self, ip: float, ea: float) -> float:
        """Calculate electronegativity (χ = (IP + EA)/2)."""
        return (ip + ea) / 2
    
    
    def calculate_solvent_kamlet_taft(self, alpha: float, beta: float, pi_star: float) -> float:
        """
        Calculate Kamlet-Taft solvent parameter.
        
        Source: Kamlet, M. J.; Taft, R. W. J. Am. Chem. Soc. 1976, 98, 377-383.
        """
        return alpha * 0.3 + beta * 0.3 + pi_star * 0.4
    
    
    def calculate_steric_a_value(self, substituent: str) -> float:
        """Calculate A-value (steric parameter) for a substituent."""
        a_values = {
            'methyl': self.a_value_methyl,
            'ethyl': self.a_value_ethyl,
            'isopropyl': self.a_value_isopropyl,
            'tertbutyl': self.a_value_tertbutyl
        }
        return a_values.get(substituent.lower(), 1.74)
    
    def calculate_hammett_sigma_plus(self, substituent: str) -> float:
        """Calculate σ+ (Brown's sigma plus) for a substituent."""
        sigma_plus = HAMMETT_SIGMA.get(substituent, {}).get('sigma_plus', 0)
        return sigma_plus * self.sigma_plus_coeff
    
    def calculate_hammett_sigma_minus(self, substituent: str) -> float:
        """Calculate σ- (sigma minus) for a substituent."""
        sigma_minus = HAMMETT_SIGMA.get(substituent, {}).get('sigma_minus', 0)
        return sigma_minus * self.sigma_minus_coeff
    
    def calculate_taft_steric_parameter(self, substituent: str) -> float:
        """Calculate Taft steric parameter (Es) for a substituent."""
        return HAMMETT_SIGMA.get(substituent, {}).get('taft_es', 0)
    
    
    def calculate_oxidative_addition_barrier(self, sigma_p: float, steric_bulk: float) -> float:
        """Calculate an oxidative-addition barrier estimate. [PARAM] The base
        value and sensitivity terms are illustrative literature-range
        placeholders, NOT DFT output for the input molecule - see the
        <mechanistic> block in info.xml."""
        barrier = self.oa_barrier + (sigma_p * self.oa_barrier_sigma_effect) + (steric_bulk * self.oa_barrier_steric_effect)
        return max(0, barrier)
    
    def calculate_transmetalation_barrier(self, base_strength: float, boronic_bulk: float) -> float:
        """Calculate a transmetalation barrier estimate. [PARAM] Illustrative
        literature-range placeholder, NOT DFT output - see the
        <mechanistic> block in info.xml."""
        barrier = self.tm_barrier + (base_strength * self.tm_barrier_base_effect) + (boronic_bulk * self.tm_barrier_boronic_effect)
        return max(0, barrier)
    
    def calculate_reductive_elimination_barrier(self, steric_bulk: float) -> float:
        """Calculate a reductive-elimination barrier estimate. [PARAM]
        Illustrative literature-range placeholder, NOT DFT output - see the
        <mechanistic> block in info.xml."""
        barrier = self.re_barrier + (steric_bulk * self.re_barrier_steric_effect)
        return max(0, barrier)
    
    
    def temperature_factor(self, temp: float, experimental: bool = False) -> float:
        """
        Calculate temperature factor using Arrhenius equation.

        Standard mode: no artificial "max_temp" ceiling, but genuine thermal
        decomposition kinetics (Arrhenius-based smooth decay past
        degradation_threshold) are applied as real chemistry.

        Experimental mode (v7.2.3): degradation ceiling, Arrhenius bell curve,
        and Eyring correction against an "optimal" temperature are disabled.
        The growth law is the real Arrhenius rate ratio k(T)/k(T_ref), passed
        through its natural logistic saturation — higher temperature always
        yields a strictly higher factor, with diminishing returns, bounded
        so it does not diverge across the 25-250 C window. Models the
        theoretical maximum catalytic activation scenario where side-reactions
        and substrate decomposition are assumed negligible.

        Sources:
        - Arrhenius, S. Z. Phys. Chem. 1889, 4, 226-248.
        - Eyring, H. J. Chem. Phys. 1935, 3, 107-115.
        """
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
    
    def time_factor(self, time_hours: float, experimental: bool = False) -> float:
        """
        Calculate time factor using pseudo-first-order kinetics.

        Standard mode: pseudo-first-order saturation kinetics — reaction
        progress flattens naturally past the saturation point (diminishing
        returns) with no hard stop.

        Experimental mode (v7.2.3): plateau_factor and diffusion-limit
        penalties are disabled. The growth law is the real integrated
        pseudo-first-order conversion 1 - exp(-k*t) — longer time always
        yields a strictly higher factor, naturally saturating toward complete
        conversion (not an artificial cap, since yield cannot exceed 100%
        conversion). Models the theoretical scenario of a perfectly sustained
        reaction with no product inhibition or catalyst deactivation.

        Source: Fersht, A. Structure and Mechanism in Protein Science, 1999.
        """
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
    
    def catalyst_factor(self, quantity: float, experimental: bool = False) -> float:
        """
        Calculate catalyst factor using Michaelis-Menten kinetics.

        Standard mode: catalytic activity is modeled via Michaelis-Menten
        saturation kinetics — naturally approaches a plateau (diminishing
        returns) with a degradation penalty beyond the threshold quantity.

        Experimental mode (v7.2.3): degradation threshold penalty is disabled.
        The growth law is the real Michaelis-Menten saturation q/(k_m+q) —
        more catalyst always yields a strictly higher factor, naturally
        saturating toward v_max (not an artificial cap, since turnover rate
        cannot exceed the catalyst's intrinsic maximum). Models the ideal
        scenario of a perfectly stable, fully soluble catalyst with no ligand
        decomposition, poisoning, or aggregation at high loading.

        Source: Michaelis, L.; Menten, M. L. Biochem. Z. 1913, 49, 333-369.
        """
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
    
    def steric_factor(self, conditions: Dict) -> float:
        """
        Calculate steric factor using Eliel A-values.
        
        Source: Eliel, E. L. et al. Stereochemistry of Organic Compounds, 1994.
        """
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
        """
        Calculate electronic factor using Hammett/Taft LFER.
        
        Source: Hansch, C.; Leo, A.; Taft, R. W. Chem. Rev. 1991, 91, 165-195.
        """
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
        """
        Calculate HSAB compatibility factor.
        
        Source: Pearson, R. G. J. Am. Chem. Soc. 1963, 85, 3533-3539.
        """
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
    
    def solvent_factor(self, solv1: str, solv2: str = '') -> float:
        """
        Calculate solvent factor using Kamlet-Taft parameters.
        
        Source: Reichardt, C. Chem. Rev. 1994, 94, 2319-2358.
        """
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
    
    def base_factor(self, base: str) -> float:
        """
        Calculate base factor using pKa and solubility properties.
        
        Source: Bordwell, F. G. Acc. Chem. Res. 1988, 21, 456-463.
        """
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
    
    def mechanistic_factor(self, conditions: Dict) -> Dict:
        """
        Calculate mechanistic factor using illustrative step-barrier constants.

        [PARAM] NOT a DFT calculation - no quantum-chemistry code runs here.
        The barrier constants (oxidative_addition_barrier=26.5, etc.) are
        illustrative values in the range typically reported for Pd(0)/Pd(II)
        cross-coupling cycles in the computational literature, chosen for
        chemical plausibility. See PARAMETER_PROVENANCE.md.
        """
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
    
    def elecproxy_factor(self, conditions: Dict) -> float:
        """
        Calculate a Hammett-sigma-driven electronic PROXY factor.

        [PARAM] NOT a DFT calculation. No quantum-chemistry package (e.g.
        Gaussian, ORCA, PySCF) runs anywhere in this codebase. HOMO/LUMO
        "energies" here are a linear function of the Hammett sigma_p
        constant (base_homo_energy - sigma_p * homo_shift_factor), not
        per-molecule quantum-chemical output. Only the DEFINITIONS of
        chemical potential / hardness / electrophilicity from HOMO-LUMO
        gaps are the genuine conceptual-DFT framework of Parr, R. G.;
        Pearson, R. G. J. Am. Chem. Soc. 1983, 105, 7512-7516 - the numbers
        fed into those definitions are engineering placeholders, not DFT
        output. If real HOMO/LUMO values are needed for a publication, run
        an actual DFT package on the specific input molecule.
        """
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
    
    def physchem_factor(self, conditions: Dict) -> float:
        """
        Calculate a continuous "typical substrate" physicochemical proxy
        factor from MW/LogP/TPSA/rotatable-bond count.

        [PARAM] Renamed/redesigned from "qsar_factor" in v7.4.0: the old
        version scored these descriptors against Lipinski/Ghose/Veber
        Rule-of-Five-style pass/fail cutoffs plus a QED composite score -
        real, correctly-cited rules, but derived for oral drug
        bioavailability, with no established relationship to Suzuki-
        Miyaura yield. That borrowed-authority scoring has been removed.
        What remains is a plain symmetric penalty for each descriptor's
        distance from a configurable "typical substrate" center (see
        <physicochemical_proxy> in info.xml) - an openly engineering-
        constant proxy, not a drug-likeness rule.
        """
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
    
    
    
    def calculate_yield(self, conditions: Dict) -> Dict:
        """
        Calculate yield using the full chemical intuition engine.
        
        This combines all factors (temperature, time, catalyst, steric,
        electronic, HSAB, solvent, base, mechanistic, Hammett-electronic-
        proxy, physicochemical proxy) using the feature importance weights
        from the XML configuration. "docking" was removed entirely (see
        CHANGELOG) as a category error, not renamed like the DFT-labelled
        and QSAR-labelled terms were.
        """
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
    
    def get_yield_class(self, yield_val: float) -> Tuple[str, str]:
        """Classify yield into qualitative categories."""
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
        """Get yield statistics."""
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
        logger.success("FeatureEngineer initialized with FULL 500+ features")
    
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
        """Extract molecular descriptors from SMILES string."""
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
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw dataframe."""
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
        
        logger.info(f"Feature engineering complete: {len(df.columns)} columns (was {original_cols})")
        return df
    
    def select_features(self, df: pd.DataFrame, target: str = 'yield') -> pd.DataFrame:
        """Select most important features using mutual information."""
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
        self.shap_values = None
        self.calibration_model = None
        self.confidence_intervals = None
        self.fallback_model = None
        self.training_data_warning = None
        self.honest_cv_performance = None
        self.feature_columns_used = []
        self.min_samples_per_feature = self.config.get_int(
            'ml_training_safeguards/min_samples_per_feature', 10)
        self.min_samples_for_full_ensemble = self.config.get_int(
            'ml_training_safeguards/min_samples_for_full_ensemble', 30)
        self.honest_cv_max_folds = self.config.get_int(
            'ml_training_safeguards/honest_cv_max_folds', 5)
        logger.success("SuzukiPredictor initialized with FULL XML integration")
    
    def _load_weights(self) -> Dict:
        """Load ensemble weights from configuration."""
        try:
            w = self.config.get_dict('model_parameters/Ensemble/weights')
            if w:
                return {k: float(v) for k, v in w.items() if float(v) > 0}
        except Exception as e:
            logger.warning(f"Could not load ensemble weights: {e}")
        
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
    
    def validate_csv(self, filepath: str) -> Tuple[bool, List[str]]:
        """Validate CSV file structure."""
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
        """Load and validate CSV data."""
        valid, missing = self.validate_csv(filepath)
        if not valid:
            missing_required = [m for m in missing if 'optional' not in m]
            if missing_required:
                raise ValueError(f"CSV validation failed. Missing required columns: {', '.join(missing_required)}")
        
        try:
            self.df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(self.df)} rows from {filepath}")
            
            if 'yield' not in self.df.columns:
                raise ValueError("'yield' column not found - cannot train model without yield data")
            
            if self.df['yield'].isnull().all():
                raise ValueError("All yield values are missing - cannot train model")
            
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
            
            usable_df, failed_df, rejected_df = classify_and_filter_rows(self.df)
            self.failed_reactions_df = failed_df
            self.rejected_rows_df = rejected_df
            self.df = usable_df
            
            if len(rejected_df) > 0:
                logger.warning(
                    f"Rejected {len(rejected_df)} row(s) from the dataset due to missing "
                    f"required (non-nullable) fields. These rows were NOT used in any way."
                )
            if len(failed_df) > 0:
                logger.info(
                    f"Recorded {len(failed_df)} row(s) as failed reactions (null yield). "
                    f"Kept for audit only, excluded from regression training."
                )
            
            if len(self.df) < 5:
                raise ValueError(f"Dataset must have at least 5 rows with valid yield data. Current: {len(self.df)}")
            
            self.df = self.fe.engineer_features(self.df)
            
            self._prepare_features()
            return self.df
            
        except Exception as e:
            logger.error(f"Load error: {str(e)}")
            raise
    
    def _prepare_features(self):
        """Prepare features for model training."""
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
            'hsab_soft_soft_interaction_score', 'hsab_pearson_class_match',
            'mechanistic_oxidative_addition_liability', 'reaction_rate_indicator',
            'elecproxy_homo_energy', 'elecproxy_lumo_energy', 'elecproxy_gap_energy',
            'elecproxy_chemical_potential', 'elecproxy_absolute_hardness', 'elecproxy_electrophilicity',
            'elecproxy_fukui_plus', 'elecproxy_fukui_minus',
            'physchem_proxy_score'
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
        X_categorical = _dedupe_columns(X_categorical)
        
        self.X = pd.concat([X_numeric, X_categorical], axis=1) if not X_categorical.empty else X_numeric
        self.y = self.df['yield'].values
        self.feature_columns = [clean_feature_name(c) for c in self.X.columns]
        self.X.columns = self.feature_columns
        self.X = _dedupe_columns(self.X)
        self.feature_columns = list(self.X.columns)
        
        if self.X.isnull().any().any():
            self.X = self.X.fillna(0)
        
        logger.info(f"Prepared {len(self.feature_columns)} features")
    
    def _create_model(self, name: str):
        """Create a scikit-learn model instance."""
        try:
            params = self.config.get_model_params(name)
            
            invalid_params = ['feature_importance_type', 'early_stopping_rounds', 'early_stopping']
            for ip in invalid_params:
                if ip in params:
                    del params[ip]
            
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
                from sklearn.ensemble import ExtraTreesRegressor
                return ExtraTreesRegressor(**params)
            elif name == 'Gaussian_Process' and GP_AVAILABLE:
                if 'kernel' in params and isinstance(params['kernel'], str):
                    try:
                        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
                        params['kernel'] = eval(params['kernel'])
                    except:
                        params['kernel'] = RBF(1.0) + WhiteKernel(0.1)
                return GaussianProcessRegressor(**params)
            elif name == 'SVR':
                from sklearn.svm import SVR
                return SVR(**params)
            elif name == 'Neural_Network':
                from sklearn.neural_network import MLPRegressor
                return MLPRegressor(**params)
            elif name == 'Ridge':
                from sklearn.linear_model import Ridge
                return Ridge(**params)
            elif name == 'Lasso':
                from sklearn.linear_model import Lasso
                return Lasso(**params)
            elif name == 'ElasticNet':
                from sklearn.linear_model import ElasticNet
                return ElasticNet(**params)
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
    

    def train(self, model_type: str = 'Ensemble') -> Dict:
        """Train the ML model ensemble."""
        try:
            if self.X is None or len(self.X) == 0:
                raise ValueError("Data must be loaded first")
            
            if not self.is_enriched:
                raise ValueError(
                    "Cannot train ML model on basic dataset.\n"
                    "Please use dataset_routes.py to enrich your data first."
                )
            
            if len(self.y) == 0 or np.all(np.isnan(self.y)):
                raise ValueError("No valid yield data available for training")

            n_samples = len(self.X)
            n_features_available = len(self.feature_columns)

            if n_samples < self.min_samples_per_feature:
                raise ValueError(
                    f"Refusing to train: only {n_samples} labeled reactions available. "
                    f"A meaningful train/test evaluation needs at least "
                    f"{self.min_samples_per_feature} reactions. With this little data, "
                    f"use the heuristic/experimental scoring mode instead, and report it "
                    f"as a rule-based estimate, not an ML prediction."
                )

            self.training_data_warning = None
            max_features_for_sample_size = max(1, n_samples // self.min_samples_per_feature)
            if n_samples < self.min_samples_for_full_ensemble or n_features_available > max_features_for_sample_size:
                self.training_data_warning = (
                    f"LOW-DATA REGIME: {n_samples} samples for {n_features_available} candidate "
                    f"features (ratio {n_samples / max(n_features_available,1):.2f} samples/feature). "
                    f"Feature count reduced to {max_features_for_sample_size} via univariate "
                    f"selection (SelectKBest/f_regression) to reduce (not eliminate) overfitting "
                    f"risk. Any R2/MAE/RMSE reported from this fit describes in-sample behavior "
                    f"on a tiny, non-representative test split and MUST be reported with this "
                    f"caveat - it is not evidence of generalizable predictive accuracy."
                )
                logger.warning(self.training_data_warning)

            logger.info(f"Starting training with {n_samples} samples, {n_features_available} features")
            
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
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, self.y, test_size=test_size, random_state=42
            )
            
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
                        'Ridge', 'Lasso', 'ElasticNet'
                    ]
            else:
                model_names = [model_type]
            
            for name in model_names:
                try:
                    logger.info(f"Training {name}...")
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
                            
                            logger.success(f"{name}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
                except Exception as e:
                    logger.warning(f"Could not train {name}: {str(e)}")
            
            if not models:
                logger.warning("No models trained successfully. Trying fallback model (Ridge)...")
                try:
                    from sklearn.linear_model import Ridge
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
                        logger.success(f"Ridge_Fallback: R2={performances['Ridge_Fallback']['r2']:.4f}")
                        self.fallback_model = 'Ridge_Fallback'
                except Exception as e:
                    logger.error(f"Fallback model also failed: {e}")
                    return {'success': False, 'message': 'No models could be trained'}
            
            if not models:
                return {'success': False, 'message': 'No models could be trained'}
            
            self.models = models
            self.is_trained = True
            self.model_performance = performances
            
            if performances:
                best_name = max(performances.items(), key=lambda x: x[1].get('r2', 0))[0]
                self.best_model = best_name
                logger.success(f"Best model: {best_name} (R2={performances[best_name]['r2']:.4f})")
            
            self.cv_results = self._perform_cross_validation()
            self.feature_importance = self._calculate_feature_importance()

            self.honest_cv_performance = None
            if self.training_data_warning is not None:
                try:
                    X_raw = self.X.copy()
                    X_raw = pd.DataFrame(
                        SimpleImputer(strategy='mean').fit_transform(X_raw),
                        columns=self.feature_columns
                    )
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
                                logger.warning(f"Honest CV fold failed for {name}: {fold_err}")
                        valid = ~np.isnan(oof_pred)
                        if valid.sum() >= 2:
                            honest_results[name] = {
                                'r2_cv_honest': float(r2_score(y_arr[valid], oof_pred[valid])),
                                'mae_cv_honest': float(mean_absolute_error(y_arr[valid], oof_pred[valid])),
                                'n_folds': n_splits,
                                'n_evaluated': int(valid.sum())
                            }
                    self.honest_cv_performance = honest_results
                    if honest_results:
                        logger.info(f"Honest nested-CV metrics (report these, not the single-split ones): {honest_results}")
                except Exception as e:
                    logger.warning(f"Could not compute honest nested-CV metrics: {e}")

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
    
    def _perform_cross_validation(self) -> Dict:
        """Perform cross-validation on trained models."""
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
                    logger.debug(f"CV failed for {name}: {e}")
            
            return cv_results
            
        except Exception as e:
            logger.warning(f"Cross-validation error: {e}")
            return {}
    
    def _calculate_feature_importance(self) -> Dict:
        """Calculate feature importance using permutation importance."""
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
                return {
                    'top_10': sorted_importance[:10],
                    'all': importance_dict
                }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {}
    
    def _ensemble_predict(self, X) -> np.ndarray:
        """Make ensemble prediction using weighted average."""
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
    
    def _bootstrap_uncertainty(self, X, n_bootstrap: int = 100) -> Dict:
        """Calculate prediction uncertainty using bootstrap."""
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
            
            return {
                'mean': mean_pred,
                'std': std_pred,
                'lower_ci': lower,
                'upper_ci': upper
            }
            
        except Exception as e:
            logger.warning(f"Bootstrap uncertainty calculation failed: {e}")
            return {}
    
    def predict(self, conditions: Dict) -> Dict:
        """
        Make a prediction for given reaction conditions.

        Standard mode: combines chemical intuition and ML predictions, with
        degradation/plateau/diffusion ceiling terms enabled (monotonic only
        up to a physically-motivated optimum, then penalized - see
        PARAMETER_PROVENANCE.md for which constants are [LIT] vs [PARAM]).

        Experimental mode (conditions['experimental_mode'] == True):
        - Physical operating bounds (temp/time/quantity range) still apply -
          "experimental" does not mean chemically-impossible inputs are
          allowed.
        - The ceiling/degradation/plateau/diffusion-limit terms are disabled;
          each factor instead uses the real named kinetic equation FORM with
          no ceiling: Arrhenius temperature dependence, integrated pseudo-
          first-order time dependence, Michaelis-Menten-style catalyst
          saturation (NOTE: this docstring previously and incorrectly said
          "arbitrary logarithmic growth model" - that described an older,
          already-removed implementation; see v7.2.3 changelog above).
        - This still strictly increases with input (no artificial cap), but
          via the real kinetic FORM, not an ad-hoc curve.
        - Final yield is clipped to [0, 100] only; the 100% hard physical
          ceiling still applies as a thermodynamic constraint.
        - increment_factor/base_offset constants used here are [PARAM]
          engineering calibrations, not literature-fitted values - do not
          describe experimental-mode output as "experimentally validated."
        """
        try:
            experimental = bool(conditions.get('experimental_mode', False))
            if experimental:
                logger.info("Making EXPERIMENTAL MODE prediction (real kinetic forms, no artificial ceiling)...")
            else:
                logger.info("Making standard-mode prediction (chemical intuition + ML ensemble)...")

            temp_in = conditions.get('temp', 80)
            time_in = conditions.get('time', 24)
            qty_in = conditions.get('quantity', 0.0025)
            bounds_violations = []
            if temp_in < self.chemical.exp_temp_min or temp_in > self.chemical.exp_temp_max:
                bounds_violations.append(
                    f"temp={temp_in} (izin verilen aralık: {self.chemical.exp_temp_min}-{self.chemical.exp_temp_max} C)"
                )
            if time_in < self.chemical.exp_time_min or time_in > self.chemical.exp_time_max:
                bounds_violations.append(
                    f"time={time_in} (izin verilen aralık: {self.chemical.exp_time_min}-{self.chemical.exp_time_max} h)"
                )
            if qty_in < self.chemical.exp_quantity_min or qty_in > self.chemical.exp_quantity_max:
                bounds_violations.append(
                    f"quantity={qty_in} (izin verilen aralık: {self.chemical.exp_quantity_min}-{self.chemical.exp_quantity_max})"
                )
            if bounds_violations:
                return {
                    'success': False,
                    'message': (
                        "Girilen değer(ler) gerçek Suzuki-Miyaura tepkime koşullarının fiziksel "
                        "sınırları dışında, deneysel mod dahil kabul edilemez: " + "; ".join(bounds_violations)
                    )
                }

            last_pred = get_last_prediction_for_conditions(conditions)
            
            elecproxy_params = self.chemical.calculate_electronic_proxy_parameters(conditions.get('subs1_smiles', ''))
            elecproxy_params2 = self.chemical.calculate_electronic_proxy_parameters(conditions.get('subs2_smiles', ''))
            physchem_params = self.chemical.calculate_physicochemical_proxy_parameters(conditions.get('subs1_smiles', ''))
            
            chemical_result = self.chemical.calculate_yield(conditions)
            chemical_yield = chemical_result['yield']
            
            logger.debug(f"Chemical yield: {chemical_yield:.4f}%")
            
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
                        logger.debug(f"ML yield: {ml_yield:.4f}%")
                except Exception as e:
                    logger.warning(f"ML prediction failed: {str(e)}")
            
            if ml_yield is not None and not np.isnan(ml_yield) and self.is_enriched:
                base_yield = 0.6 * ml_yield + 0.4 * chemical_yield
                model_name = 'Ensemble'
            else:
                base_yield = chemical_yield
                model_name = 'Chemical Intuition'
                logger.info("Using only Chemical Intuition")
            
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
                
                increase_applied = False
                increase_details = []
                
                if prev_temp is not None and current_temp > prev_temp:
                    temp_increase = calculate_logarithmic_increase(prev_temp, current_temp, prev_yield, max_increase=2.0)
                    if temp_increase > 0:
                        final_yield += temp_increase
                        increase_applied = True
                        increase_details.append(f"temp +{temp_increase:.4f}% (log)")
                        logger.debug(f"Temperature increase: {temp_increase:.4f}%")
                
                if prev_time is not None and current_time > prev_time:
                    time_increase = calculate_logarithmic_increase(prev_time, current_time, prev_yield, max_increase=1.5)
                    if time_increase > 0:
                        final_yield += time_increase
                        increase_applied = True
                        increase_details.append(f"time +{time_increase:.4f}% (log)")
                        logger.debug(f"Time increase: {time_increase:.4f}%")
                
                if prev_quantity is not None and current_quantity > prev_quantity:
                    qty_increase = calculate_logarithmic_increase(prev_quantity, current_quantity, prev_yield, max_increase=1.5)
                    if qty_increase > 0:
                        final_yield += qty_increase
                        increase_applied = True
                        increase_details.append(f"catalyst +{qty_increase:.4f}% (log)")
                        logger.debug(f"Catalyst increase: {qty_increase:.4f}%")
                
                if increase_applied:
                    logger.info(f"Logarithmic increases applied: {', '.join(increase_details)}")
                    logger.info(f"Yield increased from {base_yield:.4f}% to {final_yield:.4f}%")
            
            dominated = get_dominated_history_max_yield(conditions)
            if dominated is not None and final_yield < dominated['yield']:
                logger.info(
                    f"Monotonicity floor enforced: raising {final_yield:.4f}% to "
                    f"{dominated['yield']:.4f}% (matches or exceeds a prior run at "
                    f"temp={dominated['temp']}, time={dominated['time']}, "
                    f"quantity={dominated['quantity']})"
                )
                final_yield = dominated['yield']
            
            final_yield = np.clip(final_yield, 0, 100)
            
            confidence_interval = None
            prediction_interval = None
            
            if uncertainty:
                ci_lower = final_yield - 1.96 * uncertainty.get('std', [0])[0]
                ci_upper = final_yield + 1.96 * uncertainty.get('std', [0])[0]
                confidence_interval = {
                    'lower': max(0, ci_lower),
                    'upper': min(100, ci_upper)
                }
                
                pi_lower = final_yield - 1.645 * uncertainty.get('std', [0])[0]
                pi_upper = final_yield + 1.645 * uncertainty.get('std', [0])[0]
                prediction_interval = {
                    'lower': max(0, pi_lower),
                    'upper': min(100, pi_upper)
                }
            
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
            logger.success(f"Final prediction: {final_yield:.4f}% ({yield_class}){exp_label}")

            experimental_details = None
            if experimental:
                experimental_details = {
                    'mode': 'experimental',
                    'description': (
                        'Physical operating bounds still apply (see validation range). '
                        'Degradation thresholds, saturation/plateau penalties, and diffusion '
                        'limits are disabled; each of the three factors instead uses the real '
                        'named kinetic equation form with no ceiling: Arrhenius temperature '
                        'dependence, integrated pseudo-first-order time dependence, and '
                        'Michaelis-Menten-style catalyst saturation. Increasing any of the '
                        'three parameters always increases yield (up to the 100% physical '
                        'cap). increment_factor/base_offset constants are engineering '
                        'calibrations, not literature-fitted values - see PARAMETER_PROVENANCE.md.'
                    ),
                    'temp_factor_raw': float(chemical_result.get('temp_factor', 1.0)),
                    'time_factor_raw': float(chemical_result.get('time_factor', 1.0)),
                    'catalyst_factor_raw': float(chemical_result.get('catalyst_factor', 1.0)),
                    'monotonicity_guaranteed': True,
                    'upper_bound': '100 % (thermodynamic limit only)'
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
    
    def _create_feature_vector(self, conditions: Dict) -> np.ndarray:
        """Create feature vector from reaction conditions."""
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
    
    def _calculate_confidence(self, ml_yield: float, chemical_yield: float, final_yield: float) -> float:
        """Calculate prediction confidence score."""
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
    
    def optimize_catalyst(self, conditions: Dict) -> List[Tuple[str, float, float]]:
        """Optimize catalyst selection for given conditions."""
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
                    'Pd(PhCN)2Cl2', 'Pd(PPh3)4', 'PdCl2(MeCN)2',
                    'PdCl2(COD)', 'Pd(TFA)2', 'Pd(OPiv)2'
                ]
            
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
    
    def get_best_model(self) -> Tuple[str, Dict]:
        """Get the best performing model."""
        if not self.model_performance:
            return None, {}
        best = max(self.model_performance.items(), key=lambda x: x[1].get('r2', 0))
        return best[0], best[1]
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance scores."""
        return self.feature_importance
    
    def analyze_residuals(self) -> Dict:
        """Analyze prediction residuals."""
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
            logger.warning(f"Residual analysis failed: {e}")
            return {}
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model to file."""
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
            logger.success(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Save model error: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model from file."""
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
            
            logger.success(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Load model error: {e}")
            return False


@predict_ml_bp.route('/api/prediction_history', methods=['GET'])
@error_handler
def get_prediction_history():
    """Get prediction history."""
    load_prediction_history()
    return jsonify({
        'success': True,
        'history': PREDICTION_HISTORY,
        'count': len(PREDICTION_HISTORY)
    })

@predict_ml_bp.route('/api/clear_history', methods=['POST'])
@error_handler
def clear_prediction_history():
    """Clear prediction history."""
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = []
    if os.path.exists(PREDICTION_HISTORY_FILE):
        os.remove(PREDICTION_HISTORY_FILE)
    return jsonify({
        'success': True,
        'message': 'Prediction history cleared'
    })

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
                'is_enriched': False
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
            'required_columns': REQUIRED_COLUMNS
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
            'visualizations': {
                'created': len(images) > 0,
                'image_count': len(images),
                'directory': os.path.dirname(images[0]) if images else None
            },
            'is_enriched': bool(PREDICTOR.is_enriched),
            'fallback_used': result.get('fallback_used', False)
        })
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
    
    data = get_json_body()
    model_name = clean_text(data.get('model_name'), max_length=64)
    
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

    # Only allow known model names/keys through — never forward an arbitrary
    # client-supplied string as a lookup key into the trainer.
    allowed_keys = set(model_map.values())
    key = model_map.get(model_name, model_name)
    if key not in allowed_keys:
        return jsonify({'success': False, 'message': f'Unknown model: {model_name}'})

    logger.info(f"Changing model to: {key}")
    
    if PREDICTOR.df is not None:
        logger.info("Re-preparing features before training...")
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
            'visualizations': {
                'created': len(images) > 0,
                'image_count': len(images),
                'directory': os.path.dirname(images[0]) if images else None
            },
            'fallback_used': result.get('fallback_used', False)
        })
    else:
        return jsonify({
            'success': False,
            'message': result.get('message', 'Failed to train model')
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
        return jsonify({'success': False, 'message': 'Solvent 1 is required for Suzuki-Miyaura reaction'})

    try:
        temp = to_float(data['temp'], field_name='temp')
        time_h = to_float(data['time'], field_name='time')
        quantity = to_float(data['quantity'], field_name='quantity')
        sigma_m = to_float(data.get('sigma_m', 0), default=0.0, field_name='sigma_m')
        sigma_p = to_float(data.get('sigma_p', 0), default=0.0, field_name='sigma_p')
        taft_es = to_float(data.get('taft_es', 0), default=0.0, field_name='taft_es')
    except ValueError as e:
        return jsonify({'success': False, 'message': str(e)}), 400

    logger.info(f"Prediction request: temp={temp}, time={time_h}, catalyst={data['catalizor']}")

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

    logger.info(f"Optimization request: temp={temp}, time={time_h}")

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
    return jsonify({
        'success': True,
        'feature_importance': importance.get('top_10', []),
        'all_features': importance.get('all', {})
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
        'cache_size': len(CACHE),
        'cache_hit': CACHE_HIT,
        'cache_miss': CACHE_MISS,
        'log_count': len(logger.logs),
        'fallback_used': PREDICTOR.fallback_model is not None if PREDICTOR else False,
        'history_count': len(PREDICTION_HISTORY)
    })


def create_result_images():
    """Create visualization images for model results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
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
            logger.warning("No predictions available for visualization")
            return []
        
        r2 = r2_score(y_true, ensemble_pred)
        mae = mean_absolute_error(y_true, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        ax1.scatter(y_true, ensemble_pred, alpha=0.6, s=50, color='#2563EB')
        min_val = min(y_true.min(), ensemble_pred.min())
        max_val = max(y_true.max(), ensemble_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Yield (%)', fontsize=12)
        ax1.set_ylabel('Predicted Yield (%)', fontsize=12)
        ax1.set_title(f'Parity Plot\nR² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = axes[0, 1]
        residuals = y_true - ensemble_pred
        n, bins, patches = ax2.hist(residuals, bins=20, color='#8B5CF6', edgecolor='white', alpha=0.7, density=True)
        mu, std = np.mean(residuals), np.std(residuals)
        x_normal = np.linspace(residuals.min(), residuals.max(), 100)
        y_normal = (1/(std * np.sqrt(2*np.pi))) * np.exp(-(x_normal - mu)**2 / (2*std**2))
        ax2.plot(x_normal, y_normal, 'r-', linewidth=2, label='Normal Distribution')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Residual', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title(f'Residual Distribution\nMean = {mu:.4f}, Std = {std:.4f}', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
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
                
                ax3.set_xlabel('Models', fontsize=12)
                ax3.set_ylabel('Score / Error', fontsize=12)
                ax3.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
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
                
                ax4.set_xlabel('Importance', fontsize=12)
                ax4.set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold')
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
                    rect = plt.Rectangle(
                        (child.get_x(), child.get_y()),
                        child.get_width(), child.get_height(),
                        facecolor=child.get_facecolor(),
                        edgecolor='white',
                        linewidth=1.5
                    )
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
            logger.info(f"Saved plot: {filepath}")
        
        plt.close('all')
        
        info_file = os.path.join(images_dir, 'info.txt')
        with open(info_file, 'w') as f:
            f.write(f"Performance Visualizations\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Models: {len(PREDICTOR.models)}\n")
            f.write(f"Best Model: {PREDICTOR.best_model}\n")
            f.write(f"R² Score: {r2:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"Images: {', '.join([os.path.basename(f) for f in plot_files])}\n")
        
        logger.success(f"Created {len(plot_files)} visualization images")
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
        logger.info("Default config created")
    
    load_prediction_history()


init_app()
