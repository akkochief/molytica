![Model](https://github.com/akkochief/molytica/blob/main/img/Molytica4.png)
# Molytica — Suzuki-Miyaura Reaction Analyzer

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9cf.svg)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow.svg)](https://catboost.ai)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-brightgreen.svg)](https://www.rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A Flask-based analysis platform for Suzuki-Miyaura cross-coupling reactions. It takes reaction data (substrates, catalyst, base, solvent, temperature/time) and produces yield predictions using both a **physicochemical model** (an Eyring/Hammett/Taft/HSAB-based rule engine) and a **13-model ML ensemble**.

This README was rewritten after actually reading `routes/predict_ml_routes.py` — everything below reflects what the code actually does, not aspirational feature-list marketing copy.

## What it actually does

### 1. Chemistry engine — `ChemicalCalculator`
The code has embedded reference tables (Hammett sigma constants, Taft steric parameters, ligand properties like cone angle/TEP/Tolman angle, base pKa/solubility values, Kamlet-Taft solvent parameters). From these it computes:
- Eyring-equation reaction rate, Gibbs free energy, and equilibrium constant
- LFER (Linear Free Energy Relationships) from Hammett σ (σm, σp, σ+, σ−) and Taft Es steric parameters
- HSAB (hard-soft acid-base) compatibility: absolute hardness, chemical potential, electronegativity
- A rule-based `calculate_yield()` function that combines temperature, time, catalyst loading, steric, electronic, HSAB, and mechanistic factors into an "expected" yield — independent of the ML models

### 2. Feature engineering — `FeatureEngineer`
`engineer_features()` derives 80+ features from the raw CSV:
- Temperature×time interactions (product, ratio, log, difference, etc.)
- Log/sqrt/square/cube/exponential transforms of catalyst quantity
- SMILES-derived features (length and molecular descriptors) for substrates, product, catalyst, base, and both solvents
- Steric, LogP, molecular weight, ring count, and halogen count differences/ratios/sums between substrates
- Composite signals like `hammett_effect`, `taft_effect`, and `mechanistic_predictor` derived from Hammett σp and Taft Es

Note: this pipeline only runs on **enriched** datasets — `is_enriched_dataset()` checks for columns like `subs1_SMILES_*` and `hsab_*`; if they're missing, training is refused and the user is pointed to `dataset_routes.py` to enrich the data first.

### 3. ML prediction engine — `SuzukiPredictor`
Calling `train()` fits 13 models at once and compares them on a held-out test set using R², MAE, and RMSE:

`Random Forest, Gradient Boosting, Hist Gradient Boosting, XGBoost, LightGBM, CatBoost, Extra Trees, KNN, Ridge, Lasso, ElasticNet, SVR, Neural Network (MLP)`

Features are scaled with `StandardScaler`, missing values filled with `SimpleImputer`. Categorical columns (`catalizor`, `base`, `solv1`, `solv2`) are one-hot encoded. The best model is auto-selected by R²; there's also a weighted **ensemble average** (e.g. Random Forest 18%, Hist Gradient Boosting 18%, XGBoost 14%, etc.). The final prediction blends the ML output with `ChemicalCalculator`'s rule-based yield and returns a confidence score (`_calculate_confidence`).

### 4. Configuration — `ConfigManager` (XML)
Model hyperparameters, ensemble weights, and chemistry constants are read from `config/info.xml`, not hardcoded. The file is auto-created with defaults if missing, and can be read, updated, and reset via the API.

### 5. Other infrastructure
- A simple in-memory `Logger` (INFO/SUCCESS/DEBUG/WARNING/ERROR levels), queryable via `/api/get_logs`
- A TTL + max-size `cache_result` decorator with hit/miss counters
- Saving/loading trained models to disk with `joblib`
- Exporting results as CSV, JSON, or Excel
- A SMILES validation endpoint

## Real API endpoints (`/predict_ml/...`)

| Endpoint | Method | What it does |
|---|---|---|
| `/api/get_csv_files` | GET | Lists CSVs under `static/datasets/` |
| `/api/upload_csv` | POST | Uploads a CSV |
| `/api/load_data` | POST | Reads the CSV, checks if it's enriched, runs feature engineering |
| `/api/change_model` / `/api/save_model` / `/api/load_model` / `/api/list_models` | — | Model training/saving/loading |
| `/api/make_prediction` | POST | Yield prediction for a single reaction |
| `/api/optimize_catalyst` | POST | Suggests the best catalyst for given conditions |
| `/api/model_performance` / `/api/model_comparison` | — | R²/MAE/RMSE comparison |
| `/api/feature_importance` | GET | Feature importance ranking |
| `/api/get_xml_config` / `/api/update_xml_config` / `/api/reset_xml_config` | — | XML config management |
| `/api/validate_smiles` | POST | SMILES validation |
| `/api/export_results` | POST | CSV/JSON/Excel export |
| `/api/get_logs` / `/api/clear_logs` | — | View/clear application logs |
| `/api/health` | GET | Predictor/config/data status, cache and log counters |

## Project structure (actual directory tree)

```
molytica/
├── main.py
├── requirements.txt
├── config/
│   └── info.xml                # model parameters, ensemble weights, chemistry constants
├── routes/
│   ├── predict_ml_routes.py    # the ML/chemistry engine this README is based on
│   ├── predict_routes.py
│   ├── dataset_routes.py       # data enrichment (produces the enriched dataset)
│   ├── compare_routes.py
│   ├── csv_routes.py
│   ├── xlsx_routes.py
│   ├── manual_routes.py
│   └── help_routes.py
├── static/
│   ├── datasets/                # raw and enriched CSVs
│   ├── models/                  # trained models saved via joblib
│   └── images/YYYYMMDD_HHMMSS/  # parity/residual/importance plots per run
└── templates/
    ├── index.html
    ├── predict.html
    └── predict_ml.html
```

> `predict_routes.py`, `dataset_routes.py`, `compare_routes.py`, `csv_routes.py`, `xlsx_routes.py`, `manual_routes.py`, and `help_routes.py` were not read in this review; the descriptions above come from filenames and from what `predict_ml_routes.py` references (e.g. the "dataset not enriched" error points users to `dataset_routes.py`).

## Installation

```bash
git clone https://github.com/akkochief/molytica.git
cd molytica
pip install -r requirements.txt
python main.py
```

The app runs at `http://127.0.0.1:5000`.

## Expected CSV format (raw data, before enrichment)

| Column | Description | Example |
|---|---|---|
| `Ar-B(OH)2` | Boronic acid | `phenylboronic acid` |
| `Ar-X` | Aryl halide | `bromobenzene` |
| `product` | Expected product | `1,1'-biphenyl` |
| `catalizor` | Catalyst SMILES | `I[Pd](I)([N]1=CC=CC=C1)...` |
| `base` | Base SMILES/name | `k2co3` |
| `solv1`, `solv2` | Solvents | `water`, `propan-2-ol` |
| `amount` | Catalyst amount (mol) | `0.0025` |
| `centigrades` | Temperature (°C) | `40` |
| `minute` | Time (min) | `120` |
| `cycle` | Reaction cycle number | `88` |
| `yield` | Experimental yield (%) | `81` |

`predict_ml` doesn't accept a raw CSV for training directly — it first needs the enriched columns (`subs1_SMILES_*`, `hsab_*`, etc.) added by `dataset_routes.py`; otherwise `load_data()` raises an error.

## Scientific foundation

- Suzuki-Miyaura mechanism: Pd(0)/Pd(II) catalytic cycle
- Hammett/Taft LFER parameters
- HSAB theory (hardness, chemical potential)
- Eyring transition-state kinetics, Gibbs free energy
- Kamlet-Taft solvent parameters (α, β, π*)
- Random Forest / gradient boosting family (XGBoost, LightGBM, CatBoost, Hist-GB) and classical regression models (Ridge, Lasso, ElasticNet, SVR, KNN, MLP)

## References

1. Miyaura, N.; Suzuki, A. *Chem. Rev.* **1995**, *95*, 2457-2483.
2. Martin, R.; Buchwald, S. L. *Acc. Chem. Res.* **2008**, *41*, 1461-1473.
3. Fortman, G. C.; Nolan, S. P. *Chem. Soc. Rev.* **2011**, *40*, 5151-5169.
4. Lennox, A. J. J.; Lloyd-Jones, G. C. *Chem. Soc. Rev.* **2014**, *43*, 412-433.
5. Ahneman, D. T.; Estrada, J. G.; Lin, S.; Dreher, S. D.; Doyle, A. G. *Science* **2018**, *360*, 186-190.

## License

MIT License — see [LICENSE](LICENSE)

## Note

This software is intended for academic and research purposes. Contact the developers before commercial use.
