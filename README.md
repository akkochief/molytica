# Molytica — Suzuki-Miyaura Reaction Analyzer

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9cf.svg)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow.svg)](https://catboost.ai)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-brightgreen.svg)](https://www.rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A Flask-based analysis platform for Suzuki-Miyaura cross-coupling reactions. It takes reaction data (substrates, catalyst, base, solvent, temperature/time) and produces yield predictions using both a **physicochemical model** (Eyring/Hammett/Taft/HSAB-based rule engine) and a **13-model ML ensemble** with academic DFT/QSAR/QSPR descriptors.

## Core Capabilities

### 1. Physicochemical Engine — `ChemicalCalculator`

The system includes embedded reference tables for:
- **Hammett sigma constants** (σₘ, σₚ, σ⁺, σ⁻) for substituent effects
- **Taft steric parameters** (Eₛ) for steric effects
- **Ligand properties** (cone angle, TEP, Tolman angle, denticity, pKₐ)
- **Base properties** (pKₐ, solubility, cation radius, hygroscopicity)
- **Kamlet-Taft solvent parameters** (α, β, π*)

Computes:
- **Eyring equation** reaction rate, Gibbs free energy (ΔG), and equilibrium constant
- **LFER** (Linear Free Energy Relationships) from Hammett σ and Taft Eₛ
- **HSAB** (hard-soft acid-base) compatibility: absolute hardness, chemical potential, electronegativity
- **Michaelis-Menten kinetics** for catalyst efficiency
- **A-values** for steric analysis (methyl, ethyl, isopropyl, tert-butyl)

The rule-based `calculate_yield()` combines temperature, time, catalyst loading, steric, electronic, HSAB, solvent, base, and mechanistic factors into an independent yield estimate.

### 2. DFT & QSAR Academic Features

| Class | Features |
|-------|----------|
| **AcademicDFTCalculator** | Fukui indices (f⁺, f⁻, f⁰), HOMO-LUMO energies, chemical potential (μ), absolute hardness (η), electronegativity (χ), electrophilicity index (ω), nucleophilicity index |
| **QSARCalculator** | 2D QSAR descriptors (MW, LogP, TPSA, MR, HBA, HBD, RotBonds), Lipinski/Ghose/Veber rule compliance, QED drug-likeness score |
| **MolecularDockingCalculator** | Predicted binding affinity, hydrophobic/electrostatic/H-bond contributions, entropic penalty |

### 3. Feature Engineering — `FeatureEngineer`

`engineer_features()` derives **500+ features** from enriched CSV data:
- Temperature × time interactions (product, ratio, log transforms, difference)
- Catalyst quantity transforms (log, sqrt, square, cube, exponential, inverse)
- SMILES-derived molecular descriptors for substrates, product, catalyst, base, solvents
- Steric, LogP, MW, ring count, halogen count differences/ratios/sums
- DFT-based composite features (`dft_homo_sum`, `dft_gap_sum`, `dft_chemical_potential_avg`)
- QSAR-derived features (`qsar_mr`, `qsar_drug_likeness`)
- Mechanistic predictors (`hammett_effect`, `taft_effect`, `electronic_softness`)

**Note:** Training requires **enriched datasets** with columns like `subs1_SMILES_*`, `subs2_SMILES_*`, `hsab_*`, `dft_*`, `qsar_*`. Basic datasets trigger an error and redirect to `dataset_routes.py`.

### 4. ML Prediction Engine — `SuzukiPredictor`

`train()` fits **13 models** simultaneously and compares them on a held-out test set using R², MAE, and RMSE:

| Model Family | Specific Models |
|--------------|-----------------|
| Tree-based | Random Forest, Gradient Boosting, Hist Gradient Boosting, Extra Trees |
| Gradient Boosted | XGBoost, LightGBM, CatBoost |
| Linear | Ridge, Lasso, ElasticNet |
| Other | KNN, SVR, Neural Network (MLP), Gaussian Process |

**Ensemble Strategy:**
- Weighted average ensemble (e.g., Random Forest 16%, Hist-GB 16%, XGBoost 12%, LightGBM 8%, CatBoost 8%)
- Stacking with Random Forest as meta-model (optional)
- Soft voting for final predictions

**Academic Analytics:**
- **Cross-validation** (5-fold KFold) with R² scores
- **SHAP** model interpretability (TreeExplainer)
- **ANOVA** statistical analysis (top 5 features)
- **Confidence intervals** (95%, t-distribution)
- **Prediction intervals** (90%, normal distribution)
- **Residual analysis** with Shapiro-Wilk normality test

### 5. Configuration — `ConfigManager` (XML)

All model hyperparameters, ensemble weights, and chemistry constants are read from `config/info.xml`. The file is auto-created with defaults if missing and supports:
- Full XML parsing with caching
- Type-aware value parsing (int, float, bool, string)
- Get/set/update via REST API
- Config backup/restore on update

### 6. Visualization Engine

Automatically generates **4 academic-level plots** after training:
1. **DFT HOMO-LUMO Energy Diagram** — Energy levels for substrates and product with gap values
2. **HSAB Pearson Compatibility Matrix** — Soft-soft/hard-hard matching heatmap
3. **Mechanistic Barrier Analysis** — OA/TM/RE activation barriers, rate constants, sensitivities
4. **QSAR Drug-Likeness Radar Plot** — Lipinski rule compliance with QED scores

## API Endpoints (`/predict_ml/...`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/get_csv_files` | GET | List CSVs in `static/datasets/` |
| `/api/upload_csv` | POST | Upload a CSV file |
| `/api/load_data` | POST | Load and validate enriched CSV, run feature engineering, train ensemble |
| `/api/change_model` | POST | Switch to a specific model (Random Forest, XGBoost, etc.) |
| `/api/save_model` | POST | Save trained model with joblib |
| `/api/load_model` | POST | Load a saved model |
| `/api/list_models` | GET | List saved models |
| `/api/make_prediction` | POST | Predict yield for a single reaction with DFT/QSAR/docking details |
| `/api/optimize_catalyst` | POST | Suggest optimal catalyst for given conditions |
| `/api/model_performance` | GET | Get R²/MAE/RMSE and academic analyses (CV, ANOVA, CIs) |
| `/api/model_comparison` | POST | Compare all 13 model performances |
| `/api/feature_importance` | GET | Feature importance ranking (permutation importance) |
| `/api/get_xml_config` | GET | Get current XML configuration |
| `/api/update_xml_config` | POST | Update XML configuration with validation |
| `/api/reset_xml_config` | POST | Reset XML to defaults |
| `/api/get_config_summary` | GET | Config summary and model list |
| `/api/validate_smiles` | POST | Validate a SMILES string |
| `/api/get_data_info` | POST | Get loaded dataset information |
| `/api/export_results` | POST | Export results as CSV/JSON/Excel |
| `/api/clear_cache` | POST | Clear in-memory cache |
| `/api/get_logs` | GET | Get application logs (filter by level) |
| `/api/clear_logs` | POST | Clear logs |
| `/api/get_yield_stats` | GET | Yield statistics from config |
| `/api/get_model_list` | GET | List available model names |
| `/api/update_visualizations` | POST | Regenerate academic visualizations |
| `/api/health` | GET | Health check with cache/status/academic features |

## Project Structure

```
molytica/
├── main.py
├── requirements.txt
├── config/
│   └── info.xml                     # Full XML config (200+ parameters)
├── routes/
│   ├── predict_ml_routes.py         # ML + Chemistry + Academic features
│   ├── predict_routes.py
│   ├── dataset_routes.py            # Data enrichment (generates academic columns)
│   ├── compare_routes.py
│   ├── csv_routes.py
│   ├── xlsx_routes.py
│   ├── manual_routes.py
│   └── help_routes.py
├── static/
│   ├── datasets/                    # Raw and enriched CSVs
│   ├── models/                      # Trained models (joblib .pkl)
│   └── images/YYYYMMDD_HHMMSS/      # Academic visualizations per run
│       ├── resim1_dft_diagram.png
│       ├── resim2_hsab_heatmap.png
│       ├── resim3_mechanistic_analysis.png
│       └── resim4_qsar_radar.png
└── templates/
    ├── index.html
    ├── predict.html
    └── predict_ml.html
```

## Expected CSV Format (Before Enrichment)

| Column | Description | Example |
|--------|-------------|---------|
| `Ar-B(OH)2` / `subs1` | Boronic acid SMILES | `OB(O)c1ccccc1` |
| `Ar-X` / `subs2` | Aryl halide SMILES | `Brc1ccccc1` |
| `product` | Product SMILES | `c1ccccc1-c2ccccc2` |
| `catalizor` | Catalyst SMILES | `[Pd]` or `OAc-[Pd]-OAc` |
| `base` | Base name/SMILES | `k2co3` |
| `solv1`, `solv2` | Solvent names | `water`, `propan-2-ol` |
| `quantity` / `amount` | Catalyst amount (mol) | `0.0025` |
| `temp` / `centigrades` | Temperature (°C) | `40` |
| `time` / `minute` | Time (min) | `120` |
| `yield` | Experimental yield (%) | `81` |

**Important:** Training requires **enriched data** with academic columns (`subs1_SMILES_*`, `hsab_*`, `dft_*`, etc.) — use `dataset_routes.py` to generate these from a basic CSV.

## Installation

```bash
git clone https://github.com/akkochief/molytica.git
cd molytica
pip install -r requirements.txt
python main.py
```

The app runs at `http://127.0.0.1:5000`.

## Scientific Foundation

- **Suzuki-Miyaura mechanism**: Pd(0)/Pd(II) catalytic cycle (OA, TM, RE)
- **Hammett/Taft LFER**: Substituent effects (σₘ, σₚ, σ⁺, σ⁻, Eₛ)
- **HSAB theory**: Hardness, chemical potential, electrophilicity
- **Eyring transition-state kinetics**: ΔH‡, ΔS‡, reaction rates
- **Kamlet-Taft solvent parameters**: α, β, π*
- **DFT-based descriptors**: HOMO-LUMO, Fukui indices
- **QSAR/QSPR**: 2D descriptors, drug-likeness rules
- **ML Ensemble**: Random Forest, XGBoost, LightGBM, CatBoost, Hist-GB, SVR, KNN, Ridge, Lasso, ElasticNet, MLP, Gaussian Process

## Key Academic Features Summary

| Category | Features |
|----------|----------|
| **Electronic** | Hammett σ (m, p, plus, minus), Taft Eₛ, LFER, conjugation/inductive/resonance effects |
| **Steric** | A-values, ring penalties, ortho/meta/para penalties, rotatable bonds, molecular volume |
| **DFT** | HOMO/LUMO energies, gap, chemical potential, hardness, electrophilicity, Fukui indices |
| **QSAR** | MW, LogP, TPSA, MR, HBA, HBD, RotBonds, QED, drug-likeness score |
| **HSAB** | Softness, hardness, chemical potential, compatibility scores |
| **Kinetics** | Eyring rates, Gibbs energy, equilibrium constants, Michaelis-Menten |
| **Solvent** | Dielectric, donor number, polarity, Kamlet-Taft α/β/π*, Reichardt Eₜ(30) |

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

## Drive 
Link : https://drive.google.com/file/d/14_TyiuhB_WluTO_udYXPrO9DOz4VurRA/view?usp=sharing
