# Molytica — Suzuki-Miyaura Reaction Analyzer

<img width="1280" height="640" alt="image" src="https://github.com/user-attachments/assets/158b1b61-ba79-43e2-ad3c-068b9cbbb88b" />

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-9cf.svg)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow.svg)](https://catboost.ai)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-brightgreen.svg)](https://www.rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

A Flask-based analysis platform for Suzuki-Miyaura cross-coupling reactions.
It takes reaction data (substrates, catalyst, base, solvent, temperature/
time) and produces yield predictions using both a **heuristic
physicochemical model** (Eyring/Hammett/Taft/HSAB-inspired rule engine) and
a **13-model ML ensemble** trained on RDKit-derived molecular descriptors.

> **Read this before citing anything from this repo in a paper, poster, or
> report:** the physicochemical model is a heuristic scoring function, not
> a validated first-principles model, and the shipped training dataset is
> very small (~14-15 rows). Every constant in `config/info.xml` is tagged
> `[LIT]` (a real literature value/equation form, with a citation) or
> `[PARAM]` (an author-chosen engineering constant, not measured or fitted
> to any cited source). **Full audit: [`PARAMETER_PROVENANCE.md`](PARAMETER_PROVENANCE.md).**
> As of v7.4.0, no module in this codebase is named after a real
> computational technique (DFT, docking) unless that technique is actually
> being run — see [What changed in v7.4.0](#what-changed-in-v740) below.

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
- **HSAB** (hard-soft acid-base) compatibility: absolute hardness, chemical potential, electronegativity — the *framework* is Pearson's (1963); the specific numeric softness/hardness values used are illustrative placeholders on a self-defined scale, not a tabulated reference
- **Michaelis-Menten-style saturation** for catalyst loading (used by analogy, not as a literal enzyme-kinetics claim)
- **A-values** for steric analysis (methyl, ethyl, isopropyl, tert-butyl, cyclohexyl, phenyl)
- A **Hammett-sigma-driven electronic proxy** (chemical potential / hardness / electrophilicity / Fukui-index-style terms) — see below
- A **continuous physicochemical proxy** (MW/LogP/TPSA/rotatable-bond-based) — see below
- Three **illustrative mechanistic-barrier estimates** (oxidative addition / transmetalation / reductive elimination), in the range typically reported for Pd(0)/Pd(II) cycles, but not calculated for the specific input molecule

All of the above are combined by `ChemicalCalculator.calculate_yield()` into
a single rule-based yield estimate, independent of the ML ensemble below.

### 2. Electronic & Physicochemical Proxy Descriptors

| Method | What it actually does |
|--------|------------------------|
| `calculate_electronic_proxy_parameters()` | A Hammett-σ-driven **proxy** for HOMO/LUMO-style electronic descriptors (chemical potential, hardness, electrophilicity, Fukui-index-style terms). **This is not a DFT calculation** — no quantum-chemistry package (Gaussian, ORCA, PySCF, ...) runs anywhere in this codebase. Only the *definitions* used (Parr & Pearson conceptual DFT, 1983) are the genuine literature framework; the numbers fed into them are a linear function of the Hammett σₚ constant. |
| `calculate_physicochemical_proxy_parameters()` | Real RDKit-computed MW/LogP/TPSA/HBA/HBD/rotatable-bond descriptors for the actual input molecule, combined into a continuous "distance from a typical-substrate center" proxy score. Prior versions of this codebase scored these against Lipinski/Ghose/Veber Rule-of-Five-style drug-likeness cutoffs and a QED composite score — those rules are real and were correctly cited, but they were derived for oral drug bioavailability, not Suzuki-Miyaura yield, so they have been **removed outright** (v7.4.0), not kept with a disclaimer. |

A "docking" scoring module (hydrophobic/electrostatic/H-bond/entropic terms
styled after AutoDock Vina) existed in earlier versions and has been
**deleted entirely** (v7.4.0), not renamed: molecular docking scores a
small molecule against a protein binding pocket, and this pipeline has no
protein target, so the concept does not apply to a homogeneous
Pd-catalyzed Suzuki coupling between two small molecules. No AutoDock Vina
run, or any other docking engine, has ever executed in this codebase.

### 3. Feature Engineering — `FeatureEngineer`

`engineer_features()` derives several hundred features from the enriched
CSV/SMILES data, including:
- Temperature × time interactions (product, ratio, log transforms, difference)
- Catalyst quantity transforms (log, sqrt, square, cube, exponential, inverse)
- SMILES-derived molecular descriptors for substrates (`subs1_*`/`subs2_*`: MW, LogP, TPSA, rings, aromatic rings, HBA/HBD, rotatable bonds, kappa shape indices, complexity, fraction Csp3, halogen/heteroatom counts, Hammett σ, Taft Eₛ)
- Steric, LogP, MW, ring-count, halogen-count differences/ratios/sums
- Hammett/Taft-derived composite features (`hammett_effect`, `taft_effect`, `electronic_softness`)
- Electronic-proxy composite features (`elecproxy_homo_energy`, `elecproxy_gap_energy`, `elecproxy_chemical_potential`, etc. — see naming note above)
- A single continuous `physchem_proxy_score` feature (see above)

**Note:** Training requires **enriched datasets** with columns like
`subs1_SMILES_*`, `subs2_SMILES_*`, `hsab_*`. Basic datasets trigger an
error and redirect to `dataset_routes.py`.

### 4. ML Prediction Engine — `SuzukiPredictor`

`train()` fits up to **13 models** and compares them on a held-out test set
using R², MAE, and RMSE:

| Model Family | Specific Models |
|--------------|-----------------|
| Tree-based | Random Forest, Gradient Boosting, Hist Gradient Boosting, Extra Trees |
| Gradient Boosted | XGBoost, LightGBM, CatBoost |
| Linear | Ridge, Lasso, ElasticNet |
| Other | KNN, SVR, Neural Network (MLP), Gaussian Process |

**Low-data safeguards (see `<ml_training_safeguards>` in `info.xml`):** with
the shipped dataset (~14-15 rows), fitting all 13 models — several of which
can reach near-zero training error on a handful of rows regardless of any
real signal — and reporting a single train/test split R² as "performance"
is a well-known statistical failure mode. As of this pass, `train()`:
- refuses to fit anything below a configurable minimum sample count,
- restricts the ensemble to Ridge/Lasso/ElasticNet below a configurable
  "full ensemble" sample threshold,
- caps feature count via `SelectKBest` so the sample-to-feature ratio
  stays above a configurable minimum,
- reports an `honest_cv_performance` nested-CV metric (feature selection
  re-fit *inside* each fold) alongside the legacy single-split
  `performance` dict — **the nested-CV number is the one that should be
  cited**, not the single-split number.

**Ensemble strategy:**
- Weighted average ensemble (weights configurable in `info.xml`)
- Optional stacking with Random Forest as meta-model
- Soft voting for final predictions

**Statistical analyses:**
- Cross-validation (K-fold) R² scores
- SHAP model interpretability (TreeExplainer)
- ANOVA on top features
- Confidence intervals (t-distribution) and prediction intervals (normal approximation)
- Residual analysis with Shapiro-Wilk normality test

### 5. Configuration — `ConfigManager` (XML)

All model hyperparameters, ensemble weights, and physicochemical constants
are read from `config/info.xml` — nothing described above is hardcoded as a
"magic number" in the Python source; every weight, threshold, bonus, and
penalty has a corresponding, editable XML key with a `default` fallback
only used if that key is absent. The file is auto-created with defaults if
missing, and supports:
- Full XML parsing with caching
- Type-aware value parsing (int, float, bool, string)
- Get/set/update via REST API
- Config backup/restore on update

**What is and isn't XML-editable:** the *coefficients* (weights, centers,
scales, thresholds, bonuses, penalties) for every factor are XML-editable.
The *functional form* of each factor (e.g. "Arrhenius-shaped", "distance-
from-center penalty", which RDKit descriptors are computed, which 13
algorithms make up the ensemble) is Python code, not XML — that is a
routine config-vs-architecture split, not a limitation specific to this
project. See `PARAMETER_PROVENANCE.md` for exactly which numbers in
`info.xml` are literature values vs. engineering constants.

### 6. Visualization Engine

Generates plots after training, including:
1. **Electronic-proxy HOMO-LUMO energy diagram** — energy levels for substrates/product from the Hammett-based proxy described above (not a DFT-calculated diagram)
2. **HSAB Pearson compatibility matrix** — soft-soft/hard-hard matching heatmap, using the illustrative HSAB placeholder values
3. **Mechanistic barrier analysis** — the three illustrative OA/TM/RE barrier estimates, rate constants, sensitivities
4. **Physicochemical proxy plot** — the continuous MW/LogP/TPSA/rotatable-bond proxy score (replaces the earlier "QSAR drug-likeness radar / Lipinski compliance" plot, which has been removed along with the underlying rule set)

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
| `/api/make_prediction` | POST | Predict yield for a single reaction, with the full factor breakdown (electronic proxy / physicochemical proxy / HSAB / mechanistic / etc.) |
| `/api/optimize_catalyst` | POST | Suggest optimal catalyst for given conditions |
| `/api/model_performance` | GET | Get R²/MAE/RMSE plus `honest_cv_performance` and other statistical analyses (CV, ANOVA, CIs) |
| `/api/model_comparison` | POST | Compare all trained model performances |
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
| `/api/get_yield_stats` | GET | Yield statistics from config (`[DATA]` — should be re-derived whenever the training CSV changes) |
| `/api/get_model_list` | GET | List available model names |
| `/api/update_visualizations` | POST | Regenerate visualizations |
| `/api/health` | GET | Health check with cache/status |

## Project Structure

```
molytica/
├── main.py
├── requirements.txt
├── config/
│   └── info.xml                     # Full XML config (weights, thresholds, citations)
├── PARAMETER_PROVENANCE.md          # [LIT] vs [PARAM] audit for every constant in info.xml
├── routes/
│   ├── predict_ml_routes.py         # ML + physicochemical engine
│   ├── predict_routes.py
│   ├── dataset_routes.py            # Data enrichment (generates derived feature columns)
│   ├── compare_routes.py
│   ├── csv_routes.py
│   ├── xlsx_routes.py
│   ├── manual_routes.py
│   └── help_routes.py
├── static/
│   ├── datasets/                    # Raw and enriched CSVs
│   ├── models/                      # Trained models (joblib .pkl)
│   └── images/YYYYMMDD_HHMMSS/      # Generated visualizations per run
│       ├── resim1_elecproxy_diagram.png
│       ├── resim2_hsab_heatmap.png
│       ├── resim3_mechanistic_analysis.png
│       └── resim4_physchem_proxy_plot.png
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

**Important:** Training requires **enriched data** with derived columns
(`subs1_SMILES_*`, `hsab_*`, etc.) — use `dataset_routes.py` to generate
these from a basic CSV.

## Installation

```bash
git clone https://github.com/akkochief/molytica.git
cd molytica
pip install -r requirements.txt
python main.py
```

The app runs at `http://127.0.0.1:5000`.

## Scientific Foundation

- **Suzuki-Miyaura mechanism**: Pd(0)/Pd(II) catalytic cycle (OA, TM, RE) — the three barrier *estimates* used are illustrative, not per-molecule calculated (see `PARAMETER_PROVENANCE.md` §5)
- **Hammett/Taft LFER**: substituent effects (σₘ, σₚ, σ⁺, σ⁻, Eₛ) — genuinely literature-grounded
- **HSAB theory** (Pearson, 1963): hardness, chemical potential, electrophilicity — framework is real; the specific numeric scale used is an illustrative placeholder, not a tabulated reference
- **Eyring transition-state kinetics**: ΔH‡, ΔS‡, reaction rates
- **Kamlet-Taft solvent parameters**: α, β, π*
- **A conceptual-DFT-*inspired* electronic proxy**: HOMO-LUMO-style, chemical-potential/hardness/electrophilicity definitions (Parr & Pearson, 1983) applied to a Hammett-σ-derived proxy — **not an actual DFT calculation**
- **A continuous physicochemical proxy**: real RDKit MW/LogP/TPSA/rotatable-bond descriptors, no borrowed drug-likeness rule set
- **ML Ensemble**: Random Forest, XGBoost, LightGBM, CatBoost, Hist-GB, SVR, KNN, Ridge, Lasso, ElasticNet, MLP, Gaussian Process — restricted to linear models automatically at low sample counts (see §4)

## What changed in v7.4.0

A full academic-integrity pass renamed or removed every module whose name
implied a computational technique that isn't actually performed:

- **"DFT" → "electronic proxy" (renamed, not removed).** The Hammett-σ-
  based formula is unchanged; only the misleading `dft_*` naming (config
  keys, function names, feature columns) was fixed, since no DFT/quantum-
  chemistry package ever ran in this codebase.
- **"Docking" (removed entirely, not renamed).** Molecular docking scores
  a small molecule against a protein binding pocket; this pipeline has no
  protein target, so no honestly-labelled version of "docking score" would
  be meaningful here. The module, its config block, and its feature
  columns are gone.
- **"QSAR" → continuous "physicochemical proxy" (redesigned, not just
  renamed).** Lipinski/Ghose/Veber Rule-of-Five-style pass/fail cutoffs and
  the QED composite score were deleted outright — those rules are real but
  were derived for oral drug bioavailability, not coupling yield. What
  remains is a plain, continuous penalty based on real RDKit MW/LogP/TPSA/
  rotatable-bond descriptors, with no borrowed rule names.
- **Added `PARAMETER_PROVENANCE.md`**, previously referenced by the code
  but missing from the repo — the full `[LIT]`/`[PARAM]`/`[REMOVED]` audit
  for every constant in `info.xml`.
- **Low-data ML safeguards** (minimum sample thresholds, restricted
  ensemble at low N, nested-CV `honest_cv_performance` metric) — see §4.

None of this changes the numeric output of the electronic-proxy or
mechanistic-barrier calculations; it changes what they're honestly called
and removes the module (docking) and the borrowed rule set (QSAR
drug-likeness) that had no defensible chemical basis for this reaction.

## References

1. Miyaura, N.; Suzuki, A. *Chem. Rev.* **1995**, *95*, 2457-2483.
2. Martin, R.; Buchwald, S. L. *Acc. Chem. Res.* **2008**, *41*, 1461-1473.
3. Fortman, G. C.; Nolan, S. P. *Chem. Soc. Rev.* **2011**, *40*, 5151-5169.
4. Lennox, A. J. J.; Lloyd-Jones, G. C. *Chem. Soc. Rev.* **2014**, *43*, 412-433.
5. Ahneman, D. T.; Estrada, J. G.; Lin, S.; Dreher, S. D.; Doyle, A. G. *Science* **2018**, *360*, 186-190.
6. Parr, R. G.; Pearson, R. G. *J. Am. Chem. Soc.* **1983**, *105*, 7512-7516. (conceptual-DFT definitions used by the electronic proxy)
7. Pearson, R. G. *J. Am. Chem. Soc.* **1963**, *85*, 3533-3539. (HSAB framework)

Full parameter-by-parameter citations: see [`PARAMETER_PROVENANCE.md`](PARAMETER_PROVENANCE.md).

## License

MIT License — see [LICENSE](LICENSE)

## Note

This software is intended for academic and research purposes, and the
physicochemical model should be described as a heuristic, literature-
*inspired* scoring function, not a validated predictive model, in any
academic write-up — see the provenance audit above. Contact the developers
before commercial use.

## Drive
Link : https://drive.google.com/file/d/14_TyiuhB_WluTO_udYXPrO9DOz4VurRA/view?usp=sharing
