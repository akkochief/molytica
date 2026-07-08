![Model](https://github.com/sefaakkoc/molytica/blob/main/img/Molytica4.png)
# Molytica (Suzuki-Miyaura Reaction Analyzer)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red.svg)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-purple.svg)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-lightblue.svg)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-darkgreen.svg)](https://matplotlib.org)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-yellow.svg)](https://www.rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Passing-success.svg)](https://github.com/sefaakkoc/molytica)
[![Version](https://img.shields.io/badge/Version-2.1.0-blue.svg)](https://github.com/sefaakkoc/molytica/releases)
[![Platform](https://img.shields.io/badge/Platform-Web-lightgrey.svg)](https://github.com/sefaakkoc/molytica)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)](https://github.com/sefaakkoc/molytica)

A comprehensive web-based analysis platform for analyzing, optimizing, and simulating Suzuki-Miyaura cross-coupling reactions with additional catalyst analysis and yield prediction pages.

## Features

- **Reaction Analysis**: Mechanistic analysis of Suzuki-Miyaura reactions
- **Catalyst Optimization**: Selection and optimization of palladium catalysts
- **Catalyst Analysis Page**: Comprehensive catalyst database with performance analytics
- **Yield Prediction Page**: Advanced ML-based yield forecasting models
- **Yield Calculation**: Theoretical vs experimental yield comparison
- **Side Product Analysis**: Identification and minimization of potential side products
- **Reaction Conditions**: Temperature, solvent, and base optimization
- **Molecular Visualization**: 2D/3D molecular structure rendering
- **Responsive Design**: Optimized experience across all devices

## New Additional Pages

### Catalyst Analysis Page
- **Catalyst Database Browser**: Explore comprehensive catalyst library
- **Performance Metrics**: Compare catalyst efficiency across different reaction types
- **Structure-Activity Relationships**: Visualize how catalyst structure affects performance
- **Recommendation Engine**: Get catalyst suggestions based on your specific reaction
- **Custom Catalyst Entry**: Add and evaluate your own catalyst designs

### Yield Prediction Page
- **ML-Powered Predictions**: Advanced machine learning models for yield forecasting
- **Multi-Parameter Analysis**: Consider temperature, time, catalyst loading, and solvent effects
- **Confidence Intervals**: Statistical confidence ranges for predictions
- **Historical Data Integration**: Learn from thousands of previous reactions
- **Real-time Optimization**: Interactive parameter adjustment with instant feedback
- **Batch Prediction Mode**: Predict yields for multiple reactions simultaneously

## Requirements

```
Flask
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
gitpython
requests
scipy
lightgbm
rdkit
plotly
pubchempy
ipython
jupyter
tqdm
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sefaakkoc/molytica.git
cd molytica
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python run.py
```

The application will be available at `http://0.0.0.0:5000`.

## Project Structure

```
molytica/
├───code
├───outputs
├───static
│   ├───assets
│   ├───datasets
│   ├───images
│   ├───model
│   ├───outputs
│   └───xml_data
└───templates
    ├───index.html
    ├───predict.html
    └───[other templates]
```

## Scientific Foundation

This application is based on the following scientific principles:

- **Suzuki-Miyaura Mechanism**: Pd(0)/Pd(II) catalytic cycle
- **DFT Calculations**: Transition state energies
- **QSAR Models**: Reactivity predictions
- **Kinetic Analysis**: Rate constant calculations
- **Machine Learning**: Random Forest, XGBoost, and Neural Network models for yield prediction

## Supported Reaction Types

- Aryl-Aryl coupling
- Alkyl-Aryl coupling (limited)

## Example Workflows

### Standard Analysis
1. Input aryl halide and boronic acid SMILES
2. Select catalyst from database
3. Choose reaction conditions
4. Run mechanistic analysis
5. Review yield predictions and side products

### Catalyst Analysis Workflow
1. Navigate to **Catalyst Analysis** page
2. Browse catalyst database or input custom catalyst
3. Compare performance metrics across different substrates
4. Use recommendation engine for optimal catalyst selection
5. Export catalyst performance reports

### Yield Prediction Workflow
1. Access **Yield Prediction** page
2. Input reaction parameters (substrates, catalyst, conditions)
3. Run ML prediction models
4. Analyze confidence intervals and feature importance
5. Optimize conditions using real-time parameter adjustment
6. Export prediction results and optimization suggestions

### Batch Processing
1. Upload CSV file with multiple reactions
2. Apply optimization algorithms
3. Export results as Image And Text

## Dataset Requirements

### CSV File Format
For batch analysis, your CSV file should contain the following columns:

| Column Name | Description | Example |
|------------|-------------|---------|
| `Ar-B(OH)2` | Boronic acid compound name | `phenylboronic acid` |
| `Ar-X` | Aryl halide compound name | `bromobenzene` |
| `product` | Expected product name | `1,1'-biphenyl` |
| `catalizor` | Catalyst SMILES notation | `I[Pd](I)([N]1=CC=CC=C1)C(N2C)N(C)C3=C2N(C)C(N(C)C3=O)=O` |
| `base` | Base SMILES notation | `[K].[K]OO[C]=O` |
| `solv1` | Primary solvent | `water` |
| `solv2` | Secondary solvent | `propan-2-ol` |
| `amount` | Catalyst amount (mol) | `0.0025` |
| `centigrades` | Temperature in °C | `40` |
| `minute` | Reaction time in minutes | `120` |
| `cycle` | Reaction cycle number | `88` |
| `yield` | Experimental yield (%) | `81` |

### Sample CSV Structure
```csv
Ar-B(OH)2,Ar-X,product,catalizor,base,solv1,solv2,amount,centigrades,minute,cycle,yield
phenylboronic acid,bromobenzene,1,1'-biphenyl,I[Pd](I)([N]1=CC=CC=C1)C(N2C)N(C)C3=C2N(C)C(N(C)C3=O)=O,[K].[K]OO[C]=O,water,propan-2-ol,0.0025,40,120,88,81
4-methylphenylboronic acid,4-bromobenzaldehyde,4'-methyl-[1,1'-biphenyl]-4-carbaldehyde,I[Pd](I)([N]1=CC=CC=C1)C(N2C)N(C)C3=C2N(C)C(N(C)C3=O)=O,[K].[K]OO[C]=O,water,ethanol,0.003,60,180,92,78
```

### Analysis Process

- Analysis may take **2-5 minutes** per reaction depending on complexity
- Large datasets (>50 reactions) may require **10-30 minutes** of processing time
- **Catalyst analysis** typically completes in **30-60 seconds** per catalyst
- **Yield predictions** are generated in **5-15 seconds** per reaction
- Do not close the browser window during analysis
- Results will be displayed progressively as each reaction completes
- Failed analyses will be marked with error messages for troubleshooting

## Navigation

The platform now includes the following main sections:

- **Home**: Main dashboard and quick analysis
- **Reaction Analysis**: Detailed mechanistic studies
- **Catalyst Analysis**: Catalyst database and optimization tools
- **Yield Prediction**: ML-based yield forecasting
- **Batch Processing**: Multiple reaction analysis
- **Results Archive**: Historical data and reports

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Lead Developer** - [Sefa Akkoc](https://github.com/sefaakkoc)
- **Chemistry Advisor** - Assoc. Prof. Dr. Mitat Akkoc

## References

1. Miyaura, N.; Suzuki, A. *Chem. Rev.* **1995**, *95*, 2457-2483.
2. Martin, R.; Buchwald, S. L. *Acc. Chem. Res.* **2008**, *41*, 1461-1473.
3. Fortman, G. C.; Nolan, S. P. *Chem. Soc. Rev.* **2011**, *40*, 5151-5169.
4. Lennox, A. J. J.; Lloyd-Jones, G. C. *Chem. Soc. Rev.* **2014**, *43*, 412-433.
5. Ahneman, D. T.; Estrada, J. G.; Lin, S.; Dreher, S. D.; Doyle, A. G. *Science* **2018**, *360*, 186-190.

## Support

For issues and questions:
- Open an [Issue](https://github.com/sefaakkoc/molytica/issues)
- Check the [Wiki](https://github.com/sefaakkoc/molytica/wiki)

**Note**: This software is intended for academic and research purposes. Please contact the developers before commercial use.

---

## Recent Updates (v0.2)

### New Features
- **Catalyst Analysis Page**: Comprehensive catalyst database with performance analytics
- **Yield Prediction Page**: Advanced ML models for accurate yield forecasting
- **Enhanced Navigation**: Improved user interface with dedicated pages for specialized analyses
- **Real-time Optimization**: Interactive parameter adjustment tools
- **Extended Database**: Larger catalyst and reaction database for better predictions

### Improvements
- **Catalyst Analysis Page**: New dedicated page for catalyst selection and performance comparison
- **Yield Prediction Page**: Advanced ML-based yield forecasting with real-time optimization
- **Enhanced User Interface**: Improved navigation with dedicated pages for specialized analyses
- **Faster Processing**: Optimized algorithms for quicker analysis results
- **Better Error Handling**: Improved user feedback and troubleshooting guidance
- **Real-time Parameter Adjustment**: Interactive optimization tools for reaction conditions
