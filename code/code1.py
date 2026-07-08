import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import os
import shap
from datetime import datetime
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_dir = os.path.join(base_dir, 'static', 'images')
dataset_dir = os.path.join(base_dir, 'datasets')
outputs_dir = os.path.join(base_dir, 'static', 'outputs')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_subdir = os.path.join(images_dir, f'ml_{timestamp}')
os.makedirs(output_subdir, exist_ok=True)
try:
    df = pd.read_csv(os.path.join('out.csv'), sep='\t')
    if len(df.columns) == 1:
        df = pd.read_csv(os.path.join('out.csv'), sep=',')   
    print("Found these columns in your CSV file:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    col_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'catal' in col_lower:
            col_map['catalyst'] = col
        elif 'ar-x' in col_lower or 'halide' in col_lower:
            col_map['aryl_halide'] = col
        elif 'yield' in col_lower:
            col_map['yield'] = col
        elif 'product' in col_lower:
            col_map['product'] = col
    if 'catalyst' not in col_map and len(df.columns) >= 4:
        col_map['catalyst'] = df.columns[3]
    if 'aryl_halide' not in col_map and len(df.columns) >= 2:
        col_map['aryl_halide'] = df.columns[1]
    if 'yield' not in col_map and len(df.columns) >= 12:
        col_map['yield'] = df.columns[11]
    if 'product' not in col_map and len(df.columns) >= 3:
        col_map['product'] = df.columns[2]
    print("\nAutomatically identified column mapping:")
    for k, v in col_map.items():
        print(f"{k}: {v}")
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()
df_ml = df.copy()
df_ml['catalyst_type'] = df_ml[col_map['catalyst']].astype(str).apply(
    lambda x: 'Cl-Pd' if 'Cl[Pd]' in x else 'I-Pd' if 'I[Pd]' in x else 'Other'
)
df_ml['halide_type'] = df_ml[col_map['aryl_halide']].astype(str).apply(
    lambda x: 'Br' if 'bromo' in x.lower() 
    else 'I' if 'iodo' in x.lower() 
    else 'Cl' if 'chloro' in x.lower() 
    else 'Other'
)
functional_groups = {
    'methyl': 'Me',
    'methoxy': 'OMe',
    'fluoro': 'F',
    'aldehyde': 'CHO',
    'nitro': 'NO2',
    'carbonitrile': 'CN',
    'acetyl': 'COMe',
    'formyl': 'CHO'
}
df_ml['functional_group'] = 'H'
for key, val in functional_groups.items():
    mask = df_ml[col_map['aryl_halide']].astype(str).str.contains(key, case=False, na=False)
    df_ml.loc[mask, 'functional_group'] = val
X = pd.get_dummies(df_ml[['catalyst_type', 'halide_type', 'functional_group']])
y = df_ml[col_map['yield']]
y = pd.to_numeric(y, errors='coerce')
if y.isna().any():
    print(f"Warning: {y.isna().sum()} yield values could not be converted to numbers")
    y = y.fillna(y.mean())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
with open(os.path.join(outputs_dir, 'asd.txt'), 'w') as f:
    f.write("=== Suzuki-Miyaura Coupling ML Analysis ===\n\n")
    f.write(f"Original columns in CSV: {df.columns.tolist()}\n")
    f.write(f"Column mapping used: {col_map}\n\n")
    f.write(f"Model Performance (MAE): {mae:.2f}\n\n")
    f.write("=== Feature Importance ===\n")
    f.write(feature_importance.to_string())
    f.write("\n\n=== Top Predictors of High Yield ===\n")
    f.write(f"1. {feature_importance.iloc[0]['feature']}\n")
    f.write(f"2. {feature_importance.iloc[1]['feature']}\n")
    f.write(f"3. {feature_importance.iloc[2]['feature']}\n")
    f.write("\n=== Recommendations ===\n")
    f.write("- Consider using the catalyst type that showed highest yields\n")
    f.write("- Optimize reactions with the most productive functional groups\n")
    f.write("- Focus on the halide type that performed best\n")
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.figure(figsize=(10,6))
sns.boxplot(x='catalyst_type', y=y, data=df_ml)
plt.title('Yield Distribution by Catalyst Type')
plt.ylabel('Yield (%)')
plt.savefig(os.path.join(output_subdir, 'solvent_effect.png'), bbox_inches='tight', dpi=300)
plt.close()
plt.figure(figsize=(12,6))
order = df_ml.groupby('functional_group')[y.name].mean().sort_values(ascending=False).index
sns.barplot(x='functional_group', y=y, hue='catalyst_type', data=df_ml, order=order)
plt.title('Yield by Functional Group and Catalyst')
plt.ylabel('Yield (%)')
plt.legend(title='Catalyst Type')
plt.savefig(os.path.join(output_subdir, 'qq_plot.png'), bbox_inches='tight', dpi=300)
plt.close()
plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Permutation Feature Importance')
plt.savefig(os.path.join(output_subdir, 'residuals.png'), bbox_inches='tight', dpi=300)
plt.close()
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title('SHAP Feature Importance')
plt.savefig(os.path.join(output_subdir, 'actual_vs_predicted.png'), bbox_inches='tight', dpi=300)
plt.close()
