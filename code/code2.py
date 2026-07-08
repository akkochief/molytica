import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import shap
import os
from datetime import datetime
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_dir = os.path.join(base_dir, 'static', 'images')
dataset_dir = os.path.join(base_dir, 'datasets')
outputs_dir = os.path.join(base_dir, 'static', 'outputs')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_subdir = os.path.join(images_dir, f'ml_{timestamp}')
os.makedirs(output_subdir, exist_ok=True)
sns.set(style="whitegrid", palette="muted")
plt.rcParams['figure.facecolor'] = 'white'
def load_data(filepath):
    df = pd.read_csv(filepath, sep='\t') if '\t' in open(filepath).readline() else pd.read_csv(filepath)
    col_map = {
        'catalyst': next((c for c in df.columns if 'catal' in c.lower()), df.columns[3]),
        'aryl_halide': next((c for c in df.columns if 'ar-x' in c.lower() or 'halide' in c.lower()), df.columns[1]),
        'yield': next((c for c in df.columns if 'yield' in c.lower()), df.columns[11]),
        'product': next((c for c in df.columns if 'product' in c.lower()), df.columns[2])
    }
    return df, col_map
def extract_features(df, col_map):
    df_ml = df.copy()
    df_ml['catalyst_type'] = df_ml[col_map['catalyst']].astype(str).apply(
        lambda x: 'Cl-Pd' if 'Cl[Pd]' in x else 'I-Pd' if 'I[Pd]' in x else 'Other'
    )
    df_ml['halide_type'] = df_ml[col_map['aryl_halide']].astype(str).apply(
        lambda x: 'Br' if 'bromo' in x.lower() else 
                 'I' if 'iodo' in x.lower() else 
                 'Cl' if 'chloro' in x.lower() else 'Other'
    )
    functional_groups = {
        'methyl': 'Me', 'methoxy': 'OMe', 'fluoro': 'F',
        'aldehyde': 'CHO', 'nitro': 'NO2', 'carbonitrile': 'CN',
        'acetyl': 'COMe', 'formyl': 'CHO'
    }
    df_ml['functional_group'] = 'H'
    for key, val in functional_groups.items():
        mask = df_ml[col_map['aryl_halide']].astype(str).str.contains(key, case=False, na=False)
        df_ml.loc[mask, 'functional_group'] = val
    return df_ml
def train_and_evaluate(X, y):
    X_encoded = pd.get_dummies(X, columns=['catalyst_type', 'halide_type', 'functional_group'])
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_encoded, y)
    importance = pd.DataFrame({
        'feature': X_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_encoded)
    return model, importance, shap_values, X_encoded
def create_visualizations(df_ml, importance, shap_values, X_encoded, target_col):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='catalyst_type', y=target_col, data=df_ml)
    plt.title('Yield by Catalyst Type')
    plt.savefig(os.path.join(output_subdir, 'solvent_effect.png'), bbox_inches='tight', dpi=300)
    plt.close()
    plt.figure(figsize=(12, 6))
    order = df_ml.groupby('functional_group')[target_col].mean().sort_values(ascending=False).index
    sns.barplot(x='functional_group', y=target_col, hue='catalyst_type', data=df_ml, order=order)
    plt.title('Yield by Functional Group and Catalyst')
    plt.savefig(os.path.join(output_subdir, 'qq_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=importance.head(10))
    plt.title('Top 10 Important Features')
    plt.savefig(os.path.join(output_subdir, 'residuals.png'), bbox_inches='tight', dpi=300)
    plt.close()
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_encoded, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.savefig(os.path.join(output_subdir, 'actual_vs_predicted.png'), bbox_inches='tight', dpi=300)
    plt.close()
def main():
    df, col_map = load_data(os.path.join('out.csv'))
    print(f"Using columns: {col_map}")
    df_ml = extract_features(df, col_map)
    y = pd.to_numeric(df_ml[col_map['yield']], errors='coerce')
    if y.isna().any():
        y = y.fillna(y.mean())
    X = df_ml[['catalyst_type', 'halide_type', 'functional_group']]
    model, importance, shap_values, X_encoded = train_and_evaluate(X, y)
    create_visualizations(df_ml, importance, shap_values, X_encoded, col_map['yield'])
    results = {
        'model_performance': f"MAE: {mean_absolute_error(y, model.predict(X_encoded)):.2f}",
        'top_features': importance.head(10).to_dict(),
        'recommendations': [
            f"Use {importance.iloc[0]['feature']} for best results",
            f"Optimize conditions with {importance.iloc[1]['feature']}",
            f"Consider impact of {importance.iloc[2]['feature']}"
        ]
    }
    with open(os.path.join(outputs_dir, 'asd.txt'), 'w') as f:
        for k, v in results.items():
            f.write(f"=== {k.upper()} ===\n{str(v)}\n\n")
if __name__ == "__main__":
    main()