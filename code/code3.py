import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import seaborn as sns
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
images_dir = os.path.join(base_dir, 'static', 'images')
dataset_dir = os.path.join(base_dir, 'datasets')
outputs_dir = os.path.join(base_dir, 'static', 'outputs')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_subdir = os.path.join(images_dir, f'ml_{timestamp}')
os.makedirs(output_subdir, exist_ok=True)

def load_data(file_name):
    df = pd.read_csv(file_name, sep='\t' if '\t' in open(file_name).readline() else ',')
    column_mapping = {
        'catalyst': [c for c in df.columns if 'catal' in c.lower()][0],
        'halide': [c for c in df.columns if 'ar-x' in c.lower() or 'halide' in c.lower()][0],
        'yield': [c for c in df.columns if 'yield' in c.lower()][0]
    }
    return df, column_mapping
def extract_features(df, column_mapping):
    df = df.copy()
    df['catalyst_type'] = df[column_mapping['catalyst']].astype(str).apply(
        lambda x: 'Cl-Pd' if 'Cl[Pd]' in x else 'I-Pd'
    )
    df['halide_type'] = df[column_mapping['halide']].astype(str).apply(
        lambda x: 'Br' if 'bromo' in x.lower() else 'I' if 'iodo' in x.lower() else 'Cl'
    )
    functional_groups = {
        'methyl': 'Me', 'methoxy': 'OMe', 'fluoro': 'F',
        'aldehyde': 'CHO', 'nitro': 'NO2', 'acetyl': 'COMe'
    }
    df['functional_group'] = 'H'
    for key, val in functional_groups.items():
        df.loc[df[column_mapping['halide']].str.contains(key, case=False, na=False), 
              'functional_group'] = val
    return df
def build_model(df, target_column):
    le = LabelEncoder()
    X = df[['catalyst_type', 'halide_type', 'functional_group']].apply(le.fit_transform)
    y = pd.to_numeric(df[target_column], errors='coerce').fillna(0)
    model = LassoCV(cv=5, random_state=42)
    model.fit(X, y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_,
        'absolute_impact': np.abs(model.coef_)
    }).sort_values('absolute_impact', ascending=False)
    return model, importance
def visualize(df, importance, target_column):
    plt.figure(figsize=(10,6))
    sns.barplot(x='catalyst_type', y=target_column, data=df, ci=None)
    plt.title('Yield by Catalyst Type')
    plt.savefig(os.path.join(output_subdir, 'solvent_effect.png'), bbox_inches='tight', dpi=300)
    plt.close()
    plt.figure(figsize=(12,6))
    sns.boxplot(x='halide_type', y=target_column, hue='catalyst_type', data=df)
    plt.title('Halide and Catalyst Interaction')
    plt.savefig(os.path.join(output_subdir, 'qq_plot.png'), bbox_inches='tight', dpi=300)
    plt.close()
    plt.figure(figsize=(10,6))
    sns.barplot(x='absolute_impact', y='feature', data=importance)
    plt.title('Feature Impact on Yield')
    plt.savefig(os.path.join(output_subdir, 'residuals.png'), bbox_inches='tight', dpi=300)
    plt.close()
    plt.figure(figsize=(12,6))
    sns.violinplot(x='functional_group', y=target_column, data=df)
    plt.title('Yield Distribution by Functional Group')
    plt.savefig(os.path.join(output_subdir, 'actual_vs_predicted.png'), bbox_inches='tight', dpi=300)
    plt.close()
def main_workflow():
    df, column_mapping = load_data(os.path.join('out.csv'))
    print(f"Columns used: {column_mapping}")
    df = extract_features(df, column_mapping)
    model, importance = build_model(df, column_mapping['yield'])
    visualize(df, importance, column_mapping['yield'])
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write("=== MODEL RESULTS ===\n\n")
        f.write(f"R² Score: {model.score(df[['catalyst_type', 'halide_type', 'functional_group']].apply(LabelEncoder().fit_transform), pd.to_numeric(df[column_mapping['yield']], errors='coerce').fillna(0)):.3f}\n\n")
        f.write("=== IMPORTANT FEATURES ===\n")
        f.write(importance.to_string())
        f.write("\n\n=== RECOMMENDATIONS ===\n")
        f.write(f"1. {importance.iloc[0]['feature']} is the most important factor\n")
        f.write(f"2. {importance.iloc[1]['feature']} should be optimized\n")
        f.write(f"3. {importance.iloc[2]['feature']} effect should be considered\n")
if __name__ == "__main__":
    main_workflow()