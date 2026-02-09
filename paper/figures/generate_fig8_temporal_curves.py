import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
FLUOR_PATH = 'data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx'
META_PATH = 'data/plant_metadata.csv'
OUT_DIR = 'paper/figures'
OS_PATH = os.path.join(OUT_DIR, 'fig8_temporal_curves') # Base name

# Round to DAG mapping (from src/data/dataset.py)
ROUND_TO_DAG = {
    2: 4, 3: 5, 4: 6, 5: 7, 6: 10, 7: 12, 
    8: 13, 9: 14, 10: 17, 11: 19, 12: 20, 13: 21, 
    14: 24, 15: 27, 16: 28, 17: 29, 18: 31, 19: 33, 
    20: 34, 21: 35, 22: 38, 23: 38
}

# Classic PAM parameters mapped to closest equivalents
# Fv/Fm -> QY_max (Dark-adapted maximum quantum yield)
# Vitality -> Rfd_Lss (Fluorescence Decrease Ratio in light steady state)
# Phi_PSII -> QY_Lss (Quantum yield of PSII in light steady state)
# qP -> qP_Lss (Photochemical quenching in light steady state)
CLASSIC_PARAMS_MAP = {
    'QY_max': 'Fv/Fm (QY_max)',
    'Rfd_Lss': 'Vitality Index (Rfd_Lss)', 
    'QY_Lss': 'ΦPSII (QY_Lss)',
    'qP_Lss': 'qP (qP_Lss)'
}
CLASSIC_KEYS = list(CLASSIC_PARAMS_MAP.keys())

# Plot settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
BLUE = "#3498DB"  # WHC-80
RED = "#E74C3C"   # WHC-30

# ------------------------------------------------------------------------------
# Data Loading & Processing
# ------------------------------------------------------------------------------
def load_data():
    print(f"Loading fluorescence data from {FLUOR_PATH}...")
    df_fluor = pd.read_excel(FLUOR_PATH)
    
    print(f"Loading metadata from {META_PATH}...")
    df_meta = pd.read_csv(META_PATH)
    
    # Clean IDs
    df_fluor['Plant ID'] = df_fluor['Plant ID'].astype(str).str.strip()
    df_meta['plant_id'] = df_meta['plant_id'].astype(str).str.strip()
    
    # Merge
    # We only need Treatment from metadata
    df_merged = df_fluor.merge(
        df_meta[['plant_id', 'treatment']], 
        left_on='Plant ID', 
        right_on='plant_id', 
        how='left'
    )
    
    # Filter for valid treatments
    df_merged = df_merged[df_merged['treatment'].isin(['WHC-80', 'WHC-30'])]
    
    # Map Round to DAG
    df_merged['DAG'] = df_merged['Round Order'].map(ROUND_TO_DAG)
    
    # Drop rows without DAG (e.g. Round 1 or >23 if any)
    df_merged = df_merged.dropna(subset=['DAG'])
    
    return df_merged

def select_data_driven_params(df, classic_keys, top_k=3):
    print("Selecting data-driven parameters...")
    
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Round Order', 'DAG', 'Replicate', 'Accession Num', 'treatment_code']
    candidate_cols = [c for c in numeric_cols if c not in classic_keys and c not in exclude_cols]
    
    # Create binary treatment column for correlation
    df['treatment_code'] = df['treatment'].map({'WHC-80': 0, 'WHC-30': 1})
    
    results = []
    
    # Compute max Spearman correlation across timepoints
    for col in candidate_cols:
        if col in ['treatment_code']: continue
        
        # Check if column has enough variance
        if df[col].std() < 1e-8:
            continue
            
        max_r = 0
        for dag in df['DAG'].unique():
            df_dag = df[df['DAG'] == dag]
            if len(df_dag) < 10: continue
            
            # Clean NaNs
            valid = df_dag[[col, 'treatment_code']].dropna()
            if len(valid) < 10: continue
            
            try:
                r, p = spearmanr(valid[col], valid['treatment_code'])
                if not np.isnan(r):
                    max_r = max(max_r, abs(r))
            except:
                pass
        
        results.append((col, max_r))
    
    # Sort by max correlation descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    top_params = [x[0] for x in results[:top_k]]
    print(f"Top {top_k} data-driven params: {top_params} (max |r|: {[f'{x[1]:.2f}' for x in results[:top_k]]})")
    
    return top_params

def normalize_zscore(df, params):
    df_norm = df.copy()
    for col in params:
        mean = df[col].mean()
        std = df[col].std()
        if std < 1e-8: std = 1.0
        df_norm[col] = (df[col] - mean) / std
    return df_norm

# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------
def plot_curves(df, classic_keys, data_driven_keys):
    all_params = classic_keys + data_driven_keys
    n_params = len(all_params)
    
    # Setup grid
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
    axes = axes.flatten()
    
    # Normalize data
    print("Normalizing data (z-score)...")
    df_norm = normalize_zscore(df, all_params)
    
    # Plot each parameter
    for i, param in enumerate(all_params):
        ax = axes[i]
        
        # Determine title
        if param in CLASSIC_PARAMS_MAP:
            title = CLASSIC_PARAMS_MAP[param]
        else:
            title = param
            
        # Plot with Seaborn
        sns.lineplot(
            data=df_norm,
            x='DAG',
            y=param,
            hue='treatment',
            hue_order=['WHC-80', 'WHC-30'],
            palette={'WHC-80': BLUE, 'WHC-30': RED},
            errorbar=('ci', 95),
            ax=ax,
            legend=(i == 0) # Only legend on first plot
        )
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel("Z-Score")
        ax.set_xlabel("DAG")
        
        # Add zero line
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
        
    # Adjust layout
    plt.tight_layout()
    
    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = f"{OS_PATH}.pdf"
    png_path = f"{OS_PATH}.png"
    
    print(f"Saving to {pdf_path}...")
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    return pdf_path

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    df = load_data()
    
    # Check if classic params exist
    available_classic = [p for p in CLASSIC_KEYS if p in df.columns]
    missing_classic = [p for p in CLASSIC_KEYS if p not in df.columns]
    
    if missing_classic:
        print(f"Warning: Missing classic params: {missing_classic}")
    
    # Select data-driven params
    data_driven = select_data_driven_params(df, available_classic, top_k=3)
    
    # Generate plot
    plot_curves(df, available_classic, data_driven)
    
    # Append to notepad
    try:
        notepad_path = '.sisyphus/notepads/feature-analysis/learnings.md'
        with open(notepad_path, 'a') as f:
            f.write(f"\n## [{pd.Timestamp.now()}] Task 4: Fig 8 Temporal Curves\n")
            f.write(f"- Classic params used: {available_classic}\n")
            f.write(f"- Data-driven params selected: {data_driven}\n")
            f.write(f"- Normalization: z-score (global)\n")
            f.write(f"- Saved: fig8_temporal_curves.pdf\n")
    except Exception as e:
        print(f"Could not write to notepad: {e}")

if __name__ == "__main__":
    main()
