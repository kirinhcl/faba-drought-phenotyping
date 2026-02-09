#!/usr/bin/env python3
"""
Generate a heatmap of Spearman rank correlations between fluorescence parameters
and drought treatment (WHC-30 vs WHC-80) across timepoints (DAG).

Dependencies: pandas, numpy, seaborn, matplotlib, scipy, openpyxl
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path('data')
OUTPUT_DIR = Path('paper/figures')
METADATA_PATH = DATA_DIR / 'plant_metadata.csv'
FLUOR_PATH = DATA_DIR / 'TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx'
LEARNINGS_PATH = Path('.sisyphus/notepads/feature-analysis/learnings.md')

# Round to DAG mapping (from src/data/dataset.py)
ROUND_TO_DAG = {
    2: 4, 3: 5, 4: 6, 5: 7, 6: 10, 7: 12,
    8: 13, 9: 14, 10: 17, 11: 19, 12: 20, 13: 21,
    14: 24, 15: 27, 16: 28, 17: 29, 18: 31, 19: 33,
    20: 34, 21: 35, 22: 38, 23: 38
}

CORR_THRESHOLD = 0.3  # |r| > 0.3 to keep parameter

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from {METADATA_PATH}...")
    meta_df = pd.read_csv(METADATA_PATH)
    # Strip whitespace from plant_id
    meta_df['plant_id'] = meta_df['plant_id'].str.strip()
    
    # Create plant_id -> treatment map
    # We want correlate with DROUGHT treatment (WHC-30).
    # WHC-30 = 1 (Treatment), WHC-80 = 0 (Control).
    treatment_map = meta_df.set_index('plant_id')['treatment'].to_dict()
    
    print(f"Loading fluorescence data from {FLUOR_PATH}...")
    fluor_df = pd.read_excel(FLUOR_PATH)
    
    # Clean Plant ID in fluorescence data
    if 'Plant ID' not in fluor_df.columns:
        raise KeyError("Column 'Plant ID' not found in fluorescence data.")
    fluor_df['Plant ID'] = fluor_df['Plant ID'].astype(str).str.strip()
    
    # Map treatment
    fluor_df['treatment'] = fluor_df['Plant ID'].map(treatment_map)
    
    # Filter out rows with no treatment info (if any)
    initial_len = len(fluor_df)
    fluor_df = fluor_df.dropna(subset=['treatment'])
    if len(fluor_df) < initial_len:
        print(f"Dropped {initial_len - len(fluor_df)} rows missing treatment metadata.")

    # Create binary drought indicator: WHC-30 is True (1), WHC-80 is False (0)
    fluor_df['is_drought'] = (fluor_df['treatment'] == 'WHC-30').astype(int)
    
    # Map Round Order to DAG
    if 'Round Order' not in fluor_df.columns:
        raise KeyError("Column 'Round Order' not found in fluorescence data.")
    
    fluor_df['DAG'] = fluor_df['Round Order'].map(ROUND_TO_DAG)
    # Drop rows where Round Order is not in our mapping (e.g. round 1 if present)
    fluor_df = fluor_df.dropna(subset=['DAG'])
    fluor_df['DAG'] = fluor_df['DAG'].astype(int)
    
    # Identify fluorescence parameters
    # As per src/data/dataset.py: columns from index 15 onwards that are numeric
    # Let's double check index 15 logic on the loaded df.
    # Note: adding 'treatment', 'is_drought', 'DAG' added columns to the end.
    # But we want the original columns.
    # Let's stick to the logic: index 15 onwards of the ORIGINAL dataframe structure.
    # However, since we added columns, let's just inspect columns 15 to (end-3).
    # Or better, just identify numeric columns that are NOT metadata.
    
    exclude_cols = {'Plant ID', 'Round Order', 'treatment', 'is_drought', 'DAG', 'Tray ID', 'Genotype', 'Replicate'}
    # Also usually the first ~15 columns are metadata in these Phenospex files.
    # Let's rely on numeric check and exclusion.
    
    # Get original columns from file read
    original_cols = pd.read_excel(FLUOR_PATH, nrows=0).columns.tolist()
    potential_params = original_cols[15:] # Index 15 onwards
    
    fluor_params = []
    for col in potential_params:
        if col in fluor_df.columns and pd.api.types.is_numeric_dtype(fluor_df[col]):
             fluor_params.append(col)
             
    print(f"Identified {len(fluor_params)} fluorescence parameters.")
    
    # Compute correlations per DAG
    print("Computing correlations...")
    unique_dags = sorted(fluor_df['DAG'].unique())
    
    # Storage for correlation matrix: rows=params, cols=DAGs
    corr_matrix = pd.DataFrame(index=fluor_params, columns=unique_dags, dtype=float)
    
    for dag in unique_dags:
        dag_df = fluor_df[fluor_df['DAG'] == dag]
        
        # We need variance in treatment to compute correlation
        if dag_df['is_drought'].nunique() < 2:
            print(f"Skipping DAG {dag}: No variation in treatment (all same).")
            continue
            
        y = dag_df['is_drought'].values
        
        for param in fluor_params:
            x = dag_df[param].values
            
            # Handle NaNs: drop indices where either x or y is NaN
            mask = ~np.isnan(x) & ~np.isnan(y)
            if np.sum(mask) < 5: # Need reasonable sample size
                continue
                
            # Spearman correlation
            # We want correlation between parameter value and drought status (0 or 1).
            # If rho > 0: Higher parameter value <-> Drought
            # If rho < 0: Lower parameter value <-> Drought
            rho, _ = spearmanr(x[mask], y[mask])
            corr_matrix.loc[param, dag] = rho

    # Drop params that are all NaN (if any)
    corr_matrix = corr_matrix.dropna(how='all', axis=0)
    
    # Filter parameters
    # Keep if max(abs(rho)) > 0.3 across any timepoint
    max_abs_corr = corr_matrix.abs().max(axis=1)
    keep_mask = max_abs_corr > CORR_THRESHOLD
    filtered_corr = corr_matrix.loc[keep_mask]
    
    print(f"Keeping {len(filtered_corr)} parameters with |r| > {CORR_THRESHOLD}")
    
    if len(filtered_corr) == 0:
        print("No parameters met the correlation threshold. Exiting.")
        return

    # Sort/Cluster for visualization
    # We'll let clustermap handle the clustering, but let's fill NaNs with 0 for clustering
    # (though clustermap might handle NaNs or we might need to mask them)
    # Ideally, we shouldn't have NaNs in the middle if data is good, but let's fillna(0) for the plot
    plot_data = filtered_corr.fillna(0)
    
    # Setup plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    # Figure size: Width needs to accommodate DAGs, Height for params
    # 14 width, 10 height as requested
    figsize = (14, 10)
    
    # Create Clustermap
    # col_cluster=False to keep DAGs ordered chronologically
    # row_cluster=True to group similar parameters
    g = sns.clustermap(
        plot_data,
        col_cluster=False,
        row_cluster=True,
        cmap="RdBu_r", # Diverging: Blue (neg) -> White -> Red (pos)
        center=0,
        vmin=-1,
        vmax=1,
        figsize=figsize,
        cbar_pos=(0.02, 0.8, 0.03, 0.15), # Position colorbar
        dendrogram_ratio=(0.15, 0.05) # Adjust dendrogram sizes
    )
    
    # Customize axes
    g.ax_heatmap.set_xlabel("Days After Germination (DAG)")
    g.ax_heatmap.set_ylabel("Fluorescence Parameters")
    g.fig.suptitle("Spearman Correlation Between Fluorescence Parameters and Drought Treatment", y=0.98)
    
    # Annotate significant correlations
    # We can iterate over the axes to add text for high correlations
    ax = g.ax_heatmap
    
    # Iterate over the data to add annotations
    # Note: data in ax is reordered by clustering. 
    # filtered_corr is original index. g.data2d is the plotted data (reordered)
    # We want to annotate if |r| > 0.5
    
    # Loop over rows and cols of the plotted data
    for i in range(plot_data.shape[0]):
        for j in range(plot_data.shape[1]):
            val = g.data2d.iloc[i, j]
            if abs(val) > 0.5:
                ax.text(j + 0.5, i + 0.5, f"{val:.2f}", 
                        ha="center", va="center", color="black" if abs(val) < 0.7 else "white",
                        fontsize=8)

    # Save outputs
    pdf_path = OUTPUT_DIR / 'fig7_fluorescence_heatmap.pdf'
    png_path = OUTPUT_DIR / 'fig7_fluorescence_heatmap.png'
    
    print(f"Saving {pdf_path}...")
    g.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"Saving {png_path}...")
    g.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Record learnings
    top_params = max_abs_corr[keep_mask].sort_values(ascending=False).head(3)
    top_params_str = ", ".join([f"{idx} ({val:.2f})" for idx, val in top_params.items()])
    
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    learning_entry = (
        f"\n## [{timestamp}] Task 3: Fig 7 Fluorescence Heatmap\n"
        f"- Total fluorescence parameters: {len(fluor_params)}\n"
        f"- Parameters with |r| > {CORR_THRESHOLD}: {len(filtered_corr)}\n"
        f"- Top 3 correlated params: {top_params_str}\n"
        f"- Correlation metric: Spearman rank\n"
    )
    
    with open(LEARNINGS_PATH, 'a') as f:
        f.write(learning_entry)
        
    print("Done.")

if __name__ == "__main__":
    main()
