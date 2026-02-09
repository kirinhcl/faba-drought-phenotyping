import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import sys
import os

# --- Configuration ---
H5_PATH = "features/dinov2_features.h5"
METADATA_PATH = "data/plant_metadata.csv"
OUTPUT_PDF = "paper/figures/fig6_embeddings.pdf"
OUTPUT_PNG = "paper/figures/fig6_embeddings.png"
NOTEPAD_PATH = ".sisyphus/notepads/feature-analysis/learnings.md"

# Style settings
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
COLORS = {"WHC-80": "#3498DB", "WHC-30": "#E74C3C"}
LABELS = {"WHC-80": "Control (WHC-80)", "WHC-30": "Drought (WHC-30)"}
ALPHA = 0.3

# Round to DAG mapping (for potential use, though main aggregation is per timepoint)
ROUND_TO_DAG = {
    2: 4, 3: 5, 4: 6, 5: 7, 6: 10, 7: 12, 8: 13, 9: 14, 10: 17, 11: 19, 12: 20, 13: 21,
    14: 24, 15: 27, 16: 28, 17: 29, 18: 31, 19: 33, 20: 34, 21: 35, 22: 38, 23: 38
}

def load_data():
    """Loads embeddings and metadata."""
    print(f"Loading metadata from {METADATA_PATH}...")
    metadata = pd.read_csv(METADATA_PATH)
    # Strip whitespace from plant_id
    metadata['plant_id'] = metadata['plant_id'].astype(str).str.strip()
    
    # Create mapping from plant_id to treatment
    plant_treatment_map = dict(zip(metadata['plant_id'], metadata['treatment']))
    
    print(f"Loading embeddings from {H5_PATH}...")
    embeddings = []
    treatments = []
    plant_ids = []
    rounds = []

    with h5py.File(H5_PATH, 'r') as f:
        # Iterate over all plants in the H5 file
        for plant_id in f.keys():
            if plant_id not in plant_treatment_map:
                continue
            
            treatment = plant_treatment_map[plant_id]
            plant_group = f[plant_id]
            
            # Iterate over all rounds for this plant
            for round_key in plant_group.keys():
                round_group = plant_group[round_key]
                
                # Check for side views
                views = []
                for view_name in ['side_000', 'side_120', 'side_240']:
                    if view_name in round_group:
                        views.append(round_group[view_name][:])
                
                if len(views) == 3:
                    # Aggregate: Mean across 3 side views
                    mean_embedding = np.mean(views, axis=0)
                    
                    embeddings.append(mean_embedding)
                    treatments.append(treatment)
                    plant_ids.append(plant_id)
                    rounds.append(int(round_key))

    X = np.array(embeddings)
    y = np.array(treatments)
    print(f"Loaded {len(X)} embeddings. Shape: {X.shape}")
    return X, y

def compute_tsne(X):
    print("Computing t-SNE (perplexity=30)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)
    return X_tsne

def compute_umap(X):
    print("Computing UMAP (n_neighbors=15)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    return X_umap

def plot_embeddings(X_tsne, X_umap, y):
    print("Creating figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Order for plotting to ensure consistent colors/legend
    categories = ["WHC-80", "WHC-30"]
    
    # Helper to plot on an axis
    def plot_on_ax(ax, data, title):
        for cat in categories:
            mask = (y == cat)
            ax.scatter(
                data[mask, 0], 
                data[mask, 1], 
                c=COLORS[cat], 
                label=LABELS[cat], 
                alpha=ALPHA, 
                s=10,
                edgecolors='none'  # Remove outline for cleaner look with transparency
            )
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False) # Remove grid for cleaner embedding plots usually
        # Re-enable grid if strictly required by style "whitegrid" but usually embeddings look better without tick-grids
        # The prompt asks for style="whitegrid", so I will leave the grid or not? 
        # Usually scatter plots for embeddings remove ticks. I'll remove ticks but keep the whitegrid background.
        # sns.despine(left=True, bottom=True)

    # Left Panel: t-SNE
    plot_on_ax(axes[0], X_tsne, "t-SNE")
    
    # Right Panel: UMAP
    plot_on_ax(axes[1], X_umap, "UMAP")
    
    # Add legend to the figure (or one of the axes)
    # We add it to the first axis or outside? 
    # Usually better to have one common legend or one in each if it fits.
    # Given the layout, putting it in the upper right of one might obscure data.
    # Let's put it on the UMAP plot or just use the first one.
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=True)
    
    plt.tight_layout()
    
    # Adjust for legend at bottom
    plt.subplots_adjust(bottom=0.15) 
    
    print(f"Saving to {OUTPUT_PDF}...")
    plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches='tight')
    print(f"Saving to {OUTPUT_PNG}...")
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
    print("Figures saved.")

def append_learnings(n_embeddings):
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    content = f"""
## [{timestamp}] Task 2: Fig 6 Embeddings
- Total embeddings aggregated: {n_embeddings}
- t-SNE perplexity: 30
- UMAP n_neighbors: 15
- Visual separation: [Pending visual inspection]
"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(NOTEPAD_PATH), exist_ok=True)
    
    with open(NOTEPAD_PATH, "a") as f:
        f.write(content)
    print(f"Appended findings to {NOTEPAD_PATH}")

def main():
    X, y = load_data()
    
    X_tsne = compute_tsne(X)
    X_umap = compute_umap(X)
    
    plot_embeddings(X_tsne, X_umap, y)
    append_learnings(len(X))

if __name__ == "__main__":
    main()
