"""Embedding visualization via t-SNE and UMAP."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_and_visualize_embeddings(
    model: torch.nn.Module,
    dataset: Any,
    output_dir: Path,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """Extract CLS embeddings and compute dimensionality reduction.
    
    Args:
        model: Trained model
        dataset: Full dataset
        output_dir: Directory to save results
        device: Device to run on
    
    Returns:
        Metadata dict with plant IDs and labels
    """
    from src.data.collate import faba_collate_fn
    
    model_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(model_device)
    model.eval()
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        collate_fn=faba_collate_fn,
    )
    
    # Extract embeddings
    embeddings = []
    metadata = {
        'plant_ids': [],
        'accessions': [],
        'treatments': [],
        'categories': [],
        'dag_targets': [],
    }
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for batch in loader:
            # Move to device
            batch_gpu = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            outputs = model(batch_gpu)
            
            # Extract CLS embedding (256-dim)
            cls_emb = outputs['cls_embedding']
            embeddings.append(cls_emb.cpu().numpy())
            
            # Collect metadata
            for i in range(len(batch['plant_id'])):
                metadata['plant_ids'].append(batch['plant_id'][i])
                metadata['accessions'].append(batch['accession'][i])
                metadata['treatments'].append(batch['treatment'][i])
                
                dag_cat = int(batch['dag_category'][i].cpu().item())
                if dag_cat == 0:
                    category = 'Early'
                elif dag_cat == 1:
                    category = 'Mid'
                elif dag_cat == 2:
                    category = 'Late'
                else:
                    category = 'Control'
                metadata['categories'].append(category)
                
                dag_target = float(batch['dag_target'][i].cpu().item())
                metadata['dag_targets'].append(dag_target)
    
    # Concatenate embeddings
    embeddings_np = np.concatenate(embeddings, axis=0)  # (N, 256)
    
    print(f"  → Extracted {len(embeddings_np)} embeddings (dim={embeddings_np.shape[1]})")
    
    # Compute t-SNE
    print("Computing t-SNE...")
    from sklearn.manifold import TSNE
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_tsne = tsne.fit_transform(embeddings_np)
    
    # Save t-SNE
    tsne_path = output_dir / 'embeddings_tsne.npy'
    np.save(tsne_path, embeddings_tsne)
    print(f"  → Saved t-SNE to {tsne_path}")
    
    # Compute UMAP (if available)
    try:
        import umap
        print("Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_umap = reducer.fit_transform(embeddings_np)
        
        umap_path = output_dir / 'embeddings_umap.npy'
        np.save(umap_path, embeddings_umap)
        print(f"  → Saved UMAP to {umap_path}")
    except ImportError:
        print("  → UMAP not available (install with: pip install umap-learn)")
        embeddings_umap = None
    
    # Save metadata
    metadata_path = output_dir / 'embedding_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  → Saved metadata to {metadata_path}")
    
    return metadata
