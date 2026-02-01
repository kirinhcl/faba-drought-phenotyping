"""Temporal attention analysis for XAI interpretability."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ROUND_TO_DAG


def extract_attention_maps(
    model: torch.nn.Module,
    dataset: Any,
    fold_results_dir: Path,
    device: str = 'cuda',
) -> Dict[str, Dict[str, Any]]:
    """Extract temporal attention maps from model.
    
    For each test plant, extract CLS token's attention over temporal positions.
    Aggregate per genotype (average across 3 drought reps).
    
    Args:
        model: Trained FabaDroughtModel or FabaDroughtStudent
        dataset: Full dataset
        fold_results_dir: Directory with best_model_state.pt
        device: Device to run on
    
    Returns:
        {genotype: {
            'attention_map': (22,) array - genotype-averaged attention,
            'plant_attention_maps': {plant_id: (22,) array}
        }}
    """
    # Load best model
    model_path = fold_results_dir / 'best_model_state.pt'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.to(model_device)
    model.load_state_dict(torch.load(model_path, map_location=model_device, weights_only=True))
    model.eval()
    
    # Create dataloader for all plants
    from src.data.collate import faba_collate_fn
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=faba_collate_fn,
    )
    
    # Extract attention maps per plant
    plant_attention_maps: Dict[str, np.ndarray] = {}
    plant_metadata: Dict[str, Dict[str, str]] = {}
    
    with torch.no_grad():
        for batch in loader:
            # Move to device
            batch_gpu = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass
            outputs = model(batch_gpu)
            
            # Extract attention weights
            # attention_weights is a list of (B, H, T+1, T+1) tensors (one per layer)
            attention_weights = outputs['attention_weights']
            
            if attention_weights is None or len(attention_weights) == 0:
                continue
            
            # Average across layers and heads
            # For each layer: (B=1, H, T+1, T+1)
            # Take row 0 (CLS token attention), cols 1: (temporal positions)
            plant_id = batch['plant_id'][0]
            accession = batch['accession'][0]
            treatment = batch['treatment'][0]
            
            # Collect attention from all layers
            layer_attentions = []
            for attn in attention_weights:
                # attn: (1, H, T+1, T+1)
                cls_attn = attn[0, :, 0, 1:]  # (H, T)
                cls_attn_mean = cls_attn.mean(dim=0)  # (T,)
                layer_attentions.append(cls_attn_mean)
            
            # Average across layers
            avg_attention = torch.stack(layer_attentions, dim=0).mean(dim=0)  # (T,)
            
            # Convert to numpy and pad to 22 if needed
            attn_np = avg_attention.cpu().numpy()
            if len(attn_np) < 22:
                attn_np = np.pad(attn_np, (0, 22 - len(attn_np)), mode='constant', constant_values=0)
            elif len(attn_np) > 22:
                attn_np = attn_np[:22]
            
            plant_attention_maps[plant_id] = attn_np
            plant_metadata[plant_id] = {
                'accession': accession,
                'treatment': treatment,
            }
    
    # Aggregate per genotype (average across drought reps)
    genotype_attention: Dict[str, Dict[str, Any]] = {}
    
    for plant_id, attn_map in plant_attention_maps.items():
        accession = plant_metadata[plant_id]['accession']
        treatment = plant_metadata[plant_id]['treatment']
        
        if accession not in genotype_attention:
            genotype_attention[accession] = {
                'attention_map': np.zeros(22, dtype=np.float32),
                'plant_attention_maps': {},
                'treatment_counts': {'WHC-30': 0, 'WHC-80': 0},
            }
        
        genotype_attention[accession]['plant_attention_maps'][plant_id] = attn_map
        genotype_attention[accession]['treatment_counts'][treatment] += 1
        
        # Only aggregate drought plants for genotype mean
        if treatment == 'WHC-30':
            genotype_attention[accession]['attention_map'] += attn_map
    
    # Normalize by number of drought plants
    for accession, data in genotype_attention.items():
        n_drought = data['treatment_counts']['WHC-30']
        if n_drought > 0:
            data['attention_map'] /= n_drought
    
    return genotype_attention


def find_attention_peak(
    attention_map: np.ndarray,
    round_to_dag: Dict[int, int] = ROUND_TO_DAG,
) -> float:
    """Find DAG value corresponding to highest attention weight.
    
    Args:
        attention_map: (22,) array of attention weights
        round_to_dag: Mapping from round index to DAG value
    
    Returns:
        peak_dag: DAG value at attention peak
    """
    # Find peak index (0-indexed for round 2-23)
    peak_idx = int(np.argmax(attention_map))
    
    # Convert to round number (2-23)
    peak_round = peak_idx + 2
    
    # Get DAG value
    peak_dag = float(round_to_dag.get(peak_round, 0))
    
    return peak_dag
