"""Custom collate function for Faba Bean drought phenotyping dataset.

Handles batching of multimodal time-series data with fixed dimensions.
"""

from typing import Any, Dict, List

import torch


def faba_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a batch of samples from FabaDroughtDataset.
    
    Stacks numeric tensors along batch dimension and collects string fields as lists.
    
    Args:
        batch: List of dicts from FabaDroughtDataset.__getitem__()
    
    Returns:
        Dict with batched tensors and lists of strings:
            images: (B, T=22, V=4, D=768) float32
            image_mask: (B, T=22, V=4) bool
            fluorescence: (B, T=22, F) float32
            fluor_mask: (B, T=22) bool
            environment: (B, T=22, E=5) float32
            vi: (B, T=22, VI=11) float32
            temporal_positions: (B, T=22) float32
            dag_target: (B,) float32
            dag_category: (B,) long
            fw_target: (B,) float32
            dw_target: (B,) float32
            trajectory_target: (B, T=22) float32
            trajectory_mask: (B, T=22) bool
            plant_id: List[str]
            treatment: List[str]
            accession: List[str]
    """
    # Stack fixed-size tensors
    result: Dict[str, Any] = {
        'images': torch.stack([item['images'] for item in batch]),
        'image_mask': torch.stack([item['image_mask'] for item in batch]),
        'fluorescence': torch.stack([item['fluorescence'] for item in batch]),
        'fluor_mask': torch.stack([item['fluor_mask'] for item in batch]),
        'environment': torch.stack([item['environment'] for item in batch]),
        'vi': torch.stack([item['vi'] for item in batch]),
        'temporal_positions': torch.stack([item['temporal_positions'] for item in batch]),
        'dag_target': torch.tensor([item['dag_target'] for item in batch], dtype=torch.float32),
        'dag_category': torch.stack([item['dag_category'] for item in batch]),
        'fw_target': torch.tensor([item['fw_target'] for item in batch], dtype=torch.float32),
        'dw_target': torch.tensor([item['dw_target'] for item in batch], dtype=torch.float32),
        'trajectory_target': torch.stack([item['trajectory_target'] for item in batch]),
        'trajectory_mask': torch.stack([item['trajectory_mask'] for item in batch]),
    }
    
    # Collect string fields as lists
    result['plant_id'] = [item['plant_id'] for item in batch]
    result['treatment'] = [item['treatment'] for item in batch]
    result['accession'] = [item['accession'] for item in batch]
    
    return result
