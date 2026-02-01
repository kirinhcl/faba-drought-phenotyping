"""Fluorescence change point detection for pre-symptomatic analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.data.dataset import ROUND_TO_DAG


def detect_fluorescence_changepoints(
    config: Any,
) -> Dict[str, float]:
    """Detect fluorescence change points for each drought genotype.
    
    Uses Fv/Fm steady-state as indicator of photosynthetic stress.
    Builds control baseline from WHC-80 plants, detects first round where
    drought plant's Fv/Fm deviates by >2 std from control mean.
    
    Args:
        config: OmegaConf config with data.raw.fluorescence path
    
    Returns:
        {genotype: fluor_change_dag} - median across 3 drought reps
    """
    # Load fluorescence data
    fluor_path = Path(config.data.raw.fluorescence)
    df = pd.read_excel(fluor_path)
    
    # Normalize Plant ID
    df['Plant ID'] = df['Plant ID'].apply(lambda x: str(x).strip())
    
    # Find Fv/Fm column (steady-state variant if multiple)
    fvfm_col: Optional[str] = None
    for col in df.columns:
        if 'Fv/Fm' in col:
            # Prefer steady-state variant
            if 'ss' in col.lower() or 'steady' in col.lower():
                fvfm_col = col
                break
            elif fvfm_col is None:
                fvfm_col = col
    
    if fvfm_col is None:
        raise ValueError("Fv/Fm column not found in fluorescence data")
    
    print(f"Using fluorescence indicator: {fvfm_col}")
    
    # Build control baseline per round
    control_baseline: Dict[int, Dict[str, float]] = {}
    
    for round_num in df['Round Order'].unique():
        round_df = df[df['Round Order'] == round_num]
        
        # Get WHC-80 plants for this round
        control_plants = []
        for _, row in round_df.iterrows():
            plant_id = str(row['Plant ID'])
            # WHC-80 plants have plant_id ending with -1, -2, -3 for reps
            # But treatment is more reliable - we need to infer from plant_id pattern
            # Assume plants ending in 4-6 are WHC-30, 1-3 are WHC-80
            if plant_id[-1] in ['1', '2', '3']:
                fvfm_val = row[fvfm_col]
                if pd.notna(fvfm_val):
                    control_plants.append(float(fvfm_val))
        
        if len(control_plants) > 0:
            control_baseline[int(round_num)] = {
                'mean': float(np.mean(control_plants)),
                'std': float(np.std(control_plants)),
            }
    
    # Detect change points for each drought plant
    plant_changepoints: Dict[str, float] = {}
    
    for plant_id in df['Plant ID'].unique():
        plant_id_str = str(plant_id)
        
        # Skip control plants (assume ending in 1-3 are control)
        if plant_id_str[-1] in ['1', '2', '3']:
            continue
        
        plant_df = df[df['Plant ID'] == plant_id_str].sort_values('Round Order')
        
        fluor_change_round: Optional[int] = None
        
        for _, row in plant_df.iterrows():
            round_num = int(row['Round Order'])
            fvfm_val = row[fvfm_col]
            
            if pd.isna(fvfm_val):
                continue
            
            if round_num not in control_baseline:
                continue
            
            baseline = control_baseline[round_num]
            z_score = abs((float(fvfm_val) - baseline['mean']) / max(baseline['std'], 1e-6))
            
            if z_score > 2.0:
                fluor_change_round = round_num
                break
        
        # Convert to DAG
        if fluor_change_round is not None:
            fluor_change_dag = float(ROUND_TO_DAG.get(fluor_change_round, 0))
            plant_changepoints[plant_id_str] = fluor_change_dag
    
    # Aggregate per genotype (median across reps)
    # Need to infer accession from plant_id pattern
    # Assuming plant_id format: <accession>-<rep>-<treatment_marker>
    genotype_changepoints: Dict[str, float] = {}
    
    # Group plants by accession
    accession_plants: Dict[str, list[float]] = {}
    
    for plant_id, change_dag in plant_changepoints.items():
        # Extract accession from plant_id (remove last 2 chars: -X)
        accession = plant_id[:-2] if len(plant_id) > 2 else plant_id
        
        if accession not in accession_plants:
            accession_plants[accession] = []
        accession_plants[accession].append(change_dag)
    
    # Compute median per accession
    for accession, dags in accession_plants.items():
        if len(dags) > 0:
            genotype_changepoints[accession] = float(np.median(dags))
    
    return genotype_changepoints
