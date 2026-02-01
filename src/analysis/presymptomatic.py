"""Pre-symptomatic quantification via three-way triangulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.analysis.attention import find_attention_peak


def compute_triangulation(
    attention_data: Dict[str, Dict[str, Any]],
    fluor_changepoints: Dict[str, float],
    plant_metadata: pd.DataFrame,
    output_dir: Path,
) -> Dict[str, Any]:
    """Compute three-way triangulation of drought onset detection.
    
    For each drought genotype, align three timestamps:
    1. fluor_change_dag: Fluorescence Fv/Fm change point
    2. attention_peak_dag: Model attention peak
    3. human_dag: Human-annotated drought onset
    
    Test ordering hypothesis: fluor ≤ attention ≤ human
    Compute pre-symptomatic lead time and model-fluorescence gap.
    
    Args:
        attention_data: Output from extract_attention_maps()
        fluor_changepoints: Output from detect_fluorescence_changepoints()
        plant_metadata: Plant metadata DataFrame
        output_dir: Directory to save results
    
    Returns:
        Triangulation summary statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize accessions in metadata
    import unicodedata
    plant_metadata['accession'] = plant_metadata['accession'].apply(
        lambda x: unicodedata.normalize('NFC', str(x))
    )
    
    # Build genotype to human DAG mapping
    genotype_human_dag: Dict[str, float] = {}
    
    for _, row in plant_metadata.iterrows():
        if row['treatment'] != 'WHC-30':
            continue
        
        accession = str(row['accession'])
        dag_onset = row['dag_drought_onset']
        
        if pd.notna(dag_onset):
            genotype_human_dag[accession] = float(dag_onset)
    
    # Collect triangulation data
    triangulation_records = []
    
    for accession in attention_data.keys():
        # Get attention peak
        attention_map = attention_data[accession]['attention_map']
        attention_peak_dag = find_attention_peak(attention_map)
        
        # Get fluorescence change point
        fluor_change_dag = fluor_changepoints.get(accession)
        
        # Get human annotation
        human_dag = genotype_human_dag.get(accession)
        
        if fluor_change_dag is None or human_dag is None:
            continue
        
        # Compute metrics
        ordering_correct = (fluor_change_dag <= attention_peak_dag <= human_dag)
        lead_time_days = human_dag - attention_peak_dag
        model_fluor_gap_days = abs(attention_peak_dag - fluor_change_dag)
        is_presymptomatic = (attention_peak_dag < human_dag - 2)  # ≥2 timepoints early
        
        triangulation_records.append({
            'accession': accession,
            'fluor_change_dag': float(fluor_change_dag),
            'attention_peak_dag': float(attention_peak_dag),
            'human_dag': float(human_dag),
            'ordering_correct': bool(ordering_correct),
            'lead_time_days': float(lead_time_days),
            'model_fluor_gap_days': float(model_fluor_gap_days),
            'is_presymptomatic': bool(is_presymptomatic),
        })
    
    # Compute summary statistics
    n_total = len(triangulation_records)
    n_correct_ordering = sum(r['ordering_correct'] for r in triangulation_records)
    n_presymptomatic = sum(r['is_presymptomatic'] for r in triangulation_records)
    
    lead_times = [r['lead_time_days'] for r in triangulation_records]
    model_fluor_gaps = [r['model_fluor_gap_days'] for r in triangulation_records]
    
    # Pearson correlation between fluorescence and attention
    fluor_vals = [r['fluor_change_dag'] for r in triangulation_records]
    attn_vals = [r['attention_peak_dag'] for r in triangulation_records]
    
    if len(fluor_vals) > 1:
        fluor_attention_pearson_r = float(np.corrcoef(fluor_vals, attn_vals)[0, 1])
    else:
        fluor_attention_pearson_r = 0.0
    
    summary = {
        'n_total_drought_genotypes': n_total,
        'n_correct_ordering': n_correct_ordering,
        'ordering_accuracy': n_correct_ordering / max(n_total, 1),
        'n_presymptomatic': n_presymptomatic,
        'presymptomatic_rate': n_presymptomatic / max(n_total, 1),
        'mean_fluor_change_dag': float(np.mean([r['fluor_change_dag'] for r in triangulation_records])),
        'mean_attention_peak_dag': float(np.mean([r['attention_peak_dag'] for r in triangulation_records])),
        'mean_human_dag': float(np.mean([r['human_dag'] for r in triangulation_records])),
        'mean_lead_time_days': float(np.mean(lead_times)),
        'std_lead_time_days': float(np.std(lead_times)),
        'mean_model_fluor_gap_days': float(np.mean(model_fluor_gaps)),
        'std_model_fluor_gap_days': float(np.std(model_fluor_gaps)),
        'fluor_attention_pearson_r': fluor_attention_pearson_r,
    }
    
    # Negative control: WHC-80 plants should not show attention peaks at drought timepoints
    whc80_attention_peaks = []
    for accession, data in attention_data.items():
        if data['treatment_counts']['WHC-80'] > 0:
            attention_map = data['attention_map']
            peak_dag = find_attention_peak(attention_map)
            whc80_attention_peaks.append(peak_dag)
    
    summary['negative_control_whc80_mean_peak_dag'] = float(np.mean(whc80_attention_peaks)) if whc80_attention_peaks else 0.0
    summary['negative_control_whc80_std_peak_dag'] = float(np.std(whc80_attention_peaks)) if whc80_attention_peaks else 0.0
    
    # Save detailed records
    triangulation_path = output_dir / 'triangulation_summary.json'
    with open(triangulation_path, 'w') as f:
        json.dump({
            'summary': summary,
            'records': triangulation_records,
        }, f, indent=2)
    
    # Save presymptomatic summary
    presymptomatic_summary = {
        'n_presymptomatic': n_presymptomatic,
        'n_total': n_total,
        'presymptomatic_rate': n_presymptomatic / max(n_total, 1),
        'presymptomatic_genotypes': [
            r['accession'] for r in triangulation_records if r['is_presymptomatic']
        ],
    }
    
    presymp_path = output_dir / 'presymptomatic_summary.json'
    with open(presymp_path, 'w') as f:
        json.dump(presymptomatic_summary, f, indent=2)
    
    print(f"\nTriangulation Results:")
    print(f"  Total genotypes: {n_total}")
    print(f"  Correct ordering (fluor ≤ attn ≤ human): {n_correct_ordering}/{n_total} ({summary['ordering_accuracy']:.1%})")
    print(f"  Pre-symptomatic detections: {n_presymptomatic}/{n_total} ({summary['presymptomatic_rate']:.1%})")
    print(f"  Mean lead time: {summary['mean_lead_time_days']:.1f} ± {summary['std_lead_time_days']:.1f} days")
    print(f"  Fluorescence-Attention correlation: r = {fluor_attention_pearson_r:.3f}")
    print(f"\nResults saved to:")
    print(f"  {triangulation_path}")
    print(f"  {presymp_path}")
    
    return summary
