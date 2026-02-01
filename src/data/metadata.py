"""
Generate canonical metadata files from raw experimental data.

This module reads raw Excel files and image directories, reconciles naming
inconsistencies, and outputs two CSV files:
- data/plant_metadata.csv: Per-plant metadata (264 rows, 15 columns)
- data/timepoint_metadata.csv: Per-timepoint metadata (22 rows, 9 columns)
"""

import json
import os
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


ROUND_TO_DAG = {
    2: (3, 10, "2024-10-14"),
    3: (5, 12, "2024-10-16"),
    4: (6, 13, "2024-10-17"),
    5: (7, 14, "2024-10-18"),
    6: (10, 17, "2024-10-21"),
    7: (12, 19, "2024-10-23"),
    8: (13, 20, "2024-10-24"),
    9: (14, 21, "2024-10-25"),
    10: (17, 24, "2024-10-28"),
    11: (19, 26, "2024-10-30"),
    12: (20, 27, "2024-10-31"),
    13: (21, 28, "2024-11-01"),
    14: (24, 31, "2024-11-04"),
    15: (27, 34, "2024-11-07"),
    16: (28, 35, "2024-11-08"),
    17: (29, 36, "2024-11-09"),
    18: (31, 38, "2024-11-11"),
    19: (33, 40, "2024-11-13"),
    20: (34, 41, "2024-11-14"),
    21: (35, 42, "2024-11-15"),
    22: (38, 45, "2024-11-18"),
    23: (38, 45, "2024-11-18"),
}

REP1_ROUNDS = [2, 3, 6, 7, 10, 11, 14, 17, 18, 19, 23]
REP2_ROUNDS = [2, 4, 6, 8, 10, 12, 14, 15, 18, 20, 22, 23]
REP3_ROUNDS = [2, 5, 6, 9, 10, 13, 14, 16, 18, 21, 23]


def normalize_unicode(s: str) -> str:
    # macOS HFS+ uses NFD decomposition; NFC ensures consistent matching across filesystem and Excel
    return unicodedata.normalize('NFC', s)


def canonical_to_slug(canonical: str) -> str:
    return canonical.replace('/', '_')


def slug_to_canonical_pattern(slug: str) -> str:
    return slug.replace('_', '/').lower()


def normalize_treatment(treatment: str) -> str:
    # FabaDr_Obs uses WHC-70%/WHC-40%, but canonical output is WHC-80/WHC-30
    treatment = treatment.strip()
    if treatment in ['WHC-70%', 'WHC-70']:
        return 'WHC-80'
    elif treatment in ['WHC-40%', 'WHC-40']:
        return 'WHC-30'
    elif treatment in ['WHC-80%', 'WHC-80']:
        return 'WHC-80'
    elif treatment in ['WHC-30%', 'WHC-30']:
        return 'WHC-30'
    else:
        raise ValueError(f"Unknown treatment: {treatment}")


def scan_plant_images(plant_dir: Path) -> Tuple[List[int], int, int]:
    rounds_set = set()
    num_side = 0
    num_top = 0
    
    for angle in ['000', '120', '240']:
        angle_dir = plant_dir / angle
        if angle_dir.exists():
            for img_file in angle_dir.glob('124-*-RGB1-*-FishEyeMasked.png'):
                num_side += 1
                parts = img_file.stem.split('-')
                if len(parts) >= 2:
                    try:
                        round_num = int(parts[1])
                        rounds_set.add(round_num)
                    except ValueError:
                        pass
    
    return sorted(rounds_set), num_side, 0


def scan_top_view_images(plant_dir: Path) -> int:
    if not plant_dir.exists():
        return 0
    return len(list(plant_dir.glob('124-*-RGB2-FishEyeMasked.png')))


def load_drought_impact() -> pd.DataFrame:
    df = pd.read_excel('data/SinglePoint Datasets/Drought_Impact(DAG).xlsx')
    df['Accession Name'] = df['Accession Name'].apply(normalize_unicode)
    return df


def load_endpoint_data() -> pd.DataFrame:
    df = pd.read_excel('data/00-Misc/EndPoint_Raw_FW&DW.xlsx')
    df = df.rename(columns={'TrayID': 'Plant ID'})
    
    df['Plant+Bag FW (g)'] = pd.to_numeric(df['Plant+Bag FW (g)'], errors='coerce')
    df['P+B DW (g)'] = pd.to_numeric(df['P+B DW (g)'], errors='coerce')
    
    df['FW'] = df['Plant+Bag FW (g)'] - df['Bag Weight (g)']
    df['DW'] = df['P+B DW (g)'] - df['Bag Weight (g)']
    
    return df[['Plant ID', 'FW', 'DW', 'Treatment']]


def load_dead_plants() -> Set[str]:
    df = pd.read_excel('data/00-Misc/FabaDr_Obs.xlsx', sheet_name='RGB1')
    dead = df[
        (df['Outlier'] == 'Yes') & 
        (df['Obs'].str.contains('plant died', case=False, na=False))
    ]
    return set(dead['Plant ID'].unique())


def match_accession_name(slug: str, canonical_names: List[str]) -> str:
    slug_normalized = normalize_unicode(slug).lower()
    
    for canonical in canonical_names:
        canonical_normalized = normalize_unicode(canonical).lower()
        canonical_as_slug = canonical_normalized.replace('/', '_')
        
        if slug_normalized == canonical_as_slug:
            return canonical
    
    return slug.replace('_', '/')


def generate_plant_metadata() -> pd.DataFrame:
    drought_df = load_drought_impact()
    endpoint_df = load_endpoint_data()
    dead_plants = load_dead_plants()
    
    canonical_names = drought_df['Accession Name'].tolist()
    dag_lookup = dict(zip(drought_df['Accession Name'], drought_df['Drought Impact (DAG)']))
    category_lookup = dict(zip(drought_df['Accession Name'], drought_df['Stress Impact']))
    
    endpoint_df['Treatment'] = endpoint_df['Treatment'].apply(normalize_treatment)
    fw_lookup = dict(zip(endpoint_df['Plant ID'], endpoint_df['FW']))
    dw_lookup = dict(zip(endpoint_df['Plant ID'], endpoint_df['DW']))
    
    side_view_root = Path('data/img/side_view')
    top_view_root = Path('data/img/top_view')
    
    records = []
    
    for acc_dir in sorted(side_view_root.iterdir()):
        if not acc_dir.is_dir():
            continue
        
        acc_parts = acc_dir.name.split(' - ', 1)
        if len(acc_parts) != 2:
            continue
        
        accession_slug = acc_parts[1]
        accession = match_accession_name(accession_slug, canonical_names)
        
        for treatment_dir in sorted(acc_dir.iterdir()):
            if not treatment_dir.is_dir():
                continue
            
            treatment = normalize_treatment(treatment_dir.name)
            
            for plant_dir in sorted(treatment_dir.iterdir()):
                if not plant_dir.is_dir():
                    continue
                
                plant_parts = plant_dir.name.split(' - ', 1)
                if len(plant_parts) != 2:
                    continue
                
                replicate = int(plant_parts[0].replace('Rep-', ''))
                plant_id = plant_parts[1]
                
                available_rounds, num_side, _ = scan_plant_images(plant_dir)
                
                top_plant_dir = top_view_root / acc_dir.name / treatment_dir.name / plant_dir.name
                num_top = scan_top_view_images(top_plant_dir)
                
                dag_onset = dag_lookup.get(accession) if treatment == 'WHC-30' else None
                drought_cat = category_lookup.get(accession) if treatment == 'WHC-30' else None
                
                fw = fw_lookup.get(plant_id)
                dw = dw_lookup.get(plant_id)
                
                is_dead = plant_id in dead_plants
                
                image_side_dir = f"data/img/side_view/{acc_dir.name}/{treatment_dir.name}/{plant_dir.name}"
                image_top_dir = f"data/img/top_view/{acc_dir.name}/{treatment_dir.name}/{plant_dir.name}"
                
                records.append({
                    'plant_id': plant_id,
                    'accession': accession,
                    'accession_slug': accession_slug,
                    'treatment': treatment,
                    'replicate': replicate,
                    'dag_drought_onset': dag_onset,
                    'drought_category': drought_cat,
                    'fw_g': fw,
                    'dw_g': dw,
                    'image_side_dir': image_side_dir,
                    'image_top_dir': image_top_dir,
                    'available_timepoints': json.dumps(available_rounds),
                    'num_images_side': num_side,
                    'num_images_top': num_top,
                    'is_dead': is_dead,
                })
    
    df = pd.DataFrame(records)
    df = df.sort_values('plant_id').reset_index(drop=True)
    
    return df


def generate_timepoint_metadata() -> pd.DataFrame:
    fcq_df = pd.read_excel('data/TimeCourse Datasets/FCQ_FabaDr_Auto.xlsx')
    fluor_rounds = set(fcq_df['Round Order'].unique())
    
    records = []
    
    for round_num in sorted(ROUND_TO_DAG.keys()):
        dag, das, date = ROUND_TO_DAG[round_num]
        
        has_fluor = round_num in fluor_rounds
        rep1_images = round_num in REP1_ROUNDS
        rep2_images = round_num in REP2_ROUNDS
        rep3_images = round_num in REP3_ROUNDS
        has_images = rep1_images or rep2_images or rep3_images
        
        records.append({
            'round': round_num,
            'dag': dag,
            'das': das,
            'date': date,
            'has_images': has_images,
            'has_fluorescence': has_fluor,
            'rep1_has_images': rep1_images,
            'rep2_has_images': rep2_images,
            'rep3_has_images': rep3_images,
        })
    
    return pd.DataFrame(records)


def generate_all_metadata():
    print("Generating plant metadata...")
    plant_meta = generate_plant_metadata()
    plant_meta.to_csv('data/plant_metadata.csv', index=False)
    print(f"✓ Created data/plant_metadata.csv ({len(plant_meta)} rows)")
    
    print("\nGenerating timepoint metadata...")
    timepoint_meta = generate_timepoint_metadata()
    timepoint_meta.to_csv('data/timepoint_metadata.csv', index=False)
    print(f"✓ Created data/timepoint_metadata.csv ({len(timepoint_meta)} rows)")
    
    print("\n=== Summary ===")
    print(f"Plants: {len(plant_meta)}")
    print(f"Accessions: {plant_meta['accession'].nunique()}")
    print(f"Treatments: {plant_meta['treatment'].unique().tolist()}")
    print(f"Dead plants: {plant_meta['is_dead'].sum()}")
    print(f"Timepoints: {len(timepoint_meta)}")


if __name__ == '__main__':
    generate_all_metadata()
