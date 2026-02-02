"""PyTorch dataset for Faba Bean drought phenotyping multimodal time-series data.

Loads pre-extracted image features, fluorescence measurements, environment data,
watering records, and trajectory targets for all 264 plants across 22 timepoints.
"""

from __future__ import annotations

import json
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.config import load_config


# Canonical round to DAG mapping (simplified from metadata)
ROUND_TO_DAG = {
    2: 4, 3: 5, 4: 6, 5: 7, 6: 10, 7: 12,
    8: 13, 9: 14, 10: 17, 11: 19, 12: 20, 13: 21,
    14: 24, 15: 27, 16: 28, 17: 29, 18: 31, 19: 33,
    20: 34, 21: 35, 22: 38, 23: 38
}

# Round to measurement date mapping
ROUND_TO_DATE = {
    2: '2024-10-15', 3: '2024-10-16', 4: '2024-10-17', 5: '2024-10-18',
    6: '2024-10-21', 7: '2024-10-23', 8: '2024-10-24', 9: '2024-10-25',
    10: '2024-10-28', 11: '2024-10-30', 12: '2024-10-31', 13: '2024-11-01',
    14: '2024-11-04', 15: '2024-11-07', 16: '2024-11-08', 17: '2024-11-09',
    18: '2024-11-11', 19: '2024-11-13', 20: '2024-11-14', 21: '2024-11-15',
    22: '2024-11-18', 23: '2024-11-18'
}

EXPERIMENT_START = '2024-10-11'


def _get_plant_id_col(df: pd.DataFrame) -> str:
    """Return the plant ID column name, preferring 'Plant ID' then 'Tray ID'."""
    if 'Plant ID' in df.columns:
        return 'Plant ID'
    if 'Tray ID' in df.columns:
        return 'Tray ID'
    raise KeyError(f"No plant identifier column found. Columns: {list(df.columns)}")


class FabaDroughtDataset(Dataset[Dict[str, Any]]):
    """Multimodal time-series dataset for faba bean drought phenotyping.
    
    Loads:
    - Pre-extracted image features (DINOv2/CLIP/BioCLIP) from HDF5
    - Fluorescence measurements (93 parameters)
    - Environment data (5 parameters: light, temp, humidity)
    - Watering records (5 parameters per plant)
    - Trajectory targets (digital biomass norm)
    - Endpoint targets (DAG onset, biomass)
    
    Args:
        config_path_or_cfg: Path to YAML config file or OmegaConf DictConfig
    
    Returns:
        Dict with keys:
            images: (T=22, V=4, D=768) float32 - Image features
            image_mask: (T=22, V=4) bool - Valid image features
            fluorescence: (T=22, F=fluor_dim) float32 - Fluorescence params
            fluor_mask: (T=22,) bool - Valid fluorescence measurements
            environment: (T=22, E=5) float32 - Environment params
            watering: (T=22, W=5) float32 - Watering params
            temporal_positions: (T=22,) float32 - DAG values
            dag_target: float - Drought onset DAG (NaN for WHC-80)
            dag_category: long - 0=Early, 1=Mid, 2=Late (-1 for WHC-80)
            fw_target: float - Fresh weight in grams (NaN if missing)
            dw_target: float - Dry weight in grams (NaN if missing)
            trajectory_target: (T=22,) float32 - Digital biomass norm
            trajectory_mask: (T=22,) bool - Valid trajectory points
            plant_id: str
            treatment: str - 'WHC-80' or 'WHC-30'
            accession: str
    """
    
    def __init__(self, config_path_or_cfg: Union[str, DictConfig]) -> None:
        # Load config
        if isinstance(config_path_or_cfg, str):
            self.cfg = load_config(config_path_or_cfg)
        else:
            self.cfg = config_path_or_cfg
        
        # Load plant metadata
        metadata_path = Path(self.cfg.data.plant_metadata)
        self.plant_metadata = pd.read_csv(metadata_path)
        self.plant_metadata['accession'] = self.plant_metadata['accession'].apply(
            lambda x: unicodedata.normalize('NFC', str(x))
        )
        
        # Map image encoder to feature file
        encoder_to_file = {
            "facebook/dinov2-base": "dinov2_features.h5",
            "openai/clip-vit-base-patch16": "clip_features.h5",
            "imageomics/bioclip": "bioclip_features.h5",
        }
        feature_file = encoder_to_file[self.cfg.model.image_encoder]
        feature_path = Path(self.cfg.data.feature_dir) / feature_file
        
        # Open HDF5 file handle (keep open for performance)
        self.h5 = h5py.File(feature_path, 'r')
        
        # Load fluorescence data
        self.fluor_data, self.fluor_dim = self._load_fluorescence()
        
        # Load environment data (global per round)
        self.env_data = self._load_environment()
        
        # Load watering data (per plant per round)
        self.watering_data = self._load_watering()
        
        # Load trajectory targets (digital biomass norm)
        self.trajectory_data = self._load_trajectory()
        
        # Store temporal positions (DAG values for rounds 2-23)
        self.temporal_positions = torch.tensor(
            [ROUND_TO_DAG[r] for r in range(2, 24)],
            dtype=torch.float32
        )
        
        # Build plant list
        self.plants = list(self.plant_metadata.iterrows())
    
    def __len__(self) -> int:
        return len(self.plants)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        plant_idx, plant_row = self.plants[idx]
        # Access DataFrame at index for type-safe access
        plant_id = str(self.plant_metadata.at[plant_idx, 'plant_id'])
        treatment = str(self.plant_metadata.at[plant_idx, 'treatment'])
        accession = str(self.plant_metadata.at[plant_idx, 'accession'])
        
        # Parse available rounds from JSON
        available_rounds_str = str(self.plant_metadata.at[plant_idx, 'available_timepoints'])
        available_rounds = json.loads(available_rounds_str)
        
        # Initialize tensors
        T, V, D = 22, 4, self.cfg.model.encoder_output_dim
        images = torch.zeros((T, V, D), dtype=torch.float32)
        image_mask = torch.zeros((T, V), dtype=torch.bool)
        fluorescence = torch.zeros((T, self.fluor_dim), dtype=torch.float32)
        fluor_mask = torch.zeros(T, dtype=torch.bool)
        environment = torch.zeros((T, 5), dtype=torch.float32)
        watering = torch.zeros((T, 5), dtype=torch.float32)
        trajectory_target = torch.zeros(T, dtype=torch.float32)
        trajectory_mask = torch.zeros(T, dtype=torch.bool)
        
        # Load image features from HDF5
        view_keys = ['side_000', 'side_120', 'side_240', 'top']
        for t_idx, round_num in enumerate(range(2, 24)):
            if plant_id in self.h5:
                plant_group = self.h5[plant_id]
                if str(round_num) in plant_group:
                    round_group = plant_group[str(round_num)]
                    for v_idx, view_key in enumerate(view_keys):
                        if view_key in round_group:
                            feature = round_group[view_key][:]
                            images[t_idx, v_idx] = torch.from_numpy(feature)
                            image_mask[t_idx, v_idx] = True
        
        # Load fluorescence measurements
        if plant_id in self.fluor_data:
            for round_num, fluor_vec in self.fluor_data[plant_id].items():
                if 2 <= round_num <= 23:
                    t_idx = round_num - 2
                    fluorescence[t_idx] = torch.from_numpy(fluor_vec)
                    fluor_mask[t_idx] = True
        
        # Load environment data (global per round)
        for t_idx, round_num in enumerate(range(2, 24)):
            if round_num in self.env_data:
                environment[t_idx] = torch.from_numpy(self.env_data[round_num])
        
        # Load watering data (per plant)
        if plant_id in self.watering_data:
            for round_num, water_vec in self.watering_data[plant_id].items():
                if 2 <= round_num <= 23:
                    t_idx = round_num - 2
                    watering[t_idx] = torch.from_numpy(water_vec)
        
        # Load trajectory targets
        if plant_id in self.trajectory_data:
            for round_num, biomass_val in self.trajectory_data[plant_id].items():
                if 2 <= round_num <= 23:
                    t_idx = round_num - 2
                    trajectory_target[t_idx] = biomass_val
                    trajectory_mask[t_idx] = True
        
        # Endpoint targets
        dag_onset_val = self.plant_metadata.at[plant_idx, 'dag_drought_onset']
        try:
            dag_target = float(dag_onset_val)
        except (ValueError, TypeError):
            dag_target = float('nan')
        
        fw_val = self.plant_metadata.at[plant_idx, 'fw_g']
        try:
            fw_target = float(fw_val)
        except (ValueError, TypeError):
            fw_target = float('nan')
        
        dw_val = self.plant_metadata.at[plant_idx, 'dw_g']
        try:
            dw_target = float(dw_val)
        except (ValueError, TypeError):
            dw_target = float('nan')
        
        # DAG category (Early=0, Mid=1, Late=2, WHC-80=-1)
        drought_cat = self.plant_metadata.at[plant_idx, 'drought_category']
        try:
            is_valid_cat = drought_cat == drought_cat  # NaN != NaN, so this checks for non-NaN
        except Exception:
            is_valid_cat = False
        
        if is_valid_cat and drought_cat is not None:
            cat_map = {'Early': 0, 'Mid': 1, 'Late': 2}
            dag_category = cat_map.get(str(drought_cat), -1)
        else:
            dag_category = -1
        
        return {
            'images': images,
            'image_mask': image_mask,
            'fluorescence': fluorescence,
            'fluor_mask': fluor_mask,
            'environment': environment,
            'watering': watering,
            'temporal_positions': self.temporal_positions.clone(),
            'dag_target': dag_target,
            'dag_category': torch.tensor(dag_category, dtype=torch.long),
            'fw_target': fw_target,
            'dw_target': dw_target,
            'trajectory_target': trajectory_target,
            'trajectory_mask': trajectory_mask,
            'plant_id': plant_id,
            'treatment': treatment,
            'accession': accession,
        }
    
    def _load_fluorescence(self) -> tuple[Dict[str, Dict[int, npt.NDArray[np.float32]]], int]:
        """Load fluorescence data from Excel file.
        
        Returns:
            (fluor_data, fluor_dim) where:
            - fluor_data: {plant_id: {round_order: np.array(fluor_dim,)}}
            - fluor_dim: number of fluorescence parameters
        """
        fluor_path = Path(self.cfg.data.raw.fluorescence)
        df = pd.read_excel(fluor_path)
        
        # Normalize Plant ID for matching
        df['Plant ID'] = df['Plant ID'].apply(lambda x: str(x).strip())
        
        # Identify fluorescence parameter columns (columns from index 15 onwards that are numeric)
        fluor_cols = []
        for col_idx in range(15, len(df.columns)):
            col_name = df.columns[col_idx]
            if pd.api.types.is_numeric_dtype(df[col_name]):
                fluor_cols.append(col_name)
        
        fluor_dim = len(fluor_cols)
        
        # Build nested dict: {plant_id: {round: array}}
        fluor_data = {}
        for _, row in df.iterrows():
            plant_id = str(row['Plant ID'])
            round_order = int(row['Round Order'])
            fluor_values = np.array(row[fluor_cols].tolist(), dtype=np.float32)
            np.nan_to_num(fluor_values, copy=False, nan=0.0)
            
            if plant_id not in fluor_data:
                fluor_data[plant_id] = {}
            fluor_data[plant_id][round_order] = fluor_values
        
        return fluor_data, fluor_dim
    
    def _load_environment(self) -> Dict[int, npt.NDArray[np.float32]]:
        """Load environment data aggregated per round.
        
        Returns:
            {round: np.array(5,)} - Environment parameters for each round
        """
        env_path = Path(self.cfg.data.raw.env_data)
        df = pd.read_excel(env_path)
        
        # Environment columns
        env_cols = [
            'li1_Buffer_uE',
            't1_Buffer_C',
            'rh1_Buffer_%',
            't2_Tunnel_C',
            'rh2_Tunnel_%'
        ]
        
        # Aggregate to daily means
        # Assuming first column is timestamp
        timestamp_col = df.columns[0]
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df['date'] = df[timestamp_col].dt.date
        
        daily_means = df.groupby('date')[env_cols].mean()
        
        # Map to rounds using ROUND_TO_DATE
        env_data = {}
        for round_num, date_str in ROUND_TO_DATE.items():
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            if date in daily_means.index:
                env_data[round_num] = daily_means.loc[date].values.astype(np.float32)
        
        return env_data
    
    def _load_watering(self) -> Dict[str, Dict[int, npt.NDArray[np.float32]]]:
        """Load watering data aggregated per plant per round window.
        
        Returns:
            {plant_id: {round: np.array(5,)}} - Watering parameters
            5 values: Water Added (sum), WHC.Bf (last), WHC.Af (last),
                     Water Loss (sum), Water Loss per Hours (mean)
        """
        water_path = Path(self.cfg.data.raw.watering)
        df = pd.read_excel(water_path)
        
        # Clean Plant ID
        df['Plant ID'] = df['Plant ID'].apply(lambda x: str(x).strip())
        date_col = 'Measuring Date' if 'Measuring Date' in df.columns else 'Date'
        df['Date'] = pd.to_datetime(df[date_col]).dt.date
        
        watering_data = {}
        
        # Get unique plants
        plants = df['Plant ID'].unique()
        
        for plant_id in plants:
            plant_df = df[df['Plant ID'] == plant_id]
            watering_data[plant_id] = {}
            
            for round_num in range(2, 24):
                # Define window for this round
                if round_num == 2:
                    start_date = datetime.strptime(EXPERIMENT_START, '%Y-%m-%d').date()
                else:
                    prev_date = datetime.strptime(ROUND_TO_DATE[round_num - 1], '%Y-%m-%d').date()
                    start_date = prev_date + timedelta(days=1)
                
                end_date = datetime.strptime(ROUND_TO_DATE[round_num], '%Y-%m-%d').date()
                
                # Filter data in window
                window_df = plant_df[
                    (plant_df['Date'] >= start_date) &
                    (plant_df['Date'] <= end_date)
                ]
                
                if len(window_df) > 0:
                    water_added = float(window_df['Water Added'].sum())
                    whc_bf_list = window_df['WHC.Bf'].tolist()
                    whc_bf = float(whc_bf_list[-1])
                    whc_af_list = window_df['WHC.Af'].tolist()
                    whc_af = float(whc_af_list[-1])
                    water_loss = float(window_df['Water Loss'].sum())
                    water_loss_per_hr = float(window_df['Water Loss per Hours'].mean())
                    
                    watering_data[plant_id][round_num] = np.array([
                        water_added, whc_bf, whc_af, water_loss, water_loss_per_hr
                    ], dtype=np.float32)
                elif round_num == 23:
                    # Round 23 has same date as Round 22, copy Round 22 values
                    if 22 in watering_data[plant_id]:
                        watering_data[plant_id][23] = watering_data[plant_id][22].copy()
        
        return watering_data
    
    def _load_trajectory(self) -> Dict[str, Dict[int, float]]:
        """Load digital biomass trajectory targets.
        
        Returns:
            {plant_id: {round_order: float}} - Digital biomass norm values
        """
        traj_path = Path(self.cfg.data.raw.digital_biomass_norm)
        df = pd.read_excel(traj_path)
        
        plant_id_col = _get_plant_id_col(df)
        df[plant_id_col] = df[plant_id_col].apply(lambda x: str(x).strip())
        
        trajectory_data = {}
        for _, row in df.iterrows():
            plant_id = row[plant_id_col]
            round_order = int(row['Round Order'])
            biomass_norm = float(row['Digital Biomass Norm (e+2)'])
            
            if plant_id not in trajectory_data:
                trajectory_data[plant_id] = {}
            trajectory_data[plant_id][round_order] = biomass_norm
        
        return trajectory_data
    
    def __del__(self):
        """Close HDF5 file on cleanup."""
        if hasattr(self, 'h5'):
            self.h5.close()
