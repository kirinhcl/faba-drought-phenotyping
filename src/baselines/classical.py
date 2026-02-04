"""Classical ML baselines (XGBoost, Random Forest, DINOv2+RF) for Faba Bean drought phenotyping.

Implements Leave-One-Genotype-Out (LOGO) cross-validation with:
- XGBoost on endpoint features
- Random Forest on endpoint features
- Random Forest on DINOv2 averaged features
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.training.cv import LogoCV


def load_endpoint_features(config: DictConfig) -> pd.DataFrame:
    """Load pre-computed endpoint feature matrix.
    
    Source: data/EndPoint Datasets/EndPoint_CorrelationData-WithoutOutliers.xlsx
    This is the easiest baseline — already a feature matrix.
    
    Args:
        config: OmegaConf config with data.raw.endpoint_corr path
        
    Returns:
        DataFrame indexed by plant_id with feature columns
    """
    df = pd.read_excel(config.data.raw.endpoint_corr)
    
    # Rename Tray ID to plant_id for consistency
    if 'Tray ID' in df.columns:
        df = df.rename(columns={'Tray ID': 'plant_id'})
    
    # Set plant_id as index
    df = df.set_index('plant_id')
    
    # Handle NaN values with median imputation
    imputer = SimpleImputer(strategy='median')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    return df


def load_dinov2_mean_features(config: DictConfig) -> pd.DataFrame:
    """Load DINOv2 features averaged across timepoints and views.
    
    Source: features/dinov2_features.h5
    For each plant: average CLS token across all available (round, view) pairs → 768-dim vector
    
    Args:
        config: OmegaConf config with data.feature_dir path
        
    Returns:
        DataFrame with columns dim_0...dim_767, indexed by plant_id
    """
    import h5py  # type: ignore[import-not-found]
    
    h5_path = f"{config.data.feature_dir}dinov2_features.h5"
    
    # Expected feature dimension (DINOv2-B/14)
    EXPECTED_DIM = 768
    
    plant_features = {}
    
    with h5py.File(h5_path, 'r') as f:
        for plant_id in f.keys():
            plant_group = f[plant_id]
            
            # Collect all features across rounds and views
            all_features = []
            
            for round_key in plant_group.keys():
                round_group = plant_group[round_key]
                for view_key in round_group.keys():
                    feature_vec = round_group[view_key][()]
                    
                    # Ensure feature is 1D and has expected dimension
                    feature_vec = np.asarray(feature_vec).flatten()
                    if feature_vec.shape[0] == EXPECTED_DIM:
                        all_features.append(feature_vec)
            
            if len(all_features) > 0:
                # Stack into (N, 768) array then average
                stacked = np.stack(all_features, axis=0)
                mean_feature = np.mean(stacked, axis=0)
                plant_features[plant_id] = mean_feature
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(plant_features, orient='index')
    df.index.name = 'plant_id'
    
    # Rename columns to dim_0, dim_1, ...
    df.columns = [f'dim_{i}' for i in range(df.shape[1])]
    
    return df


class ClassicalBaselines:
    """Run XGBoost, RF, and DINOv2+RF baselines with LOGO-CV."""
    
    def __init__(self, config: DictConfig) -> None:
        """Initialize baselines with config and load features.
        
        Args:
            config: OmegaConf config object
        """
        self.cfg = config
        self.plant_metadata = pd.read_csv(config.data.plant_metadata)
        
        # Load features
        self.endpoint_features = load_endpoint_features(config)
        self.dinov2_features = load_dinov2_mean_features(config)
    
    def _get_feature_columns_for_task(
        self,
        features: pd.DataFrame,
        task: str
    ) -> list[str]:
        """Get feature columns, excluding targets based on task.
        
        Args:
            features: Feature DataFrame
            task: 'dag' or 'biomass'
            
        Returns:
            List of feature column names to use
        """
        exclude_cols = set()
        
        # Columns to exclude for DAG prediction
        if task == 'dag':
            # Exclude FW and DW (targets for biomass)
            exclude_cols.update(['Fresh Weight', 'Dry Weight'])
            # Exclude DAG-derived columns
            exclude_cols.update(['Drought Impact (DAG)', 'Stress Impact'])
        
        # Columns to exclude for biomass prediction
        elif task == 'biomass':
            # Exclude FW and DW (they ARE the targets)
            exclude_cols.update(['Fresh Weight', 'Dry Weight'])
            # Exclude DAG ground truth
            exclude_cols.update(['Drought Impact (DAG)', 'Stress Impact'])
        
        # Also exclude metadata columns
        exclude_cols.update([
            'Accession Name', 'Accession Number', 'Treatment', 'Replicate'
        ])
        
        feature_cols = [
            col for col in features.columns
            if col not in exclude_cols and features[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        
        return feature_cols
    
    def _run_xgboost(self) -> dict[str, Any]:
        """Run XGBoost baseline with LOGO-CV.
        
        Returns:
            Dictionary with metrics: dag_mae, dag_rmse, biomass_r2, ranking_spearman
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
        
        logo_cv = LogoCV(self.plant_metadata, n_folds=44, seed=self.cfg.seed)
        
        dag_preds = {}
        dag_true = {}
        biomass_preds = {}
        biomass_true = {}
        
        for train_idx, val_idx, test_idx in logo_cv.split():
            # Get train/test data
            train_plants = self.plant_metadata.iloc[train_idx]['plant_id'].values
            test_plants = self.plant_metadata.iloc[test_idx]['plant_id'].values
            
            # Get features and targets
            X_train = self.endpoint_features.loc[train_plants]
            X_test = self.endpoint_features.loc[test_plants]
            
            # DAG prediction (train on WHC-30 only)
            whc30_train = self.plant_metadata.iloc[train_idx]
            whc30_train = whc30_train[whc30_train['treatment'] == 'WHC-30']
            
            # Filter to plants that exist in endpoint_features
            available_plants = set(self.endpoint_features.index)
            whc30_train = whc30_train[whc30_train['plant_id'].isin(available_plants)]
            whc30_train_plants = whc30_train['plant_id'].values
            
            if len(whc30_train_plants) > 0:
                X_dag_train = self.endpoint_features.loc[whc30_train_plants]
                y_dag_train = whc30_train['dag_drought_onset'].values
                
                # Remove NaN targets
                valid_mask = ~pd.isna(y_dag_train)
                X_dag_train = X_dag_train[valid_mask]
                y_dag_train = y_dag_train[valid_mask]
                
                if y_dag_train.shape[0] > 0:
                    dag_feature_cols = self._get_feature_columns_for_task(
                        self.endpoint_features, 'dag'
                    )
                    X_dag_train = X_dag_train[dag_feature_cols]
                    X_dag_test = X_test[dag_feature_cols]
                    
                    dag_model = xgb.XGBRegressor(
                        max_depth=6,
                        n_estimators=100,
                        learning_rate=0.1,
                        random_state=self.cfg.seed,
                        verbosity=0
                    )
                    dag_model.fit(X_dag_train, y_dag_train)
                    
                    y_dag_pred = dag_model.predict(X_dag_test)
                    
                    for plant_id, pred in zip(test_plants, y_dag_pred):
                        dag_preds[plant_id] = pred
                    
                    # Get true DAG for test plants
                    test_metadata = self.plant_metadata[
                        (self.plant_metadata['plant_id'].isin(test_plants)) &
                        (self.plant_metadata['treatment'] == 'WHC-30')
                    ]
                    for _, row in test_metadata.iterrows():
                        if pd.notna(row['dag_drought_onset']):
                            dag_true[row['plant_id']] = row['dag_drought_onset']
            
            # Biomass prediction (train on all plants)
            biomass_feature_cols = self._get_feature_columns_for_task(
                self.endpoint_features, 'biomass'
            )
            X_biomass_train = X_train[biomass_feature_cols]
            X_biomass_test = X_test[biomass_feature_cols]
            
            # FW prediction
            y_fw_train = self.plant_metadata.iloc[train_idx]['fw_g'].values
            valid_mask = ~pd.isna(y_fw_train)
            X_biomass_train_valid = X_biomass_train[valid_mask]
            y_fw_train_valid = y_fw_train[valid_mask]
            
            if y_fw_train_valid.shape[0] > 0:
                fw_model = xgb.XGBRegressor(
                    max_depth=6,
                    n_estimators=100,
                    learning_rate=0.1,
                    random_state=self.cfg.seed,
                    verbosity=0
                )
                fw_model.fit(X_biomass_train_valid, y_fw_train_valid)
                
                y_fw_pred = fw_model.predict(X_biomass_test)
                
                for plant_id, pred in zip(test_plants, y_fw_pred):
                    biomass_preds[plant_id] = pred
                
                # Get true FW for test plants
                test_metadata = self.plant_metadata[
                    self.plant_metadata['plant_id'].isin(test_plants)
                ]
                for _, row in test_metadata.iterrows():
                    if pd.notna(row['fw_g']):
                        biomass_true[row['plant_id']] = row['fw_g']
        
        # Compute metrics
        results = {}
        
        # DAG metrics
        if len(dag_preds) > 0 and len(dag_true) > 0:
            common_plants = set(dag_preds.keys()) & set(dag_true.keys())
            if len(common_plants) > 0:
                dag_pred_vals = np.array([dag_preds[p] for p in common_plants])
                dag_true_vals = np.array([dag_true[p] for p in common_plants])
                
                results['dag_mae'] = float(mean_absolute_error(dag_true_vals, dag_pred_vals))
                results['dag_rmse'] = float(np.sqrt(mean_squared_error(dag_true_vals, dag_pred_vals)))
        
        # Biomass metrics
        if len(biomass_preds) > 0 and len(biomass_true) > 0:
            common_plants = set(biomass_preds.keys()) & set(biomass_true.keys())
            if len(common_plants) > 0:
                biomass_pred_vals = np.array([biomass_preds[p] for p in common_plants])
                biomass_true_vals = np.array([biomass_true[p] for p in common_plants])
                
                results['biomass_r2'] = float(r2_score(biomass_true_vals, biomass_pred_vals))
        
        # Ranking metric
        ranking_spearman = self._compute_ranking_metric(dag_preds, dag_true)
        if ranking_spearman is not None:
            results['ranking_spearman'] = ranking_spearman
        
        return results
    
    def _run_rf(self) -> dict[str, Any]:
        """Run Random Forest baseline with LOGO-CV.
        
        Returns:
            Dictionary with metrics: dag_mae, dag_rmse, biomass_r2, ranking_spearman
        """
        logo_cv = LogoCV(self.plant_metadata, n_folds=44, seed=self.cfg.seed)
        
        dag_preds = {}
        dag_true = {}
        biomass_preds = {}
        biomass_true = {}
        
        for train_idx, val_idx, test_idx in logo_cv.split():
            # Get train/test data
            train_plants = self.plant_metadata.iloc[train_idx]['plant_id'].values
            test_plants = self.plant_metadata.iloc[test_idx]['plant_id'].values
            
            # Get features and targets
            X_train = self.endpoint_features.loc[train_plants]
            X_test = self.endpoint_features.loc[test_plants]
            
            # DAG prediction (train on WHC-30 only)
            whc30_train = self.plant_metadata.iloc[train_idx]
            whc30_train = whc30_train[whc30_train['treatment'] == 'WHC-30']
            
            # Filter to plants that exist in endpoint_features
            available_plants = set(self.endpoint_features.index)
            whc30_train = whc30_train[whc30_train['plant_id'].isin(available_plants)]
            whc30_train_plants = whc30_train['plant_id'].values
            
            if len(whc30_train_plants) > 0:
                X_dag_train = self.endpoint_features.loc[whc30_train_plants]
                y_dag_train = whc30_train['dag_drought_onset'].values
                
                # Remove NaN targets
                valid_mask = ~pd.isna(y_dag_train)
                X_dag_train = X_dag_train[valid_mask]
                y_dag_train = y_dag_train[valid_mask]
                
                if y_dag_train.shape[0] > 0:
                    dag_feature_cols = self._get_feature_columns_for_task(
                        self.endpoint_features, 'dag'
                    )
                    X_dag_train = X_dag_train[dag_feature_cols]
                    X_dag_test = X_test[dag_feature_cols]
                    
                    dag_model = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=None,
                        random_state=self.cfg.seed,
                        n_jobs=-1
                    )
                    dag_model.fit(X_dag_train, y_dag_train)
                    
                    y_dag_pred = dag_model.predict(X_dag_test)
                    
                    for plant_id, pred in zip(test_plants, y_dag_pred):
                        dag_preds[plant_id] = pred
                    
                    # Get true DAG for test plants
                    test_metadata = self.plant_metadata[
                        (self.plant_metadata['plant_id'].isin(test_plants)) &
                        (self.plant_metadata['treatment'] == 'WHC-30')
                    ]
                    for _, row in test_metadata.iterrows():
                        if pd.notna(row['dag_drought_onset']):
                            dag_true[row['plant_id']] = row['dag_drought_onset']
            
            # Biomass prediction (train on all plants)
            biomass_feature_cols = self._get_feature_columns_for_task(
                self.endpoint_features, 'biomass'
            )
            X_biomass_train = X_train[biomass_feature_cols]
            X_biomass_test = X_test[biomass_feature_cols]
            
            # FW prediction
            y_fw_train = self.plant_metadata.iloc[train_idx]['fw_g'].values
            valid_mask = ~pd.isna(y_fw_train)
            X_biomass_train_valid = X_biomass_train[valid_mask]
            y_fw_train_valid = y_fw_train[valid_mask]
            
            if y_fw_train_valid.shape[0] > 0:
                fw_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    random_state=self.cfg.seed,
                    n_jobs=-1
                )
                fw_model.fit(X_biomass_train_valid, y_fw_train_valid)
                
                y_fw_pred = fw_model.predict(X_biomass_test)
                
                for plant_id, pred in zip(test_plants, y_fw_pred):
                    biomass_preds[plant_id] = pred
                
                # Get true FW for test plants
                test_metadata = self.plant_metadata[
                    self.plant_metadata['plant_id'].isin(test_plants)
                ]
                for _, row in test_metadata.iterrows():
                    if pd.notna(row['fw_g']):
                        biomass_true[row['plant_id']] = row['fw_g']
        
        # Compute metrics
        results = {}
        
        # DAG metrics
        if len(dag_preds) > 0 and len(dag_true) > 0:
            common_plants = set(dag_preds.keys()) & set(dag_true.keys())
            if len(common_plants) > 0:
                dag_pred_vals = np.array([dag_preds[p] for p in common_plants])
                dag_true_vals = np.array([dag_true[p] for p in common_plants])
                
                results['dag_mae'] = float(mean_absolute_error(dag_true_vals, dag_pred_vals))
                results['dag_rmse'] = float(np.sqrt(mean_squared_error(dag_true_vals, dag_pred_vals)))
        
        # Biomass metrics
        if len(biomass_preds) > 0 and len(biomass_true) > 0:
            common_plants = set(biomass_preds.keys()) & set(biomass_true.keys())
            if len(common_plants) > 0:
                biomass_pred_vals = np.array([biomass_preds[p] for p in common_plants])
                biomass_true_vals = np.array([biomass_true[p] for p in common_plants])
                
                results['biomass_r2'] = float(r2_score(biomass_true_vals, biomass_pred_vals))
        
        # Ranking metric
        ranking_spearman = self._compute_ranking_metric(dag_preds, dag_true)
        if ranking_spearman is not None:
            results['ranking_spearman'] = ranking_spearman
        
        return results
    
    def _run_dinov2_rf(self) -> dict[str, Any]:
        """Run Random Forest on DINOv2 features with LOGO-CV.
        
        Returns:
            Dictionary with metrics: dag_mae, dag_rmse, biomass_r2, ranking_spearman
        """
        logo_cv = LogoCV(self.plant_metadata, n_folds=44, seed=self.cfg.seed)
        
        dag_preds = {}
        dag_true = {}
        biomass_preds = {}
        biomass_true = {}
        
        for train_idx, val_idx, test_idx in logo_cv.split():
            # Get train/test data
            train_plants = self.plant_metadata.iloc[train_idx]['plant_id'].values
            test_plants = self.plant_metadata.iloc[test_idx]['plant_id'].values
            
            # Get DINOv2 features
            X_train = self.dinov2_features.loc[train_plants]
            X_test = self.dinov2_features.loc[test_plants]
            
            # DAG prediction (train on WHC-30 only)
            whc30_train = self.plant_metadata.iloc[train_idx]
            whc30_train = whc30_train[whc30_train['treatment'] == 'WHC-30']
            
            # Filter to plants that exist in dinov2_features
            available_plants = set(self.dinov2_features.index)
            whc30_train = whc30_train[whc30_train['plant_id'].isin(available_plants)]
            whc30_train_plants = whc30_train['plant_id'].values
            
            if len(whc30_train_plants) > 0:
                X_dag_train = self.dinov2_features.loc[whc30_train_plants]
                y_dag_train = whc30_train['dag_drought_onset'].values
                
                # Remove NaN targets
                valid_mask = ~pd.isna(y_dag_train)
                X_dag_train = X_dag_train[valid_mask]
                y_dag_train = y_dag_train[valid_mask]
                
                if y_dag_train.shape[0] > 0:
                    X_dag_test = X_test
                    
                    dag_model = RandomForestRegressor(
                        n_estimators=200,
                        max_depth=None,
                        random_state=self.cfg.seed,
                        n_jobs=-1
                    )
                    dag_model.fit(X_dag_train, y_dag_train)
                    
                    y_dag_pred = dag_model.predict(X_dag_test)
                    
                    for plant_id, pred in zip(test_plants, y_dag_pred):
                        dag_preds[plant_id] = pred
                    
                    # Get true DAG for test plants
                    test_metadata = self.plant_metadata[
                        (self.plant_metadata['plant_id'].isin(test_plants)) &
                        (self.plant_metadata['treatment'] == 'WHC-30')
                    ]
                    for _, row in test_metadata.iterrows():
                        if pd.notna(row['dag_drought_onset']):
                            dag_true[row['plant_id']] = row['dag_drought_onset']
            
            # Biomass prediction (train on all plants)
            X_biomass_train = X_train
            X_biomass_test = X_test
            
            # FW prediction
            y_fw_train = self.plant_metadata.iloc[train_idx]['fw_g'].values
            valid_mask = ~pd.isna(y_fw_train)
            X_biomass_train_valid = X_biomass_train[valid_mask]
            y_fw_train_valid = y_fw_train[valid_mask]
            
            if y_fw_train_valid.shape[0] > 0:
                fw_model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=None,
                    random_state=self.cfg.seed,
                    n_jobs=-1
                )
                fw_model.fit(X_biomass_train_valid, y_fw_train_valid)
                
                y_fw_pred = fw_model.predict(X_biomass_test)
                
                for plant_id, pred in zip(test_plants, y_fw_pred):
                    biomass_preds[plant_id] = pred
                
                # Get true FW for test plants
                test_metadata = self.plant_metadata[
                    self.plant_metadata['plant_id'].isin(test_plants)
                ]
                for _, row in test_metadata.iterrows():
                    if pd.notna(row['fw_g']):
                        biomass_true[row['plant_id']] = row['fw_g']
        
        # Compute metrics
        results = {}
        
        # DAG metrics
        if len(dag_preds) > 0 and len(dag_true) > 0:
            common_plants = set(dag_preds.keys()) & set(dag_true.keys())
            if len(common_plants) > 0:
                dag_pred_vals = np.array([dag_preds[p] for p in common_plants])
                dag_true_vals = np.array([dag_true[p] for p in common_plants])
                
                results['dag_mae'] = float(mean_absolute_error(dag_true_vals, dag_pred_vals))
                results['dag_rmse'] = float(np.sqrt(mean_squared_error(dag_true_vals, dag_pred_vals)))
        
        # Biomass metrics
        if len(biomass_preds) > 0 and len(biomass_true) > 0:
            common_plants = set(biomass_preds.keys()) & set(biomass_true.keys())
            if len(common_plants) > 0:
                biomass_pred_vals = np.array([biomass_preds[p] for p in common_plants])
                biomass_true_vals = np.array([biomass_true[p] for p in common_plants])
                
                results['biomass_r2'] = float(r2_score(biomass_true_vals, biomass_pred_vals))
        
        # Ranking metric
        ranking_spearman = self._compute_ranking_metric(dag_preds, dag_true)
        if ranking_spearman is not None:
            results['ranking_spearman'] = ranking_spearman
        
        return results
    
    def _compute_ranking_metric(
        self,
        predictions: dict[str, float],
        ground_truth: dict[str, float]
    ) -> Optional[float]:
        """Compute Spearman correlation for genotype DAG ranking.
        
        For each genotype: average predicted DAG across its 3 drought reps → rank
        Compare rank vector to true DAG rank vector using Spearman ρ
        
        Args:
            predictions: {plant_id: predicted_dag}
            ground_truth: {plant_id: true_dag}
            
        Returns:
            Spearman correlation coefficient or None if insufficient data
        """
        if len(predictions) == 0 or len(ground_truth) == 0:
            return None
        
        # Group by genotype
        genotype_preds = {}
        genotype_true = {}
        
        for plant_id, pred in predictions.items():
            # Find genotype for this plant
            plant_row = self.plant_metadata[self.plant_metadata['plant_id'] == plant_id]
            if len(plant_row) > 0:
                genotype = plant_row.iloc[0]['accession']
                if genotype not in genotype_preds:
                    genotype_preds[genotype] = []
                genotype_preds[genotype].append(pred)
        
        for plant_id, true_val in ground_truth.items():
            # Find genotype for this plant
            plant_row = self.plant_metadata[self.plant_metadata['plant_id'] == plant_id]
            if len(plant_row) > 0:
                genotype = plant_row.iloc[0]['accession']
                if genotype not in genotype_true:
                    genotype_true[genotype] = []
                genotype_true[genotype].append(true_val)
        
        # Average across replicates
        genotype_pred_avg = {g: np.mean(v) for g, v in genotype_preds.items()}
        genotype_true_avg = {g: np.mean(v) for g, v in genotype_true.items()}
        
        # Find common genotypes
        common_genotypes = set(genotype_pred_avg.keys()) & set(genotype_true_avg.keys())
        
        if len(common_genotypes) < 3:
            return None
        
        pred_vals = np.array([genotype_pred_avg[g] for g in common_genotypes])
        true_vals = np.array([genotype_true_avg[g] for g in common_genotypes])
        
        rho, _ = spearmanr(pred_vals, true_vals)
        
        return float(rho)
    
    def run_all(self) -> dict[str, dict[str, Any]]:
        """Run all 3 baselines and return results.
        
        Returns:
            Dictionary with keys: 'xgboost_tabular', 'rf_tabular', 'dinov2_rf'
            Each value is a dict with metrics
        """
        results = {}
        results['xgboost_tabular'] = self._run_xgboost()
        results['rf_tabular'] = self._run_rf()
        results['dinov2_rf'] = self._run_dinov2_rf()
        return results
