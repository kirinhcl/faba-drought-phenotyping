"""Leave-One-Genotype-Out (LOGO) cross-validation for Faba Bean drought phenotyping.

Implements stratified LOGO CV where each fold uses one genotype for testing,
three genotypes (one from each drought category) for validation, and the
remaining 40 genotypes for training.
"""

from typing import Any, Generator, Tuple, cast

import numpy as np
import numpy.typing as npt
import pandas as pd


class LogoCV:
    """Leave-One-Genotype-Out cross-validation with stratified validation set.
    
    Each fold:
    - Test: 1 genotype (6 plants: 3 WHC-80 + 3 WHC-30)
    - Validation: 3 genotypes (1 Early + 1 Mid + 1 Late, 18 plants total)
    - Train: 40 genotypes (240 plants)
    
    Args:
        plant_metadata_df: DataFrame with columns: accession, drought_category, etc.
        n_folds: Number of folds (default 44, one per genotype)
        stratify_col: Column name for stratification (default 'drought_category')
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        plant_metadata_df: pd.DataFrame,
        n_folds: int = 44,
        stratify_col: str = 'drought_category',
        seed: int = 42
    ) -> None:
        self.df = plant_metadata_df
        self.n_folds = n_folds
        self.stratify_col = stratify_col
        self.seed = seed
        
        # Build genotype list and category mapping
        self.genotypes, self.genotype_to_category = self._build_genotype_map()
        
        # Group genotypes by category
        self.category_to_genotypes = self._group_by_category()
    
    def _build_genotype_map(self) -> Tuple[list[str], dict[str, str]]:
        """Build list of unique genotypes and map each to its drought category.
        
        Returns:
            (genotypes, genotype_to_category)
        """
        genotypes = sorted(self.df['accession'].unique().tolist())
        genotype_to_category: dict[str, str] = {}
        
        for genotype in genotypes:
            # Get category from WHC-30 plants (should be consistent across replicates)
            genotype_df = self.df[
                (self.df['accession'] == genotype) &
                (self.df['treatment'] == 'WHC-30')
            ]
            
            if len(genotype_df) > 0:
                cat_values = genotype_df[self.stratify_col]
                non_null_cats = [c for c in cat_values if pd.notna(c)]
                if len(non_null_cats) > 0:
                    genotype_to_category[genotype] = str(non_null_cats[0])
                else:
                    genotype_to_category[genotype] = 'Unknown'
            else:
                # Genotype only has WHC-80 plants (should not happen)
                genotype_to_category[genotype] = 'Unknown'
        
        return genotypes, genotype_to_category
    
    def _group_by_category(self) -> dict[str, list[str]]:
        """Group genotypes by their drought category.
        
        Returns:
            {category: [genotype1, genotype2, ...]}
        """
        category_to_genotypes = {}
        for genotype, category in self.genotype_to_category.items():
            if category not in category_to_genotypes:
                category_to_genotypes[category] = []
            category_to_genotypes[category].append(genotype)
        
        return category_to_genotypes
    
    def split(self) -> Generator[Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]], None, None]:
        """Generate train/val/test splits for each fold.
        
        Yields:
            (train_indices, val_indices, test_indices) for each fold
            All indices are integer positions in the plant metadata DataFrame
        """
        rng = np.random.RandomState(self.seed)
        
        # Iterate through each genotype as test set
        for test_genotype in self.genotypes:
            test_category = self.genotype_to_category[test_genotype]
            
            test_idx_list = self.df[self.df['accession'] == test_genotype].index.tolist()
            test_indices = cast(npt.NDArray[np.int_], np.array(test_idx_list, dtype=np.int64))
            
            remaining_genotypes = [g for g in self.genotypes if g != test_genotype]
            
            val_genotypes = []
            for category in ['Early', 'Mid', 'Late']:
                category_genotypes = [
                    g for g in self.category_to_genotypes.get(category, [])
                    if g != test_genotype
                ]
                
                if len(category_genotypes) > 0:
                    val_genotype = rng.choice(category_genotypes)
                    val_genotypes.append(val_genotype)
            
            val_idx_list = self.df[self.df['accession'].isin(val_genotypes)].index.tolist()
            val_indices = cast(npt.NDArray[np.int_], np.array(val_idx_list, dtype=np.int64))
            
            train_genotypes = [g for g in remaining_genotypes if g not in val_genotypes]
            train_idx_list = self.df[self.df['accession'].isin(train_genotypes)].index.tolist()
            train_indices = cast(npt.NDArray[np.int_], np.array(train_idx_list, dtype=np.int64))
            
            yield train_indices, val_indices, test_indices  # pyright: ignore[reportReturnType]
