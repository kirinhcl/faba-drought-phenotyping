"""Custom samplers for genotype-aware batching."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler


class GenotypeBatchSampler(Sampler[list[int]]):
    """Batch sampler that groups plants by genotype.
    
    Ensures each batch contains complete genotype groups (all 6 plants:
    3 WHC-80 + 3 WHC-30 replicates). This enables proper genotype-level
    aggregation during training.
    
    Args:
        plant_metadata: DataFrame with 'accession' column
        genotypes_per_batch: Number of genotypes per batch (default 2-3)
        shuffle: Whether to shuffle genotypes each epoch
        seed: Random seed for reproducibility
        drop_last: Whether to drop incomplete final batch
    """
    
    def __init__(
        self,
        plant_metadata: pd.DataFrame,
        genotypes_per_batch: int = 2,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.metadata = plant_metadata
        self.genotypes_per_batch = genotypes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        self.genotype_to_indices = self._build_genotype_index()
        self.genotypes = list(self.genotype_to_indices.keys())
    
    def _build_genotype_index(self) -> dict[str, list[int]]:
        """Build mapping from genotype to plant indices."""
        genotype_to_indices: dict[str, list[int]] = {}
        for idx, row in self.metadata.iterrows():
            genotype = row['accession']
            if genotype not in genotype_to_indices:
                genotype_to_indices[genotype] = []
            genotype_to_indices[genotype].append(int(idx))
        return genotype_to_indices
    
    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        
        genotype_order = self.genotypes.copy()
        if self.shuffle:
            rng.shuffle(genotype_order)
        
        batch: list[int] = []
        genotypes_in_batch = 0
        
        for genotype in genotype_order:
            indices = self.genotype_to_indices[genotype]
            batch.extend(indices)
            genotypes_in_batch += 1
            
            if genotypes_in_batch >= self.genotypes_per_batch:
                yield batch
                batch = []
                genotypes_in_batch = 0
        
        if batch and not self.drop_last:
            yield batch
        
        self.epoch += 1
    
    def __len__(self) -> int:
        n_genotypes = len(self.genotypes)
        n_batches = n_genotypes // self.genotypes_per_batch
        if not self.drop_last and n_genotypes % self.genotypes_per_batch > 0:
            n_batches += 1
        return n_batches


class GenotypeSubsetSampler(Sampler[list[int]]):
    """Batch sampler for a subset of genotypes (train/val/test splits).
    
    Args:
        plant_metadata: Full DataFrame with 'accession' column
        subset_indices: Indices of plants in this subset
        genotypes_per_batch: Number of genotypes per batch
        shuffle: Whether to shuffle genotypes each epoch
        seed: Random seed
    """
    
    def __init__(
        self,
        plant_metadata: pd.DataFrame,
        subset_indices: list[int],
        genotypes_per_batch: int = 2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.metadata = plant_metadata
        self.subset_indices = set(subset_indices)
        self.genotypes_per_batch = genotypes_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        self.genotype_to_indices = self._build_genotype_index()
        self.genotypes = list(self.genotype_to_indices.keys())
    
    def _build_genotype_index(self) -> dict[str, list[int]]:
        """Build mapping from genotype to plant indices (subset only)."""
        genotype_to_indices: dict[str, list[int]] = {}
        for idx in self.subset_indices:
            row = self.metadata.iloc[idx]
            genotype = row['accession']
            if genotype not in genotype_to_indices:
                genotype_to_indices[genotype] = []
            genotype_to_indices[genotype].append(int(idx))
        return genotype_to_indices
    
    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.RandomState(self.seed + self.epoch)
        
        genotype_order = self.genotypes.copy()
        if self.shuffle:
            rng.shuffle(genotype_order)
        
        batch: list[int] = []
        genotypes_in_batch = 0
        
        for genotype in genotype_order:
            indices = self.genotype_to_indices[genotype]
            batch.extend(indices)
            genotypes_in_batch += 1
            
            if genotypes_in_batch >= self.genotypes_per_batch:
                yield batch
                batch = []
                genotypes_in_batch = 0
        
        if batch:
            yield batch
        
        self.epoch += 1
    
    def __len__(self) -> int:
        n_genotypes = len(self.genotypes)
        n_batches = n_genotypes // self.genotypes_per_batch
        if n_genotypes % self.genotypes_per_batch > 0:
            n_batches += 1
        return n_batches
