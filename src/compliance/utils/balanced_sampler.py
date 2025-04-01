import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Optional
class ComplianceBalancedSampler:
    """Balanced sampling from multiple compliance datasets with reproducibility and stratification"""
    
    def __init__(self, datasets, seed=None, stratify_by=None):
        if not all(hasattr(dataset, '__iter__') for dataset in datasets.values()):
            raise TypeError("All datasets must be iterable")
            
        self.datasets = datasets
        self.seed = seed
        self.stratify_by = stratify_by
        
        # Set random seed for reproducibility
        if seed is not None:
            import random
            random.seed(seed)
            import numpy as np
            np.random.seed(seed)
            import torch
            torch.manual_seed(seed)
        
        self.samplers = {name: iter(dataset) for name, dataset in datasets.items()}
        self.weights = self._calculate_sampling_weights()
        
        # Setup stratification if requested
        if stratify_by:
            self.stratification_values = self._get_stratification_values(stratify_by)
    
    def _get_stratification_values(self, stratify_by):
        """Get unique values for stratification attribute across all datasets"""
        values = set()
        for dataset in self.datasets.values():
            for sample in dataset:
                if stratify_by in sample:
                    values.add(sample[stratify_by])
        return values
    
    def get_stratified_batches(self, batch_size):
        """Get batches with stratified sampling across the stratification attribute"""
        if not self.stratify_by:
            return self.get_batches(batch_size)
            
        # Implementation of stratified sampling
        # Ensures even representation of each stratification value

        
    def _calculate_sampling_weights(self):
        """Calculate sampling weights for different datasets"""
        # Give higher weight to framework-specific and edge cases
        weights = {
            "general": 0.3,
            "framework_specific": 0.4,
            "edge_cases": 0.2,
            "adversarial": 0.1
        }
        
        # Normalize weights for actual datasets present
        total = sum(weights[k] for k in self.datasets.keys() if k in weights)
        return {k: weights.get(k, 0.1) / total for k in self.datasets.keys()}
    
    def get_batches(self, batch_size):
        """Get batches sampled from datasets according to weights"""
        # Determine number of samples from each dataset
        total_samples = 0
        dataset_samples = {}
        
        for name, weight in self.weights.items():
            dataset_samples[name] = max(1, int(batch_size * weight))
            total_samples += dataset_samples[name]
            
        # Adjust to match batch size exactly
        diff = batch_size - total_samples
        if diff != 0:
            # Distribute difference among datasets
            for name in sorted(self.weights, key=self.weights.get, reverse=True):
                dataset_samples[name] += 1
                diff -= 1
                if diff == 0:
                    break
        
        # Create infinite iterator
        while True:
            # Sample from each dataset
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            regulatory_constraints = {}
            
            for name, count in dataset_samples.items():
                for _ in range(count):
                    try:
                        sample = next(self.samplers[name])
                    except StopIteration:
                        # Reset sampler if exhausted
                        self.samplers[name] = iter(self.datasets[name])
                        sample = next(self.samplers[name])
                        
                    # Add to batch
                    for key in batch:
                        if key in sample:
                            batch[key].append(sample[key])
                            
                    # Collect regulatory constraints
                    if "regulatory_constraints" in sample:
                        for constraint, value in sample["regulatory_constraints"].items():
                            if constraint not in regulatory_constraints:
                                regulatory_constraints[constraint] = []
                            regulatory_constraints[constraint].append(value)
            
            # Combine batch elements
            for key in batch:
                batch[key] = torch.stack(batch[key]) if batch[key] else None
                
            # Add regulatory constraints
            batch["regulatory_constraints"] = regulatory_constraints
            
            yield batch

