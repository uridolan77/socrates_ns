import torch
from typing import List, Dict, Any

# Mock data samples (replace with more realistic data structure if needed)
# Each sample typically includes input_ids, attention_mask, and labels for training.
# We add 'regulatory_constraints' and 'input_texts' as examples based on the
# distillation code's potential needs during loss calculation or evaluation.

MOCK_SAMPLE_1 = {
    "input_ids": torch.randint(0, 50000, (1, 128)), # Batch dim 1, sequence length 128
    "attention_mask": torch.ones((1, 128), dtype=torch.long),
    "labels": torch.randint(0, 50000, (1, 128)), # Dummy labels
    "regulatory_constraints": {
        "data_minimization": {"personal_data_tokens": ["email", "phone"]},
        "input_ids": torch.randint(0, 50000, (1, 128)) # Example input IDs for comparison
    },
    "input_texts": ["This is general compliance text sample 1."] # Added for evaluation
}

MOCK_SAMPLE_2 = {
    "input_ids": torch.randint(0, 50000, (1, 128)),
    "attention_mask": torch.ones((1, 128), dtype=torch.long),
    "labels": torch.randint(0, 50000, (1, 128)),
    "regulatory_constraints": {
        "purpose_limitation": {"stated_purpose": "customer support inquiry"}
    },
    "input_texts": ["This is general compliance text sample 2."]
}

MOCK_FRAMEWORK_SAMPLE = {
    "input_ids": torch.randint(0, 50000, (1, 128)),
    "attention_mask": torch.ones((1, 128), dtype=torch.long),
    "labels": torch.randint(0, 50000, (1, 128)),
    "regulatory_constraints": {
        "phi_protection": {"phi_tokens": ["patient name", "diagnosis code"]},
        "phi_redaction_map": {"John Doe": "[PATIENT_NAME]"}
    },
    "input_texts": ["Framework specific sample (e.g., HIPAA)."]
}

MOCK_EDGE_CASE_SAMPLE = {
    "input_ids": torch.randint(0, 50000, (1, 128)),
    "attention_mask": torch.ones((1, 128), dtype=torch.long),
    "labels": torch.randint(0, 50000, (1, 128)),
    "regulatory_constraints": {
        "data_minimization": {"personal_data_tokens": ["ssn"]},
        "input_ids": torch.randint(0, 50000, (1, 128))
    },
    "input_texts": ["Edge case sample involving sensitive data."]
}

MOCK_ADVERSARIAL_SAMPLE = {
    "input_ids": torch.randint(0, 50000, (1, 128)),
    "attention_mask": torch.ones((1, 128), dtype=torch.long),
    "labels": torch.randint(0, 50000, (1, 128)),
    "regulatory_constraints": {
        "phi_protection": {"phi_tokens": ["medical record number"]},
        "context_allowed_phi": False # Example constraint
    },
    "input_texts": ["Adversarial sample trying to elicit PHI."]
}

# Mock Dataset Class (Optional, can also just return lists)
class MockComplianceDataset:
    """ Simple iterable mock dataset """
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

# --- Mock Loading Functions ---

def load_general_compliance_dataset() -> MockComplianceDataset:
    """
    Mocks loading a general compliance dataset.
    Returns an iterable (like a DataLoader or list) of sample dictionaries.
    """
    print("MOCK: Loading general compliance dataset...")
    # In a real scenario, load data from files/database
    mock_data = [MOCK_SAMPLE_1, MOCK_SAMPLE_2] * 5 # Repeat for more samples
    return MockComplianceDataset(mock_data)

def load_framework_dataset(regulatory_framework: str) -> MockComplianceDataset:
    """
    Mocks loading a dataset specific to a regulatory framework.
    Returns an iterable of sample dictionaries.
    """
    print(f"MOCK: Loading framework-specific dataset for {regulatory_framework}...")
    # Load data specific to the framework
    mock_data = [MOCK_FRAMEWORK_SAMPLE] * 10 # Repeat for more samples
    # Add framework identifier to samples if needed
    for sample in mock_data:
        sample['framework'] = regulatory_framework
    return MockComplianceDataset(mock_data)

def load_compliance_edge_cases(regulatory_framework: str) -> MockComplianceDataset:
    """
    Mocks loading edge case examples for compliance testing.
    Returns an iterable of sample dictionaries.
    """
    print(f"MOCK: Loading edge cases for {regulatory_framework}...")
    # Load specific edge case data
    mock_data = [MOCK_EDGE_CASE_SAMPLE] * 5
    return MockComplianceDataset(mock_data)

def load_compliance_adversarial(regulatory_framework: str) -> MockComplianceDataset:
    """
    Mocks loading adversarial examples designed to test compliance boundaries.
    Returns an iterable of sample dictionaries.
    """
    print(f"MOCK: Loading adversarial examples for {regulatory_framework}...")
    # Load adversarial data
    mock_data = [MOCK_ADVERSARIAL_SAMPLE] * 5
    return MockComplianceDataset(mock_data)

# --- Example of how to use these mocks ---
if __name__ == '__main__':
    # Example usage within the context of ComplianceAwareDistillation init
    class MockDistillation:
        def __init__(self, framework):
            self.regulatory_framework = framework
            self.framework_datasets = self._load_framework_datasets()
            print(f"\nLoaded datasets for {framework}:")
            for name, dataset in self.framework_datasets.items():
                print(f"- {name}: {len(dataset)} samples")

        def _load_framework_datasets(self):
            datasets = {
                "general": load_general_compliance_dataset(),
                "framework_specific": load_framework_dataset(self.regulatory_framework),
                "edge_cases": load_compliance_edge_cases(self.regulatory_framework),
                "adversarial": load_compliance_adversarial(self.regulatory_framework)
            }
            return datasets

    # Instantiate for GDPR
    mock_distiller_gdpr = MockDistillation("GDPR")

    # Instantiate for HIPAA
    mock_distiller_hipaa = MockDistillation("HIPAA")

    # Example of iterating through a mock dataset
    print("\nIterating through general dataset:")
    general_dataset = load_general_compliance_dataset()
    for i, sample in enumerate(general_dataset):
        if i >= 2: # Print first 2 samples
            break
        print(f"Sample {i+1}: Keys={list(sample.keys())}, Input shape={sample['input_ids'].shape}")

