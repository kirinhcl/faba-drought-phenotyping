#!/usr/bin/env python3
"""Validate environment setup before training stress detection model."""

import sys
from pathlib import Path

def check_imports():
    """Check all required imports."""
    print("Checking imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA version: {torch.version.cuda}")
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy: {e}")
        return False
    
    try:
        import pandas
        print(f"  ✓ Pandas {pandas.__version__}")
    except ImportError as e:
        print(f"  ✗ Pandas: {e}")
        return False
    
    try:
        from omegaconf import OmegaConf
        print(f"  ✓ OmegaConf")
    except ImportError as e:
        print(f"  ✗ OmegaConf: {e}")
        return False
    
    try:
        from sklearn.metrics import accuracy_score
        print(f"  ✓ scikit-learn")
    except ImportError as e:
        print(f"  ✗ scikit-learn: {e}")
        return False
    
    return True

def check_files():
    """Check all required files exist."""
    print("\nChecking files...")
    required_files = [
        "configs/stress.yaml",
        "src/model/gating.py",
        "src/model/stress_model.py",
        "src/training/stress_loss.py",
        "src/data/dataset.py",
        "src/data/collate.py",
        "scripts/train_stress.py",
        "scripts/evaluate_stress.py",
        "scripts/slurm/train_stress.sh",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def check_model():
    """Check model can be instantiated."""
    print("\nChecking model...")
    try:
        from src.model.stress_model import StressDetectionModel
        from src.utils.config import load_config
        
        cfg = load_config("configs/stress.yaml")
        model = StressDetectionModel(cfg)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created: {num_params:,} parameters")
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False

def check_dataset():
    """Check dataset can be loaded."""
    print("\nChecking dataset...")
    try:
        from src.data.dataset import FabaDroughtDataset
        from src.utils.config import load_config
        
        cfg = load_config("configs/stress.yaml")
        dataset = FabaDroughtDataset(cfg)
        
        print(f"  ✓ Dataset loaded: {len(dataset)} samples")
        
        # Check first sample
        sample = dataset[0]
        print(f"  ✓ stress_labels shape: {sample['stress_labels'].shape}")
        print(f"  ✓ stress_mask shape: {sample['stress_mask'].shape}")
        print(f"  ✓ Sample keys: {list(sample.keys())}")
        
        return True
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}")
        return False

def check_loss():
    """Check loss can be computed."""
    print("\nChecking loss...")
    try:
        import torch
        from src.training.stress_loss import StressLoss
        
        loss_fn = StressLoss()
        
        # Create dummy data
        predictions = {
            'stress_logits': torch.randn(2, 22)
        }
        targets = {
            'stress_labels': torch.randint(0, 2, (2, 22)),
            'stress_mask': torch.ones(2, 22, dtype=torch.bool)
        }
        
        loss, loss_dict = loss_fn(predictions, targets)
        print(f"  ✓ Loss computed: {loss.item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ Loss computation failed: {e}")
        return False

def check_directories():
    """Check output directories can be created."""
    print("\nChecking directories...")
    try:
        Path("results/stress/checkpoints").mkdir(parents=True, exist_ok=True)
        print(f"  ✓ results/stress/checkpoints/")
        
        Path("logs").mkdir(parents=True, exist_ok=True)
        print(f"  ✓ logs/")
        
        return True
    except Exception as e:
        print(f"  ✗ Directory creation failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("="*80)
    print("Stress Detection Model - Environment Validation")
    print("="*80)
    
    checks = [
        ("Imports", check_imports),
        ("Files", check_files),
        ("Model", check_model),
        ("Dataset", check_dataset),
        ("Loss", check_loss),
        ("Directories", check_directories),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            results[name] = False
    
    print("\n" + "="*80)
    print("Validation Summary")
    print("="*80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All checks passed! Ready to train.")
        return 0
    else:
        print("\n✗ Some checks failed. Fix issues before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
