#!/usr/bin/env python3
"""Run classical ML baselines for comparison."""

import argparse
import json
from pathlib import Path

from src.utils.config import load_config
from src.baselines.classical import ClassicalBaselines


def main() -> None:
    """Run baselines and save results."""
    parser = argparse.ArgumentParser(
        description='Run classical ML baselines (XGBoost, RF, DINOv2+RF)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/baselines/',
        help='Output directory for results (default: results/baselines/)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run baselines
    print("Initializing baselines...")
    baselines = ClassicalBaselines(config)
    
    print("Running all baselines with LOGO-CV...")
    results = baselines.run_all()
    
    # Save results
    for baseline_name, baseline_results in results.items():
        output_path = output_dir / f"{baseline_name}_results.json"
        with open(output_path, 'w') as f:
            json.dump(baseline_results, f, indent=2)
        print(f"✓ Saved {baseline_name} results to {output_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY")
    print("="*80)
    
    for baseline_name, baseline_results in results.items():
        print(f"\n{baseline_name}:")
        for metric_name, metric_value in baseline_results.items():
            if isinstance(metric_value, float):
                print(f"  {metric_name}: {metric_value:.4f}")
            else:
                print(f"  {metric_name}: {metric_value}")


if __name__ == '__main__':
    main()
