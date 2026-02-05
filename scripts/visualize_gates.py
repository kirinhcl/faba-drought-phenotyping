#!/usr/bin/env python3
"""Visualize modality gates from trained stress detection model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_gates(checkpoint_dir: Path, fold_id: int) -> np.ndarray:
    """Load modality gates for a specific fold.
    
    Args:
        checkpoint_dir: Path to checkpoints directory
        fold_id: Fold ID to load
    
    Returns:
        gates: Array of shape (N, T=22, 4) with modality weights
    """
    gates_path = checkpoint_dir / f"fold_{fold_id}" / "test_modality_gates.npy"
    
    if not gates_path.exists():
        raise FileNotFoundError(f"Gates not found: {gates_path}")
    
    gates = np.load(gates_path)
    print(f"Loaded gates: {gates.shape}")
    
    # Verify gates sum to 1
    gate_sums = gates.sum(axis=-1)
    if not np.allclose(gate_sums, 1.0):
        print(f"Warning: Gates don't sum to 1.0 (range: [{gate_sums.min():.3f}, {gate_sums.max():.3f}])")
    
    return gates


def plot_mean_gates(gates: np.ndarray, output_path: Path):
    """Plot mean modality gates over time.
    
    Args:
        gates: Array of shape (N, T=22, 4)
        output_path: Path to save figure
    """
    # Average across samples
    mean_gates = gates.mean(axis=0)  # (T=22, 4)
    std_gates = gates.std(axis=0)
    
    modality_names = ['Image', 'Fluorescence', 'Environment', 'Vegetation Index']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'd']
    
    plt.figure(figsize=(14, 6))
    
    timesteps = np.arange(22)
    
    for i, (name, color, marker) in enumerate(zip(modality_names, colors, markers)):
        plt.plot(timesteps, mean_gates[:, i], label=name, color=color, 
                marker=marker, linewidth=2, markersize=6, alpha=0.8)
        plt.fill_between(timesteps, 
                        mean_gates[:, i] - std_gates[:, i],
                        mean_gates[:, i] + std_gates[:, i],
                        color=color, alpha=0.2)
    
    plt.xlabel('Timestep (Round)', fontsize=12)
    plt.ylabel('Gate Weight', fontsize=12)
    plt.title('Mean Modality Gate Weights Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gate_heatmap(gates: np.ndarray, output_path: Path):
    """Plot heatmap of modality gates.
    
    Args:
        gates: Array of shape (N, T=22, 4)
        output_path: Path to save figure
    """
    mean_gates = gates.mean(axis=0).T  # (4, T=22)
    
    modality_names = ['Image', 'Fluorescence', 'Environment', 'Veg. Index']
    
    plt.figure(figsize=(14, 5))
    
    im = plt.imshow(mean_gates, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    
    plt.colorbar(im, label='Gate Weight')
    plt.yticks(range(4), modality_names)
    plt.xlabel('Timestep (Round)', fontsize=12)
    plt.ylabel('Modality', fontsize=12)
    plt.title('Modality Gate Weights Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(4):
        for j in range(22):
            text = plt.text(j, i, f'{mean_gates[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_gate_distribution(gates: np.ndarray, output_path: Path):
    """Plot distribution of gate weights.
    
    Args:
        gates: Array of shape (N, T=22, 4)
        output_path: Path to save figure
    """
    modality_names = ['Image', 'Fluorescence', 'Environment', 'Veg. Index']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, color, ax) in enumerate(zip(modality_names, colors, axes)):
        gate_values = gates[:, :, i].flatten()
        
        ax.hist(gate_values, bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(gate_values.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {gate_values.mean():.3f}')
        ax.set_xlabel('Gate Weight', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{name} Gate Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    """Main visualization entry point."""
    parser = argparse.ArgumentParser(description="Visualize modality gates")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Path to checkpoints directory")
    parser.add_argument("--fold", type=int, default=0,
                       help="Fold ID to visualize (default: 0)")
    parser.add_argument("--output_dir", type=str, default="results/stress/visualizations",
                       help="Output directory for plots")
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("Modality Gates Visualization")
    print("="*80)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Fold: {args.fold}")
    print(f"Output dir: {output_dir}")
    print()
    
    # Load gates
    gates = load_gates(checkpoint_dir, args.fold)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_mean_gates(gates, output_dir / f"gates_mean_fold{args.fold}.png")
    plot_gate_heatmap(gates, output_dir / f"gates_heatmap_fold{args.fold}.png")
    plot_gate_distribution(gates, output_dir / f"gates_distribution_fold{args.fold}.png")
    
    print("\n✓ All visualizations complete!")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
