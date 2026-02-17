#!/usr/bin/env python3
"""
Generate Figure 2: Main Results (F1 Score and Onset MAE)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory if it doesn't exist
output_dir = '/Users/chenghao/Downloads/Faba/paper/figures'
os.makedirs(output_dir, exist_ok=True)

# Data
x_labels = [
    'LUPIN',
    'LUPIN-D',
    'Image Only',
    'Fluor Only',
    'Env Only',
    'VI Only',
    'Drop Image',
    'Drop Fluor',
    'Drop Env',
    'Drop VI',
    'Causal Mask',
    'No Temporal',
    'Concat Fusion',
    'Random Forest',
    'Logistic Reg.',
    'XGBoost'
]

f1_scores = [0.660, 0.667, 0.624, 0.651, 0.434, 0.600, 0.662, 0.636, 0.645, 0.632, 0.637, 0.642, 0.649, 0.622, 0.584, 0.576]
onset_mae = [8.3, 7.4, 8.7, 8.6, 10.8, 8.7, 8.4, 8.7, 8.2, 8.7, 8.7, 8.2, 8.2, 7.1, 8.5, 7.4]

# Color mapping
color_map = {
    'full': '#3c4d5b',        # CLIP - Dark Charcoal Slate
    'distilled': '#2ca02c',   # Distilled - Green
    'single': '#44a0d8',      # Single-Modality - Bright Blue
    'loo': '#eb5f51',         # Leave-One-Out - Red-Orange
    'arch': '#a466b7',        # Architecture - Purple
    'baseline': '#9ba6a7'     # Baselines - Muted Grey
}

colors = []
for i in range(len(x_labels)):
    if i == 0:
        colors.append(color_map['full'])
    elif i == 1:
        colors.append(color_map['distilled'])
    elif 2 <= i <= 5:
        colors.append(color_map['single'])
    elif 6 <= i <= 9:
        colors.append(color_map['loo'])
    elif 10 <= i <= 12:
        colors.append(color_map['arch'])
    else:  # 13-15
        colors.append(color_map['baseline'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=300)

x = np.arange(len(x_labels))
width = 0.6

# Panel A: F1 Score
bars1 = ax1.bar(x, f1_scores, width=width, color=colors, edgecolor='none')
ax1.set_ylabel('F1 Score', fontsize=9, fontfamily='sans-serif')
ax1.set_xlabel('', fontsize=9, fontfamily='sans-serif')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7, fontfamily='sans-serif')
ax1.set_ylim(0.0, 0.8)
ax1.grid(axis='y', linestyle='--', color='grey', alpha=0.3)
ax1.set_axisbelow(True)

# Add reference line at CLIP value for F1
clip_f1 = f1_scores[0]
ax1.axhline(y=clip_f1, color='grey', linestyle='--', alpha=0.5, linewidth=1)

# Panel label (A)
ax1.text(0.02, 0.98, '(A)', transform=ax1.transAxes, fontsize=10, fontweight='bold',
         verticalalignment='top', fontfamily='sans-serif')

# Panel B: Onset MAE
bars2 = ax2.bar(x, onset_mae, width=width, color=colors, edgecolor='none')
ax2.set_ylabel('Onset MAE (days)', fontsize=9, fontfamily='sans-serif')
ax2.set_xlabel('', fontsize=9, fontfamily='sans-serif')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7, fontfamily='sans-serif')
ax2.set_ylim(0, 12)
ax2.grid(axis='y', linestyle='--', color='grey', alpha=0.3)
ax2.set_axisbelow(True)

# Add reference line at CLIP value for MAE
clip_mae = onset_mae[0]
ax2.axhline(y=clip_mae, color='grey', linestyle='--', alpha=0.5, linewidth=1)

# Panel label (B)
ax2.text(0.02, 0.98, '(B)', transform=ax2.transAxes, fontsize=10, fontweight='bold',
         verticalalignment='top', fontfamily='sans-serif')

# Create legend
legend_labels = ['LUPIN', 'LUPIN-D', 'Single-Modality', 'Leave-One-Out', 'Architecture', 'Baselines']
legend_colors = ['#3c4d5b', '#2ca02c', '#44a0d8', '#eb5f51', '#a466b7', '#9ba6a7']
legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in legend_colors]

fig.legend(legend_handles, legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
           ncol=6, fontsize=7, frameon=True, fancybox=False, edgecolor='black', framealpha=1.0)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save figure
pdf_path = os.path.join(output_dir, 'fig2_main_results.pdf')
png_path = os.path.join(output_dir, 'fig2_main_results.png')

plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

print(f"Figure saved to:")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")

plt.close()
