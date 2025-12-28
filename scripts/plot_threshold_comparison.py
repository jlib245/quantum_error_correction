"""
Threshold comparison plot for different code sizes and y_ratios.
Models: FFNN, CNN, Transformer, 4ch_CNN_Transformer (ViT_LUT_Concat)
"""

import matplotlib.pyplot as plt
import numpy as np
import re
import os
from pathlib import Path
from scipy.interpolate import interp1d

# Style settings
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

# Model name mapping
MODEL_NAMES = {
    'MWPM': 'MWPM',
    'FFNN': 'FFNN',
    'CNN': 'CNN',
    'Transformer': 'Transformer',
    'ViT_LUT_Concat': '4ch_CNN_Transformer'
}

# Colors for models
MODEL_COLORS = {
    'MWPM': '#7f7f7f',
    'FFNN': '#1f77b4',
    'CNN': '#2ca02c',
    'Transformer': '#ff7f0e',
    '4ch_CNN_Transformer': '#d62728'
}

# Markers for models
MODEL_MARKERS = {
    'MWPM': 's',
    'FFNN': 'o',
    'CNN': '^',
    'Transformer': 'D',
    '4ch_CNN_Transformer': 'p'
}


def parse_log_file(filepath):
    """Parse comparison log file and extract LER data for each model."""
    results = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract L and y_ratio from header
    L_match = re.search(r'Code distance L=(\d+)', content)
    y_match = re.search(r'Y-ratio: ([\d.]+)', content)

    if not L_match or not y_match:
        return None, None, None

    L = int(L_match.group(1))
    y_ratio = float(y_match.group(1))

    # Parse summary table
    summary_match = re.search(r'COMPARISON SUMMARY.*?={50,}(.*?)={50,}', content, re.DOTALL)
    if not summary_match:
        return L, y_ratio, {}

    summary = summary_match.group(1)
    lines = [l.strip() for l in summary.strip().split('\n') if l.strip() and not l.startswith('-')]

    if len(lines) < 2:
        return L, y_ratio, {}

    # Parse header
    header = lines[0].split()

    # Find column indices for models we care about
    model_cols = {}
    for i, col in enumerate(header):
        if 'LER' in col:
            model_name = col.replace('_LER', '').replace('LER', '')
            if model_name == '':
                continue
            # Map to our names
            if model_name in ['MWPM', 'Transformer', 'FFNN', 'CNN']:
                model_cols[model_name] = i
            elif model_name == 'ViT_LUT_Concat':
                model_cols['ViT_LUT_Concat'] = i

    # Parse data rows
    p_values = []
    for model in model_cols:
        results[model] = []

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            p = float(parts[0])
            p_values.append(p)

            for model, col_idx in model_cols.items():
                if col_idx < len(parts):
                    ler = float(parts[col_idx])
                    results[model].append(ler)
        except (ValueError, IndexError):
            continue

    # Convert to numpy arrays
    for model in results:
        results[model] = np.array(results[model])

    return L, y_ratio, {'p_values': np.array(p_values), 'ler': results}


def find_threshold(p_values, ler_values):
    """Find threshold where LER crosses the un-coded line (p)."""
    if len(p_values) < 2 or len(ler_values) < 2:
        return None

    # Un-coded: LER = p
    # Find where ler_values crosses p_values
    diff = ler_values - p_values[:len(ler_values)]

    # Find sign changes
    for i in range(len(diff) - 1):
        if diff[i] < 0 and diff[i+1] >= 0:
            # Linear interpolation
            p1, p2 = p_values[i], p_values[i+1]
            d1, d2 = diff[i], diff[i+1]
            threshold = p1 - d1 * (p2 - p1) / (d2 - d1)
            return threshold

    # If no crossing found, check if always below
    if np.all(diff < 0):
        return p_values[-1]  # threshold beyond measured range

    return None


def collect_all_data(experiments_dir):
    """Collect data from all experiment directories."""
    all_data = {}

    exp_path = Path(experiments_dir)

    for log_file in exp_path.glob('L*_correlated_y*/comparison_*.log'):
        L, y_ratio, data = parse_log_file(log_file)

        if L is None or not data or 'p_values' not in data:
            continue

        key = (L, y_ratio)

        # Keep the most recent/complete data
        if key not in all_data or len(data.get('p_values', [])) > len(all_data[key].get('p_values', [])):
            all_data[key] = data

    return all_data


def plot_threshold_by_L(experiments_dir, output_path):
    """
    Plot threshold curves for each L (rows) and different y_ratios.
    """
    all_data = collect_all_data(experiments_dir)

    if not all_data:
        print("No data found!")
        return

    # Get unique L values and y_ratios
    L_values = sorted(set(k[0] for k in all_data.keys()))
    y_ratios = sorted(set(k[1] for k in all_data.keys()))

    print(f"Found L values: {L_values}")
    print(f"Found y_ratios: {y_ratios}")

    # Create figure: 1 row, len(L_values) columns
    fig, axes = plt.subplots(1, len(L_values), figsize=(5*len(L_values), 4.5))

    if len(L_values) == 1:
        axes = [axes]

    models_to_plot = ['MWPM', 'FFNN', 'CNN', 'Transformer', 'ViT_LUT_Concat']

    for idx, L in enumerate(L_values):
        ax = axes[idx]

        # Find data for this L with highest y_ratio available
        available_y = [y for (l, y) in all_data.keys() if l == L]
        if not available_y:
            continue

        # Use highest y_ratio for this L
        y_ratio = max(available_y)
        data = all_data[(L, y_ratio)]

        p_values = data['p_values']

        # Plot un-coded line
        ax.plot(p_values, p_values, 'k--', linewidth=2, label='Un-coded', alpha=0.7)

        # Plot each model
        for model in models_to_plot:
            if model not in data['ler']:
                continue

            ler = data['ler'][model]
            display_name = MODEL_NAMES.get(model, model)
            color = MODEL_COLORS.get(display_name, '#333333')
            marker = MODEL_MARKERS.get(display_name, 'o')

            ax.plot(p_values[:len(ler)], ler,
                   color=color, marker=marker, markersize=5,
                   linewidth=1.5, label=display_name)

        ax.set_xlabel('Physical Error Rate (p)')
        ax.set_ylabel('Logical Error Rate')
        ax.set_title(f'L={L}, Y-ratio={y_ratio:.0%}')
        ax.set_xlim([0, max(p_values)])
        ax.set_ylim([0, 0.25])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_by_y_ratio_per_L(experiments_dir, output_path):
    """
    Plot: each column = different L, each subplot shows all y_ratios for that L.
    One row showing threshold comparison across y_ratios.
    """
    all_data = collect_all_data(experiments_dir)

    if not all_data:
        print("No data found!")
        return

    L_values = sorted(set(k[0] for k in all_data.keys()))

    fig, axes = plt.subplots(1, len(L_values), figsize=(5*len(L_values), 4.5))

    if len(L_values) == 1:
        axes = [axes]

    models_to_plot = ['MWPM', 'FFNN', 'CNN', 'Transformer', 'ViT_LUT_Concat']

    for idx, L in enumerate(L_values):
        ax = axes[idx]

        # Get all y_ratios for this L
        y_ratios_for_L = sorted([y for (l, y) in all_data.keys() if l == L])

        if not y_ratios_for_L:
            continue

        # Calculate thresholds for each model at each y_ratio
        thresholds = {MODEL_NAMES.get(m, m): [] for m in models_to_plot}
        y_ratio_labels = []

        for y_ratio in y_ratios_for_L:
            data = all_data.get((L, y_ratio))
            if not data:
                continue

            y_ratio_labels.append(y_ratio)
            p_values = data['p_values']

            for model in models_to_plot:
                display_name = MODEL_NAMES.get(model, model)
                if model in data['ler']:
                    ler = data['ler'][model]
                    th = find_threshold(p_values[:len(ler)], ler)
                    thresholds[display_name].append(th if th else 0)
                else:
                    thresholds[display_name].append(0)

        # Plot thresholds vs y_ratio
        x = np.arange(len(y_ratio_labels))
        width = 0.15

        for i, (model_name, th_values) in enumerate(thresholds.items()):
            if not th_values or all(v == 0 for v in th_values):
                continue

            color = MODEL_COLORS.get(model_name, '#333333')
            offset = (i - len(thresholds)/2) * width

            ax.bar(x + offset, th_values, width, label=model_name, color=color, alpha=0.8)

        ax.set_xlabel('Y-ratio')
        ax.set_ylabel('Threshold')
        ax.set_title(f'L={L}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{y:.0%}' for y in y_ratio_labels])
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_ler_curves_row(experiments_dir, output_path):
    """
    Main plot: One row with L=3, L=5, L=7 (each with their highest y_ratio).
    Shows LER vs p curves with threshold crossing un-coded line.
    """
    all_data = collect_all_data(experiments_dir)

    if not all_data:
        print("No data found!")
        return

    L_values = sorted(set(k[0] for k in all_data.keys()))

    fig, axes = plt.subplots(1, len(L_values), figsize=(5*len(L_values), 4))

    if len(L_values) == 1:
        axes = [axes]

    models_to_plot = ['MWPM', 'FFNN', 'CNN', 'Transformer', 'ViT_LUT_Concat']

    for idx, L in enumerate(L_values):
        ax = axes[idx]

        # Get highest y_ratio for this L
        y_ratios_for_L = [y for (l, y) in all_data.keys() if l == L]
        if not y_ratios_for_L:
            continue

        y_ratio = max(y_ratios_for_L)
        data = all_data[(L, y_ratio)]
        p_values = data['p_values']

        # Plot un-coded line (threshold reference)
        ax.plot(p_values, p_values, 'k--', linewidth=2.5, label='Un-coded', zorder=10)

        # Plot each model
        for model in models_to_plot:
            if model not in data['ler']:
                continue

            ler = data['ler'][model]
            display_name = MODEL_NAMES.get(model, model)
            color = MODEL_COLORS.get(display_name, '#333333')
            marker = MODEL_MARKERS.get(display_name, 'o')

            ax.plot(p_values[:len(ler)], ler,
                   color=color, marker=marker, markersize=6,
                   linewidth=2, label=display_name)

            # Find and annotate threshold
            th = find_threshold(p_values[:len(ler)], ler)
            if th and th < p_values[-1]:
                ax.axvline(x=th, color=color, linestyle=':', alpha=0.5)

        ax.set_xlabel('Physical Error Rate (p)')
        if idx == 0:
            ax.set_ylabel('Logical Error Rate')
        ax.set_title(f'L={L}, Y-ratio={y_ratio:.0%}')
        ax.set_xlim([0, max(p_values) + 0.01])
        ax.set_ylim([0, 0.22])
        ax.grid(True, alpha=0.3)

        if idx == len(L_values) - 1:
            ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    experiments_dir = 'experiments'

    # Main plot: LER curves in one row
    plot_ler_curves_row(experiments_dir, 'figures/threshold_comparison_row.png')

    # Additional: threshold by L
    plot_threshold_by_L(experiments_dir, 'figures/threshold_by_L.png')

    # Additional: threshold bar chart by y_ratio
    plot_by_y_ratio_per_L(experiments_dir, 'figures/threshold_by_y_ratio.png')

    print("All plots generated!")
