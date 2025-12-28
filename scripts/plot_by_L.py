"""
각 코드 크기(L)별로 Y-ratio 비교 플롯 생성
L=3.png, L=5.png, L=7.png
"""

import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

# Model display names
MODEL_NAMES = {
    'MWPM': 'MWPM',
    'FFNN': 'FFNN',
    'CNN': 'CNN',
    'Transformer': 'Transformer',
    'ViT_LUT_Concat': '4ch_CNN_Transformer'
}

MODEL_COLORS = {
    'MWPM': '#7f7f7f',
    'FFNN': '#1f77b4',
    'CNN': '#2ca02c',
    'Transformer': '#ff7f0e',
    '4ch_CNN_Transformer': '#d62728'
}

MODEL_MARKERS = {
    'MWPM': 's',
    'FFNN': 'o',
    'CNN': '^',
    'Transformer': 'D',
    '4ch_CNN_Transformer': 'p'
}


def parse_model_section(content, model_name):
    """Parse a specific model section and extract p_values and LER."""
    # Find model section
    pattern = rf'{model_name}.*?Evaluation\n={50,}(.*?)(?=\n={50,}|Plot saved|$)'
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

    if not match:
        return [], []

    section = match.group(1)
    p_values = []
    ler_values = []
    current_p = None

    for line in section.split('\n'):
        p_match = re.search(r'Testing p=([\d.]+)', line)
        if p_match:
            current_p = float(p_match.group(1))

        ler_match = re.search(r'LER:\s*([\d.e+-]+)', line)
        if ler_match and current_p is not None:
            p_values.append(current_p)
            ler_values.append(float(ler_match.group(1)))
            current_p = None  # Reset to avoid duplicates

    return p_values, ler_values


def parse_log_file(filepath):
    """Parse log file and extract data."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Get L and y_ratio
    L_match = re.search(r'Code distance L=(\d+)', content)
    y_match = re.search(r'Y-ratio: ([\d.]+)', content)

    if not L_match or not y_match:
        return None, None, None

    L = int(L_match.group(1))
    y_ratio = float(y_match.group(1))

    # Parse each model
    results = {'p_values': None, 'ler': {}}

    model_patterns = {
        'MWPM': 'MWPM Decoder',
        'Transformer': 'TRANSFORMER Model',
        'FFNN': 'FFNN Model',
        'CNN': 'CNN Model',
        'ViT_LUT_Concat': 'VIT_LUT_CONCAT Model'
    }

    for model_key, pattern in model_patterns.items():
        p_vals, ler_vals = parse_model_section(content, pattern)

        if ler_vals:
            results['ler'][model_key] = np.array(ler_vals)
            if results['p_values'] is None:
                results['p_values'] = np.array(p_vals)

    return L, y_ratio, results


def collect_data(experiments_dir):
    """Collect all data organized by L and y_ratio."""
    all_data = {}
    exp_path = Path(experiments_dir)

    for log_file in exp_path.glob('L*_correlated_y*/comparison_*.log'):
        L, y_ratio, data = parse_log_file(log_file)

        if L is None or not data or not data.get('ler'):
            continue

        key = (L, y_ratio)

        # Keep most complete data
        if key not in all_data:
            all_data[key] = data
        elif len(data.get('ler', {})) > len(all_data[key].get('ler', {})):
            all_data[key] = data

    return all_data


def plot_for_L(L, all_data, output_dir):
    """Create plot for a specific L value with all y_ratios."""

    # Get all y_ratios for this L
    y_ratios = sorted([y for (l, y) in all_data.keys() if l == L])

    if not y_ratios:
        print(f"No data for L={L}")
        return

    print(f"L={L}: y_ratios = {[f'{y:.0%}' for y in y_ratios]}")

    # Create subplots: 1 row, len(y_ratios) columns
    n_cols = len(y_ratios)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5*n_cols, 4))

    if n_cols == 1:
        axes = [axes]

    models_to_plot = ['MWPM', 'FFNN', 'CNN', 'Transformer', 'ViT_LUT_Concat']

    for idx, y_ratio in enumerate(y_ratios):
        ax = axes[idx]
        data = all_data.get((L, y_ratio))

        if not data or data['p_values'] is None or len(data['p_values']) == 0:
            ax.set_title(f'Y-ratio={y_ratio:.0%}\n(No data)')
            continue

        p_values = data['p_values']

        # Plot un-coded line
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
                   color=color, marker=marker, markersize=5,
                   linewidth=1.8, label=display_name)

        ax.set_xlabel('Physical Error Rate (p)')
        if idx == 0:
            ax.set_ylabel('Logical Error Rate')

        y_pct = int(round(y_ratio * 100))
        ax.set_title(f'Y-ratio = {y_pct}%')
        ax.set_xlim([0, max(p_values) + 0.005])
        ax.set_ylim([0, 0.20])
        ax.grid(True, alpha=0.3)

        # Legend only on last plot
        if idx == n_cols - 1:
            ax.legend(loc='upper left', fontsize=8)

    fig.suptitle(f'Surface Code L={L}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = Path(output_dir) / f'L{L}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    experiments_dir = 'experiments'
    output_dir = 'figures'

    Path(output_dir).mkdir(exist_ok=True)

    all_data = collect_data(experiments_dir)

    print(f"Found {len(all_data)} experiment results")
    for key in sorted(all_data.keys()):
        models = list(all_data[key]['ler'].keys())
        print(f"  L={key[0]}, y={key[1]:.2f}: {models}")

    # Get unique L values
    L_values = sorted(set(k[0] for k in all_data.keys()))
    print(f"\nL values: {L_values}")

    # Create plot for each L
    for L in L_values:
        plot_for_L(L, all_data, output_dir)

    print("\nDone!")
