import matplotlib.pyplot as plt
import numpy as np

# L=3 Data (p = 0.07 ~ 0.13)
p = np.array([0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13])

def find_threshold(p_vals, ler_vals):
    """Find threshold where LER crosses un-coded line (p)"""
    p_arr = np.array(p_vals)
    ler_arr = np.array(ler_vals)
    diff = ler_arr - p_arr
    for i in range(len(diff)-1):
        if diff[i] < 0 and diff[i+1] >= 0:
            p1, p2 = p_arr[i], p_arr[i+1]
            d1, d2 = diff[i], diff[i+1]
            return p1 - d1 * (p2 - p1) / (d2 - d1)
    if np.all(diff < 0):
        return p_arr[-1]
    return None

L3_y33 = {
    'MWPM': [0.0579, 0.0819, 0.098, 0.1133, 0.1251, 0.1482, 0.1738],
    'FFNN': [0.052, 0.066, 0.083, 0.097, 0.124, 0.138, 0.155],
    'CNN': [0.055, 0.069, 0.085, 0.098, 0.128, 0.141, 0.157],
    'Transformer': [0.052, 0.066, 0.083, 0.097, 0.124, 0.138, 0.155],
    '4ch_CNN_Transformer': [0.053, 0.070, 0.086, 0.097, 0.127, 0.140, 0.159]
}

L3_y50 = {
    'MWPM': [0.0761, 0.0961, 0.1133, 0.1364, 0.1675, 0.1835, 0.2057],
    'FFNN': [0.0544, 0.0678, 0.0846, 0.0949, 0.1254, 0.1376, 0.1581],
    'CNN': [0.0554, 0.0688, 0.0848, 0.0982, 0.1277, 0.1414, 0.1567],
    'Transformer': [0.0522, 0.0669, 0.0837, 0.0971, 0.1242, 0.1381, 0.1597],
    '4ch_CNN_Transformer': [0.0529, 0.0700, 0.0861, 0.0973, 0.1268, 0.1396, 0.1594]
}

L3_y75 = {
    'MWPM': [0.0971, 0.1208, 0.1473, 0.1768, 0.2120, 0.2307, 0.2556],
    'FFNN': [0.0510, 0.0697, 0.0852, 0.0968, 0.1258, 0.1381, 0.1557],
    'CNN': [0.0529, 0.0703, 0.0881, 0.1038, 0.1319, 0.1450, 0.1617],
    'Transformer': [0.0478, 0.0670, 0.0855, 0.0980, 0.1218, 0.1376, 0.1559],
    '4ch_CNN_Transformer': [0.0499, 0.0701, 0.0853, 0.1015, 0.1269, 0.1386, 0.1575]
}

L3_y100 = {
    'MWPM': [0.1217, 0.1527, 0.1852, 0.2171, 0.2553, 0.2838, 0.3112],
    'FFNN': [0.0538, 0.0697, 0.0845, 0.0965, 0.1272, 0.1401, 0.1614],
    'CNN': [0.0575, 0.0717, 0.0894, 0.1056, 0.1332, 0.1491, 0.1694],
    'Transformer': [0.0494, 0.0663, 0.0811, 0.0964, 0.1204, 0.1343, 0.1545],
    '4ch_CNN_Transformer': [0.0524, 0.0686, 0.0845, 0.0983, 0.1254, 0.1395, 0.1599]
}

models = ['MWPM', 'FFNN', 'CNN', 'Transformer', '4ch_CNN_Transformer']
colors = {'MWPM': '#7f7f7f', 'FFNN': '#1f77b4', 'CNN': '#2ca02c', 'Transformer': '#ff7f0e', '4ch_CNN_Transformer': '#d62728'}
markers = {'MWPM': 's', 'FFNN': 'o', 'CNN': '^', 'Transformer': 'D', '4ch_CNN_Transformer': 'p'}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
L3_data = [L3_y33, L3_y50, L3_y75, L3_y100]
y_labels = ['33%', '50%', '75%', '100%']

for idx, (data, y_label) in enumerate(zip(L3_data, y_labels)):
    ax = axes[idx]

    # Un-coded line
    ax.plot(p, p, 'k--', linewidth=2, label='Un-coded', zorder=10)

    for model in models:
        ler = data[model]
        ax.plot(p, ler, color=colors[model], marker=markers[model],
                markersize=5, linewidth=1.5, label=model)

        # Find and mark threshold
        th = find_threshold(p, ler)
        if th and th < p[-1]:
            ax.axvline(x=th, color=colors[model], linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Physical Error Rate (p)')
    if idx == 0:
        ax.set_ylabel('Logical Error Rate')
    ax.set_title(f'Y-ratio = {y_label}')
    ax.set_xlim([0.07, 0.13])
    ax.set_ylim([0.04, 0.35])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=7)

fig.suptitle('Surface Code L=3', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/L3.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: figures/L3.png")
