import matplotlib.pyplot as plt
import numpy as np

# L=7 Data (p = 0.07 ~ 0.13)
p = np.array([0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13])

L7_y33 = {
    'MWPM': [0.0267, 0.0396, 0.0584, 0.0818, 0.1015, 0.1302, 0.1655],
    'FFNN': [0.0236, 0.0396, 0.0549, 0.0735, 0.1016, 0.1215, 0.1588],
    'CNN': [0.0194, 0.0301, 0.0429, 0.0575, 0.0822, 0.1017, 0.1322],
    'Transformer': [0.0195, 0.0270, 0.0417, 0.0577, 0.0788, 0.0964, 0.1315],
    '4ch_CNN_Transformer': [0.0167, 0.0268, 0.0415, 0.0586, 0.0754, 0.0996, 0.1269]
}

L7_y50 = {
    'MWPM': [0.0385, 0.0582, 0.0784, 0.1142, 0.1452, 0.182, 0.2162],
    'FFNN': [0.0282, 0.0454, 0.066, 0.0909, 0.1168, 0.147, 0.179],
    'CNN': [0.0169, 0.0338, 0.0502, 0.0625, 0.0895, 0.1173, 0.1481],
    'Transformer': [0.0174, 0.0308, 0.0447, 0.0573, 0.0819, 0.1103, 0.1303],
    '4ch_CNN_Transformer': [0.0163, 0.0285, 0.0434, 0.0522, 0.0761, 0.1053, 0.1267]
}

L7_y75 = {
    'MWPM': [0.0617, 0.0914, 0.1224, 0.1717, 0.2061, 0.257, 0.2997],
    'FFNN': [0.0361, 0.0568, 0.0821, 0.1147, 0.1488, 0.1883, 0.2301],
    'CNN': [0.0248, 0.0387, 0.0592, 0.0812, 0.1097, 0.1457, 0.1821],
    'Transformer': [0.0216, 0.0311, 0.0503, 0.0658, 0.0922, 0.1235, 0.1568],
    '4ch_CNN_Transformer': [0.0191, 0.0281, 0.046, 0.0579, 0.0896, 0.1163, 0.149]
}

L7_y100 = {
    'MWPM': [0.0868, 0.1312, 0.1708, 0.2358, 0.2783, 0.3369, 0.3811],
    'FFNN': [0.052, 0.0741, 0.1105, 0.1435, 0.1879, 0.2223, 0.2739],
    'CNN': [0.0323, 0.0503, 0.0795, 0.1052, 0.1418, 0.1761, 0.2242],
    'Transformer': [0.0237, 0.0353, 0.0616, 0.0823, 0.113, 0.1408, 0.1781],
    '4ch_CNN_Transformer': [0.0206, 0.0334, 0.0568, 0.0729, 0.1069, 0.1382, 0.1762]
}

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
        return p_arr[-1]  # Beyond range
    return None

models = ['MWPM', 'FFNN', 'CNN', 'Transformer', '4ch_CNN_Transformer']
colors = {'MWPM': '#7f7f7f', 'FFNN': '#1f77b4', 'CNN': '#2ca02c', 'Transformer': '#ff7f0e', '4ch_CNN_Transformer': '#d62728'}
markers = {'MWPM': 's', 'FFNN': 'o', 'CNN': '^', 'Transformer': 'D', '4ch_CNN_Transformer': 'p'}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
L7_data = [L7_y33, L7_y50, L7_y75, L7_y100]
y_labels = ['33%', '50%', '75%', '100%']

for idx, (data, y_label) in enumerate(zip(L7_data, y_labels)):
    ax = axes[idx]

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
    ax.set_ylim([0.015, 0.4])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=7)

fig.suptitle('Surface Code L=7', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/L7.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: figures/L7.png")
