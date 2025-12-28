import matplotlib.pyplot as plt
import numpy as np

# L=5 Data (p = 0.07 ~ 0.13)
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

L5_y33 = {
    'MWPM': [0.0377, 0.0548, 0.0752, 0.097, 0.1193, 0.1401, 0.1731],
    'FFNN': [0.0275, 0.0427, 0.0561, 0.0766, 0.0958, 0.1133, 0.1393],
    'CNN': [0.0283, 0.0408, 0.0525, 0.0703, 0.0903, 0.1101, 0.1339],
    'Transformer': [0.0286, 0.0407, 0.0505, 0.0725, 0.0908, 0.113, 0.138],
    '4ch_CNN_Transformer': [0.0273, 0.0402, 0.0545, 0.0723, 0.0907, 0.1104, 0.1344]
}

L5_y50 = {
    'MWPM': [0.0518, 0.0734, 0.097, 0.1229, 0.1559, 0.1831, 0.2156],
    'FFNN': [0.0313, 0.0412, 0.0638, 0.0809, 0.1023, 0.1209, 0.1497],
    'CNN': [0.026, 0.0362, 0.0597, 0.0739, 0.0921, 0.113, 0.1386],
    'Transformer': [0.0252, 0.0359, 0.0592, 0.0714, 0.089, 0.1131, 0.1343],
    '4ch_CNN_Transformer': [0.0275, 0.036, 0.0592, 0.073, 0.0955, 0.1153, 0.1404]
}

L5_y75 = {
    'MWPM': [0.0689, 0.1029, 0.1403, 0.1742, 0.2128, 0.2377, 0.2812],
    'FFNN': [0.0329, 0.0449, 0.0675, 0.0867, 0.1114, 0.1358, 0.1651],
    'CNN': [0.0268, 0.0376, 0.059, 0.0731, 0.0957, 0.1179, 0.1509],
    'Transformer': [0.0258, 0.0341, 0.0526, 0.0651, 0.0899, 0.1085, 0.1357],
    '4ch_CNN_Transformer': [0.0275, 0.0365, 0.0553, 0.0713, 0.0976, 0.1211, 0.1445]
}

L5_y100 = {
    'MWPM': [0.0916, 0.1396, 0.1831, 0.2244, 0.2657, 0.3019, 0.3547],
    'FFNN': [0.0359, 0.0503, 0.0777, 0.0983, 0.1271, 0.1464, 0.1825],
    'CNN': [0.0307, 0.0413, 0.065, 0.08, 0.1062, 0.1283, 0.16],
    'Transformer': [0.0269, 0.0324, 0.0538, 0.0673, 0.0917, 0.1071, 0.1365],
    '4ch_CNN_Transformer': [0.03, 0.0378, 0.0606, 0.077, 0.1034, 0.1257, 0.1516]
}

models = ['MWPM', 'FFNN', 'CNN', 'Transformer', '4ch_CNN_Transformer']
colors = {'MWPM': '#7f7f7f', 'FFNN': '#1f77b4', 'CNN': '#2ca02c', 'Transformer': '#ff7f0e', '4ch_CNN_Transformer': '#d62728'}
markers = {'MWPM': 's', 'FFNN': 'o', 'CNN': '^', 'Transformer': 'D', '4ch_CNN_Transformer': 'p'}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
L5_data = [L5_y33, L5_y50, L5_y75, L5_y100]
y_labels = ['33%', '50%', '75%', '100%']

for idx, (data, y_label) in enumerate(zip(L5_data, y_labels)):
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
    ax.set_ylim([0.02, 0.4])
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=7)

fig.suptitle('Surface Code L=5', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/L5.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: figures/L5.png")
