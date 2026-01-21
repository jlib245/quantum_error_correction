import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. 데이터 정의 (Data Definitions)
# ---------------------------------------------------------
# (데이터는 이전과 동일합니다)
data_l3 = {
    'p_error': [0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130],
    'MWPM': [6.147e-2, 7.787e-2, 9.494e-2, 1.135e-1, 1.340e-1, 1.543e-1, 1.723e-1],
    'ViT_LUT_CONCAT': [5.641e-2, 6.915e-2, 8.589e-2, 1.025e-1, 1.204e-1, 1.394e-1, 1.581e-1],
    'Jung_CNN': [5.550e-2, 6.973e-2, 8.644e-2, 1.020e-1, 1.212e-1, 1.394e-1, 1.581e-1]
}
data_l5 = {
    'p_error': [0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130],
    'MWPM': [3.955e-2, 5.509e-2, 7.486e-2, 9.527e-2, 1.191e-1, 1.429e-1, 1.702e-1],
    'ViT_LUT_CONCAT': [2.675e-2, 3.924e-2, 5.310e-2, 7.055e-2, 8.906e-2, 1.104e-1, 1.323e-1],
    'Jung_CNN': [2.778e-2, 4.049e-2, 5.618e-2, 7.350e-2, 9.229e-2, 1.146e-1, 1.376e-1]
}
data_l7 = {
    'p_error': [0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140],
    'MWPM': [2.602e-2, 4.062e-2, 5.576e-2, 7.873e-2, 1.039e-1, 1.310e-1, 1.635e-1, 1.972e-1],
    'ViT_LUT_CONCAT': [1.344e-2, 2.264e-2, 3.453e-2, 5.078e-2, 6.804e-2, 9.217e-2, 1.166e-1, 1.429e-1],
    'Jung_CNN': [1.717000e-02, 2.833000e-02, 4.263000e-02, 6.278000e-02, 8.290000e-02, 1.097700e-01, 1.360500e-01, 1.680300e-01]
}
data_l7_y100 = {
    'p_error': [0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130],
    'MWPM': [8.873e-2, 1.2989e-1, 1.735e-1, 2.2227e-1, 2.7891e-1, 3.305e-1, 3.8536e-1],
    'ViT_LUT_CONCAT': [1.427e-2, 2.496e-2, 3.988e-2, 6.024e-2, 8.483e-2, 1.1222e-1, 1.4697e-1],
    'Jung_CNN': [2.861000e-02, 4.687000e-02, 7.162000e-02, 1.017300e-01, 1.355200e-01 , 1.758500e-01, 2.183500e-01]
}

# ---------------------------------------------------------
# 2. 그래프 설정
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

x_limit = (0.07, 0.13)
# Colors
color_mwpm = '#1f77b4' 
color_cnn = '#ff7f0e'  
color_prop = '#d62728' 

# ---------------------------------------------------------
# [Graphs 1-3] Standard Plots
# ---------------------------------------------------------
standard_plots = [
    (data_l3, "L=3", axes[0, 0]),
    (data_l5, "L=5", axes[0, 1]),
    (data_l7, "L=7 (Depolarizing)", axes[1, 0])
]

for data, title, ax in standard_plots:
    df = pd.DataFrame(data)
    df = df[df['p_error'] <= 0.13]
    
    ax.plot(df['p_error'], df['p_error'], color='black', linestyle=':', linewidth=1.5, label='Pseudo Threshold')

    ax.plot(df['p_error'], df['MWPM'], marker='o', linestyle='--', color=color_mwpm, label='MWPM', linewidth=2, markersize=8)
    ax.plot(df['p_error'], df['Jung_CNN'], marker='s', linestyle='-.', color=color_cnn, label='CNN', linewidth=2, markersize=8)
    ax.plot(df['p_error'], df['ViT_LUT_CONCAT'], marker='^', linestyle='-', color=color_prop, label='Proposed', linewidth=2, markersize=8)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Physical Error Rate ($p$)', fontsize=12)
    ax.set_ylabel('Logical Error Rate ($P_L$)', fontsize=12)
    ax.set_yscale('log')
    ax.set_xlim(x_limit)
    ax.grid(True, which="both", ls="-", alpha=0.4)
    ax.legend(fontsize=10)
    ax.tick_params(axis='both', which='major', labelsize=11)

# ---------------------------------------------------------
# [Graph 4] Comparison Plot (Markers + Distinction)
# ---------------------------------------------------------
ax4 = axes[1, 1]
df_dep = pd.DataFrame(data_l7)
df_dep = df_dep[df_dep['p_error'] <= 0.13]
df_y = pd.DataFrame(data_l7_y100)
df_y = df_y[df_y['p_error'] <= 0.13]
x_vals = df_dep['p_error']

# Pseudo Threshold
ax4.plot(x_vals, x_vals, color='black', linestyle=':', linewidth=1.5, label='Pseudo Threshold')

# --- Plotting Function to reduce repetition ---
def plot_comparison(ax, x, y_dep, y_y100, color, marker, label_base):
    # 1. Depolarizing (Filled Marker)
    ax.plot(x, y_dep, marker=marker, linestyle='-', color=color, 
            label=f'{label_base} (Depol)', 
            linewidth=1.5, markersize=8, alpha=0.8)
    
    # 2. Y-100% (Hollow Marker - markerfacecolor='white')
    ax.plot(x, y_y100, marker=marker, linestyle='--', color=color, 
            label=f'{label_base} (Y-100%)', 
            linewidth=1.5, markersize=8, markerfacecolor='white', markeredgewidth=2, alpha=0.8)
    
    # 3. Shading
    ax.fill_between(x, y_dep, y_y100, color=color, alpha=0.15)

# Execute Plotting
plot_comparison(ax4, x_vals, df_dep['MWPM'], df_y['MWPM'], color_mwpm, 'o', 'MWPM')
plot_comparison(ax4, x_vals, df_dep['Jung_CNN'], df_y['Jung_CNN'], color_cnn, 's', 'CNN')
plot_comparison(ax4, x_vals, df_dep['ViT_LUT_CONCAT'], df_y['ViT_LUT_CONCAT'], color_prop, '^', 'Proposed')

# Styling Graph 4
ax4.set_title("L=7 Performance Range (Depolarizing vs Y-Noise)", fontsize=14, fontweight='bold')
ax4.set_xlabel('Physical Error Rate ($p$)', fontsize=12)
ax4.set_ylabel('Logical Error Rate ($P_L$)', fontsize=12)
ax4.set_yscale('log')
ax4.set_xlim(x_limit)
ax4.grid(True, which="both", ls="-", alpha=0.4)

# Legend settings (2 columns to fit all entries nicely)
ax4.legend(fontsize=9, ncol=2, loc='lower right')
ax4.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
filename = "Comparison_Final_With_Markers.png"
plt.savefig(filename, dpi=300, bbox_inches='tight')
print(f"Saved: {filename}")
