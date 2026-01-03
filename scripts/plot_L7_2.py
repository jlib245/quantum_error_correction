import matplotlib.pyplot as plt
import numpy as np

# Data from the user
p_errors = np.array([0.070, 0.080, 0.090, 0.100, 0.110, 0.120, 0.130, 0.140])
mwpm_lers = np.array([2.602000e-02, 4.062000e-02, 5.576000e-02, 7.873000e-02, 1.038600e-01, 1.309900e-01, 1.634800e-01, 1.971500e-01])
vit_lers = np.array([1.352000e-02, 2.339000e-02, 3.469000e-02, 5.258000e-02, 6.943000e-02, 9.319000e-02, 1.171200e-01, 1.434900e-01])
jung_cnn_lers = np.array([1.796000e-02, 3.029000e-02, 4.410000e-02, 6.480000e-02, 8.591000e-02, 1.139700e-01, 1.411100e-01, 1.736400e-01])

# Plotting
plt.figure(figsize=(10, 7)) # Size suitable for a paper figure

# Using markers and different line styles for clarity in black & white printing
plt.plot(p_errors, mwpm_lers, marker='s', linestyle='--', color='gray', label='MWPM (Baseline)', markersize=8)
plt.plot(p_errors, jung_cnn_lers, marker='^', linestyle='-.', color='tab:orange', label='Jung et al. (CNN)', markersize=8)
plt.plot(p_errors, vit_lers, marker='o', linestyle='-', color='tab:blue', label='ViT_LUT_CONCAT (Ours)', markersize=9, linewidth=2.5)

# Axes labels and title
plt.xlabel(r'Physical Error Rate ($p$)', fontsize=14)
plt.ylabel(r'Logical Error Rate ($p_L$)', fontsize=14)
plt.title(r'Decoding Performance Comparison ($L=7$, Depolarizing)', fontsize=16)

# Log scale for y-axis is standard in QEC papers
plt.yscale('log') 

# Grid and legend
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)

# Tick parameters
plt.xticks(p_errors, fontsize=12) # Use the exact data points for x-ticks
plt.yticks(fontsize=12)

# Annotation for the extrapolation point (p=0.14)
plt.annotate('Extrapolation Point', xy=(0.140, vit_lers[-1]), xytext=(0.125, 0.16),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

# Saving the figure in high resolution for publication
plt.tight_layout()
plt.savefig('ler_comparison_L7_logscale_final.pdf', dpi=300) # PDF is best for LaTeX papers
plt.show()