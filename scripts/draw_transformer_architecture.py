"""
Transformer Architecture Visualization for QEC Paper/Presentation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def draw_transformer_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#E3F2FD',      # Light blue
        'embedding': '#BBDEFB',   # Blue
        'attention': '#FFCDD2',   # Light red
        'ffn': '#C8E6C9',         # Light green
        'norm': '#FFF9C4',        # Light yellow
        'output': '#E1BEE7',      # Light purple
        'cls': '#FFE0B2',         # Light orange
        'arrow': '#424242',       # Dark gray
        'border': '#37474F',      # Dark blue-gray
    }

    def draw_box(x, y, w, h, text, color, fontsize=10, bold=False):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color, edgecolor=colors['border'],
                             linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight)

    def draw_arrow(x1, y1, x2, y2, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle=style, color=colors['arrow'], lw=1.5))

    def draw_add_norm(x, y):
        """Draw Add & Norm block"""
        draw_box(x, y, 2.5, 0.4, 'Add & Norm', colors['norm'], fontsize=9)

    # ============== Title ==============
    ax.text(5, 15.5, 'QEC Transformer Decoder', ha='center', va='center',
            fontsize=16, fontweight='bold')

    # ============== Output ==============
    draw_box(3.5, 14.5, 3, 0.6, 'Output: Logits (4)', colors['output'], fontsize=11, bold=True)

    # Linear classifier
    draw_arrow(5, 14.5, 5, 14.1)
    draw_box(3.75, 13.5, 2.5, 0.6, 'Linear (d → 4)', colors['output'], fontsize=10)

    # [CLS] output
    draw_arrow(5, 13.5, 5, 13.1)
    draw_box(4, 12.6, 2, 0.5, '[CLS] output', colors['cls'], fontsize=9)

    # ============== Transformer Encoder Block ==============
    # Outer box for encoder
    encoder_box = FancyBboxPatch((1.5, 5.5), 7, 6.8,
                                  boxstyle="round,pad=0.02,rounding_size=0.2",
                                  facecolor='white', edgecolor=colors['border'],
                                  linewidth=2, linestyle='--')
    ax.add_patch(encoder_box)
    ax.text(8.2, 11.8, '×N', fontsize=14, fontweight='bold', color=colors['border'])

    draw_arrow(5, 12.6, 5, 12.0)

    # Top Add & Norm
    draw_add_norm(3.75, 11.5)

    # FFN residual connection
    ax.plot([3.75, 3.2, 3.2], [11.7, 11.7, 10.3], color=colors['arrow'], lw=1.5)
    ax.plot([3.2, 3.75], [10.3, 10.3], color=colors['arrow'], lw=1.5)
    draw_arrow(6.25, 11.5, 6.8, 11.5)
    ax.plot([6.8, 6.8, 6.25], [11.5, 11.9, 11.9], color=colors['arrow'], lw=1.5)

    # FFN block
    draw_arrow(5, 11.5, 5, 11.1)
    draw_box(3.25, 10.0, 3.5, 1.1, '', colors['ffn'], fontsize=10)
    ax.text(5, 10.75, 'Feed-Forward', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 10.35, 'd → 4d → d (GELU)', ha='center', va='center', fontsize=8)

    # Middle Add & Norm
    draw_arrow(5, 10.0, 5, 9.6)
    draw_add_norm(3.75, 9.1)

    # Attention residual connection
    ax.plot([3.75, 3.2, 3.2], [9.3, 9.3, 7.2], color=colors['arrow'], lw=1.5)
    ax.plot([3.2, 3.75], [7.2, 7.2], color=colors['arrow'], lw=1.5)
    draw_arrow(6.25, 9.1, 6.8, 9.1)
    ax.plot([6.8, 6.8, 6.25], [9.1, 9.5, 9.5], color=colors['arrow'], lw=1.5)

    # Multi-Head Attention block
    draw_arrow(5, 9.1, 5, 8.7)
    draw_box(3.0, 6.8, 4, 1.9, '', colors['attention'], fontsize=10)
    ax.text(5, 8.15, 'Multi-Head', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 7.75, 'Self-Attention', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, 7.25, '+ 2D Relative Position Bias', ha='center', va='center', fontsize=8)

    # Input to encoder
    draw_arrow(5, 6.8, 5, 6.4)
    ax.text(5, 6.1, '...', ha='center', va='center', fontsize=14)
    draw_arrow(5, 5.9, 5, 5.5)

    # ============== Positional Encoding ==============
    draw_box(2.5, 4.6, 5, 0.6, '2D Positional Encoding', colors['embedding'], fontsize=10, bold=True)

    # ============== Concatenation ==============
    draw_arrow(5, 4.6, 5, 4.2)

    # Concat symbol
    ax.text(5, 3.95, '⊕', ha='center', va='center', fontsize=16)

    # Three input branches
    # CLS token
    draw_arrow(2.5, 3.95, 3.5, 3.95)
    draw_box(1.0, 3.0, 2.5, 0.6, '[CLS] Token', colors['cls'], fontsize=9)
    ax.text(2.25, 2.65, '(learnable)', ha='center', va='center', fontsize=7)

    # Token embedding (middle)
    draw_arrow(5, 3.65, 5, 3.3)
    draw_box(3.5, 2.6, 3, 0.7, 'Token Embedding', colors['embedding'], fontsize=9)
    ax.text(5, 2.25, '(0/1 → d)', ha='center', va='center', fontsize=8)

    # Type embedding (right)
    draw_arrow(7.5, 3.95, 6.5, 3.95)
    draw_box(6.5, 3.0, 2.5, 0.6, 'Type Embedding', colors['embedding'], fontsize=9)
    ax.text(7.75, 2.65, '(Z/X → d)', ha='center', va='center', fontsize=7)

    # Connections to syndrome
    draw_arrow(5, 2.25, 5, 1.9)
    ax.plot([7.75, 7.75, 5], [3.0, 1.6, 1.6], color=colors['arrow'], lw=1.5)
    draw_arrow(5, 1.6, 5, 1.9)

    # ============== Input ==============
    draw_box(2.5, 0.7, 5, 0.8, 'Syndrome Vector  s ∈ {0,1}ⁿ', colors['input'], fontsize=11, bold=True)
    draw_arrow(5, 1.5, 5, 1.6)

    # Add legend for special components
    legend_y = 0.1
    ax.text(1, legend_y, 'n = nz + nx (Z and X stabilizers)', fontsize=8, style='italic')

    plt.tight_layout()
    plt.savefig('figures/transformer_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/transformer_architecture.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figures/transformer_architecture.png")
    print("Saved: figures/transformer_architecture.pdf")
    plt.show()


def draw_detailed_attention():
    """Draw detailed attention mechanism with relative position bias"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    colors = {
        'input': '#E3F2FD',
        'linear': '#BBDEFB',
        'attention': '#FFCDD2',
        'bias': '#C8E6C9',
        'output': '#E1BEE7',
        'arrow': '#424242',
        'border': '#37474F',
    }

    def draw_box(x, y, w, h, text, color, fontsize=10):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=color, edgecolor=colors['border'],
                             linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize)

    # Title
    ax.text(6, 7.5, 'Multi-Head Self-Attention with 2D Relative Position Bias',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Input
    draw_box(5, 0.3, 2, 0.5, 'Input X', colors['input'], fontsize=10)

    # Q, K, V projections
    ax.annotate('', xy=(2.5, 1.3), xytext=(5.5, 0.8),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(6, 1.3), xytext=(6, 0.8),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(9.5, 1.3), xytext=(6.5, 0.8),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    draw_box(1.5, 1.3, 2, 0.6, 'Wq', colors['linear'], fontsize=10)
    draw_box(5, 1.3, 2, 0.6, 'Wk', colors['linear'], fontsize=10)
    draw_box(8.5, 1.3, 2, 0.6, 'Wv', colors['linear'], fontsize=10)

    # Q, K, V
    ax.annotate('', xy=(2.5, 2.4), xytext=(2.5, 1.9),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(6, 2.4), xytext=(6, 1.9),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(9.5, 2.4), xytext=(9.5, 1.9),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    ax.text(2.5, 2.2, 'Q', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6, 2.2, 'K', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(9.5, 2.2, 'V', ha='center', va='center', fontsize=11, fontweight='bold')

    # QK^T / sqrt(d)
    ax.annotate('', xy=(4.25, 3.0), xytext=(2.5, 2.5),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(4.25, 3.0), xytext=(6, 2.5),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    draw_box(3, 3.0, 2.5, 0.7, 'QKᵀ / √d', colors['attention'], fontsize=10)

    # Relative position bias
    draw_box(7, 3.0, 3.5, 0.7, '2D Rel. Pos. Bias', colors['bias'], fontsize=10)
    ax.text(8.75, 2.6, '(distance → bucket → bias)', ha='center', va='center', fontsize=8)

    # Add
    ax.annotate('', xy=(5.5, 4.2), xytext=(4.25, 3.7),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(5.5, 4.2), xytext=(8.75, 3.7),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    circle = Circle((5.5, 4.4), 0.25, facecolor='white', edgecolor=colors['border'], lw=1.5)
    ax.add_patch(circle)
    ax.text(5.5, 4.4, '+', ha='center', va='center', fontsize=14)

    # Softmax
    ax.annotate('', xy=(5.5, 5.1), xytext=(5.5, 4.65),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    draw_box(4.25, 5.1, 2.5, 0.6, 'Softmax', colors['attention'], fontsize=10)

    # Attention weights × V
    ax.annotate('', xy=(6.75, 5.7), xytext=(6.75, 5.4),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.annotate('', xy=(7.5, 5.7), xytext=(9.5, 2.5),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5,
                              connectionstyle="arc3,rad=0.2"))

    circle2 = Circle((7.25, 5.9), 0.25, facecolor='white', edgecolor=colors['border'], lw=1.5)
    ax.add_patch(circle2)
    ax.text(7.25, 5.9, '×', ha='center', va='center', fontsize=14)

    # Output projection
    ax.annotate('', xy=(6, 6.5), xytext=(7.25, 6.15),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    draw_box(5, 6.5, 2, 0.6, 'Wo', colors['linear'], fontsize=10)

    # Output
    ax.annotate('', xy=(6, 7.4), xytext=(6, 7.1),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))

    plt.tight_layout()
    plt.savefig('figures/attention_mechanism.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/attention_mechanism.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figures/attention_mechanism.png")
    print("Saved: figures/attention_mechanism.pdf")
    plt.show()


def draw_surface_code_embedding():
    """Draw how syndrome is embedded with 2D position"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Surface code layout
    ax1 = axes[0]
    ax1.set_xlim(-0.5, 4.5)
    ax1.set_ylim(-0.5, 4.5)
    ax1.set_aspect('equal')
    ax1.set_title('Surface Code (L=3)\nStabilizer Layout', fontsize=12, fontweight='bold')

    # Draw grid
    for i in range(5):
        ax1.axhline(y=i, color='lightgray', lw=0.5)
        ax1.axvline(x=i, color='lightgray', lw=0.5)

    # Z stabilizers (blue squares) - plaquettes
    z_positions = [(1, 1), (3, 1), (1, 3), (3, 3)]
    for i, (x, y) in enumerate(z_positions):
        rect = FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8,
                              boxstyle="round,pad=0.02",
                              facecolor='#2196F3', edgecolor='#1565C0', lw=2)
        ax1.add_patch(rect)
        ax1.text(x, y, f'Z{i+1}', ha='center', va='center', color='white',
                fontsize=10, fontweight='bold')

    # X stabilizers (red circles) - vertices
    x_positions = [(0, 0), (2, 0), (0, 2), (2, 2), (4, 2), (2, 4), (4, 4)]
    for i, (x, y) in enumerate(x_positions):
        circle = Circle((x, y), 0.35, facecolor='#F44336', edgecolor='#C62828', lw=2)
        ax1.add_patch(circle)
        ax1.text(x, y, f'X{i+1}', ha='center', va='center', color='white',
                fontsize=8, fontweight='bold')

    # Data qubits (small gray dots)
    for i in range(5):
        for j in range(5):
            if (i + j) % 2 == 0:  # data qubits on even positions
                ax1.plot(i, j, 'o', color='gray', markersize=5)

    ax1.set_xticks([])
    ax1.set_yticks([])

    # Legend
    z_patch = mpatches.Patch(color='#2196F3', label='Z Stabilizer')
    x_patch = mpatches.Patch(color='#F44336', label='X Stabilizer')
    ax1.legend(handles=[z_patch, x_patch], loc='upper right', fontsize=9)

    # Right: Token sequence
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    ax2.set_title('Transformer Input Sequence', fontsize=12, fontweight='bold')

    # Token boxes
    token_colors = ['#FFE0B2'] + ['#2196F3']*4 + ['#F44336']*7  # CLS + Z + X
    token_labels = ['[CLS]', 'Z₁', 'Z₂', 'Z₃', 'Z₄', 'X₁', 'X₂', 'X₃', 'X₄', 'X₅', 'X₆', 'X₇']

    for i, (color, label) in enumerate(zip(token_colors, token_labels)):
        x = 0.5 + (i % 6) * 1.5
        y = 6.5 - (i // 6) * 2

        box = FancyBboxPatch((x, y), 1.2, 0.8,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='#37474F', lw=1.5)
        ax2.add_patch(box)
        text_color = 'black' if color == '#FFE0B2' else 'white'
        ax2.text(x + 0.6, y + 0.4, label, ha='center', va='center',
                color=text_color, fontsize=9, fontweight='bold')

    # Annotations
    ax2.text(5, 4.8, 'Each token has:', ha='center', fontsize=10)
    ax2.text(5, 4.3, '• Syndrome value embedding (0 or 1)', ha='center', fontsize=9)
    ax2.text(5, 3.8, '• Type embedding (Z or X)', ha='center', fontsize=9)
    ax2.text(5, 3.3, '• 2D Position encoding (x, y coords)', ha='center', fontsize=9)

    # Arrow showing 2D position
    ax2.annotate('', xy=(3.5, 2.5), xytext=(2, 6.9),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.5,
                               connectionstyle="arc3,rad=-0.3"))
    ax2.text(1.5, 2.3, 'Position from\nSurface Code', ha='center', fontsize=8, color='#1565C0')

    plt.tight_layout()
    plt.savefig('figures/surface_code_embedding.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('figures/surface_code_embedding.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: figures/surface_code_embedding.png")
    print("Saved: figures/surface_code_embedding.pdf")
    plt.show()


if __name__ == '__main__':
    import os
    os.makedirs('figures', exist_ok=True)

    print("Drawing Transformer Architecture...")
    draw_transformer_architecture()

    print("\nDrawing Attention Mechanism...")
    draw_detailed_attention()

    print("\nDrawing Surface Code Embedding...")
    draw_surface_code_embedding()

    print("\nAll figures saved!")
