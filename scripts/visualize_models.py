 # scripts/visualize_models.py

import torch
import sys
sys.path.insert(0, '.')

# pip install torchviz graphviz 필요
from torchviz import make_dot

from qec.core.codes import Get_surface_Code
from qec.models.ffnn import ECC_FFNN
from qec.models.cnn import ECC_CNN
from qec.models.transformer import ECC_Transformer

def setup_code(L=5):
    """코드 설정"""
    Hx, Hz, Lx, Lz = Get_surface_Code(L)

    class Code:
        pass

    code = Code()
    code.H_z = torch.from_numpy(Hz).long()
    code.H_x = torch.from_numpy(Hx).long()
    code.L_z = torch.from_numpy(Lz).long()
    code.L_x = torch.from_numpy(Lx).long()
    code.pc_matrix = torch.block_diag(code.H_z, code.H_x)
    code.logic_matrix = torch.block_diag(code.L_z, code.L_x)

    return code

def visualize_model(model, model_name, input_size, L=5):
    """모델 시각화"""
    # 더미 입력 (0 또는 1의 syndrome)
    x = torch.randint(0, 2, (1, input_size)).float()

    # Forward pass
    y = model(x)

    # 시각화
    dot = make_dot(y, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    dot.render(f'figures/model_{model_name}_L{L}', format='png', cleanup=True)
    print(f"Saved: figures/model_{model_name}_L{L}.png")

def main():
    L = 5
    code = setup_code(L)

    # Args 설정
    class Args:
        pass

    args = Args()
    args.code = code
    args.code_L = L
    args.L = L
    args.hidden_size = 256
    args.d_model = 128
    args.h = 16
    args.N_dec = 6
    args.no_g = 0
    args.no_mask = 0

    syndrome_size = code.pc_matrix.shape[0]  # 24 for L=5

    # 1. FFNN
    print("Visualizing FFNN...")
    ffnn = ECC_FFNN(args, dropout=0.2, label_smoothing=0.1)
    visualize_model(ffnn, 'FFNN', syndrome_size, L)

    # 2. CNN
    print("Visualizing CNN...")
    cnn = ECC_CNN(args, dropout=0.2, label_smoothing=0.1)
    visualize_model(cnn, 'CNN', syndrome_size, L)

    # 3. Transformer
    print("Visualizing Transformer...")
    transformer = ECC_Transformer(args, dropout=0.2, label_smoothing=0.1)
    visualize_model(transformer, 'Transformer', syndrome_size, L)

    print("\nDone! Check figures/ folder.")

if __name__ == '__main__':
    main()