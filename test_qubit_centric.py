"""
Test script for Qubit-Centric models
"""
import torch
import numpy as np
import time

# Add path
import sys
sys.path.insert(0, '/home/jiindesktop/doc_ubt/quantum_error_correction')

from qec.training.common import (
    load_surface_code,
    create_surface_code_pure_error_lut,
    simple_decoder_C_torch,
)


def test_lut_residual():
    """Test LUT Residual model logic"""
    print("=" * 60)
    print("Testing LUT Residual Logic")
    print("=" * 60)

    device = torch.device('cpu')
    L = 3

    # Load code
    code = load_surface_code(L, device)
    H_z = code.H_z.float()
    H_x = code.H_x.float()

    print(f"\nCode L={L}")
    print(f"H_z shape: {H_z.shape}")  # (n_z, n_qubits)
    print(f"H_x shape: {H_x.shape}")  # (n_x, n_qubits)

    # Create LUT
    x_error_lut = create_surface_code_pure_error_lut(L, 'X_only', device)
    z_error_lut = create_surface_code_pure_error_lut(L, 'Z_only', device)

    # Precompute HtH
    HtH_z = H_z.T @ H_z
    HtH_x = H_x.T @ H_x

    print(f"\nHtH_z shape: {HtH_z.shape}")
    print(f"HtH_z:\n{HtH_z}")

    n_z = H_z.shape[0]
    n_x = H_x.shape[0]
    n_qubits = H_z.shape[1]

    # Test 1: Single X error at qubit 4 (center)
    print("\n" + "-" * 40)
    print("Test 1: Single X error at qubit 4")
    print("-" * 40)

    e_x = torch.zeros(n_qubits)
    e_x[4] = 1  # X error at center qubit
    e_z = torch.zeros(n_qubits)

    # Compute syndrome
    s_z = (H_z @ e_x) % 2  # Z stabilizers detect X errors
    s_x = (H_x @ e_z) % 2  # X stabilizers detect Z errors
    syndrome = torch.cat([s_z, s_x])

    print(f"Error e_x: {e_x.tolist()}")
    print(f"Syndrome s_z: {s_z.tolist()}")
    print(f"Syndrome s_x: {s_x.tolist()}")

    # LUT prediction
    lut_error = simple_decoder_C_torch(syndrome.byte(), x_error_lut, z_error_lut, H_z.long(), H_x.long())
    lut_e_z = lut_error[:n_qubits].float()
    lut_e_x = lut_error[n_qubits:].float()

    print(f"LUT e_z: {lut_e_z.tolist()}")
    print(f"LUT e_x: {lut_e_x.tolist()}")

    # Real qubit count: H.T @ syndrome
    real_z_count = H_z.T @ s_z
    real_x_count = H_x.T @ s_x

    print(f"\nReal Z count (H_z.T @ s_z): {real_z_count.tolist()}")
    print(f"Real X count (H_x.T @ s_x): {real_x_count.tolist()}")

    # LUT qubit count: HtH @ lut_error
    lut_z_count = HtH_z @ lut_e_x  # X error triggers Z stabilizers
    lut_x_count = HtH_x @ lut_e_z  # Z error triggers X stabilizers

    print(f"LUT Z count (HtH_z @ lut_e_x): {lut_z_count.tolist()}")
    print(f"LUT X count (HtH_x @ lut_e_z): {lut_x_count.tolist()}")

    # Residual
    residual_z = real_z_count - lut_z_count
    residual_x = real_x_count - lut_x_count

    print(f"\nResidual Z: {residual_z.tolist()}")
    print(f"Residual X: {residual_x.tolist()}")

    # Check if LUT is correct
    lut_syndrome_z = (H_z @ lut_e_x) % 2
    print(f"\nLUT syndrome from lut_e_x: {lut_syndrome_z.tolist()}")
    print(f"Original syndrome s_z:     {s_z.tolist()}")
    print(f"Syndromes match: {torch.allclose(lut_syndrome_z, s_z)}")

    # Test 2: Multiple errors
    print("\n" + "-" * 40)
    print("Test 2: Multiple X errors at qubit 0, 4")
    print("-" * 40)

    e_x = torch.zeros(n_qubits)
    e_x[0] = 1
    e_x[4] = 1
    e_z = torch.zeros(n_qubits)

    s_z = (H_z @ e_x) % 2
    s_x = (H_x @ e_z) % 2
    syndrome = torch.cat([s_z, s_x])

    print(f"Error e_x: {e_x.tolist()}")
    print(f"Syndrome s_z: {s_z.tolist()}")

    lut_error = simple_decoder_C_torch(syndrome.byte(), x_error_lut, z_error_lut, H_z.long(), H_x.long())
    lut_e_x = lut_error[n_qubits:].float()

    print(f"LUT e_x: {lut_e_x.tolist()}")

    real_z_count = H_z.T @ s_z
    lut_z_count = HtH_z @ lut_e_x
    residual_z = real_z_count - lut_z_count

    print(f"Real Z count: {real_z_count.tolist()}")
    print(f"LUT Z count: {lut_z_count.tolist()}")
    print(f"Residual Z: {residual_z.tolist()}")

    # Test 3: Batch LUT lookup speed
    print("\n" + "-" * 40)
    print("Test 3: Batch LUT lookup speed")
    print("-" * 40)

    batch_size = 1024
    # Random syndromes
    syndromes = torch.randint(0, 2, (batch_size, n_z + n_x)).float()

    start = time.time()
    for b in range(batch_size):
        syndrome = syndromes[b]
        lut_error = simple_decoder_C_torch(syndrome.byte(), x_error_lut, z_error_lut, H_z.long(), H_x.long())
    elapsed = time.time() - start

    print(f"Batch size: {batch_size}")
    print(f"Time for {batch_size} LUT lookups: {elapsed:.3f}s")
    print(f"Time per lookup: {elapsed/batch_size*1000:.3f}ms")


def test_qubit_centric_output():
    """Test QubitCentric model output shapes"""
    print("\n" + "=" * 60)
    print("Testing Model Output Shapes")
    print("=" * 60)

    device = torch.device('cpu')
    L = 5

    # Create args
    class Args:
        code_L = L
        d_model = 128

    args = Args()

    # Load code
    code = load_surface_code(L, device)
    args.code = code

    x_error_lut = create_surface_code_pure_error_lut(L, 'X_only', device)
    z_error_lut = create_surface_code_pure_error_lut(L, 'Z_only', device)

    n_z = code.H_z.shape[0]
    n_x = code.H_x.shape[0]

    # Import models
    from qec.models.qubit_centric import ECC_QubitCentric, ECC_LUT_Residual, ECC_LUT_Concat

    # Test input
    batch_size = 4
    syndrome = torch.randint(0, 2, (batch_size, n_z + n_x)).float()

    # Test QubitCentric
    print("\n1. ECC_QubitCentric")
    model1 = ECC_QubitCentric(args)
    out1 = model1(syndrome)
    print(f"   Input shape: {syndrome.shape}")
    print(f"   Output shape: {out1.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")

    # Test LUT_Residual
    print("\n2. ECC_LUT_Residual")
    model2 = ECC_LUT_Residual(args, x_error_lut, z_error_lut)

    start = time.time()
    out2 = model2(syndrome)
    elapsed = time.time() - start

    print(f"   Input shape: {syndrome.shape}")
    print(f"   Output shape: {out2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")
    print(f"   Forward time: {elapsed*1000:.1f}ms")

    # Test LUT_Concat
    print("\n3. ECC_LUT_Concat")
    model3 = ECC_LUT_Concat(args, x_error_lut, z_error_lut)
    out3 = model3(syndrome)
    print(f"   Input shape: {syndrome.shape}")
    print(f"   Output shape: {out3.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model3.parameters()):,}")


def test_residual_meaning():
    """Test what residual actually represents"""
    print("\n" + "=" * 60)
    print("Testing Residual Meaning")
    print("=" * 60)

    device = torch.device('cpu')
    L = 3

    code = load_surface_code(L, device)
    H_z = code.H_z.float()
    H_x = code.H_x.float()
    L_z = code.L_z.float()
    L_x = code.L_x.float()

    x_error_lut = create_surface_code_pure_error_lut(L, 'X_only', device)
    z_error_lut = create_surface_code_pure_error_lut(L, 'Z_only', device)

    HtH_z = H_z.T @ H_z

    n_z = H_z.shape[0]
    n_x = H_x.shape[0]
    n_qubits = H_z.shape[1]

    print(f"\nLogical X operator (L_z): {L_z.tolist()}")
    print(f"Logical Z operator (L_x): {L_x.tolist()}")

    # Case: Error that causes logical error
    print("\n" + "-" * 40)
    print("Case: Logical X error (vertical line of X errors)")
    print("-" * 40)

    # Vertical line of X errors (should cause logical X)
    e_x = torch.zeros(n_qubits)
    e_x[1] = 1  # Column 1
    e_x[4] = 1
    e_x[7] = 1
    e_z = torch.zeros(n_qubits)

    s_z = (H_z @ e_x) % 2
    s_x = (H_x @ e_z) % 2
    syndrome = torch.cat([s_z, s_x])

    print(f"Error e_x (vertical line): {e_x.view(3,3).tolist()}")
    print(f"Syndrome s_z: {s_z.tolist()}")

    # Logical error check
    l_x_flip = (L_z @ e_x) % 2
    print(f"Logical X flip (L_z @ e_x): {l_x_flip.tolist()}")

    # LUT prediction
    lut_error = simple_decoder_C_torch(syndrome.byte(), x_error_lut, z_error_lut, H_z.long(), H_x.long())
    lut_e_x = lut_error[n_qubits:].float()

    print(f"LUT e_x: {lut_e_x.view(3,3).tolist()}")

    # Physical error after LUT correction
    corrected_e_x = (e_x.long() ^ lut_e_x.long()).float()
    print(f"Corrected e_x (e_x ^ lut_e_x): {corrected_e_x.view(3,3).tolist()}")

    # Logical error after correction
    l_x_after = (L_z @ corrected_e_x) % 2
    print(f"Logical X after correction: {l_x_after.tolist()}")

    # Residual
    real_z_count = H_z.T @ s_z
    lut_z_count = HtH_z @ lut_e_x
    residual_z = real_z_count - lut_z_count

    print(f"\nReal Z count: {real_z_count.view(3,3).tolist()}")
    print(f"LUT Z count: {lut_z_count.view(3,3).tolist()}")
    print(f"Residual Z: {residual_z.view(3,3).tolist()}")

    print("\n결론: Residual이 0이 아니면 → LUT가 완벽하지 않음 → Logical error 가능성")


if __name__ == '__main__':
    test_lut_residual()
    test_qubit_centric_output()
    test_residual_meaning()
