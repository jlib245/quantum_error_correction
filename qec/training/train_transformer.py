"""
Transformer Training Script for Quantum Error Correction
"""
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import numpy as np

from qec.core.codes import Get_surface_Code


def generate_correlated_noise(n_qubits, p_total, y_ratio=0.3):
    """
    계획서 목표 노이즈 모델: 상관 오류 (Y 오류)

    Args:
        n_qubits: 물리 큐빗 수
        p_total: 전체 오류 확률
        y_ratio: Y 오류 비율 (0.0~1.0)

    Returns:
        error_X, error_Z: numpy 배열
    """
    p_Y = p_total * y_ratio
    p_X = p_total * (1 - y_ratio) / 2
    p_Z = p_total * (1 - y_ratio) / 2

    rand_samples = np.random.rand(n_qubits)

    error_vector_X = np.zeros(n_qubits, dtype=int)
    error_vector_Z = np.zeros(n_qubits, dtype=int)

    # X 오류
    error_vector_X[rand_samples < p_X] = 1
    # Y 오류 (X와 Z 동시 발생)
    error_vector_X[(rand_samples >= p_X) & (rand_samples < p_X + p_Y)] = 1
    error_vector_Z[(rand_samples >= p_X) & (rand_samples < p_X + p_Y)] = 1
    # Z 오류
    error_vector_Z[(rand_samples >= p_X + p_Y) & (rand_samples < p_X + p_Y + p_Z)] = 1

    return error_vector_X, error_vector_Z


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def create_surface_code_pure_error_lut(L, error_type, device):
    """
    Surface code를 위한 Pure error LUT 생성
    model_path.py의 _surface_cn_coordinates 로직을 참고하여 정확한 stabilizer 구조 사용
    """
    device = torch.device('cpu')
    print(f"Creating Surface code pure error LUT for L={L}, error_type={error_type}")
    cn_coord = _surface_cn_coordinates(L, error_type)
    total_syndromes = len(cn_coord)
    N = L * L
    lut = {}
    print(f"Total syndrome bits: {total_syndromes}")
    for synd_idx in range(total_syndromes):
        if synd_idx not in cn_coord:
            continue
        face_row, face_col = cn_coord[synd_idx]
        fixed_row = L // 2
        fixed_col = L // 2
        if error_type == 'X_only':
            pure_error = _get_surface_outer_path_x(face_row, face_col, fixed_row, L, device)
        elif error_type == 'Z_only':
            pure_error = _get_surface_outer_path_z(face_row, face_col, fixed_col, L, device)
        else:
            num_z_stab = sum(1 for coord in cn_coord.values() if (coord[0] + coord[1]) % 2 == 0)
            if synd_idx < num_z_stab:
                pure_error = _get_surface_outer_path_x(face_row, face_col, fixed_row, L, device)
            else:
                pure_error = _get_surface_outer_path_z(face_row, face_col, fixed_col, L, device)

        lut[synd_idx] = pure_error.cpu()
    print(f"Pure error LUT created with {len(lut)} entries for error_type='{error_type}'")
    return {k: v.to(device) for k, v in lut.items()}


def _surface_cn_coordinates(L, error_type):
    """Surface code에서 CN index를 grid coordinate로 매핑"""
    mapping = {}
    old_idx = 0
    if error_type in ["both", "X_only"]:
        interior_z, boundary_z = [], []
        for row in range(L + 1):
            for col in range(L + 1):
                if (row + col) % 2 == 0:
                    if row in [0, L]: continue
                    if 1 <= row <= L - 1 and 1 <= col <= L - 1: interior_z.append((row, col))
                    else: boundary_z.append((row, col))
        for coord in interior_z: mapping[old_idx] = coord; old_idx += 1
        right_boundary = [(r, L) for r in range(1, L) if (r + L) % 2 == 0]
        left_boundary = [(r, 0) for r in range(1, L) if (r + 0) % 2 == 0]
        alt_boundary = []
        max_len = max(len(right_boundary), len(left_boundary))
        for i in range(max_len):
            if i < len(right_boundary): alt_boundary.append(right_boundary[i])
            if i < len(left_boundary): alt_boundary.append(left_boundary[i])
        for coord in alt_boundary: mapping[old_idx] = coord; old_idx += 1
    if error_type in ["both", "Z_only"]:
        interior_x, boundary_x = [], []
        for row in range(L + 1):
            for col in range(L + 1):
                if (row + col) % 2 == 1:
                    if col in [0, L]: continue
                    if 1 <= row <= L - 1 and 1 <= col <= L - 1: interior_x.append((row, col))
                    else: boundary_x.append((row, col))
        for coord in interior_x: mapping[old_idx] = coord; old_idx += 1
        top_boundary = [(0, c) for c in range(1, L) if (0 + c) % 2 == 1]
        bottom_boundary = [(L, c) for c in range(1, L) if (L + c) % 2 == 1]
        alt_boundary = []
        max_len = max(len(top_boundary), len(bottom_boundary))
        for i in range(max_len):
            if i < len(top_boundary): alt_boundary.append(top_boundary[i])
            if i < len(bottom_boundary): alt_boundary.append(bottom_boundary[i])
        for coord in alt_boundary: mapping[old_idx] = coord; old_idx += 1
    return mapping


def _get_surface_outer_path_x(face_row, face_col, fixed_row, L, device):
    pure_error = torch.zeros(L * L, dtype=torch.int64, device=device)
    vn_col = face_col - 1 if face_col == L else face_col
    up_vn_row, down_vn_row = face_row - 1, face_row
    start_row, direction = (up_vn_row, -1) if up_vn_row < fixed_row else (down_vn_row, 1)
    r = start_row
    while 0 <= r < L:
        vn_idx = r * L + vn_col
        pure_error[vn_idx] = 1
        r += direction
    return pure_error


def _get_surface_outer_path_z(face_row, face_col, fixed_col, L, device):
    pure_error = torch.zeros(L * L, dtype=torch.int64, device=device)
    vn_row = face_row - 1 if face_row == L else face_row
    left_vn_col, right_vn_col = face_col - 1, face_col
    start_col, direction = (left_vn_col, -1) if left_vn_col < fixed_col else (right_vn_col, 1)
    c = start_col
    while 0 <= c < L:
        vn_idx = vn_row * L + c
        pure_error[vn_idx] = 2
        c += direction
    return pure_error


def simple_decoder_C_torch(syndrome_vector, x_error_basis, z_error_basis, H_z, H_x):
    device = syndrome_vector.device
    c_x = torch.zeros(H_z.shape[1], dtype=torch.uint8, device=device)
    c_z = torch.zeros(H_x.shape[1], dtype=torch.uint8, device=device)

    s_z = syndrome_vector[:H_z.shape[0]]
    s_x = syndrome_vector[H_z.shape[0]:]

    for i in range(len(s_z)):
        if s_z[i] == 1 and i in x_error_basis:
            c_x.bitwise_xor_(x_error_basis[i])
    for i in range(len(s_x)):
        if s_x[i] == 1 and i in z_error_basis:
            c_z.bitwise_xor_(z_error_basis[i])

    return torch.cat([c_z, c_x])


class QECC_Dataset(data.Dataset):
    def __init__(self, code, x_error_basis, z_error_basis, ps, len, args, seed_offset=0):
        self.ps = ps
        self.len = len
        self.args = args
        self.seed_offset = seed_offset  # Train: 0, Test: large number

        self.device = next(iter(x_error_basis.values())).device if x_error_basis else torch.device("cpu")

        self.H_z = code.H_z.to(self.device)
        self.H_x = code.H_x.to(self.device)
        self.L_z = code.L_z.to(self.device)
        self.L_x = code.L_x.to(self.device)

        self.n_phys = self.H_z.shape[1]

        self.x_error_basis_dict = x_error_basis
        self.z_error_basis_dict = z_error_basis

    def generate_noise(self, p):
        if self.args.y_ratio > 0:
            e_x_np, e_z_np = generate_correlated_noise(self.n_phys, p, self.args.y_ratio)
        else:
            rand_vals = np.random.rand(self.n_phys)
            e_z_np = (rand_vals < p/3)
            e_x_np = (p/3 <= rand_vals) & (rand_vals < 2*p/3)
            e_y_np = (2*p/3 <= rand_vals) & (rand_vals < p)
            e_z_np, e_x_np = (e_z_np + e_y_np) % 2, (e_x_np + e_y_np) % 2

        e_z = torch.from_numpy(e_z_np).to(self.device, dtype=torch.uint8)
        e_x = torch.from_numpy(e_x_np).to(self.device, dtype=torch.uint8)
        return torch.cat([e_z, e_x])

    def __getitem__(self, index):
        # index 기반 seed로 매 epoch 같은 데이터 보장
        # seed_offset으로 train/test 분리 (test는 큰 offset 사용)
        local_seed = self.args.seed + index + self.seed_offset
        np.random.seed(local_seed)
        random.seed(local_seed)

        p = random.choice(self.ps)

        e_full = self.generate_noise(p)
        while not torch.any(e_full):
             e_full = self.generate_noise(p)

        e_z_actual = e_full[:self.n_phys]
        e_x_actual = e_full[self.n_phys:]

        s_z_actual = (self.H_z.float() @ e_x_actual.float()) % 2
        s_x_actual = (self.H_x.float() @ e_z_actual.float()) % 2
        syndrome = torch.cat([s_z_actual, s_x_actual])

        pure_error_C = simple_decoder_C_torch(syndrome.type(torch.uint8), self.x_error_basis_dict, self.z_error_basis_dict, self.H_z, self.H_x)
        l_physical = pure_error_C.long() ^ e_full.long()

        l_z_physical = l_physical[:self.n_phys]
        l_x_physical = l_physical[self.n_phys:]

        l_x_flip = (self.L_z.float() @ l_x_physical.float()) % 2
        l_z_flip = (self.L_x.float() @ l_z_physical.float()) % 2

        true_class_index = (l_z_flip * 2 + l_x_flip).long()

        return syndrome.float(), true_class_index.cpu()

    def __len__(self):
        return self.len


class FixedQECC_Dataset(data.Dataset):
    """논문용 고정 데이터셋 - 저장된 데이터 로드"""
    def __init__(self, syndromes, labels):
        self.syndromes = syndromes
        self.labels = labels

    def __getitem__(self, index):
        return self.syndromes[index], self.labels[index]

    def __len__(self):
        return len(self.syndromes)


def generate_and_save_dataset(code, x_error_basis, z_error_basis, p_errors,
                               n_samples, y_ratio, save_path, seed=42):
    """
    논문용 고정 데이터셋 생성 및 저장

    Args:
        code: 코드 객체
        x_error_basis: X 에러 LUT
        z_error_basis: Z 에러 LUT
        p_errors: 에러율 리스트
        n_samples: 샘플 수
        y_ratio: Y 에러 비율
        save_path: 저장 경로
        seed: 랜덤 시드
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    device = next(iter(x_error_basis.values())).device if x_error_basis else torch.device("cpu")

    H_z = code.H_z.to(device)
    H_x = code.H_x.to(device)
    L_z = code.L_z.to(device)
    L_x = code.L_x.to(device)
    n_phys = H_z.shape[1]

    syndromes = []
    labels = []

    print(f"Generating {n_samples} samples with seed={seed}...")

    for i in range(n_samples):
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{n_samples}")

        p = random.choice(p_errors)

        # Generate noise
        if y_ratio > 0:
            e_x_np, e_z_np = generate_correlated_noise(n_phys, p, y_ratio)
        else:
            rand_vals = np.random.rand(n_phys)
            e_z_np = (rand_vals < p/3)
            e_x_np = (p/3 <= rand_vals) & (rand_vals < 2*p/3)
            e_y_np = (2*p/3 <= rand_vals) & (rand_vals < p)
            e_z_np, e_x_np = (e_z_np + e_y_np) % 2, (e_x_np + e_y_np) % 2

        e_z = torch.from_numpy(e_z_np).to(device, dtype=torch.uint8)
        e_x = torch.from_numpy(e_x_np).to(device, dtype=torch.uint8)
        e_full = torch.cat([e_z, e_x])

        # Skip if no error
        if not torch.any(e_full):
            continue

        e_z_actual = e_full[:n_phys]
        e_x_actual = e_full[n_phys:]

        # Calculate syndrome
        s_z_actual = (H_z.float() @ e_x_actual.float()) % 2
        s_x_actual = (H_x.float() @ e_z_actual.float()) % 2
        syndrome = torch.cat([s_z_actual, s_x_actual])

        # Calculate label
        pure_error_C = simple_decoder_C_torch(syndrome.type(torch.uint8),
                                              x_error_basis, z_error_basis, H_z, H_x)
        l_physical = pure_error_C.long() ^ e_full.long()

        l_z_physical = l_physical[:n_phys]
        l_x_physical = l_physical[n_phys:]

        l_x_flip = (L_z.float() @ l_x_physical.float()) % 2
        l_z_flip = (L_x.float() @ l_z_physical.float()) % 2

        true_class_index = (l_z_flip * 2 + l_x_flip).long()

        syndromes.append(syndrome.float().cpu())
        labels.append(true_class_index.cpu())

    syndromes = torch.stack(syndromes)
    labels = torch.stack(labels)

    # Save dataset
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save({
        'syndromes': syndromes,
        'labels': labels,
        'p_errors': p_errors,
        'y_ratio': y_ratio,
        'seed': seed,
        'n_samples': len(syndromes)
    }, save_path)

    print(f"Dataset saved to {save_path}")
    print(f"  Samples: {len(syndromes)}")
    print(f"  Syndrome shape: {syndromes.shape}")
    print(f"  Class distribution: {torch.bincount(labels.squeeze(), minlength=4).tolist()}")

    return syndromes, labels


def load_dataset(load_path):
    """저장된 데이터셋 로드"""
    data = torch.load(load_path)
    print(f"Dataset loaded from {load_path}")
    print(f"  Samples: {data['n_samples']}")
    print(f"  p_errors: {data['p_errors']}")
    print(f"  y_ratio: {data['y_ratio']}")
    print(f"  seed: {data['seed']}")
    return data['syndromes'], data['labels'], data


def setup_device(device_type):
    """Setup compute device."""
    if device_type == 'cpu':
        device = torch.device("cpu")
        logging.info("CPU를 사용합니다 (강제 설정).")
    elif device_type == 'cuda':
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"NVIDIA GPU (CUDA)를 사용합니다: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("CUDA를 요청했지만 사용할 수 없습니다. CPU로 fallback합니다.")
            device = torch.device("cpu")
    elif device_type == 'xpu':
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device("xpu")
            logging.info(f"Intel Arc GPU (XPU)를 사용합니다: {torch.xpu.get_device_name(0)}")
        else:
            logging.warning("XPU를 요청했지만 사용할 수 없습니다. CPU로 fallback합니다.")
            device = torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"NVIDIA GPU (CUDA)를 사용합니다: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device("xpu")
            logging.info(f"Intel Arc GPU (XPU)를 사용합니다: {torch.xpu.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            logging.info("사용 가능한 GPU가 없어 CPU를 사용합니다.")
    return device


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, path):
    """Save training checkpoint."""
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_loss': best_loss,
    }
    torch.save(checkpoint, os.path.join(path, 'checkpoint.pt'))
    torch.save(model_to_save, os.path.join(path, 'best_model'))


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load training checkpoint."""
    logging.info(f"Resuming from checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))

        logging.info(f"Checkpoint loaded, resuming from epoch {start_epoch}, LR={scheduler.get_last_lr()[0]:.2e}")
        return start_epoch, best_loss
    else:
        # Legacy model format
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint.state_dict())
        else:
            model.load_state_dict(checkpoint.state_dict())

        logging.info("Model loaded (legacy format)")
        return None, float('inf')


def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ler = cum_samples = 0
    t = time.time()

    for batch_idx, (syndrome, labels) in enumerate(train_loader):

        syndrome, labels = syndrome.to(device), labels.to(device)
        outputs = model(syndrome)

        if isinstance(model, torch.nn.DataParallel):
            loss = model.module.loss(outputs, labels)
        else:
            loss = model.loss(outputs, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()

        cum_loss += loss.item() * syndrome.shape[0]
        cum_ler += correct
        cum_samples += syndrome.shape[0]

        if (batch_idx+1) % (len(train_loader)//2) == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e} LER={1 -(cum_ler / cum_samples):.3e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples


def test(model, device, test_loader_list, ps_range_test, cum_count_lim):
    model.eval()
    test_loss_ler_list, cum_samples_all = [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ler = cum_count = 0.
            while True:
                (syndrome, labels) = next(iter(test_loader))

                syndrome, labels = syndrome.to(device), labels.to(device)
                outputs = model(syndrome)

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()

                test_ler += correct
                cum_count += syndrome.shape[0]
                if cum_count > cum_count_lim:
                    break
            cum_samples_all.append(cum_count)
            test_loss_ler_list.append(1 - (test_ler / cum_count))

            # Handle both numeric and string p values
            p_val = ps_range_test[ii]
            if isinstance(p_val, (int, float)):
                print(f'Test p={p_val:.3e}, LER={test_loss_ler_list[-1]:.3e}')
            else:
                print(f'Test p={p_val}, LER={test_loss_ler_list[-1]:.3e}')

        # Format output based on p value types
        if all(isinstance(p, (int, float)) for p in ps_range_test):
            logging.info('Test LER  ' + ' '.join(
                ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
                 in (zip(test_loss_ler_list, ps_range_test))]))
        else:
            logging.info('Test LER  ' + ' '.join(
                ['p={}: {:.2e}'.format(ebno, elem) for (elem, ebno)
                 in (zip(test_loss_ler_list, ps_range_test))]))
        logging.info(f'Mean LER = {np.mean(test_loss_ler_list):.3e}')
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ler_list


def main(args):
    # Device selection
    device = setup_device(args.device)

    args.code.logic_matrix = args.code.logic_matrix.to(device)
    args.code.pc_matrix = args.code.pc_matrix.to(device)
    code = args.code

    x_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'X_only', device)
    z_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'Z_only', device)

    assert 0 < args.repetitions

    # Dataset generation mode (for reproducible experiments)
    if args.generate_dataset:
        dataset_dir = os.path.join(args.dataset_dir, f"L{args.code_L}_y{args.y_ratio}")
        os.makedirs(dataset_dir, exist_ok=True)

        # Generate training dataset
        train_path = os.path.join(dataset_dir, f"train_{args.n_train_samples}_seed{args.seed}.pt")
        generate_and_save_dataset(
            code, x_error_basis_dict, z_error_basis_dict, args.p_errors,
            args.n_train_samples, args.y_ratio, train_path, seed=args.seed
        )

        # Generate test dataset (separate seed)
        test_path = os.path.join(dataset_dir, f"test_{args.n_test_samples}_seed{args.test_seed}.pt")
        generate_and_save_dataset(
            code, x_error_basis_dict, z_error_basis_dict, args.p_errors,
            args.n_test_samples, args.y_ratio, test_path, seed=args.test_seed
        )

        logging.info(f"\nDatasets generated in {dataset_dir}")
        logging.info("Run training with --use_fixed_dataset flag")
        return

    from qec.models.transformer import ECC_Transformer

    model = ECC_Transformer(args, dropout=0).to(device)

    # DataParallel for multi-GPU (CUDA only)
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"NVIDIA GPU {torch.cuda.device_count()}개를 DataParallel로 사용합니다.")
        model = torch.nn.DataParallel(model)
    elif device.type == 'xpu':
        logging.info("Intel Arc GPU (XPU) - 단일 GPU 모드 (권장: L>=5, batch_size>=512)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logging.info(f'PC matrix shape {code.pc_matrix.shape}')
    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')

    ps_train = args.p_errors  # Train on all error rates
    ps_test = args.p_errors

    # pin_memory=True for faster CPU->GPU data transfer (CUDA and XPU)
    use_pin_memory = device.type in ['cuda', 'xpu']

    # OPTIMIZED: Add prefetch_factor for better pipeline (1.2-1.5x faster)
    prefetch = 2 if args.workers > 0 else None

    # Generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(args.seed)

    # Create DataLoaders
    if args.use_fixed_dataset:
        # Load pre-generated fixed datasets (for reproducible experiments)
        dataset_dir = os.path.join(args.dataset_dir, f"L{args.code_L}_y{args.y_ratio}")
        train_path = os.path.join(dataset_dir, f"train_{args.n_train_samples}_seed{args.seed}.pt")
        test_path = os.path.join(dataset_dir, f"test_{args.n_test_samples}_seed{args.test_seed}.pt")

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logging.error(f"Dataset not found. Run with --generate_dataset first.")
            logging.error(f"  Expected: {train_path}")
            logging.error(f"  Expected: {test_path}")
            return

        train_syndromes, train_labels, _ = load_dataset(train_path)
        test_syndromes, test_labels, _ = load_dataset(test_path)

        train_dataloader = DataLoader(
            FixedQECC_Dataset(train_syndromes, train_labels),
            batch_size=int(args.batch_size),
            shuffle=True, num_workers=args.workers,
            persistent_workers=True if args.workers > 0 else False,
            pin_memory=use_pin_memory,
            prefetch_factor=prefetch if args.workers > 0 else None,
            
            generator=g
        )

        # Single test dataloader for fixed dataset
        test_dataloader_list = [DataLoader(
            FixedQECC_Dataset(test_syndromes, test_labels),
            batch_size=int(args.test_batch_size),
            shuffle=False, num_workers=args.workers,
            persistent_workers=True if args.workers > 0 else False,
            pin_memory=use_pin_memory,
            prefetch_factor=prefetch if args.workers > 0 else None,
            
        )]
        ps_test = ['all']  # Indicate combined test set

        logging.info(f"Using fixed dataset from {dataset_dir}")
    else:
        # Dynamic dataset generation (original behavior)
        train_dataloader = DataLoader(
            QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, ps_train,
                        len=args.samples_per_epoch, args=args),
            batch_size=int(args.batch_size),
            shuffle=True, num_workers=args.workers,
            persistent_workers=True if args.workers > 0 else False,
            pin_memory=use_pin_memory,
            prefetch_factor=prefetch if args.workers > 0 else None,
            
            generator=g
        )

        test_dataloader_list = [
            DataLoader(
                QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, [ps_test[ii]],
                            len=int(args.test_batch_size), args=args, seed_offset=10_000_000),
                batch_size=int(args.test_batch_size),
                shuffle=False, num_workers=args.workers,
                persistent_workers=True if args.workers > 0 else False,
                pin_memory=use_pin_memory,
                prefetch_factor=prefetch if args.workers > 0 else None,
                
            ) for ii in range(len(ps_test))
        ]

    best_loss = float('inf')
    patience_counter = 0
    start_epoch = args.start_epoch

    # Resume from checkpoint
    if args.resume:
        if os.path.exists(args.resume):
            loaded_epoch, loaded_best_loss = load_checkpoint(
                args.resume, model, optimizer, scheduler, device
            )
            if loaded_epoch:
                start_epoch = loaded_epoch
                best_loss = loaded_best_loss
            else:
                # Legacy format - adjust scheduler manually
                for _ in range(args.start_epoch - 1):
                    scheduler.step()
                logging.info(f"Starting from epoch {args.start_epoch}, LR={scheduler.get_last_lr()[0]:.2e}")
        else:
            logging.error(f"Checkpoint not found: {args.resume}")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        # Shuffle seed는 epoch마다 다르게 (데이터는 index 기반으로 고정)
        g.manual_seed(args.seed + epoch)

        loss = train(model, device, train_dataloader, optimizer,
                     epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()

        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, args.path)

        # Check for improvement
        if loss < best_loss - args.min_delta:
            best_loss = loss
            patience_counter = 0

            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(model_to_save, os.path.join(args.path, 'final_model'))

            logging.info('Model Saved - New best loss: {:.5e}'.format(best_loss))
        else:
            patience_counter += 1
            logging.info('No improvement. Patience: {}/{}'.format(patience_counter, args.patience if args.patience > 0 else 'disabled'))

        # Early stopping check
        if args.patience > 0 and patience_counter >= args.patience:
            logging.info(f'Early stopping triggered after {epoch} epochs (patience={args.patience})')
            logging.info(f'Best loss: {best_loss:.5e}')
            break

        if epoch % 10 == 0 or epoch in [args.epochs]:
            test(model, device, test_dataloader_list, ps_test, args.test_samples)

    model = torch.load(os.path.join(args.path, 'best_model'), weights_only=False).to(device)

    logging.info('Best model loaded')

    # Final evaluation
    if args.use_fixed_dataset:
        # Use the same fixed test dataset
        final_test_list = test_dataloader_list
        final_ps_test = ps_test
    else:
        # Generate new test data
        final_ps_test = args.p_errors
        final_test_list = [
            DataLoader(
                QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, [final_ps_test[ii]],
                            len=int(args.test_batch_size), args=args, seed_offset=10_000_000),
                batch_size=int(args.test_batch_size),
                shuffle=False, num_workers=args.workers,
                persistent_workers=True, pin_memory=use_pin_memory, prefetch_factor=prefetch
            ) for ii in range(len(final_ps_test))
        ]

    final_ler = test(model, device, final_test_list, final_ps_test, args.test_samples)

    return {
        'best_loss': best_loss,
        'final_ler': final_ler,
        'model_path': args.path
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DQEC - Transformer')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gpus', type=str, default='0', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--test_samples', type=int, default=10000,
                        help='Number of test samples for evaluation')
    parser.add_argument('--samples_per_epoch', type=int, default=100000,
                        help='Number of samples per training epoch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'xpu'],
                        help='Device to use: auto (default), cpu, cuda, or xpu')

    # Early stopping args
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (epochs). 0 = disabled (default)')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Minimum change in loss to qualify as improvement')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model to resume training from')
    parser.add_argument('--start_epoch', type=int, default=1,
                        help='Starting epoch (use with --resume)')

    # Dataset args (for reproducible experiments)
    parser.add_argument('--generate_dataset', action='store_true',
                        help='Generate and save dataset only (no training)')
    parser.add_argument('--use_fixed_dataset', action='store_true',
                        help='Use pre-generated fixed dataset for training')
    parser.add_argument('--dataset_dir', type=str, default='datasets',
                        help='Directory for saving/loading datasets')
    parser.add_argument('--n_train_samples', type=int, default=1000000,
                        help='Number of training samples for fixed dataset')
    parser.add_argument('--n_test_samples', type=int, default=100000,
                        help='Number of test samples for fixed dataset')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Number of experiment runs for statistical significance')
    parser.add_argument('--test_seed', type=int, default=12345,
                        help='Separate seed for test dataset')

    # Code args
    parser.add_argument('--code_type', type=str, default='surface',choices=['surface'])
    parser.add_argument('-L', '--code_L', type=int, default=3,help='Lattice length')
    parser.add_argument('--repetitions', type=int, default=1,help='Number of faulty repetitions. <=1 is equivalent to none.')
    parser.add_argument('--noise_type', type=str,default='independent', choices=['independent','depolarization'],help='Noise model')
    parser.add_argument('-p', '--p_errors', type=float, nargs='+',
                        default=[0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13],
                        help='Physical error rates for training/testing')
    parser.add_argument('-y', '--y_ratio', type=float, default=0.0,
                        help="Ratio of Y errors for correlated noise (default: 0.0 = independent noise)")

    # model args
    parser.add_argument('--N_dec', type=int, default=6,help='Number of QECCT self-attention modules')
    parser.add_argument('--d_model', type=int, default=128,help='QECCT dimension')
    parser.add_argument('--h', type=int, default=16,help='Number of heads')

    # qecc args
    parser.add_argument('--lambda_loss_ber', type=float, default=0.3,help='BER loss regularization')
    parser.add_argument('--lambda_loss_ler', type=float, default=1.,help='LER loss regularization')
    parser.add_argument('--lambda_loss_n_pred', type=float, default=0.3,help='g noise prediction regularization')
    parser.add_argument('--lambda_loss_log_pred', type=float, default=1,help='g noise prediction regularization')

    # ablation args
    parser.add_argument('--no_g', type=int, default=0)
    parser.add_argument('--no_mask', type=int, default=0)

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)

    if args.no_g > 0:
        args.lambda_loss_n_pred= 0.

    class Code():
        pass
    code = Code()

    Hx, Hz, Lx, Lz = eval(f'Get_surface_Code')(args.code_L)
    H_x = torch.from_numpy(Hx).long()
    H_z = torch.from_numpy(Hz).long()
    L_x = torch.from_numpy(Lx).long()
    L_z = torch.from_numpy(Lz).long()
    H = torch.block_diag(H_z, H_x)
    L = torch.block_diag(L_z, L_x)

    code.H_z = H_z
    code.H_x = H_x
    code.L_z = L_z
    code.L_x = L_x

    code.logic_matrix = L
    code.pc_matrix = H
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]
    code.code_type = args.code_type
    args.code = code
    args.L = args.code_L  # For 2D positional encoding

    model_dir = os.path.join('Final_Results_QECCT', args.code_type,
                             'Transformer_Code_L_' + str(args.code_L),
                             f'noise_model_{args.noise_type}',
                             f'repetition_{args.repetitions}',
                             datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))

    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [
        logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(message)s',
                        handlers=handlers)
    logging.info(f"Path to model/logs: {model_dir}")
    logging.info(args)

    # Environment info for reproducibility
    logging.info(f"\n=== Environment Info ===")
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"NumPy version: {np.__version__}")
    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"cuDNN version: {torch.backends.cudnn.version()}")
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        logging.info(f"XPU: {torch.xpu.get_device_name(0)}")
    else:
        logging.info("Device: CPU")
    logging.info(f"========================\n")

    # Run experiments
    if args.n_runs > 1:
        # Multiple runs for statistical significance
        all_results = []
        all_lers = []

        for run_idx in range(args.n_runs):
            run_seed = args.seed + run_idx * 1000
            set_seed(run_seed)

            logging.info(f"\n{'='*60}")
            logging.info(f"RUN {run_idx + 1}/{args.n_runs} (seed={run_seed})")
            logging.info(f"{'='*60}")

            # Update model directory for this run
            run_model_dir = os.path.join(model_dir, f"run_{run_idx + 1}")
            os.makedirs(run_model_dir, exist_ok=True)
            args.path = run_model_dir

            result = main(args)
            if result:
                all_results.append(result)
                if result.get('final_ler'):
                    all_lers.append(np.mean(result['final_ler']))

        # Print statistical summary
        if all_lers:
            logging.info(f"\n{'='*60}")
            logging.info("STATISTICAL SUMMARY")
            logging.info(f"{'='*60}")
            logging.info(f"Number of runs: {len(all_lers)}")
            logging.info(f"Mean LER: {np.mean(all_lers):.6e}")
            logging.info(f"Std LER:  {np.std(all_lers):.6e}")
            logging.info(f"Min LER:  {np.min(all_lers):.6e}")
            logging.info(f"Max LER:  {np.max(all_lers):.6e}")
            logging.info(f"\nResult: {np.mean(all_lers):.6e} ± {np.std(all_lers):.6e}")
            logging.info(f"{'='*60}")
    else:
        # Single run
        main(args)
