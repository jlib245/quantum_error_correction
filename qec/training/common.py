"""
Common utilities for QEC training scripts.
Shared by transformer.py, cnn.py, ffnn.py
"""
import random
import os
import logging
import time
import torch
import numpy as np
from torch.utils import data
from qec.core.codes import Get_surface_Code


# ============================================
# Seed & Device Setup
# ============================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


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


# ============================================
# Noise Generation
# ============================================

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


# ============================================
# LUT Generation (Surface Code)
# ============================================

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
    """LUT 기반 simple decoder"""
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


# ============================================
# Code Loading
# ============================================

class Code:
    """Code object for storing parity check and logical matrices."""
    pass


def load_surface_code(L, device):
    """Load surface code and return code object."""
    Hx, Hz, Lx, Lz = Get_surface_Code(L)

    code = Code()
    code.H_x = torch.from_numpy(Hx).long()
    code.H_z = torch.from_numpy(Hz).long()
    code.L_x = torch.from_numpy(Lx).long()
    code.L_z = torch.from_numpy(Lz).long()
    code.pc_matrix = torch.block_diag(code.H_z, code.H_x)
    code.logic_matrix = torch.block_diag(code.L_z, code.L_x)
    code.n = code.pc_matrix.shape[1]
    code.k = code.n - code.pc_matrix.shape[0]

    return code


# ============================================
# Dataset
# ============================================

class QECC_Dataset(data.Dataset):
    """Common dataset for QEC training."""

    def __init__(self, code, x_error_basis, z_error_basis, ps, length, args, seed_offset=0):
        self.ps = ps
        self.length = length
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
        # Mixed y-ratio 지원: y_ratios 리스트가 있으면 랜덤 선택
        if hasattr(self.args, 'y_ratios') and self.args.y_ratios:
            y_ratio = random.choice(self.args.y_ratios)
        else:
            y_ratio = getattr(self.args, 'y_ratio', 0.0)

        if y_ratio > 0:
            e_x_np, e_z_np = generate_correlated_noise(self.n_phys, p, y_ratio)
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
        return self.length


class FixedQECC_Dataset(data.Dataset):
    """논문용 고정 데이터셋 - 저장된 데이터 로드"""
    def __init__(self, syndromes, labels):
        self.syndromes = syndromes
        self.labels = labels

    def __getitem__(self, index):
        return self.syndromes[index], self.labels[index]

    def __len__(self):
        return len(self.syndromes)


# ============================================
# Dataset Generation & Loading
# ============================================

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


# ============================================
# Training & Testing
# ============================================

def train_epoch(model, device, train_loader, optimizer, epoch, LR):
    """Train for one epoch."""
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

        if (batch_idx+1) % max(1, len(train_loader)//2) == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e} LER={1 -(cum_ler / cum_samples):.3e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples


def test_model(model, device, test_loader_list, ps_range_test, cum_count_lim):
    """Test model on multiple p values."""
    model.eval()
    test_loss_ler_list, cum_samples_all = [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ler = cum_count = 0.
            for syndrome, labels in test_loader:
                syndrome, labels = syndrome.to(device), labels.to(device)
                outputs = model(syndrome)

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()

                test_ler += correct
                cum_count += syndrome.shape[0]
                if cum_count >= cum_count_lim:
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


# ============================================
# Checkpoint
# ============================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, path, metric_name='loss'):
    """Save training checkpoint."""
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
        'metric_name': metric_name,
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
        best_metric = checkpoint.get('best_metric', checkpoint.get('best_loss', float('inf')))

        logging.info(f"Checkpoint loaded, resuming from epoch {start_epoch}, LR={scheduler.get_last_lr()[0]:.2e}")
        return start_epoch, best_metric
    else:
        # Legacy model format
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint.state_dict())
        else:
            model.load_state_dict(checkpoint.state_dict())

        logging.info("Model loaded (legacy format)")
        return None, float('inf')


# ============================================
# Training Loop with Validation-based Early Stopping
# ============================================

def train_with_validation(model, device, train_loader, val_loaders, optimizer, scheduler,
                          args, ps_test):
    """
    Validation LER 기반 early stopping을 사용하는 training loop.

    Args:
        model: 모델
        device: 디바이스
        train_loader: 학습 데이터로더
        val_loaders: 검증 데이터로더 리스트 (각 p 값에 대해)
        optimizer: 옵티마이저
        scheduler: 스케줄러
        args: 설정 (epochs, patience, min_delta, path, test_samples 등)
        ps_test: 테스트 p 값 리스트

    Returns:
        best_val_ler: 최고 validation LER
    """
    start_epoch = 1
    best_val_ler = float('inf')

    # Resume from checkpoint
    if hasattr(args, 'resume') and args.resume and os.path.exists(args.resume):
        loaded_epoch, loaded_best = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        if loaded_epoch:
            start_epoch = loaded_epoch + 1
            best_val_ler = loaded_best
            logging.info(f"Resumed from epoch {loaded_epoch}, best_val_ler={best_val_ler:.5e}")

    patience_counter = 0
    val_interval = getattr(args, 'val_interval', 5)  # 기본 5 epoch마다 validation

    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch,
                                  scheduler.get_last_lr()[0])
        scheduler.step()

        # Validation (매 val_interval epoch 또는 마지막 epoch)
        if epoch % val_interval == 0 or epoch == args.epochs:
            val_lers = test_model(model, device, val_loaders, ps_test, args.test_samples)
            mean_val_ler = np.mean(val_lers)

            # Best model 저장 (validation LER 기준)
            if mean_val_ler < best_val_ler - args.min_delta:
                best_val_ler = mean_val_ler
                patience_counter = 0
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_ler,
                               args.path, metric_name='val_ler')
                logging.info(f'Model Saved - Best val LER: {best_val_ler:.5e}')
            else:
                patience_counter += 1
                logging.info(f'No improvement. Patience: {patience_counter}/{args.patience}')

            # Early stopping
            if args.patience > 0 and patience_counter >= args.patience:
                logging.info(f'Early stopping at epoch {epoch}')
                break

    # Final test with best model
    logging.info("Best model loaded")
    model_path = os.path.join(args.path, 'best_model')
    if os.path.exists(model_path):
        best_model = torch.load(model_path, weights_only=False).to(device)
        test_model(best_model, device, val_loaders, ps_test, args.test_samples)

    return best_val_ler
