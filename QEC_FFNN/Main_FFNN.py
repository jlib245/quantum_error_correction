"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
from __future__ import print_function
import argparse
import random
import os
from torch.utils.data import DataLoader
from torch.utils import data
from datetime import datetime
import logging
from Codes import *
from itertools import combinations
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
import shutil
####################################################################################################################################
####################################################################################################################################

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
    # (기존 코드와 동일)
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
             # 'both'의 경우를 단순화하여 X 또는 Z 하나만 생성하도록 가정
             # (원래 코드에는 'both'에 대한 복잡한 분기 로직이 있었음)
            num_z_stab = sum(1 for coord in cn_coord.values() if (coord[0] + coord[1]) % 2 == 0)
            if synd_idx < num_z_stab:
                pure_error = _get_surface_outer_path_x(face_row, face_col, fixed_row, L, device)
            else:
                pure_error = _get_surface_outer_path_z(face_row, face_col, fixed_col, L, device)

        lut[synd_idx] = pure_error.cpu()
    print(f"Pure error LUT created with {len(lut)} entries for error_type='{error_type}'")
    return {k: v.to(device) for k, v in lut.items()}


def _surface_cn_coordinates(L, error_type):
    """
    Surface code에서 CN index를 grid coordinate로 매핑
    """
    # (기존 코드와 동일)
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
    # (기존 코드와 동일)
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
    # (기존 코드와 동일)
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
            
    # 에러 벡터 e가 [e_z, e_x] 순서이므로, 순수오류 C(T)도 [c_z, c_x] 순서로 반환
    return torch.cat([c_z, c_x])

####################################################################################################################################
# --- QECC_Dataset 클래스 전체 수정 ---
####################################################################################################################################

class QECC_Dataset(data.Dataset):
    def __init__(self, code, x_error_basis, z_error_basis, ps, len, args):
        self.ps = ps
        self.len = len
        self.args = args

        # GPU 디바이스 설정
        self.device = next(iter(x_error_basis.values())).device if x_error_basis else torch.device("cpu")

        # 코드 행렬들을 GPU로 미리 이동시켜 둠
        self.H_z = code.H_z.to(self.device)
        self.H_x = code.H_x.to(self.device)
        self.L_z = code.L_z.to(self.device)
        self.L_x = code.L_x.to(self.device)
        
        self.n_phys = self.H_z.shape[1] # 물리 큐빗 수
        
        # 미리 계산된 pure error basis 저장
        self.x_error_basis_dict = x_error_basis
        self.z_error_basis_dict = z_error_basis

    def generate_noise(self, p):
        # e = [e_z, e_x] 순서로 생성
        # Y-error를 포함한 depolarizing noise 모델
        rand_vals = np.random.rand(self.n_phys)
        e_z_np = (rand_vals < p/3)
        e_x_np = (p/3 <= rand_vals) & (rand_vals < 2*p/3)
        e_y_np = (2*p/3 <= rand_vals) & (rand_vals < p)
        
        e_z_np, e_x_np = (e_z_np + e_y_np) % 2, (e_x_np + e_y_np) % 2
        
        e_z = torch.from_numpy(e_z_np).to(self.device, dtype=torch.uint8)
        e_x = torch.from_numpy(e_x_np).to(self.device, dtype=torch.uint8)
        
        return torch.cat([e_z, e_x])

    def __getitem__(self, index):
        p = random.choice(self.ps)
        
        # 1. 물리적 오류 E = [e_z, e_x] 랜덤 생성
        e_full = self.generate_noise(p)
        while not torch.any(e_full): # 오류가 없는 경우는 제외
             e_full = self.generate_noise(p)
        
        e_z_actual = e_full[:self.n_phys]
        e_x_actual = e_full[self.n_phys:]

        # 2. 신드롬 s 계산 (모델의 입력)
        s_z_actual = (self.H_z.float() @ e_x_actual.float()) % 2
        s_x_actual = (self.H_x.float() @ e_z_actual.float()) % 2
        syndrome = torch.cat([s_z_actual, s_x_actual])

        # 3. 정답 논리 클래스 L 계산
        pure_error_C = simple_decoder_C_torch(syndrome.type(torch.uint8), self.x_error_basis_dict, self.z_error_basis_dict, self.H_z, self.H_x)
        l_physical = pure_error_C.long() ^ e_full.long()
        
        l_z_physical = l_physical[:self.n_phys]
        l_x_physical = l_physical[self.n_phys:]
        
        l_x_flip = (self.L_z.float() @ l_x_physical.float()) % 2
        l_z_flip = (self.L_x.float() @ l_z_physical.float()) % 2
        
        # [l_z, l_x] -> 클래스 인덱스 [0, 1, 2, 3] 변환
        true_class_index = (l_z_flip * 2 + l_x_flip).long()
        
        # 모델 입력(신드롬)과 정답(클래스)만 CPU로 보내서 반환
        return syndrome.float(), true_class_index.cpu()
    
    def __len__(self):
        return self.len


####################################################################################################################################
####################################################################################################################################
class Binarization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors
        return grad_output*(torch.abs(x[0])<=1)

def binarization(y):
    return sign_to_bin(Binarization.apply(y))
###
def logical_flipped(L,x):
    return torch.matmul(x.float(),L.float()) % 2
###
def diff_GF2_mul(H,x):    
    H_bin = sign_to_bin(H) if -1 in H else H
    x_bin = x
    
    tmp = bin_to_sign(H_bin.unsqueeze(0)*x_bin.unsqueeze(-1))
    tmp = torch.prod(tmp,1)
    tmp = sign_to_bin(tmp)

    # assert torch.allclose(logical_flipped(H_bin,x_bin).cpu().detach().bool(), tmp.detach().cpu().bool())
    return tmp

####################################################################################################################################
####################################################################################################################################

def train(model, device, train_loader, optimizer, epoch, LR):
    model.train()
    cum_loss = cum_ber = cum_ler = cum_samples = 0
    correct = cum_loss1 = cum_loss2 = cum_loss3 = 0
    t = time.time()
    # bin_fun = binarization
    bin_fun = torch.sigmoid
    for batch_idx, (syndrome, labels) in enumerate(train_loader):
        
        syndrome, labels = syndrome.to(device), labels.to(device)
        outputs = model(syndrome)

        loss = model.module.loss(outputs, labels)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        ###
        _, predicted = torch.max(outputs.data, 1)

        correct = (predicted == labels).sum().item()

        cum_loss += loss.item() * syndrome.shape[0]
        #
        cum_ler += correct
        
        cum_samples += syndrome.shape[0]
        #
        if (batch_idx+1) % (len(train_loader)//2) == 0 or batch_idx == len(train_loader) - 1:
            logging.info(
                f'Training epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}: LR={LR:.2e}, Loss={cum_loss / cum_samples:.5e} LER={1 -(cum_ler / cum_samples):.3e}')
            logging.info(
                f'***Loss={cum_loss / cum_samples:.5e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - t}s\n')
    return cum_loss / cum_samples, cum_ber / cum_samples, cum_ler / cum_samples

####################################################################################################################################
####################################################################################################################################

def test(model, device, test_loader_list, ps_range_test, cum_count_lim=100000):
    model.eval()
    test_loss_ber_list, test_loss_ler_list, cum_samples_all = [], [], []
    t = time.time()
    with torch.no_grad():
        for ii, test_loader in enumerate(test_loader_list):
            test_ber = test_ler = cum_count = 0.
            while True:
                (syndrome, labels) = next(iter(test_loader))


                syndrome, labels = syndrome.to(device), labels.to(device)
                outputs = model(syndrome)

                ###

                _, predicted = torch.max(outputs.data, 1)

                correct = (predicted == labels).sum().item()

                #
                
                test_ler += correct
                cum_count += syndrome.shape[0]
                if cum_count > cum_count_lim:
                    break
            cum_samples_all.append(cum_count)
            test_loss_ber_list.append(test_ber / cum_count)
            test_loss_ler_list.append(1 - (test_ler / cum_count))
            print(f'Test p={ps_range_test[ii]:.3e}, LER={test_loss_ler_list[-1]:.3e}')
        ###
        logging.info('Test LER  ' + ' '.join(
            ['p={:.2e}: {:.2e}'.format(ebno, elem) for (elem, ebno)
             in
             (zip(test_loss_ler_list, ps_range_test))]))
        logging.info(f'Mean LER = {np.mean(test_loss_ler_list):.3e}')
    logging.info(f'# of testing samples: {cum_samples_all}\n Test Time {time.time() - t} s\n')
    return test_loss_ber_list, test_loss_ler_list

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.code.logic_matrix = args.code.logic_matrix.to(device) 
    args.code.pc_matrix = args.code.pc_matrix.to(device) 
    code = args.code

    x_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'X_only', device)
    z_error_basis_dict = create_surface_code_pure_error_lut(args.code_L, 'Z_only', device)

    assert 0 < args.repetitions 
    if args.repetitions > 1:
        from Model_T_measurements import ECC_Transformer
    else:
        from Model_FFNN import ECC_Transformer

    ##################################################################
    ##################################################################
    model = ECC_Transformer(args, dropout=0).to(device)
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    logging.info(f'PC matrix shape {code.pc_matrix.shape}')
    logging.info(model)
    logging.info(f'# of Parameters: {np.sum([np.prod(p.shape) for p in model.parameters()])}')
    ##################################################################
    ##################################################################

    ps_train = [0.09]
    ps_test = [0.07, 0.08, 0.09, 0.1, 0.11]


    train_dataloader = DataLoader(QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, ps_train, len=args.batch_size * 1000, args=args), batch_size=int(args.batch_size),
                                  shuffle=True, num_workers=args.workers)
    
    test_dataloader_list = [DataLoader(QECC_Dataset(code, x_error_basis_dict, z_error_basis_dict, [ps_test[ii]], len=int(args.test_batch_size),args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(ps_test))]
    
    #################################
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        loss, ber, ler = train(model, device, train_dataloader, optimizer,
                               epoch, LR=scheduler.get_last_lr()[0])
        scheduler.step()
        torch.save(model, os.path.join(args.path, 'last_model'))
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(args.path, 'best_model'))
            logging.info('Model Saved')
        if epoch % 10 == 0 or epoch in [args.epochs]:
            test(model, device, test_dataloader_list, ps_test)
    ###
    model = torch.load(os.path.join(args.path, 'best_model')).to(device)
    logging.info('Best model loaded')
    ps_test = [0.07, 0.08, 0.09, 0.1, 0.11]
    ###
    test_dataloader_list = [DataLoader(QECC_Dataset(code, [ps_test[ii]], len=int(args.test_batch_size),args=args),
                                       batch_size=int(args.test_batch_size), shuffle=False, num_workers=args.workers) for ii in range(len(ps_test))]
    test(model, device, test_dataloader_list, ps_test)

##################################################################################################################
##################################################################################################################
##################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DQEC')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gpus', type=str, default='0', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)

    # Code args
    parser.add_argument('--code_type', type=str, default='surface',choices=['surface'])
    parser.add_argument('--code_L', type=int, default=4,help='Lattice length')
    parser.add_argument('--repetitions', type=int, default=1,help='Number of faulty repetitions. <=1 is equivalent to none.')
    parser.add_argument('--noise_type', type=str,default='independent', choices=['independent','depolarization'],help='Noise model')

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

####################################################################################################################################
####################################################################################################################################

    if args.no_g > 0:
        args.lambda_loss_n_pred= 0.
    ###
    class Code():
        pass
    code = Code()

    Hx, Hz,Lx, Lz = eval(f'Get_surface_Code')(args.code_L)
    H_x = torch.from_numpy(Hx).long()
    H_z = torch.from_numpy(Hz).long()
    L_x = torch.from_numpy(Lx).long()
    L_z = torch.from_numpy(Lz).long()
    H = torch.block_diag(H_z,H_x)
    L = torch.block_diag(L_z,L_x)

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

    np.savetxt('Hx_test.txt',code.pc_matrix,fmt='%d', delimiter=',')
    np.savetxt('logX_test.txt',code.logic_matrix,fmt='%d', delimiter=',')

####################################################################################################################################
####################################################################################################################################

    model_dir = os.path.join('Final_Results_QECCT', args.code_type, 
                             'FFNN_Code_L_' + str(args.code_L) , 
                             f'noise_model_{args.noise_type}', 
                             f'repetition_{args.repetitions}' , 
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

    main(args)

