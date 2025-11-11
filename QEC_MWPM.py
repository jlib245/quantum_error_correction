import pymatching
import numpy as np
import argparse
import time
from tqdm import tqdm

# --- 1. Python Import Path & CWD 설정 ---
import sys
import os

# 스크립트가 실행되는 현재 디렉터리를 Python 경로에 추가 (Import용)
_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(_ROOT_DIR)

# 'common' 디렉터리의 절대 경로
_COMMON_DIR = os.path.join(_ROOT_DIR, 'common')
# ------------------------------------------------

try:
    # 이제 이 import 문이 성공해야 합니다.
    from common.Codes import Get_surface_Code
except ImportError as e:
    print(f"오류: 'from common.Codes import Get_surface_Code' 실패.")
    print(f"세부 오류: {e}")
    print("스크립트가 프로젝트 루트 디렉터리에 있는지,")
    print("프로젝트 루트에 'common' 폴더가 있고 그 안에 'Codes.py' 파일이 있는지,")
    print("common/Codes.py 파일 안에 'Get_surface_Code' 함수가 정의되어 있는지 확인하세요.")
    exit()

def generate_correlated_noise(n_qubits, p_total, y_ratio=0.3):
    """
    계획서 목표 노이즈 모델: 상관 오류 (Y 오류 30%)
    """
    p_Y = p_total * y_ratio
    p_X = p_total * (1 - y_ratio) / 2
    p_Z = p_total * (1 - y_ratio) / 2
    
    rand_samples = np.random.rand(n_qubits)
    
    error_vector_X = np.zeros(n_qubits, dtype=int)
    error_vector_Z = np.zeros(n_qubits, dtype=int)
    
    error_vector_X[rand_samples < p_X] = 1
    error_vector_X[(rand_samples >= p_X) & (rand_samples < p_X + p_Y)] = 1
    error_vector_Z[(rand_samples >= p_X) & (rand_samples < p_X + p_Y)] = 1
    error_vector_Z[(rand_samples >= p_X + p_Y) & (rand_samples < p_X + p_Y + p_Z)] = 1
    
    return error_vector_X, error_vector_Z

def generate_depolarizing_noise(n_qubits, p_error):
    """
    간단한 비상관 (Depolarizing) 노이즈 모델
    """
    p_channel = p_error / 3.0
    
    error_vector_X = (np.random.rand(n_qubits) < (p_channel * 2)).astype(int)
    error_vector_Z = (np.random.rand(n_qubits) < (p_channel * 2)).astype(int)
    
    return error_vector_X, error_vector_Z

def main(args):
    
    # --- 2. 코드 로드 [수정된 부분] ---
    # 원본 Codes.py가 'Codes_DB'를 현재 작업 디렉터리(CWD) 기준으로 찾으므로,
    # CWD를 'common' 디렉터리로 임시 변경합니다.
    
    original_cwd = os.getcwd() # 원래 CWD 저장
    
    try:
        os.chdir(_COMMON_DIR) # CWD를 'common'으로 변경
        
        # 이제 Get_surface_Code는 'common/Codes_DB'에서 파일을 찾습니다.
        Hx, Hz, Lx, Lz = Get_surface_Code(args.L)
        
    except Exception as e:
        print(f"오류: L{args.L} 코드 파일 로드 실패.")
        print(f"세부 오류: {e}")
        print(f"common/Codes_DB/Hx_surface_L{args.L}.txt 파일 등이 있는지 확인하세요.")
        os.chdir(original_cwd) # CWD 원상복구
        exit()
    finally:
        # try/except가 끝나면 무조건 CWD를 원래대로 복구합니다.
        os.chdir(original_cwd) 

    n_qubits = Hx.shape[1]
    print(f"L{args.L} Surface Code 로드 완료. (n_qubits: {n_qubits})")
    # -----------------------------------
    
    # --- 3. PyMatching 디코더 객체 생성 ---
    m_z = pymatching.Matching(Hx) # Z-error decoder
    m_x = pymatching.Matching(Hz) # X-error decoder
    
    print(f"\n--- L={args.L}, p_error={args.p_error}, y_ratio={args.y_ratio} MWPM 시뮬레이션 ---")
    print(f"테스트 횟수 (n_test_shots): {args.n_test_shots}")
    
    logical_error_count = 0
    total_decode_time = 0
    
    # --- 4. 시뮬레이션 루프 ---
    for _ in tqdm(range(args.n_test_shots)):
        
        # (1) 노이즈 생성
        if args.y_ratio > 0:
            error_X, error_Z = generate_correlated_noise(n_qubits, args.p_error, args.y_ratio)
        else:
            error_X, error_Z = generate_depolarizing_noise(n_qubits, args.p_error)
        
        # (2) 신드롬 계산
        syndrome_Z = Hx.dot(error_Z) % 2
        syndrome_X = Hz.dot(error_X) % 2
        
        start_time = time.perf_counter()
        
        # (3) MWPM 디코딩 수행
        correction_Z = m_z.decode(syndrome_Z)
        correction_X = m_x.decode(syndrome_X)
        
        end_time = time.perf_counter()
        total_decode_time += (end_time - start_time)
        
        # (4) 잔여 오류(Residual Error) 계산
        residual_Z = (error_Z + correction_Z) % 2
        residual_X = (error_X + correction_X) % 2
        
        # (5) 논리 오류(Logical Error) 확인
        if (Lx.dot(residual_Z) % 2 != 0) or (Lz.dot(residual_X) % 2 != 0):
            logical_error_count += 1
            
    # --- 5. 결과 집계 ---
    ler = logical_error_count / args.n_test_shots
    avg_latency = (total_decode_time / args.n_test_shots) * 1000 # ms 단위

    print("\n--- MWPM 디코더 시뮬레이션 결과 ---")
    print(f"총 논리 오류: {logical_error_count} / {args.n_test_shots}")
    print(f"논리 오류율 (LER): {ler:.8f} (p_error={args.p_error})")
    print(f"평균 디코딩 시간: {avg_latency:.6f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MWPM Decoder Baseline Simulation")
    parser.add_argument('-L', type=int, default=3, help='Code distance (L)')
    parser.add_argument('-p', '--p_error', type=float, default=0.01, help='Physical error rate (p_total)')
    parser.add_argument('-n', '--n_test_shots', type=int, default=10000, help='Number of simulation shots')
    parser.add_argument('-y', '--y_ratio', type=float, default=0.3, help="Ratio of Y errors in correlated noise (default: 0.3 for 30%)")

    args = parser.parse_args()
    main(args)