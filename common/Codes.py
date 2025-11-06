"""
Implementation of "Deep Quantum Error Correction" (DQEC), AAAI24
@author: Yoni Choukroun, choukroun.yoni@gmail.com
"""
import numpy as np
import torch
import os
##########################################################################################

def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

#############################################
def Get_toric_Code(L):

    path_pc_mat = os.path.join('Codes_DB', f'H_toric_L{str(L)}')
    path_logX_mat = os.path.join('Codes_DB', f'logX_toric_L{str(L)}')

    Hx = np.loadtxt(path_pc_mat+'.txt')
    logX = np.loadtxt(path_logX_mat+'.txt')
    
    #np.savetxt('Hx.txt',Hx,fmt='%d', delimiter=',')
    #np.savetxt('logX.txt',logX,fmt='%d', delimiter=',')
    return Hx, logX

def Get_surface_Code(L):

    path_Hx = os.path.join('Codes_DB', f'Hx_surface_L{str(L)}')
    path_Lx = os.path.join('Codes_DB', f'Lx_surface_L{str(L)}')
    path_Hz = os.path.join('Codes_DB', f'Hz_surface_L{str(L)}')
    path_Lz = os.path.join('Codes_DB', f'Lz_surface_L{str(L)}')

    Hx = np.loadtxt(path_Hx+'.txt')
    Lx = np.loadtxt(path_Lx+'.txt')
    Hz = np.loadtxt(path_Hz+'.txt')
    Lz = np.loadtxt(path_Lz+'.txt')
    
    #np.savetxt('Hx.txt',Hx,fmt='%d', delimiter=',')
    #np.savetxt('logX.txt',logX,fmt='%d', delimiter=',')
    return Hx, Hz, Lx, Lz

def Get_toric_Code(L):

    path_Hx = os.path.join('Codes_DB', f'Hx_toric_L{str(L)}')
    path_Lx = os.path.join('Codes_DB', f'Lx_toric_L{str(L)}')
    path_Hz = os.path.join('Codes_DB', f'Hz_toric_L{str(L)}')
    path_Lz = os.path.join('Codes_DB', f'Lz_toric_L{str(L)}')

    Hx = np.loadtxt(path_Hx+'.txt')
    Lx = np.loadtxt(path_Lx+'.txt')
    Hz = np.loadtxt(path_Hz+'.txt')
    Lz = np.loadtxt(path_Lz+'.txt')
    
    #np.savetxt('Hx.txt',Hx,fmt='%d', delimiter=',')
    #np.savetxt('logX.txt',logX,fmt='%d', delimiter=',')
    return Hx, Hz, Lx, Lz
#############################################
if __name__ == "__main__":
    Get_toric_Code(4)
    #Get_toric_Code(4)
    class Code:
        pass
