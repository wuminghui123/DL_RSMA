import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第二块GPU（从0开始）
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import random
import torch.nn.functional as F
import torchvision
import numpy as np
from math import *
import matplotlib.pyplot as plt
from torch.autograd import Variable
from IPython import display
import torch.utils.data as Data
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.linalg import block_diag
import datetime
from torch.nn.utils import *
from Transformer_model import *

def Hermitian(X):#torch矩阵共轭转置
    X = torch.real(X) - 1j*torch.imag(X)
    return X.transpose(-1,-2)
def kron(a,b):
    #a与b维度为[batch,N],输出应为[batch,N*N]
    batch = a.shape[0]
    a = a.reshape(batch,-1,1)
    b = b.reshape(batch,1,-1)
    c = a @ b
    return c.reshape(batch,-1) #输出的维度a在前，b在后
def kron_add(a,b):
    #a与b维度为[batch,N],输出应为[batch,N*N]
    batch = a.shape[0]
    a = a.reshape(batch,-1,1)
    b = b.reshape(batch,1,-1)
    c = a + b
    return c.reshape(batch,-1) #输出的维度a在前，b在后

### BS-RIS信道
def Channel_BS_RIS(param_list):
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]
    fc = param_list[0]
    B  = param_list[1]
    Nc = param_list[2]
    M  = param_list[3]
    N  = param_list[4]
    D_sub = param_list[5]
    D_ant = param_list[6]
    R = param_list[7]
    M_ant = param_list[8]
    h1 = param_list[9]
    h2 = param_list[10]
    c = 3e8;
    lambda_c = c/fc;
    Ay = (M-1)*D_sub + (N-1)*(D_ant);#阵列长度
    Az = Ay;

    r_BS  = (torch.tensor([0,0,h1]) - torch.tensor([0,Ay/2,Az/2])).cuda()
    r_RIS = (torch.tensor([R,0,h1]) - torch.tensor([0,Ay/2,Az/2])).cuda()

    r_BS_0  = r_BS+0
    r_RIS_0 = r_RIS+0
    D_H = torch.zeros(M_ant,M_ant).cuda()
    for m_h1 in range(M):
        for m_v1 in range(M):
            for m_h2 in range(M):
                for m_v2 in range(M):
                    for n_h1 in range(N):
                        for n_v1 in range(N):
                            for n_h2 in range(N):
                                for n_v2 in range(N):
                                    r_BS_0[1] = r_BS[1] + m_h1*D_sub + n_h1*D_ant;
                                    r_BS_0[2] = r_BS[2] + m_v1*D_sub + n_v1*D_ant;

                                    r_RIS_0[1] = r_RIS[1] + m_h2*D_sub + n_h2*D_ant;
                                    r_RIS_0[2] = r_RIS[2] + m_v2*D_sub + n_v2*D_ant;
                                    d = torch.norm(r_BS_0-r_RIS_0);

                                    index1 = m_h1*M*N*N + m_v1*N*N + n_h1*N + n_v1;
                                    index2 = m_h2*M*N*N + m_v2*N*N + n_h2*N + n_v2;
                                    D_H[index1,index2] = d;
    H_BR = torch.zeros(Nc,M_ant,M_ant)
    a = (torch.min(D_H))
    D_H_dif = D_H-a

    n = (torch.arange(0, Nc).cuda()+1)
    nn = -(Nc+1)/2
    deta = B/Nc
    fm=(nn+n)*deta+fc
    lambda_m = c/fm
    PL_BS2UE_LoS =  32.44 + (20*torch.log10(fm/1e6)).reshape(Nc,1,1) + (20*torch.log10(D_H / 1000)).reshape(1,M_ant,M_ant)
    Phase = torch.exp(1j*2*pi/lambda_m.reshape(Nc,1,1)*D_H_dif.reshape(1,M_ant,M_ant))
    H_BR = 10**(-PL_BS2UE_LoS/20) * Phase
    return H_BR


### RIS-UE信道
def Channel_RIS_UE(param_list,batch,alpha): #更快的速度
    #         param_list = [fc,B,Nc,M,N,D_sub,D_ant,R,M_ant,h1,h2,L]

    fc = param_list[0]
    B  = param_list[1]
    Nc = param_list[2]
    M  = param_list[3]
    N  = param_list[4]
    D_sub = param_list[5]
    D_ant = param_list[6]
    R = param_list[7]
    M_ant = param_list[8]
    h1 = param_list[9]
    h2 = param_list[10]

    Lp = param_list[15]
    c = 3e8;
    lambda_c = c/fc;
    Ay = (M-1)*D_sub + (N-1)*(D_ant);#阵列长度
    Az = Ay;

    r_BS  = (torch.tensor([0,0,h1]) - torch.tensor([0,Ay/2,Az/2])).cuda()
    r_RIS = (torch.tensor([R,0,h1]) - torch.tensor([0,Ay/2,Az/2])).cuda()
    r_RIS = r_RIS.double()

    r_RIS_0 = r_RIS + 0

    for l in range(Lp):
        D_RU = torch.zeros(batch,M_ant).cuda().double()
        r_UE = torch.zeros(batch,3).cuda().double()
        h2 = 4*torch.rand(batch).cuda().double()
        r_UE[:,0] = R + 0
        r_UE[:,2] = h2 + 0
        Theta_UE = pi*torch.rand(batch).cuda().double() - pi/2
        R_UE = 10*(torch.rand(batch)/2+0.5).cuda().double()
        r_UE[:,0] = r_UE[:,0] + R_UE*torch.cos(Theta_UE)
        r_UE[:,1] = r_UE[:,1] + R_UE*torch.sin(Theta_UE)
        
        for m_h1 in range(M):
            for m_v1 in range(M):
                r_RIS_0[1] = r_RIS[1] + m_h1*D_sub
                r_RIS_0[2] = r_RIS[2] + m_v1*D_sub
                D = torch.norm(r_RIS_0-r_UE,dim=1) #[batch]
                sin_h = r_UE[:,1]/D
                sin_v = (r_UE[:,2]-h1)/D
                D = D.reshape(batch,1)

                a_h = torch.range(0,N-1).reshape(1,N).cuda().double()*D_ant*sin_h.reshape(-1,1)
                a_v = torch.range(0,N-1).reshape(1,N).cuda().double()*D_ant*sin_v.reshape(-1,1)

                D_RU[:,(m_h1*M*N*N + m_v1*N*N):(m_h1*M*N*N + m_v1*N*N + N*N)] = D + kron_add(a_h,a_v)
        if l == 0:
            a,b = (torch.min(D_RU,1))
        D_RU_dif = D_RU-a.reshape(-1,1)
        # D_RU_dif = D_RU

        n = (torch.arange(0, Nc).cuda().double()+1)
        nn = -(Nc+1)/2
        deta_norm = B/Nc/c
        fm_norm=(nn+n)*deta_norm+fc/c
        fm = fm_norm * c
        PL_BS2UE_LoS =  32.44 + (20*torch.log10(fm/1e6)).reshape(1,Nc,1) + (20*torch.log10(D_RU / 1000)).reshape(-1,1,M_ant)
        Phase = torch.exp(1j*2*pi*fm_norm.reshape(1,Nc,1)*D_RU_dif.reshape(-1,1,M_ant))
        if l == 0:
            H_RU = 10**(-PL_BS2UE_LoS/20) * Phase * alpha
        else:
            H_RU = H_RU + 10**(-PL_BS2UE_LoS/20) * Phase * alpha
    return H_RU.type(torch.complex64)

def NMSE(H_RU_hat,H_RU):    
    NMSE = torch.sum(torch.abs((H_RU_hat-H_RU))**2)/torch.sum(torch.abs(H_RU)**2)
    return NMSE  

