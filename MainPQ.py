import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from numpy import random
from numba import jit,njit,prange
from scipy import fftpack
from scipy.fft import fft, ifft, fft2,fftshift
from random import randrange
import sys

sys.path.insert(0, './Scripts')

mov_dir_ax = './Snapshots/Axion/' 
mov_dir_sax = './Snapshots/Saxion/' 
mov_dir_double = './Snapshots/Double/' 
mov_dir_3ds = './Snapshots/Strings/' 
mov_dir_histo = './Snapshots/Histo/' 
mov_dir_ps1 = './Snapshots/PS/' 
mov_dir_ps2 = './Snapshots/PS_Double' 

from Input import *
from Utils import *
from Snapshots import * 
from StringID import Cores2D,Cores3D, PlaqSave,CoresAppend

if Potential == 'Mexican':
    phi1,phi2,phidot1,phidot2 = IC_Mexican(N)

if Potential == 'Thermal':
    phi1,phi2,phidot1,phidot2 = IC_Thermal(L,N,T0,meffsquared,flag_normalize = True)

K1 = Laplacian(phi1,dx) + Potential1(phi1,phi2,phidot1,t_evol)
K2 = Laplacian(phi2,dx) + Potential2(phi1,phi2,phidot2,t_evol)

final_step = light_time-int(1/DeltaRatio)+1


for tstep in range(final_step):

    phi1 = PhiSum(phi1,phidot1,K1,dtau)
    phi2 = PhiSum(phi2,phidot2,K2,dtau)

    t_evol = t_evol + dtau

    K1_next = Laplacian(phi1,dx) + Potential1(phi1,phi2,phidot1,t_evol)
    K2_next = Laplacian(phi2,dx) + Potential2(phi1,phi2,phidot2,t_evol)

    phidot1 = PhidotSum(phidot1,K1,K1_next,dtau)
    phidot2 = PhidotSum(phidot2,K2,K2_next,dtau)

    K1 = 1.0*K1_next
    K2 = 1.0*K2_next
    
    kappa = np.log(t_evol/ms) 
    R = t_evol/(ms*L) 
    time = t0*(R/R0*ms)**2.0 

#    PHI = phi1 + 1j * phi2
#    PHIDOT = phidot1 +1j * phidot2
    axion = axionize(phi1,phi2)
    saxion = saxionize(phi1,phi2)


Print_Snapshots(f_axion = False,f_saxion = False, f_double = True, f_3dstrings = False)  



