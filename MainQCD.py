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

sys.path.insert(0, './Scripts/') 

from Input import *
from Utils2 import *
from Snapshots2 import * 
from StringID2 import Cores2D,Cores3D, PlaqSave,CoresAppend

if analyse_spectrum:
    spectrum_print = open('./Output/PS/Power_spectrum_finalPQ.txt', 'w+')

if Potential == 'Mexican':
    phi1,phi2,phidot1,phidot2 = IC_Mexican(N)

if Potential == 'Thermal':
    phi1,phi2,phidot1,phidot2 = IC_Thermal(L,1/Delta,meffsquared,T0,N,flag_normalize=False)


K1 = Laplacian(phi1,dx) + Potential1(phi1,phi2,phidot1,t_evol)
K2 = Laplacian(phi2,dx) + Potential2(phi1,phi2,phidot2,t_evol)

#--------------------------------------------------------------------------------------------------------------------------------------------------

for tstep in range(0,final_step):

    phi1,phi2,phidot1,phidot2,K1,K2,t_evol = Evolve(phi1,phi2,phidot1,phidot2,K1,K2,t_evol,dtau)

    if init_QCD:

        if NDIMS == 2:

            to_check = np.arange(int(final_step*0.8),final_step,3)
            if tstep in to_check:
                num_cores = Cores2D(axion,thr)
                if num_cores == cores_final:
                    break

# Continue here ... 






