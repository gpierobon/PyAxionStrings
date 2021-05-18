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

sys.path.insert(0, '/Users/z5278074/Desktop/FastVersion/Scripts/') 
mov_dir = '/Users/z5278074/Desktop/FastVersion/Output/' 

from Input import *
from Initialize import *
from StringID import plaq_pi, cores_pi, cores_std,histo_std
from Plot import TwilightPlot,Snap_Save,FourierPlot,RdBuPlot,BinaryPlot,AxionSnapshot,SaxionSnapshot,DoubleSnapshot
from Fourier import PS_ScreenPlot,PS,PS_Plot,newPSPlot

if Potential = 'Mexican':
    phi1,phi2,phidot1,phidot2 = IC_Mexican(N)

if Potential = 'Thermal':
    phi1,phi2,phidot1,phidot2 = IC_Thermal(L,1/Delta,meffsquared,T0,N,flag_normalize=False)


K1 = Laplacian(phi1,dx) + Potential1(phi1,phi2,phidot1,t_evol)
K2 = Laplacian(phi2,dx) + Potential2(phi1,phi2,phidot2,t_evol)

final_step = light_time-int(1/DeltaRatio)+1
z_slice = randrange(N-1)

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

    if tstep > int(final_step/2):
        num_cores = cores_pi(axion,N,1)
        if num_cores == 10:
            break

fig1, ax1 = AxionSnapshot(NDIMS,axion,z_slice,t_evol)
Snap_Save(fig1,tstep,mov_dir)


L_new = 4 # physical size in units of L_1=1/(R_1H_1)
ms_new  = 3072

n = 7.0 
tc = 5

DeltaRatio = 1.0/3.0 # Courant parameter
Delta = L_new/N
Delta_tau = DeltaRatio*Delta # time step (comoving)
dx = ms_new*Delta 
dtau = ms_new*Delta_tau

t_evol = dtau/DeltaRatio - dtau # Such that in the first loop iteration, t_evol = 1 
light_time_new = int(0.5*N/DeltaRatio) # light crossing time (do not evolve beyond this timestep)

#lambdaPRS = 1000
final_step_new = light_time_new-int(1/DeltaRatio)+1

K1 = Laplacian(phi1,dx) + Potential1(phi1,phi2,phidot1,t_evol)
K2 = Laplacian(phi2,dx) + Potential2(phi1,phi2,phidot2,t_evol)

for tstep_new in range(final_step_new):

    phi1 = phi1 + dtau*(phidot1 + 0.5*K1*dtau)
    phi2 = phi2 + dtau*(phidot2 + 0.5*K2*dtau)
    
    t_evol += dtau
    
    K1_next = Laplacian(phi1,dx,N) + Potential1(phi1,phi2,phidot1,t_evol) - np.min(t_evol,tc)**n *t_evol**2.0*(phi2**2.0/(phi1**2.0+phi2**2.0)**(3.0/2.0))
    K2_next = Laplacian(phi2,dx,N) + Potential2(phi1,phi2,phidot2,t_evol) + np.min(t_evol,tc)**n *t_evol**2.0*(phi1*phi2/(phi1**2.0+phi2**2.0)**(3.0/2.0))
    
    phidot1 = phidot1 + 0.5*(K1 + K1_next)*dtau
    phidot2 = phidot2 + 0.5*(K2 + K2_next)*dtau
    K1 = 1.0*K1_next
    K2 = 1.0*K2_next


axion = axionize(phi1,phi2)
saxion = saxionize(phi1**2.0+phi2**2.0)

fig2, ax2 = AxionSnapshot(NDIMS,axion,z_slice,t_evol)
Snap_Save(fig2,tstep_new,mov_dir)












