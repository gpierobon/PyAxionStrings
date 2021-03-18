from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from numpy import random
from numba import jit,njit,prange
from scipy import fftpack
from scipy.fft import fft, ifft, fft2,fftshift

import sys
sys.path.insert(0, '/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Scripts/')
# Change with: 
#sys.path.insert(0, 'path-to-directory/PQsim/Scripts/')

from Spatial import Laplacian_2D, Laplacian_3D, Gradient_2D
from StringID import cores_pi, cores_std,histo_std
from Plot import TwilightPlot,Snap_Save,FourierPlot,RdBuPlot,BinaryPlot
from Init import init_noise, kernels
from Input import *

mov_dir = '/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Snapshots/2D/'
str_dir = '/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Scaling/Files'
# Change with: 
#dir = 'path-to-directory/PQsim/Output/Snapshots/2D/'

############################################################################
# OPEN OUTPUT FILES - uncomment to use
############################################################################


#scaling = open("/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Data/scaling.txt","w+")
#energy = open("/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Data/energy.txt","w+")
#----------------------------------------------------------------------------

phi1,phi2,phidot1,phidot2 = init_noise(N)
K1,K2 = kernels(phi1,phi2,phidot1,phidot2)

final_step = light_time-int(1/DeltaRatio)+1

for tstep in range(0,final_step):
    #----------------------------------------------------------------------------
    # KDK EVOLUTION
    #----------------------------------------------------------------------------
    
    phi1 = phi1 + dtau*(phidot1 + 0.5*K1*dtau)
    phi2 = phi2 + dtau*(phidot2 + 0.5*K2*dtau)
    t_evol = t_evol + dtau
    K1_next, K2_next = kernels(phi1,phi2,phidot1,phidot2)
    K1_next = Laplacian_2D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1)
    K2_next = Laplacian_2D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1)
    phidot1 = phidot1 + 0.5*(K1 + K1_next)*dtau
    phidot2 = phidot2 + 0.5*(K2 + K2_next)*dtau
    K1 = 1.0*K1_next
    K2 = 1.0*K2_next
    
    #----------------------------------------------------------------------------
    # TEMPORAL QUANTITIES
    #----------------------------------------------------------------------------
    
    kappa = log(t_evol/ms) # String tension (also time variable)
    R = t_evol/(ms*L) # Scale factor in L units
    time = t0*(R/R0*ms)**2.0 # Cosmic time in L units 
    
    #----------------------------------------------------------------------------
    # FIELDS 
    #----------------------------------------------------------------------------
    
    PHI = phi1 + 1j * phi2
    PHIDOT = phidot1 +1j * phidot2
    saxion = sqrt(phi1**2.0+phi2**2.0) # Or abs(PHI)
    axion = arctan2(phi1,phi2)
    
    #----------------------------------------------------------------------------
    # STRINGS AND SCALING - uncomment to use
    #----------------------------------------------------------------------------
    
#     Nchecks = 40
#     to_analyse = arange(0, final_step,int(final_step/40))
#     thr = 1 # In % value
#     if i in to_find:
#         num_cores = cores_pi(axion,N,thr) # Call cores_pi2 to use pi/2 method
#         thetasum = cores_std(axion,N,thr) # Array of values of DeltaTheta over the loop (useful for the histogram)
#         histo, ax_histo = histo_std(thetasum, log(t_evol),num_cores) # Histogram plot 
#         xi = num_cores*time**2.0/(t_evol**2.0) # Scaling parameter
#         scaling.write('%f %f \n' % (kappa,xi))
        
    #----------------------------------------------------------------------------
    # ENERGIES - uncomment to use
    #----------------------------------------------------------------------------
    
    saxiondot = PHIDOT 
    axiondot = (PHIDOT/PHI).imag
    saxiongrad = Gradient_2D(saxion,dx,N)
    axiongrad = Gradient_2D(axion,dx,N)
    saxkin = 0.5*saxiondot**2.0
    axkin = 0.5*axiondot**2.0
    pot = lambdaPRS*(saxion**2.0-fa)**2.0 # Saxion potential energy
    
############################################################################
# CLOSE OUTPUT FILES - uncomment to use
############################################################################

#scaling.close()
#energy.close()