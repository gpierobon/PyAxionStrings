from numpy import *
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from numpy import random
from numba import jit,njit,prange
from scipy import fftpack
from scipy.fft import fft, ifft, fft2,fftshift
from random import randrange
import sys

sys.path.insert(0, '/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Scripts/') 
#sys.path.insert(0, '/home/z5278074/PQaxions/Scripts/') # ON KATANA

from Spatial import Laplacian_2D, Laplacian_3D, Gradient_2D
from StringID import cores_pi, cores_std,histo_std
from Plot import TwilightPlot,Snap_Save,FourierPlot,RdBuPlot,BinaryPlot
from Plot3D import print_plaqs,plaq_coord, String_Plot
from Init import init_noise, kernels
from Input import *

mov_dir2D = '/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Snapshots/2D/'
mov_dir3D = '/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Snapshots/3D/'
#mov_dir2D = '/home/z5278074/PQaxions/Output/Snapshots/2D/' # ON KATANA
#mov_dir3D = '/home/z5278074/PQaxions/Output/Snapshots/3D/' # ON KATANA

############################################################################
# OPEN OUTPUT FILES - uncomment to use
############################################################################


#scaling2D = open("/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Data/scaling2D.txt","w+")
#scaling3D = open("/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Data/scaling3D.txt","w+")
#energy = open("/Users/z5278074/OneDrive - UNSW/AxionSimulations/PQsim/Output/Data/energy.txt","w+")
#----------------------------------------------------------------------------

phi1,phi2,phidot1,phidot2 = init_noise(N)
K1,K2 = kernels(phi1,phi2,phidot1,phidot2)

final_step = light_time-int(1/DeltaRatio)+1

# FOR 3D PLOTTING 
z_slice = randrange(N-1)

for tstep in range(0,final_step):
    
    #----------------------------------------------------------------------------
    # KDK EVOLUTION
    #----------------------------------------------------------------------------
    
    phi1 = phi1 + dtau*(phidot1 + 0.5*K1*dtau)
    phi2 = phi2 + dtau*(phidot2 + 0.5*K2*dtau)
    t_evol = t_evol + dtau
    K1_next, K2_next = kernels(phi1,phi2,phidot1,phidot2)
    if NDIMS == 2:
        K1_next = Laplacian_2D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1)
        K2_next = Laplacian_2D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1)
    if NDIMS == 3:
        K1_next = Laplacian_3D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1)
        K2_next = Laplacian_3D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1)
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
#         xi_2D = num_cores*time**2.0/(t_evol**2.0) # Scaling parameter
#         scaling2D.write('%f %f \n' % (kappa,xi_2D))
#         print(kappa,xi_3D) # if output is for array job 
#         
#         num_plaqs = plaq_pi(axion,N,thr)
#         xi_3D = num_plaqs*time**2.0/(t_evol**3.0)
#         scaling3D.write('%f %f \n' % (kappa,xi_3D))
#         print(kappa,xi_3D) # if output is for array job 
        
    #----------------------------------------------------------------------------
    # ENERGIES - uncomment to use
    #----------------------------------------------------------------------------
    
#    saxiondot = PHIDOT 
#    axiondot = (PHIDOT/PHI).imag
#    saxiongrad = Gradient_2D(saxion,dx,N)
#    axiongrad = Gradient_2D(axion,dx,N)
#    saxkin = 0.5*saxiondot**2.0
#    axkin = 0.5*axiondot**2.0
#    pot = lambdaPRS*(saxion**2.0-fa)**2.0 # Saxion potential energy

    #----------------------------------------------------------------------------
    # SNAPSHOTS - uncomment to use
    #----------------------------------------------------------------------------

    # 2D
#    fig_axion,ax_axion = TwilightPlot(axion,t_evol)
#    fig_saxion,ax_saxion = RdBuPlot(saxion,t_evol)
#    Snap_Save(fig_axion,tstep,mov_dir2D)
#    Snap_Save(fig_saxion,tstep,mov_dir2D)
    # 3D
#    fig_axion_projected,ax_axion_projected = TwilightPlot(axion_field[0:,0:,z_slice],t_evol)
#    Snap_Save(fig_axion_projected,tstep,mov_dir3D)
#   # 3D full plot
#    Nchecks = 40
#    to_plot = arange(0,final_step,int(final_step/Nchecks))
#    if tstep in to_plot:
#        num=print_plaqs(axion,N,1,tstep)
#        X,Y,Z = plaq_coord(N,tstep)
#        fig,ax = String_Plot(t_evol,X,Y,Z)
#        Snap_Save(fig,tstep,mov_dir3D)
    
############################################################################
# CLOSE OUTPUT FILES - uncomment to use
############################################################################

#scaling2D.close()
#scaling3D.close()
#energy.close()
