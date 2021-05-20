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

sys.path.insert(0, './Scripts/') 

from Input import *
from Utils2 import *
from Snapshots2 import * 
from StringID2 import Cores2D,Cores3D, PlaqSave,CoresAppend

if analyse_strings:
    string_print = open('./Output/Strings/String_scaling.txt', 'w+')

if analyse_spectrum:
    spectrum_print = open('./Output/PS/Power_spectrum.txt', 'w+')

if Potential == 'Mexican':
    phi1,phi2,phidot1,phidot2 = IC_Mexican(N)

if Potential == 'Thermal':
    phi1,phi2,phidot1,phidot2 = IC_Thermal(L,1/Delta,meffsquared,T0,N,flag_normalize=False)


K1 = Laplacian(phi1,dx) + Potential1(phi1,phi2,phidot1,t_evol)
K2 = Laplacian(phi2,dx) + Potential2(phi1,phi2,phidot2,t_evol)

for tstep in range(0,final_step):
    
    phi1 = PhiSum(phi1,phidot1,K1,dtau)
    phi2 = PhiSum(phi2,phidot2,K2,dtau)

    t_evol = t_evol + dtau

    K1_next = Laplacian(phi1,dx) + Potential1(phi1,phi2,phidot1,t_evol)
    K2_next = Laplacian(phi2,dx) + Potential2(phi1,phi2,phidot2,t_evol)

    phidot1 = PhidotSum(phidot1,K1,K1_next,dtau)
    phidot2 = PhidotSum(phidot2,K2,K2_next,dtau)

    K1 = 1.0*K1_next
    K2 = 1.0*K2_next

    #------------------------------------------------------------------------------
    # ANALYSIS 
    #------------------------------------------------------------------------------
    
    if analyse_strings: 
        
        to_analsye = np.arange(0,final_step,string_checks)

        if tstep in to_analsye:
            
            time_x = Time_Variable(t_evol)
            axion = axionize(phi1,phi2)

            if NDIMS == 2:

                R = t_evol/(ms*L)
                time = t0*(R/R0*ms)**2.0     
                num_cores = Cores2D(axion,thr)
                xi = num_cores*time**2/(t_evol**2)
                if array_job:
                    print(time_x,xi)
                else:
                    string_print.write('%f %f \n' % (time_x,xi))
            
            if NDIMS ==3:

                R = t_evol/(ms*L)
                time = t0*(R/R0*ms)**2.0
                num_cores = Cores3D(axion,thr)
                xi = num_cores*time**2/(t_evol**3)
                if array_job:
                    print(time_x,xi)
                else:
                    string_print.write('%f %f \n' % (time_x,xi))

    if analyse_spectrum:

        to_analsye = np.arange(0,final_step,number_checks)

        if tstep in to_analsye:

            PHI = phi1 + 1j * phi2
            PHIDOT = phidot1 +1j * phidot2
            axiondot = (PHIDOT/PHI).imag
            adot_screen = saxion*axiondot
            kvals, Abins = PS(adot_screen,N,L)

    if break_loop:

        if NDIMS ==2:
            to_check = np.arange(int(final_step*0.8),final_step,3)
            if tstep in to_check:
                num_cores = Cores2D(axion,thr)
                if num_cores == cores_final:
                    break




if analyse_strings:
    string_print.close()

if analyse_spectrum:


#------------------------------------------------------------------------------
# FINAL RESULT 
#------------------------------------------------------------------------------

if print_snapshot_final:
    axion = axionize(phi1,phi2)
    saxion = saxionize(phi1,phi2)
    Print_Snapshots2D(f_axion,f_saxion,f_double)
    if f_3dstrings:
        if NDIMS == 2:
            pass
        if NDIMS == 3:
            locs = PlaqSave(axion,1)
            Print_3Dstrings(locs,axion,t_evol,tstep)
