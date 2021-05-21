###############################################################################
####               Utils.py                  ##################################
###############################################################################

from numba import njit,prange
import numpy as np
from scipy import stats
import scipy.fftpack
from numpy import random
from Input import * 



# ===================================================================
# GRID SETUP 
# ===================================================================

Mpl = 2.4*10**18.0/fa_phys # Normalized to fa_phys chosen 
fa = 1 # In fa units
ms = np.sqrt(lambdaPRS)*fa # Saxion mass for PRS in ADM units
L = Delta*N # Comoving 
Delta_tau = DeltaRatio*Delta
H1 = fa/Delta

dx = ms*Delta 
dtau = ms*Delta_tau
t_evol = dtau/DeltaRatio - dtau # Such that in the first loop iteration, t_evol = 1 
light_time = int(0.5*N/DeltaRatio)
gstar = 106.75
T0 = np.sqrt(Mpl*90.0/(gstar*np.pi*np.pi)) # In units of fa
R0 = Delta*ms/(ms*L)
t0 = Delta/(2*L*ms**2.0)
meffsquared = lambdaPRS**2.0*(T0**2.0/3.0-1)
final_step = light_time-int(1/DeltaRatio)+1

# ===================================================================
# TIME VARIABLES
# ===================================================================

def Time_Variable(t_evol):

    if time_var == 'String tension':

        temp = np.log(t_evol/ms) 

    if time_var == 'Conformal time':

        temp = t_evol

    if time_var == 'Cosmic time':
        
        R = t_evol/(ms*L)
        temp = t0*(R/R0*ms)**2.0 

    if time_var == 'Scale factor':

        temp = t_evol/(ms*L)

    return temp

# ===================================================================
# INITIAL CONDITIONS - MEXICAN HAT POTENTIAL ZERO T
# ===================================================================

def IC_Mexican(N,single_precision = False):
    
    if NDIMS == 2:
        
        th = 2*np.pi*random.uniform(size=(N,N))

        if single_precision:

            th = th.astype('float32')
        
        r = random.normal(loc=1,scale=0.5,size=(N,N))

        if single_precision:

            r = r.astype('float32')

        phi1 = r*np.cos(th)
        phi2 = r*np.sin(th)
        phidot1 = np.zeros(shape=(N,N))
        phidot2 = np.zeros(shape=(N,N))
    
    if NDIMS == 3:
        
        th = 2*np.pi*random.uniform(size=(N,N,N))

        if single_precision:

            th = th.astype('float32')
        
        r = random.normal(loc=1,scale=0.5,size=(N,N,N))

        if single_precision:

            r = r.astype('float32')
        
        phi1 = r*np.cos(th)
        phi2 = r*np.sin(th)
        phidot1 = np.zeros(shape=(N,N,N))
        phidot2 = np.zeros(shape=(N,N,N))
    
    
    return phi1,phi2,phidot1,phidot2


# ===================================================================
# INITIAL CONDITIONS - THERMAL POTENTIAL
# ===================================================================

def fftind(N):
    k_ind = np.mgrid[:N, :N] - int( (N + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return(k_ind)


def IC_Thermal(L,k_scale,meffsquared,T0,N,flag_normalize = True):
    
    k_idx = fftind(N)

    k = np.sqrt(k_idx[0]**2 + k_idx[1]**2+1e-10)
    omegak = np.sqrt((k*np.pi/N)**2 + meffsquared)
    bose = 1/(np.exp(omegak/T0)-1)
    amplitude = np.sqrt(L*bose/omegak) # Power spectrum for phi
    amplitude_dot = np.sqrt(L*bose*omegak) # Power spectrum for phidot
    
    if NDIMS == 2:
        
        noise = random.normal(size = (N, N)) + 1j*random.normal(size = (N,N)) 
        
        if single_precision:
            
            noise = noise.astype('float32')
        
        gfield1 = scipy.fft.ifft2(noise*amplitude).real
        gfield2 = scipy.fft.ifft2(noise*amplitude).imag

        if flag_normalize:

            gfield1 = gfield1 - np.mean(gfield1)
            gfield1 = gfield1/np.std(gfield1)

        if flag_normalize:

            gfield2 = gfield2 - np.mean(gfield2)
            gfield2 = gfield2/np.std(gfield2)
          
        gfield1_dot = scipy.fft.ifft2(noise*amplitude_dot).real
        gfield2_dot = scipy.fft.ifft2(noise*amplitude_dot).imag

        if flag_normalize:
            
            gfield1_dot = gfield1_dot - np.mean(gfield1_dot)
            gfield1_dot = gfield1_dot/np.std(gfield1_dot)
        
        if flag_normalize:
            
            gfield2_dot = gfield2_dot - np.mean(gfield2_dot)
            gfield2_dot = gfield2_dot/np.std(gfield2_dot)
    
    if NDIMS == 3:
        
        noise = random.normal(size = (N,N,N)) + 1j*random.normal(size = (N,N,N))
        
        if single_precision:
            
            noise = noise.astype('float32')
            
        gfield1 = scipy.fft.ifftn(noise*amplitude).real
        gfield2 = scipy.fft.ifftn(noise*amplitude).imag

        if flag_normalize:

            gfield1 = gfield1 - np.mean(gfield1)
            gfield1 = gfield1/np.std(gfield1)

        if flag_normalize:

            gfield2 = gfield2 - np.mean(gfield2)
            gfield2 = gfield2/np.std(gfield2)

           
        gfield1_dot = scipy.fft.ifftn(noise*amplitude_dot).real
        gfield2_dot = scipy.fft.ifftn(noise*amplitude_dot).imag

        if flag_normalize:
            
            gfield1_dot = gfield1_dot - np.mean(gfield1_dot)
            gfield1_dot = gfield1_dot/np.std(gfield1_dot)
        
        if flag_normalize:
            
            gfield2_dot = gfield2_dot - np.mean(gfield2_dot)
            gfield2_dot = gfield2_dot/np.std(gfield2_dot)

    return gfield1,gfield2,gfield1_dot,gfield2_dot


# ===================================================================
# PARALLEL FUNCTIONS 
# ===================================================================

@njit(fastmath=True,parallel=True)
def saxionize(phi1,phi2):
    return np.sqrt(phi1**2+phi2**2)
    
@njit(fastmath=True,parallel=True)
def axionize(phi1,phi2):
    return np.arctan2(phi1,phi2)

# @njit(fastmath=True,parallel=True)
# def Potential1(phi1,phi2,phidot1,t_evol):
#     kernel1 = np.zeros_like(phi1)
#     kernel1 =  -2*(Era/t_evol)*phidot1-lambdaPRS*phi1*(phi1**2+phi2**2 -1 + (T0/L)**2/(3.0*t_evol**2))
#     return kernel1

# @njit(fastmath=True,parallel=True)
# def Potential2(phi1,phi2,phidot2,t_evol):
#     kernel2 = np.zeros_like(phi1)
#     kernel2 =  -2*(Era/t_evol)*phidot2-lambdaPRS*phi2*(phi1**2+phi2**2 -1 + (T0/L)**2/(3.0*t_evol**2))
#     return kernel2

@njit(parallel=True,fastmath=True)
def Potential1(phi1,phi2,phidot1,t_evol):

    kernel1 = np.zeros_like(phi1)
    
    if Potential == 'Mexican':

        kernel1 =  -2*(Era/t_evol)*phidot1-lambdaPRS*phi1*(phi1**2+phi2**2 -1)

    if Potential == 'Thermal':

        kernel1 =  -2*(Era/t_evol)*phidot1-lambdaPRS*phi1*(phi1**2+phi2**2 -1 + (T0/L)**2/(3.0*t_evol**2))
    
    return kernel1

@njit(parallel=True,fastmath=True)
def Potential2(phi1,phi2,phidot2,t_evol):
    
    kernel2 = np.zeros_like(phi1)

    if Potential == 'Mexican':

        kernel2 =  -2*(Era/t_evol)*phidot2-lambdaPRS*phi2*(phi1**2+phi2**2 -1)

    if Potential == 'Thermal':

        kernel2 =  -2*(Era/t_evol)*phidot2-lambdaPRS*phi2*(phi1**2+phi2**2 -1 + (T0/L)**2/(3.0*t_evol**2))
    
    return kernel2

@njit(fastmath=True,parallel=True)
def PhiSum(phi,phidot,K,dtau):
    phi += dtau*(phidot+0.5*K*dtau)
    return phi

@njit(fastmath=True,parallel=True)
def PhidotSum(phidot,K,K_next,dtau):
    phidot += 0.5*(K+ K_next)*dtau
    return phidot


@njit(nogil=True,parallel=True,fastmath=True)
def Laplacian(phi,dx):

    ddphi = np.zeros_like(phi)

    if NDIMS == 2:

        if StencilOrder == 2:
            
            for i in prange(phi.shape[0]):
                for j in prange(phi.shape[0]):

                    ddphi[i,j] = ( phi[np.mod(i+1,phi.shape[0]),j] - 2*phi[i,j] + phi[i-1,j] \
                                 + phi[i,np.mod(j+1,phi.shape[0])] - 2*phi[i,j] + phi[i,j-1] )/(dx**2.0)
    
    if NDIMS == 3:

        if StencilOrder == 2:

            for i in prange(phi.shape[0]):
                for j in prange(phi.shape[0]):
                    for k in prange(phi.shape[0]):

                        ddphi[i,j,k] = (( phi[np.mod(i+1,phi.shape[0]),j,k] - 2*phi[i,j,k] + phi[i-1,j,k])\
                                        + (( phi[i,np.mod(j+1,phi.shape[0]),k] - 2*phi[i,j,k] + phi[i,j-1,k] ))\
                                        + (( phi[i,j,np.mod(k+1,phi.shape[0])] - 2*phi[i,j,k] + phi[i,j,k-1] )))/(dx**2.0) 
    return ddphi



@njit(nogil=True,parallel=True,fastmath=True)
def Gradient(phi,dx):

    dphi = np.zeros_like(phi)

    if NDIMS == 2:

        if StencilOrder == 2:

            for i in prange(phi.shape[0]):
                for j in prange(phi.shape[0]):

                    dphi[i,j] = (phi[mod(i+1,phi.shape[0]),j]-phi[i-1,j]+phi[i,mod(j+1,phi.shape[0])]-phi[i,j-1])/(2*dx)

        if StencilOrder == 4:

            for i in range(0,phi.shape[0]):
                for j in range(0,phi.shape[0]):

                    dphi[i,j] = ((-phi[mod(i+2,phi.shape[0]),j]+8*phi[mod(i+1,phi.shape[0]),j]-8*phi[i-1,j] + phi[i-2,j])\
                     + (-phi[i,mod(j+2,phi.shape[0])] + 8*phi[i,mod(j+1,phi.shape[0])] -8*phi[i,j-1] + phi[i,j-2]))/(12*dx) 
    
    if NDIMS == 3:

        if StencilOrder == 2:

            for i in range(phi.shape[0]):
                for j in range(phi.shape[0]):
                    for k in range(phi.shape[0]):

                        dphi[i,j,k] = (phi[mod(i+1,phi.shape[0]),j,k]-phi[i-1,j,k]+phi[i,mod(j+1,phi.shape[0]),k]\
                                      -phi[i,j-1,k] + phi[i,j,mod(k+1,phi.shape[0])] - phi[i,j,k-1])/(2*dx)

    return dphi



@njit(parallel=True)
def Evolve(phi1,phi2,phidot1,phidot2,K1,K2,t_evol,dtau):

    phi1 = PhiSum(phi1,phidot1,K1,dtau)
    phi2 = PhiSum(phi2,phidot2,K2,dtau)

    t_evol = t_evol + dtau

    K1_next = Laplacian(phi1,dx) + Potential1(phi1,phi2,phidot1,t_evol)
    K2_next = Laplacian(phi2,dx) + Potential2(phi1,phi2,phidot2,t_evol)

    phidot1 = PhidotSum(phidot1,K1,K1_next,dtau)
    phidot2 = PhidotSum(phidot2,K2,K2_next,dtau)

    K1 = 1.0*K1_next
    K2 = 1.0*K2_next

    return phi1,phi2,phidot1,phidot2,K1,K2,t_evol


def PS(f,N,L):
    y = fftn(f)
    P = np.abs(y)**2
    kfreq = fftfreq(N)*N 
    
    if NDIMS == 2:

        kfreq2D = np.meshgrid(kfreq, kfreq)
        K = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
        K = K.flatten()
        P = P.flatten()
        kbins = np.arange(0.5, N//2, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Pk, a, b = stats.binned_statistic(K, P, statistic = "mean", bins = kbins)
        Pk *= pi * (kbins[1:]**2 - kbins[:-1]**2)

    if NDIMS == 3:
        
        kfreq3D = np.meshgrid(kfreq, kfreq, kfreq)
        K = np.sqrt(kfreq3D[0]**2 + kfreq3D[1]**2 + kfreq3D[2]**2) 
        K = K.flatten()
        P = P.flatten()
        kbins = np.arange(0.5, N//2, 1.)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Pk, a, b = stats.binned_statistic(K, P, statistic = "mean", bins = kbins)
        Pk *= 4/3 * np.pi * (kbins[1:]**3 - kbins[:-1]**3)
    
    return kvals*2*np.pi/L,Pk/(2*np.pi*L)**2




