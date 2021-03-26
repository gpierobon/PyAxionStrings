#===============================Init.py==============================#

# Description: Module that sets the initial conditions for the string 
#              simulation

#====================================================================#


from numpy import *
from Input import *
from Spatial import Laplacian_2D, Laplacian_3D
from scipy import fftpack
from scipy.fft import fft, fft2,fftn,ifft,ifft2,ifftn,fftshift

##############################################################################
# RANDOM WHITE NOISE IN POSITION SPACE
#
# Independent of the shape of the potential
##############################################################################


def init_noise(N):
    
    if NDIMS == 2:
        
        th = 2*pi*random.uniform(size=(N,N))
        r = random.normal(loc=1,scale=0.5,size=(N,N))
        phi1 = r*cos(th)
        phi2 = r*sin(th)
        phidot1 = zeros(shape=(N,N))
        phidot2 = zeros(shape=(N,N))
    
    elif NDIMS == 3:
        
        th = 2*pi*random.uniform(size=(N,N,N))
        r = random.normal(loc=1,scale=0.5,size=(N,N,N))
        phi1 = r*cos(th)
        phi2 = r*sin(th)
        phidot1 = zeros(shape=(N,N,N))
        phidot2 = zeros(shape=(N,N,N))
    
    else:
        
        print("Number of spatial dimensions is incorrect! Check value of NDIMS on Input.py")
    
    return phi1,phi2,phidot1,phidot2

########################################################################################
# THERMAL POWER SPECTRUM INITIALIZED WITGH A GAUSSIAN RANDOM FIELD IN FOURIER SPACE
# 
# Only pairde with evolving thermal potential
# Set Potential = 'Thermal' on Input.py
########################################################################################

def fftind(N):
    k_ind = mgrid[:N, :N] - int( (N + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return(k_ind)


def gaussian_thermal(L,k_scale,meffsquared,T0,N,flag_normalize = True):
    
    k_idx = fftind(N)

    k = sqrt(k_idx[0]**2 + k_idx[1]**2+1e-10)
    omegak = sqrt((k*pi/N)**2 + meffsquared)
    bose = 1/(exp(omegak/T0)-1)
    amplitude = sqrt(L*bose/omegak) # Power spectrum for phi
    amplitude_dot = sqrt(L*bose*omegak) # Power spectrum for phidot
    
    if NDIMS == 2:
        
        noise = random.normal(size = (N, N)) + 1j*random.normal(size = (N,N))    
        gfield1 = fft.ifft2(noise*amplitude).real
        gfield2 = fft.ifft2(noise*amplitude).imag

        if flag_normalize:

            gfield1 = gfield1 - mean(gfield1)
            gfield1 = gfield1/std(gfield1)

        if flag_normalize:

            gfield2 = gfield2 - mean(gfield2)
            gfield2 = gfield2/std(gfield2)
          
        gfield1_dot = fft.ifft2(noise*amplitude_dot).real
        gfield2_dot = fft.ifft2(noise*amplitude_dot).imag

        if flag_normalize:
            
            gfield1_dot = gfield1_dot - mean(gfield1_dot)
            gfield1_dot = gfield1_dot/std(gfield1_dot)
        
        if flag_normalize:
            
            gfield2_dot = gfield2_dot - mean(gfield2_dot)
            gfield2_dot = gfield2_dot/std(gfield2_dot)
    
    if NDIMS == 3:
        
        noise = random.normal(size = (N,N,N) + 1j*random.normal(size = (N,N,N)))
        gfield1 = fft.ifftn(noise*amplitude).real
        gfiled2 = fft.ifftn(noise*amplitude).imag

        if flag_normalize:

            gfield1 = gfield1 - mean(gfield1)
            gfield1 = gfield1/std(gfield1)

        if flag_normalize:

            gfield2 = gfield2 - mean(gfield2)
            gfield2 = gfield2/std(gfield2)

           
        gfield1_dot = fft.ifft2(noise*amplitude_dot).real
        gfield2_dot = fft.ifft2(noise*amplitude_dot).imag

        if flag_normalize:
            
            gfield1_dot = gfield1_dot - mean(gfield1_dot)
            gfield1_dot = gfield1_dot/std(gfield1_dot)
        
        if flag_normalize:
            
            gfield2_dot = gfield2_dot - mean(gfield2_dot)
            gfield2_dot = gfield2_dot/std(gfield2_dot)

    return gfield1,gfield2,gfield1_dot,gfield2_dot


def kernels(phi1,phi2,phidot1,phidot2):
    
    if Potential == 'MexicanHat':
        
        if NDIMS == 2:
            
            K1 = Laplacian_2D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1)
            K2 = Laplacian_2D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1)
        
        if NDIMS == 3:
            
            K1 = Laplacian_3D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1)
            K2 = Laplacian_3D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1)    
    
    if Potential == 'Thermal':
        
        if NDIMS == 2:
           
            K1 = Laplacian_2D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
            K2 = Laplacian_2D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
        
        if NDIMS == 3:
            
            K1 = Laplacian_3D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
            K2 = Laplacian_3D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
    
    return K1,K2 


################################################################
# OLD STUFF
################################################################

def init_thermal(N):
    
    wnoise = 2*pi*random.uniform(size=(N,N)) 
    
    dk = 2*pi/L
    kx = linspace(-pi+dk,pi,N-1)
    ky = kx
    meffsquared=lambdaPRS/L**2.0*(T0**2.0/3.0-1) 
    KX,KY = meshgrid(kx,ky)
    K = sqrt(KX**2+KY**2)
    
    phi_spectrum = thermal_phi(L,K,meffsquared,T0)
    phidot_spectrum = thermal_phidot(L,K,meffsquared,T0)
    
    init_phi = ifft(phi_spectrum*whitenoise)
    init_phidot = ifft(phidot_spectrum*whitenoise)
    
    phi1 = init_phi.real
    phi2 = init_phi.imag
    phidot1 = init_phidot.real
    phidot2 = init_phidot.imag
    
    return phi1,phi2,phidot1,phidot2

def init_thermal2(N):
    
    u1 = random.uniform(size=(N,N)) 
    u2 = random.uniform(size=(N,N))

    dk = 2*pi/L
    kx = linspace(-pi+dk,pi,N)
    ky = kx
    meffsquared=lambdaPRS/L**2.0*(T0**2.0/3.0-1) 
    KX,KY = meshgrid(kx,ky)
    K = sqrt(KX**2+KY**2)
   
    phi_spectrum = thermal_phi(L,K,meffsquared,T0)
    phidot_spectrum = thermal_phidot(L,K,meffsquared,T0)

    phi1 = ifft(phi_spectrum*sqrt(-log(u1))*cos(2*pi*u2))
    phi2 = ifft(phi_spectrum*sqrt(-log(u1))*sin(2*pi*u2))
    phidot1 = ifft(phidot_spectrum*sqrt(-log(u1))*cos(2*pi*u2))
    phidot2 = ifft(phidot_spectrum*sqrt(-log(u1))*sin(2*pi*u2))
    
    return phi1,phi2,phidot1,phidot2




