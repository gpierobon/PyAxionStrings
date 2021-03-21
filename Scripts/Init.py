from numpy import *
from Input import *
from Spatial import Laplacian_2D, Laplacian_3D


def init_noise(N):
    
    if NDIMS == 2:
        
        th = 2*pi*random.uniform(size=(N,N))
        phi1 = cos(th)
        phi2 = sin(th)
        phidot1 = zeros(shape=(N,N))
        phidot2 = zeros(shape=(N,N))
    
    elif NDIMS == 3:
        
        th = 2*pi*random.uniform(size=(N,N,N))
        phi1 = cos(th)
        phi2 = sin(th)
        phidot1 = zeros(shape=(N,N,N))
        phidot2 = zeros(shape=(N,N,N))
    
    else:
        
        print("Number of spatial dimensions is incorrect! Check value of NDIMS on Input.py")
    
    return phi1,phi2,phidot1,phidot2


def thermal_phi(L,K,meffsquared,T0):
    
    omegak = sqrt(K**2.0 + meffsquared)
    bose = 1/(exp(omegak/T0)-1)
    spectrum = sqrt(L*bose/omegak)
    
    return spectrum


def thermal_phidot(L,K,meffsquared,T0):
    
    omegak = sqrt(K**2.0 + meffsquared)
    bose = 1/(exp(omegak/T0)-1)
    spectrum = sqrt(L*bose*omegak)
    
    return spectrum


def init_thermal(N):
    
    wnoise = 2*pi*random.uniform(size=(N,N)) # Fourier space realization of white noise
    
    dk = 2*pi/L
    kx = linspace(-pi+dk,pi,N-1)
    ky = kx
    meffsquared=lambdaPRS/L**2.0*(T0**2.0/3.0-1) # L**2.0 on denominator is converting into L units 
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
    kx = linspace(-pi+dk,pi,N-1)
    ky = kx
    meffsquared=lambdaPRS/L**2.0*(T0**2.0/3.0-1) # L**2.0 on denominator is converting into L units 
    KX,KY = meshgrid(kx,ky)
    K = sqrt(KX**2+KY**2)
   
    phi_spectrum = thermal_phi(L,K,meffsquared,T0)
    phidot_spectrum = thermal_phidot(L,K,meffsquared,T0)

    phi1 = ifft(phi_spectrum*sqrt(-log(u1))*cos(2*pi*u2))
    phi2 = ifft(phi_spectrum*sqrt(-log(u1))*sin(2*pi*u2))
    phidot1 = ifft(phidot_spectrum*sqrt(-log(u1))*cos(2*pi*u2))
    phidot2 = ifft(phidot_spectrum*sqrt(-log(u1))*sin(2*pi*u2))
    
    return phi1,phi2,phidot1,phidot2



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
           
            K1 = Laplacian_2D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + T0**2.0/(3.0*t_evol**2.0))
            K2 = Laplacian_2D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + T0**2.0/(3.0*t_evol**2.0))
        
        if NDIMS == 3:
            
            K1 = Laplacian_3D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + T0**2.0/(3.0*t_evol**2.0))
            K2 = Laplacian_3D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + T0**2.0/(3.0*t_evol**2.0))
    
    return K1,K2 
