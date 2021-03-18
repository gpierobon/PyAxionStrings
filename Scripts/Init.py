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
        print("Warning: number of dimensions (NDIMS) is incorrect!")
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