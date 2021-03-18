from numpy import *
from scipy.fft import fft, ifft, fft2,fftshift
import matplotlib.pyplot as plt
from matplotlib import cm

def FourierPlot(f,N,L,index,size_x,size_y):
    y = fftshift(fft2(f))
    dk = 2*pi/L
    kx = linspace(-pi+dk,pi,N-1)
    KX,KY = meshgrid(kx,kx)
    K = sqrt(KX**2+KY**2)
    P = abs(y[1:,1:])**2
    plt.rcParams['axes.linewidth'] = 1
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=25)
    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)
    im=plt.pcolormesh(KX,KY,log10(P))
    ax.set_title(r'$\hat{\tau} = %f$' % index)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    cb = cbar(im)
    return fig,ax