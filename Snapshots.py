################################################
### Snapshots.py ###############################
################################################

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft, ifft, fft2,fftshift
import numpy as np
import random

from Input import NDIMS,N
from StringID import CoresAppend, PlaqSave

z_slice = random.randrange(N-1)

def Snap_Save(fig,index,mov_dir):
    fig.savefig(mov_dir+str(index)+'.png',bbox_inches='tight')


def cbar(mappable,extend='neither',minorticklength=8,majorticklength=10,\
            minortickwidth=2,majortickwidth=2.5,pad=0.2,side="right",orientation="vertical"):
    
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)

    cax = divider.append_axes(side, size="5%", pad=pad)
    cbar = fig.colorbar(mappable, cax=cax,extend=extend,orientation=orientation)
    cbar.ax.tick_params(which='minor',length=minorticklength,width=minortickwidth)
    cbar.ax.tick_params(which='major',length=majorticklength,width=majortickwidth)
    cbar.solids.set_edgecolor("face")


def AxionSnapshot(NDIMS,field,z_slice,index,lw=1,lfs=30,tfs=25):

    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    fig = plt.figure(figsize=(13,12))
    ax = fig.add_subplot(111)

    ax.set_title(r'$\hat{\tau} = %d$' % index)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
   

    if NDIMS == 2:

        im = ax.imshow(field,cmap=cm.twilight,origin='lower')

    if NDIMS == 3:

        im = ax.imshow(field[:,:,z_slice],cmap=cm.twilight,origin='lower')
    
    cb = cbar(im)

    return fig,ax

def SaxionSnapshot(NDIMS,field,z_slice,index,lw=1,lfs=30,tfs=25):

    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    fig = plt.figure(figsize=(13,12))
    ax = fig.add_subplot(111)

    ax.set_title(r'$\hat{\tau} = %d$' % index)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    

    if NDIMS == 2:

        im = ax.imshow(field,cmap=cm.RdBu,origin='lower',vmin=0,vmax=1)

    if NDIMS == 3:

        im = ax.imshow(field[:,:,z_slice],cmap=cm.RdBu,origin='lower',vmin=0,vmax=1)
    
    cb = cbar(im)

    return fig,ax


def DoubleSnapshot(NDIMS,field1,field2,z_slice,index,xlab1='',ylab1='',xlab2='',ylab2='',\
                 wspace=0.25,lw=1,lfs=45,tfs=23,size_x=20,size_y=11,Grid=False):
    
    plt.rcParams['axes.linewidth'] = lw
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=tfs)
    #mpl.rcParams['text.latex.preamble'] = [r'\usepackage{mathpazo}']
    
    fig, axarr = plt.subplots(1, 2,figsize=(size_x,size_y))
    gs = gridspec.GridSpec(1, 2)
    gs.update(wspace=wspace)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ax1.set_title(r'$\hat{\tau} = %d$' % index)
    ax2.set_title(r'$\hat{\tau} = %d$' % index)

    ax1.set_xlabel(xlab1,fontsize=lfs)
    ax1.set_ylabel(ylab1,fontsize=lfs)
    ax2.set_xlabel(xlab2,fontsize=lfs)
    ax2.set_ylabel(ylab2,fontsize=lfs)
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])

    if Grid:
        ax1.grid()
        ax2.grid()


    if NDIMS == 2:

        im1 = ax1.imshow(field1,cmap=cm.RdBu,origin='lower',vmin=0,vmax=1)
        im2 = ax2.imshow(field2,cmap=cm.twilight,origin='lower')

    if NDIMS == 3:

        im1 = ax1.imshow(field1[:,:,z_slice],cmap=cm.RdBu,origin='lower',vmin=0,vmax=1)
        im2 = ax2.imshow(field2[:,:,z_slice],cmap=cm.twilight,origin='lower')

    cb1 = cbar(im1)
    cb2 = cbar(im2)

    return fig,ax1,ax2


def StringSnapshot(locs,t_evol,N):
    
    X = [row[0] for row in locs]
    Y = [row[1] for row in locs]
    Z = [row[2] for row in locs]
    
    fig = plt.figure(figsize=(13,13))
    ax = Axes3D(fig)

    r1 = [-t_evol/2, t_evol/2]
    r2 = [-t_evol/2, t_evol/2]
    r3 = [-t_evol/2, t_evol/2]
    center = [t_evol/2,t_evol/2,t_evol/2]

    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        s=np.array(center)+np.array(s)
        e=np.array(center)+np.array(e)
        #ax.scatter3D(*center, color="r") 
        if np.linalg.norm(s-e) == 2*r1[1] or np.linalg.norm(s-e) == 2*r2[1] or np.linalg.norm(s-e) == 2*r3[1]:
            #print zip(s,e)
            ax.plot3D(*zip(s,e), color="black")  

    def get_fix_mins_maxs(mins, maxs):
        deltas = (maxs - mins) / 12.
        mins = mins + deltas / 4.
        maxs = maxs - deltas / 4.
    
        return [mins, maxs]
        
    plt.rcParams['axes.linewidth'] = 2
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=25)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    minmax = get_fix_mins_maxs(0,N-1)
    ax.set_xlim(minmax)
    ax.set_ylim(minmax) 
    ax.set_zlim(minmax) 

    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0
    ax.grid(False)
    ax.scatter(X,Y,Z,c=Z,marker = '.',linewidths=2,cmap=cm.RdBu)

    return fig,ax


def HistoSnapshot(s,index):

    x=pd.Series(s)
    
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=25)

    fig=plt.figure(figsize=(13,8))
    ax=fig.add_subplot(111)

    plt.yscale('log')
    hist, bins, _ = plt.hist(x, bins=200,color='g')
    ax.set_title(r'$\kappa = %f' % index)
    ax.set_xlabel(r'$\Delta\theta/2\pi$',fontsize=25)
    ax.set_ylabel(r'$N$',fontsize=25,rotation=0)
    ax.xaxis.set_label_coords(1.0, -0.08)
    ax.yaxis.set_label_coords(-0.08,0.95)

    return fig,ax    


def Print_Snapshots(f_axion = True, f_saxion = False, f_double = False, f_3dstrings = False):
    
    from MainPQ import axion,saxion,t_evol,tstep,mov_dir_ax,mov_dir_sax,mov_dir_double,mov_dir_3ds
    if f_axion:

        fig1, ax1 = AxionSnapshot(NDIMS,axion,z_slice,t_evol)
        Snap_Save(fig1,tstep,mov_dir_ax) 

    if f_saxion:

        fig2, ax2 = SaxionSnapshot(NDIMS,saxion,z_slice,t_evol)
        Snap_Save(fig2,tstep,mov_dir_sax) 

    if f_double:

        fig3, ax3a, ax3b = DoubleSnapshot(NDIMS,saxion,axion,z_slice,t_evol)
        Snap_Save(fig3,tstep,mov_dir_double) 

    if f_3dstrings:

        locs = PlaqSave(axion,1)
        fig4, ax4 = StringSnapshot(locs,t_evol,N)
        Snap_Save(fig3,tstep,mov_dir_3ds)


def Print_Histogram(field, f_histo = False):

    if f_histo:

        s = CoresAppend(field,1)
        fig5, ax5 = HistoSnapshot(s,kappa)
        Snap_Save(fig5,tstep,mov_dir_histo)



### TODO


# def Print_PowerSpectrum(field,bins,f_power = True, f_power_double = False):

# 	if f_power:

# 		fig5, ax5 = PS_Plot(field, N, L, t_evol, bins)
# 		Snap_Save(fig5,tstep, mov_dir_ps1)




