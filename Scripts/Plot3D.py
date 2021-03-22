from numpy import *
from numpy.random import *
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib import colors
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.fft import fft, ifft, fft2,fftshift
from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations


def print_plaqs(f,N,thr,index):
    accept = pi/2.0 - pi/2.0 *thr/100
    a = open("Locs/locs%d.txt" % index ,"w")
    count=0
    for i in range(0,N-1):
        for j in range(0,N-1):
            for k in range(0,N-1):
                aij=abs(f[i+1][j][k])-abs(f[i][j][k]) 
                bij=abs(f[i+1][j+1][k])-abs(f[i+1][j][k])
                cij=abs(f[i][j+1][k])-abs(f[i+1][j+1][k]) 
                dij=abs(f[i][j][k])-abs(f[i][j+1][k])
                aik=abs(f[i+1][j][k])-abs(f[i][j][k]) 
                bik=abs(f[i+1][j][k+1])-abs(f[i+1][j][k]) 
                cik=abs(f[i][j][k+1])-abs(f[i+1][j][k+1])
                dik=abs(f[i][j][k])-abs(f[i][j][k+1]) 
                ajk=abs(f[i][j+1][k])-abs(f[i][j][k]) 
                bjk=abs(f[i][j+1][k])-abs(f[i][j+1][k]) 
                cjk=abs(f[i][j][k+1])-abs(f[i][j+1][k]) 
                djk=abs(f[i][j][k])-abs(f[i][j][k+1])
                if (aij>accept or bij>accept or cij>accept or dij>accept): 
                    a.write("%s %s %s \n" % (i,j,k))
                    count += 1
                if (aik>accept or bik>accept or cik>accept or dik>accept):
                    a.write("%s %s %s \n" % (i,j,k))
                    count += 1
                if (ajk>accept or bjk>accept or cjk>accept or djk>accept):
                    a.write("%s %s %s \n" % (i,j,k))
                    count += 1            
    a.close()   
    return count

def plaq_coord(N,index):
    x=[]
    y=[]
    z=[]
    table = pd.read_table("Locs/locs%d.txt" %index, delim_whitespace=True, header=None) 
    x = table.iloc[:,-1].values.tolist() # create array with first column
    y = table.iloc[:,-2].values.tolist()
    z = table.iloc[:,-3].values.tolist()# create array with second column
    return x,y,z


def String_Plot(t_evol,X,Y,Z):
    fig = plt.figure(figsize=(13,13))
    ax = Axes3D(fig)

    r1 = [-t_evol/2, t_evol/2]
    r2 = [-t_evol/2, t_evol/2]
    r3 = [-t_evol/2, t_evol/2]
    center = [t_evol/2,t_evol/2,t_evol/2]

    for s, e in combinations(array(list(product(r1,r2,r3))), 2):
        s=array(center)+array(s)
        e=array(center)+array(e)
        #ax.scatter3D(*center, color="r") 
        if linalg.norm(s-e) == 2*r1[1] or linalg.norm(s-e) == 2*r2[1] or linalg.norm(s-e) == 2*r3[1]:
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
    
    minmax = get_fix_mins_maxs(0,99)
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
    ax.scatter(X,Y,Z,c=Z,marker = '.',linewidths=5,cmap=cm.RdBu)

    return fig,ax
