#===============================StringID.py============================#
# Created by Giovanni Pierobon 2021

# Description: Functions to identify, record and plot strings from the
# newtork evolution of an axion field 

#======================================================================#
from numpy import *
from numba import njit
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

###################################################################
# Pi method in 2D 
###################################################################
@njit
def cores_pi(f,N,thr):
    s=[]
    count = 0
    accept = 0.5 - 0.5*thr/100
    for i in range(N-1):
        for j in range(N-1):
            norm1 = (f[i][j]+pi)/(2*pi)
            norm2 = (f[i+1][j]+pi)/(2*pi)
            norm3 = (f[i+1][j+1]+pi)/(2*pi)
            norm4 = (f[i][j+1]+pi)/(2*pi)
            theta1 = min(abs(norm2-norm1),1-abs(norm2-norm1))
            theta2 = min(abs(norm3-norm2),1-abs(norm3-norm2))
            theta3 = min(abs(norm4-norm3),1-abs(norm4-norm3))
            theta_sum = theta1 + theta2 + theta3
            if theta_sum>accept:
                s.append([i,j])
    for a in range(0,len(s)-1):
        diff_y=s[a+1][1]-s[a][1]
        diff_x=s[a+1][0]-s[a][0]
        if (diff_y == 0 and diff_x == 1):
            count+=1
        if (diff_y == 1 and diff_x == 0):
            count+=1
    num=len(s)-count
    return num

###################################################################
# Pi/2 method in 2D 
###################################################################
@njit
def cores_pi2(f,N,thr):
    accept = pi/2.0 - pi/2.0 *thr/100
    s=[]
    count=0
    for i in range(0,N-1):
        for j in range(0,N-1):
            south=abs(f[i+1][j]) - abs(f[i][j])
            east=abs(f[i+1][j+1]) - abs(f[i+1][j]) 
            north=abs(f[i][j+1]) - abs(f[i+1][j+1]) 
            west=abs(f[i][j]) - abs(f[i][j+1])
            if (south>accept or east>accept or north>accept or west>accept): 
                s.append([i,j])
                s.append([i,j+1])
                s.append([i+1,j])
                s.append([i+1,j+1])
    return int(len(s)/4.0)

@njit
def cores_pi2v2(f,N,thr):
    accept = pi/2.0 - pi/2.0 *thr/100
    s=[]
    nos=[]
    count=0
    for i in range(0,N-1):
        for j in range(0,N-1):
            south=abs(f[i+1][j]) - abs(f[i][j])
            east=abs(f[i+1][j+1]) - abs(f[i+1][j]) 
            north=abs(f[i][j+1]) - abs(f[i+1][j+1]) 
            west=abs(f[i][j]) - abs(f[i][j+1])
            if (south>accept or east>accept or north>accept or west>accept): 
                s.append([i,j])
                s.append([i,j+1])
                s.append([i+1,j])
                s.append([i+1,j+1])
            else:
                nos.append([i,j])
                nos.append([i,j+1])
                nos.append([i+1,j])
                nos.append([i+1,j+1])
    return int(len(s)/4.0), int(len(nos)/4.0)

##################################################################
# Pi method in 3D
##################################################################

@njit
def plaq_pi(f,N,thr):
    s = []
    count = 0
    accept = 0.5 - 0.5*thr/100
    for i in range(N-1):
        for j in range(N-1):
            for k in range(N-1):
                norm1a = (f[i][j][k]+pi)/(2*pi)          # (i,j) plane
                norm2a = (f[i+1][j][k]+pi)/(2*pi)
                norm3a = (f[i+1][j+1][k]+pi)/(2*pi)
                norm4a = (f[i][j+1][k]+pi)/(2*pi)

                norm1b = (f[i][j][k]+pi)/(2*pi)          # (i,k) plane 
                norm2b = (f[i+1][j][k]+pi)/(2*pi)
                norm3b = (f[i+1][j][k+1]+pi)/(2*pi)
                norm4b = (f[i][j][k+1]+pi)/(2*pi)

                norm1c = (f[i][j][k]+pi)/(2*pi)             # (j,k) plane
                norm2c = (f[i][j+1][k]+pi)/(2*pi)
                norm3c = (f[i][j+1][k+1]+pi)/(2*pi)
                norm4c = (f[i][j][k+1]+pi)/(2*pi)

                theta1a = min(abs(norm2a-norm1a),1-abs(norm2a-norm1a))
                theta2a = min(abs(norm3a-norm2a),1-abs(norm3a-norm2a))
                theta3a = min(abs(norm4a-norm3a),1-abs(norm4a-norm3a))
            
                theta1b = min(abs(norm2b-norm1b),1-abs(norm2b-norm1b))
                theta2b = min(abs(norm3b-norm2b),1-abs(norm3b-norm2b))
                theta3b = min(abs(norm4b-norm3b),1-abs(norm4b-norm3b))
            
                theta1c = min(abs(norm2c-norm1c),1-abs(norm2c-norm1c))
                theta2c = min(abs(norm3c-norm2c),1-abs(norm3c-norm2c))
                theta3c = min(abs(norm4c-norm3c),1-abs(norm4c-norm3c))

                theta_sum_a = theta1a + theta2a + theta3a
                theta_sum_b = theta1b + theta2b + theta3b
                theta_sum_c = theta1c + theta2c + theta3c

                if (theta_sum_a or theta_sum_b or theta_sum_c) > accept:
                    s.append([i,j,k])
    
    for a in range(0,len(s)-1):
        diff_z = abs(s[a+1][2] - s[a][2])
        diff_y = abs(s[a+1][1] - s[a][1])
        diff_x = abs(s[a+1][0] - s[a][0])
        
        if (diff_z == 0 and diff_y == 0 and diff_x == 1):
            count+=1
        if (diff_z == 0 and diff_y == 1 and diff_x == 0):
            count+=1
        if (diff_z == 1 and diff_y == 0 and diff_x == 0):
            count+=1
    num=len(s)-count
    return num*2.0/3.0 # Contains statistical count factor from Moore paper (arXiv:1509.00026)

##################################################################
# 3D version of pi/2 method 
##################################################################
@njit 
def plaq_pi2(f,N,thr):
    accept = pi/2.0 - pi/2.0 *thr/100
    s=[]
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
                    s.append([i,j,k])
                    s.append([i,j+1,k])
                    s.append([i+1,j,k])
                    s.append([i+1,j+1,k])
                if (aik>accept or bik>accept or cik>accept or dik>accept):
                    s.append([i,j,k])
                    s.append([i,j,k+1])
                    s.append([i+1,j,k])
                    s.append([i+1,j,k+1])
                if (ajk>accept or bjk>accept or cjk>accept or djk>accept):
                    s.append([i,j,k])
                    s.append([i,j+1,k])
                    s.append([i,j,k+1])
                    s.append([i,j+1,k+1])
    return int(len(s)/6.0) # Correction of factor 2/3 (Manhattan effect)

######################################################################
# HISTOGRAM TO CHECK THE CORES FINDIND ALGORITHM
######################################################################

def cores_std(f,N,thr):
    s=[]
    accept = pi - pi*thr/100
    for i in range(N-1):
        for j in range(N-1):
            norm1 = (f[i][j]+pi)/(2*pi)
            norm2 = (f[i+1][j]+pi)/(2*pi)
            norm3 = (f[i+1][j+1]+pi)/(2*pi)
            norm4 = (f[i][j+1]+pi)/(2*pi)
            theta1 = min(abs(norm2-norm1),1-abs(norm2-norm1))
            theta2 = min(abs(norm3-norm2),1-abs(norm3-norm2))
            theta3 = min(abs(norm4-norm3),1-abs(norm4-norm3))
            theta_sum = theta1 + theta2 + theta3
            s.append(theta_sum)
    return s

def histo_std(hist_temp,index,num_cores):
    x=pd.Series(hist_temp)
    plt.rcParams['axes.linewidth'] = 2.5
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif',size=25)
    fig=plt.figure(figsize=(13,8))
    ax=fig.add_subplot(111)
    plt.yscale('log')
    hist, bins, _ = plt.hist(x, bins=200,color='g')
    ax.set_title(r'$\kappa = %f,\;\;\;\;\mathit{Cores}= %d$' % (index,num_cores))
    ax.set_xlabel(r'$\Delta\theta/2\pi$',fontsize=25)
    ax.set_ylabel(r'$N$',fontsize=25,rotation=0)
    ax.xaxis.set_label_coords(1.0, -0.08)
    ax.yaxis.set_label_coords(-0.08,0.95)
    return fig,ax    


#################################################################################
# FINDING CORES AND PLOTTING 2D
#################################################################################

def draw(f,N,index,size_x=13,size_y=12):
    s=[]
    count = 0
    accept = 0.5 - 0.5*thr/100
    for i in range(N-1):
        for j in range(N-1):
            norm1 = (f[i][j]+pi)/(2*pi)
            norm2 = (f[i+1][j]+pi)/(2*pi)
            norm3 = (f[i+1][j+1]+pi)/(2*pi)
            norm4 = (f[i][j+1]+pi)/(2*pi)
            theta1 = min(abs(norm2-norm1),1-abs(norm2-norm1))
            theta2 = min(abs(norm3-norm2),1-abs(norm3-norm2))
            theta3 = min(abs(norm4-norm3),1-abs(norm4-norm3))
            theta_sum = theta1 + theta2 + theta3
            if theta_sum>accept:
                s.append([i,j])
    x_coord=[row[1] for row in s]
    y_coord=[row[0] for row in s]
    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)
    white_field=zeros(shape=(N,N))
    ax.imshow(white_field,origin='lower',cmap=cm.binary,vmax=1)
    plt.scatter(x_coord,y_coord,c=y_coord,cmap=cm.viridis,lw=1)
    ax.set_title(r'$\hat{\tau} = %d$' % index)
    return fig,ax



