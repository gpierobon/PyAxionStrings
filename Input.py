#######################################
# SIMULATION PARAMETERS TO CHOOSE
#######################################

NDIMS = 2 # Spatial dimensions of the simulatiuo box. Set NDIMS = 2 or 3
N = 512 # Number of grid points
Delta = 1 # Default Delta=1 and H=fa
DeltaRatio = 1.0/3.0 # Time/Space step ratio 

fa_phys = 10**12.0 # Phsycal value of fa in GeV
lambdaPRS = 1 # Quartic coupling 
Potential = 'MexicanHat' # Choose potential. Options: 'MexicanHat' or 'Thermal'
Era = 1 # Era=1 for radiation domination, Era=2 for early matter domination (in PRS trick)


# TO IMPLEMENT

#alpha =1 # PRS fudge factor
#nu = 1 # Choose simulation type. Options: nu = 0 (Physical), nu = 1 (PRS)



#######################################
# INDEPENDENT PARAMETERS 
#######################################

from numpy import *
Mpl = 2.4*10**18.0/fa_phys # Normalized to fa_phys chosen 
fa = 1 # In fa units
ms = sqrt(lambdaPRS)*fa # Saxion mass for PRS in ADM units
L = Delta*N # Comoving 
Delta_tau = DeltaRatio*Delta
H1 = fa/Delta

# Code spacings
dx = ms*Delta 
dtau = ms*Delta_tau
t_evol = dtau/DeltaRatio - dtau # Such that in the first loop iteration, t_evol = 1 
light_time = int(0.5*N/DeltaRatio)
gstar = 106.75
T0 = math.sqrt(Mpl*90.0/(gstar*pi*pi)) # In units of fa
R0 = Delta*ms/(ms*L)
t0 = Delta/(2*L*ms**2.0)

        
