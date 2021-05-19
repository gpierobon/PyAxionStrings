#######################################
# Input.py ############################
#######################################

NDIMS = 3 # Spatial dimensions of the simulation box. Set NDIMS = 2 or 3
fa_phys = 10**12.0 # Phsycal value of fa in GeV
lambdaPRS = 1 # Quartic coupling 
Potential = 'Mexican' # Choose potential. Options: 'MexicanHat' or 'Thermal'

N = 64 # Number of grid points
Delta = 1 # Default Delta=1 and H=fa
DeltaRatio = 1.0/3.0 # Time/Space step ratio 
Era = 1 # Era=1 for radiation domination, Era=2 for early matter domination (in PRS trick)
StencilOrder = 2 

#--------------------------------
# SNAPSHOTS
#--------------------------------

print_snapshot_final = True

f_axion = True
f_saxion = False
f_double = False
f_3dstrings = False

#--------------------------------
# ANALYSIS
#--------------------------------

analyse_strings = False
analyse_spectrum = False
analyse_energies = False

string_checks = 40
count_strings = True
thr = 1 
