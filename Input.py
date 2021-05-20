#=================================================================================================================
#   Input.py - PyAxions (2021)
#=================================================================================================================

NDIMS = 3 # Spatial dimensions of the simulation box. Set NDIMS = 2 or 3
fa_phys = 10**12.0 # Phsycal value of fa in GeV
lambdaPRS = 1 # Quartic coupling 
Potential = 'Thermal' # Choose potential. Options: 'MexicanHat' or 'Thermal'

N = 64 # Number of grid points
Delta = 1 # Default Delta=1 and H=fa
DeltaRatio = 1.0/3.0 # Time/Space step ratio 
Era = 1 # Era=1 for radiation domination, Era=2 for early matter domination (in PRS trick)
StencilOrder = 2
time_var = 'String tension' # Options: 'String tension', 'Conformal time', 'Cosmic time', 'Scale factor' 

#-----------------------------------------------------------------------------------------------------------------
# SNAPSHOTS
#-----------------------------------------------------------------------------------------------------------------

print_snapshot_final = False

f_axion = True
f_saxion = False
f_double = False
f_3dstrings = True

#-----------------------------------------------------------------------------------------------------------------
# ANALYSIS
#-----------------------------------------------------------------------------------------------------------------

analyse_strings = True
analyse_spectrum = False
analyse_energies = False

string_checks = 40
thr = 1 
break_loop = False
cores_final = 10

#TODO: Spectrum stuff here 
#......

#-----------------------------------------------------------------------------------------------------------------
# KATANA/CLUSTER RUN
#-----------------------------------------------------------------------------------------------------------------

array_job = False


