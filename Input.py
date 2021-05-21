#=================================================================================================================
#   Input.py - PyAxions (2021)
#=================================================================================================================

NDIMS = 2 # Spatial dimensions of the simulation box. Set NDIMS = 2 or 3
fa_phys = 10**12.0 # Phsycal value of fa in GeV
lambdaPRS = 1 # Quartic coupling 
Potential = 'Thermal' # Choose potential. Options: 'MexicanHat' or 'Thermal'

N = 256 # Number of grid points
Delta = 1 # Default Delta=1 and H=fa
DeltaRatio = 1.0/3.0 # Time/Space step ratio 
Era = 1 # Options: Era = 1 for RD, Era = 2 for EMD (in PRS trick)
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
break_tstep = 442  # Ref. values: log4=160,log5=442,log5.5=731,log6=1210,log6.5= 995,log7=3289,log7.5 = 5424,log8 = 8942 

#TODO: Spectrum stuff here 
#......

#-----------------------------------------------------------------------------------------------------------------
# QCD SIMULATION INPUT 
#-----------------------------------------------------------------------------------------------------------------

init_QCD = False
cores_final = 10 # This is for 2D

# TODO: 3D simulation stopping point for QCD input 

#-----------------------------------------------------------------------------------------------------------------
# KATANA/CLUSTER RUN
#-----------------------------------------------------------------------------------------------------------------

array_job = False


