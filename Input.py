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
break_tstep = 442    #Ref. values: log4=160,log5=442,log5.5=731,log6=1210,log6.5= 995,log7=3289,log7.5 = 5424,log8 = 8942 

#TODO: Spectrum stuff here 
#......

#-----------------------------------------------------------------------------------------------------------------
# QCD SIMULATION
#-----------------------------------------------------------------------------------------------------------------

init_QCD = False
cores_final = 10 # This is for 2D, missign 3D implementation 

L_QCD = 6 
ti_QCD = 0.4
n = 2  # Ref. values: QCD lattice n=6.68, QCD generic n=7, QCD beta function n=8. ALPS: n=2,4,6,8,10,...  
#t_critical = # Axion mass reaching zero T value 
lambdaPRS_QCD = 1000 # Check Buschmann et al, Vaquero et al. 



#-----------------------------------------------------------------------------------------------------------------
# KATANA/CLUSTER RUN
#-----------------------------------------------------------------------------------------------------------------

array_job = False



