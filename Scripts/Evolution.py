from Input import *

fstep = light_time-int(1/DeltaRatio)+1

def KDK_Evolve(phi1,phi2,phidot1,phidot2,K1,K2):
    phi1 = phi1 + dtau*(phidot1 + 0.5*K1*dtau)
    phi2 = phi2 + dtau*(phidot2 + 0.5*K2*dtau)
    t_evol = t_evol + dtau
    K1_next = Laplacian_2D(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1)#- T0**2.0/(3.0*t_evol**2.0))
    K2_next = Laplacian_2D(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1)#- T0**2.0/(3.0*t_evol**2.0))  
    phidot1 = phidot1 + 0.5*(K1 + K1_next)*dtau
    phidot2 = phidot2 + 0.5*(K2 + K2_next)*dtau
    K1 = 1.0*K1_next
    K2 = 1.0*K2_next
    return 

def DKD_Evolve():
	# DRIFT KICK DRIFT SCHEME 
	return 

# MAYBE SOME OTHER EVOLVERS 