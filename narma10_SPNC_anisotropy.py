# Run narma10 for the SPNC basic class

import spnc_ml as ml
from spnc import spnc_anisotropy

# NARMA parameters
Ntrain = 1000
Ntest = 1000

# Net Parameters
Nvirt = 400
m0 = 7e-2
bias = True

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
params = {'theta': 10,'gamma' : .28,'delay_feedback' : 1,'Nvirt' : Nvirt}
spn = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transform = spn.gen_signal_fast_delayed_feedback

# DO IT
ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transform, params)


