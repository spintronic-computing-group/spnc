# -*- coding: utf-8 -*-
# Run TI46 spoken digit task for the SPNC basic class

import spnc_ml as ml
from spnc import spnc_anisotropy

# spoken digit variables
speakers = ['f1', 'f2', 'f3', 'f4', 'f5'] # blank for all
# speakers = None # None for all

# Net Parameters
Nvirt = 100
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
ml.spnc_spoken_digits(speakers,Nvirt,m0,bias,transform,params,nfft=2048)

