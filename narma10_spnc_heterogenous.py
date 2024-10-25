# Run narma10 for the SPNC basic class

import spnc_ml as ml
from spnc import spnc_anisotropy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# NARMA parameters
Ntrain = 2000
Ntest = 1000

# Net Parameters
Nvirt = 400
m0 = 0.003
bias = True

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 20
params = {'theta': 0.3,'gamma' : .113,'delay_feedback' : 0,'Nvirt' : Nvirt}
spn = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transform = spn.gen_signal_fast_delayed_feedback_varing_temp

# DO IT
(y_test,y_pred)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transform, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)