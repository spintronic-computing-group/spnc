# Run narma10 for the SPNC basic class

import spnc_ml as ml
from spnc import spnc_anisotropy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



from single_node_heterogenous_reservoir import single_node_heterogenous_reservoir

# NARMA parameters
Ntrain = 2000
Ntest = 1000

# Net Parameters
Nvirt = 40
m0 = 1e-2
bias = True

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
params = {'theta': 1/3,'gamma' : .25,'delay_feedback' : 0,'Nvirt' : Nvirt}

gamma = 0.25
beta_ref = 10
deltabeta_list = [0.0,0.0,0.0]
theta = 1/3
step = 1
beta_left = 10
beta_right = 10
weights = [1,0,0]  # weights 长度应与 deltabeta_list 匹配


# DO IT

beta_primes_temp, nrmse_temp =ml.spnc_narma10_heterogenous(Ntrain,Ntest,Nvirt,gamma, beta_prime, beta_ref, deltabeta_list,h,theta,m0,step,beta_left,beta_right,*weights, bias = bias,params = params,seed_NARMA=1234)