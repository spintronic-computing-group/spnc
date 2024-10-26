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

gamma = 0.113
beta_ref = 20
deltabeta_list = [0.1, 0.2, 0.3]
theta = 0.3
step = 1
beta_left = 18.9
beta_right = 21.1
weights = [0.3,0.4,0.3]  # weights 长度应与 deltabeta_list 匹配


# DO IT

beta_primes_temp, nrmse_temp =ml.spnc_narma10_heterogenous(Ntrain,Ntest,Nvirt,gamma, beta_prime, beta_ref, deltabeta_list,h,theta,m0,step,beta_left,beta_right,*weights, bias = bias,params = params,seed_NARMA=1234)