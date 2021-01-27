# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Sort out relative paths
import sys
from pathlib import Path
repodir = Path('../../..').resolve()
try:
    sys.path.index(str(repodir))
except ValueError:
    sys.path.append(str(repodir))

#tuple of Path variables
searchpaths = (Path.home() / 'repos', )
#tuple of repos
repos = ('machine_learning_library',)

# local imports    
from SPNC import spnc
#ML specific
from SPNC.deterministic_mask import fixed_seed_mask, max_sequences_mask
import SPNC.repo_tools
SPNC.repo_tools.repos_path_finder(searchpaths, repos) #find ml library
from single_node_res import single_node_reservoir
import ridge_regression as RR
from linear_layer import *
from mask import binary_mask
from utility import *
from NARMA10 import NARMA10
from sklearn.metrics import classification_report

'''
NARMA10 plotting
'''

# NARMA parameters
Ntrain = 2000
Ntest = 1000

# Net Parameters
Nvirt = 400
m0 = 7e-2
bias = True

# Mist parameters
seed_NARMA = 1234
fixed_mask = False

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
params = {'theta': 10,'gamma' : .28,'delay_feedback' : 1,'Nvirt' : Nvirt}
spnres = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transform = spnres.gen_signal_fast_delayed_feedback


# Lets get into it
print("seed NARMA: "+str(seed_NARMA))
u, d = NARMA10(Ntrain + Ntest,seed=seed_NARMA)

x_train = u[:Ntrain]
y_train = d[:Ntrain]
x_test = u[Ntrain:]
y_test = d[Ntrain:]

print("Samples for training: ", len(x_train))
print("Samples for test: ", len(x_test))

# Net setup
Nin = x_train[0].shape[-1]
Nout = len(np.unique(y_train))

print( 'Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)

snr = single_node_reservoir(Nin, Nout, Nvirt, m0, res = transform)
net = linear(Nin, Nout, bias = bias)

# Training
S_train, J_train = snr.transform(x_train,params)
np.size(S_train)
seed_training = 1234
RR.Kfold_train(net,S_train,y_train,5, quiet = False)

# Testing
S_test, J_test = snr.transform(x_test,params)

spacer = 0
print("Spacer NRMSE:"+str(spacer))
pred = net.forward(S_test)
np.size(pred)
error = MSE(pred, y_test)
predNRMSE = NRMSE(pred, y_test)
print(error, predNRMSE)

plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0), 'k--')
plt.plot(y_test, pred, 'o')
plt.show()



