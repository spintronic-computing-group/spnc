# -*- coding: utf-8 -*-
"""
@author: Alexander

This code runs some machine learning and saves the variables for plotting elsewhere

Local Dependancies
------------------
machine_learning_library  : v0.1.2
    This repository will need to be on your path in order to work.
    This is achieved with repo_tools module and a path find function
    Add to the searchpath and repos tuples if required


"""

import numpy as np
import matplotlib.pyplot as plt

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
NARMA10 response
'''

# NARMA parameters
Ntrain = 2000
Ntest = 1000

# Net Parameters
Nvirt = 400
m0 = 0.003
bias = True

# Misc parameters
seed_NARMA = None
fixed_mask = False
spacer = 50

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 20

theta = 0.4
gamma = 0.132
delay_feedback = 0
params = {'theta': theta, 'gamma' : gamma,'delay_feedback' : delay_feedback,'Nvirt' : Nvirt}
spnres = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transform = spnres.gen_signal_fast_delayed_feedback

spnreshigher = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime*1.1)
transformhigher = spnres.gen_signal_fast_delayed_feedback


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
snrhigher = single_node_reservoir(Nin, Nout, Nvirt, m0, res = transformhigher)
net = linear(Nin, Nout, bias = bias)

# Training
S_train, J_train = snr.transform(x_train,params)
np.size(S_train)
seed_training = 1234
RR.Kfold_train(net,S_train,y_train,10, quiet = False)

# Testing
S_test, J_test = snrhigher.transform(x_test,params)

print("Spacer NRMSE:"+str(spacer))
pred = net.forward(S_test)
np.size(pred)
error = MSE(pred[spacer:], y_test[spacer:])
predNRMSE = NRMSE(pred[spacer:], y_test[spacer:])
print(error, predNRMSE)

plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0),'k--')
plt.plot(y_test[spacer:],pred[spacer:],'o')
plt.show()

#Save data for plotting elsewhere
np.savez('data/NARMA10-temp.npz',
         Ntrain=Ntrain,
         Ntest=Ntest,
         Nvirt = Nvirt,
         gamma = gamma,
         delay_feedback = delay_feedback,
         spacer = spacer,
         x_train = x_train,
         y_train = y_train,
         x_test = x_test,
         y_test = y_test,
         Nin = Nin,
         Nout = Nout,
         S_train = S_train,
         J_train = J_train,
         S_test = S_test,
         J_test = J_test,
         pred = pred
         )
