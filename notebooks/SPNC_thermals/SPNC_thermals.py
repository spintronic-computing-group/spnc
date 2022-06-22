# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.7.6 ('SPNC')
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SPNC thermals
# Code for experimenting with the effects of thermal variation on ML performance

# %% [markdown]
# Set up imports:

# %%
"""
Import handeling and Dependancy info

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
repodir = Path('..').resolve()
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

# %% [markdown]
# Establish shared parameters

# %%
'''
NARMA10 response
'''

plt.rcParams['text.usetex'] = True

figurewidth = 3.37*2.5
figureaspect = 2/3
figureheight = figurewidth*figureaspect

# NARMA parameters
Ntrain = 100
Ntest = 50

# Net Parameters
Nvirt = 5
m0 = 0.003
bias = True

# Misc parameters
seed_NARMA = 1234
fixed_mask = False
spacer = 5

theta = 0.4
gamma = 0.132
delay_feedback = 0
params = {'theta': theta, 'gamma' : gamma,'delay_feedback' : delay_feedback,'Nvirt' : Nvirt}


def get_res(h=0.4,theta_H=90,k_s_0=0,phi=45,beta_prime=20):
    res = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
    transform = res.gen_signal_fast_delayed_feedback

    return res, transform

base_beta_prime = 20
spnres, transform = get_res(beta_prime=base_beta_prime)
spnreshigher, transformhigher = get_res(beta_prime=base_beta_prime)

# %% [markdown]
# Data and net setup

# %%
# Testing output properties

# Misc parameters
params = {'theta': 0.4, 'gamma' : 0.132,'delay_feedback' : 0,'Nvirt' : 5}
params2 = {'theta': 0.4, 'gamma' : 0.132,'delay_feedback' : 0,'Nvirt' : 5}
params3 = {'theta': 0.4, 'gamma' : 0.132,'delay_feedback' : 1,'Nvirt' : 5}
Ntrain = 100
Ntest = 50

print("seed NARMA: "+str(seed_NARMA))
u, d = NARMA10(Ntrain + Ntest,seed=seed_NARMA)

fig, ax = plt.subplots(1,figsize=[figurewidth,figureheight],dpi=200)
ax.plot(u)
ax.plot(d)
ax.legend(['Narma10 Input','Narma10 Output'])
ax.set_title('Narma10 data')

transform_1 = get_res(beta_prime=base_beta_prime)[1]
transform_2 = get_res(beta_prime=base_beta_prime)[1]
transform_3 = get_res(beta_prime=base_beta_prime/0.9)[1]

transform_4 = get_res(beta_prime=base_beta_prime)[1]
transform_5 = get_res(beta_prime=base_beta_prime/0.9)[1]

transformed_1 = transform_1(u,params)
transformed_2 = transform_2(u,params)
transformed_3 = transform_3(u,params)

transformed_4 = transform_4(u,params3)
transformed_5 = transform_5(u,params3)


fig,ax = plt.subplots(2,figsize=[figurewidth,figureheight],dpi=200)
ax[0].plot(transformed_1)
ax[0].plot(transformed_2,linestyle='dashed')
ax[0].set_title('Transformed input (mask then res) with nominally identical res')
ax[0].legend(['Transformed data 1','Transformed data 2'])
ax[0].set_ylabel('Output')
ax[1].plot(transformed_1-transformed_2)
ax[1].set_ylabel('Difference')

fig,ax = plt.subplots(2,figsize=[figurewidth,figureheight],dpi=200)
ax[0].plot(transformed_1)
ax[0].plot(transformed_3,linestyle='dashed')
ax[0].set_title('Transformed input (mask then res) reservoirs differ')
ax[0].legend([r"$\beta' = 20$",r"$\beta' = 20/0.9$"])
ax[0].set_ylabel('Output')
ax[1].plot(transformed_1-transformed_3)
ax[1].set_ylabel('Difference')

fig,ax = plt.subplots(2,figsize=[figurewidth,figureheight],dpi=200)
ax[0].plot(transformed_4)
ax[0].plot(transformed_5,linestyle='dashed')
ax[0].set_title('Transformed input (mask then res) reservoirs differ, delayed feedback')
ax[0].legend([r"$\beta' = 20$",r"$\beta' = 20/0.9$"])
ax[0].set_ylabel('Output')
ax[1].plot(transformed_4-transformed_5)
ax[1].set_ylabel('Difference')

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

# %% [markdown]
# Training and testing:

# %%
# Training at one temp, testing at another

# Misc parameters
params = {'theta': 0.4, 'gamma' : 0.132,'delay_feedback' : 0,'Nvirt' : 50}
Ntrain = 500
Ntest = 250
spacer = 50

print("seed NARMA: "+str(seed_NARMA))
u, d = NARMA10(Ntrain + Ntest,seed=seed_NARMA)


transform = get_res(beta_prime=base_beta_prime)[1]
transformhigher = get_res(beta_prime=base_beta_prime/0.9)[1]


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
RR.Kfold_train(net,S_train,y_train,10, quiet = False)

# Testing
S_test, J_test = snr.transform(x_test,params)
#test with the other transform...
snr.res = transformhigher
S_test_higher, J_test_higher = snr.transform(x_test,params)


print("Spacer NRMSE:"+str(spacer))
pred = net.forward(S_test)
pred_higher = net.forward(S_test_higher)
plt.plot(pred)
plt.plot(pred_higher)
plt.show()
np.size(pred)
error = MSE(pred[spacer:], y_test[spacer:])
predNRMSE = NRMSE(pred[spacer:], y_test[spacer:])
print('Error and NRMSE for normal testing', error, predNRMSE)

error_higher = MSE(pred_higher[spacer:], y_test[spacer:])
predNRMSE_higher = NRMSE(pred_higher[spacer:], y_test[spacer:])
print('Error and NRMSE for higher temp testing', error_higher, predNRMSE_higher)

plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0),'k--')
plt.plot(y_test[spacer:],pred[spacer:],'o')
plt.show()

plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0),'k--')
plt.plot(y_test[spacer:],pred_higher[spacer:],'o')
plt.show()

# %%
# Training at one temp, testing at another

# Misc parameters
params = {'theta': 0.4, 'gamma' : 0,'delay_feedback' : 0,'Nvirt' : 5}
Ntrain = 50
Ntest = 50
spacer = 5

print("seed NARMA: "+str(seed_NARMA))
u, d = NARMA10(Ntrain + Ntest,seed=seed_NARMA)


transform = get_res(beta_prime=base_beta_prime)[1]
transformhigher = get_res(beta_prime=base_beta_prime/1.01)[1]


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
RR.Kfold_train(net,S_train,y_train,10, quiet = False)

# Testing
S_test, J_test = snr.transform(x_test,params)
#test with the other transform...
snr.res = transformhigher
S_test_higher, J_test_higher = snr.transform(x_test,params)


print("Spacer NRMSE:"+str(spacer))
pred = net.forward(S_test)
pred_higher = net.forward(S_test_higher)
plt.plot(pred)
plt.plot(pred_higher)
plt.show()
np.size(pred)
error = MSE(pred[spacer:], y_test[spacer:])
predNRMSE = NRMSE(pred[spacer:], y_test[spacer:])
print('Error and NRMSE for normal testing', error, predNRMSE)

error_higher = MSE(pred_higher[spacer:], y_test[spacer:])
predNRMSE_higher = NRMSE(pred_higher[spacer:], y_test[spacer:])
print('Error and NRMSE for higher temp testing', error_higher, predNRMSE_higher)

plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0),'k--')
plt.plot(y_test[spacer:],pred[spacer:],'o')
plt.show()

plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0),'k--')
plt.plot(y_test[spacer:],pred_higher[spacer:],'o')
plt.show()

# %%
# Training at one temp, testing at another

# Misc parameters
params = {'theta': 0.4, 'gamma' : 0.132,'delay_feedback' : 1,'Nvirt' : 5}
Ntrain = 50
Ntest = 50
spacer = 5

print("seed NARMA: "+str(seed_NARMA))
u, d = NARMA10(Ntrain + Ntest,seed=seed_NARMA)


transform = get_res(beta_prime=base_beta_prime)[1]
transformhigher = get_res(beta_prime=base_beta_prime/1.01)[1]


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
RR.Kfold_train(net,S_train,y_train,10, quiet = False)

# Testing
S_test, J_test = snr.transform(x_test,params)
#test with the other transform...
snr.res = transformhigher
S_test_higher, J_test_higher = snr.transform(x_test,params)


print("Spacer NRMSE:"+str(spacer))
pred = net.forward(S_test)
pred_higher = net.forward(S_test_higher)
plt.plot(pred)
plt.plot(pred_higher)
plt.show()
np.size(pred)
error = MSE(pred[spacer:], y_test[spacer:])
predNRMSE = NRMSE(pred[spacer:], y_test[spacer:])
print('Error and NRMSE for normal testing', error, predNRMSE)

error_higher = MSE(pred_higher[spacer:], y_test[spacer:])
predNRMSE_higher = NRMSE(pred_higher[spacer:], y_test[spacer:])
print('Error and NRMSE for higher temp testing', error_higher, predNRMSE_higher)

plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0),'k--')
plt.plot(y_test[spacer:],pred[spacer:],'o')
plt.show()

plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0),'k--')
plt.plot(y_test[spacer:],pred_higher[spacer:],'o')
plt.show()
