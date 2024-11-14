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

# %% [markdown]
# ### Examining differences between fast (interpolated) and slow evolver

# %% [markdown]
# We've seen some difference in performance between the fast and slow evolver. This is trying to explore the origin.

# %% [markdown]
# #### Set up imports

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
# Set up some required code/functions

# %%
def NRMSE(Y,Y_pred):
    var = np.var(Y)
    return np.sqrt(np.square(Y_pred-Y).mean()/var)

def NRMSE_list(y,y_pred):
    Y = np.array(y)
    Y_pred = np.array(y_pred)
    return(NRMSE(Y,Y_pred))


# %% [markdown]
# Initalise both transforms from one reservoir and then run the two models one after another without any restart.

# %%
import spnc_ml as ml

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
spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime,restart=False)
transforms = spn.gen_signal_slow_delayed_feedback
transformf = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# %% [markdown]
# Transform from new reservoir in between runs (no restart)

# %%
import spnc_ml as ml

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


spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime,restart=False)
transforms = spn.gen_signal_slow_delayed_feedback

# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

spac = 50
NRMSE_list(y_test_f[spac:],y_pred_f[spac:])


spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime,restart=False)
transformf = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

spac = 50
NRMSE_list(y_test_f[spac:],y_pred_f[spac:])

# %% [markdown]
# Look at some of the data

# %%
spac = 50
window = 100

plt.figure()
plt.plot(y_test_s[spac:spac+window])
plt.plot(y_pred_s[spac:spac+window])

plt.figure()
plt.plot(y_test_f[spac:spac+window])
plt.plot(y_pred_f[spac:spac+window])

plt.figure()
plt.plot(y_pred_s[spac:spac+window])
plt.plot(y_pred_f[spac:spac+window])

plt.figure()
plt.plot(y_pred_s[spac:spac+window]-y_pred_f[spac:spac+window])

print(NRMSE_list(y_test_s[spac:],y_pred_s[spac:]))
print(NRMSE_list(y_test_f[spac:],y_pred_f[spac:]))

# %% [markdown]
# ##### Repeat above, but with restart on...

# %% [markdown]
# Initalise before

# %%
import spnc_ml as ml

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
spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime,restart=True)
transforms = spn.gen_signal_slow_delayed_feedback
transformf = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# %% [markdown]
# Initalise between

# %%
import spnc_ml as ml

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


spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime,restart=True)
transforms = spn.gen_signal_slow_delayed_feedback

# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

spac = 50
NRMSE_list(y_test_f[spac:],y_pred_f[spac:])


spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime,restart=True)
transformf = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

spac = 50
NRMSE_list(y_test_f[spac:],y_pred_f[spac:])

# %% [markdown]
# ##### More virtual nodes

# %% [markdown]
# Initalise before, use restart

# %%
import spnc_ml as ml

# NARMA parameters
Ntrain = 2000
Ntest = 1000

# Net Parameters
Nvirt = 400
m0 = 1e-2
bias = True

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
params = {'theta': 1/3,'gamma' : .25,'delay_feedback' : 0,'Nvirt' : Nvirt}
spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime,restart=True)
transforms = spn.gen_signal_slow_delayed_feedback
transformf = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# %% [markdown]
# Check NRMSE with spacer

# %%
spac = 50
window = 100

plt.figure()
plt.plot(y_test_s[spac:spac+window])
plt.plot(y_pred_s[spac:spac+window])

plt.figure()
plt.plot(y_test_f[spac:spac+window])
plt.plot(y_pred_f[spac:spac+window])

plt.figure()
plt.plot(y_pred_s[spac:spac+window])
plt.plot(y_pred_f[spac:spac+window])

plt.figure()
plt.plot(y_pred_s[spac:spac+window]-y_pred_f[spac:spac+window])

print(NRMSE_list(y_test_s[spac:],y_pred_s[spac:]))
print(NRMSE_list(y_test_f[spac:],y_pred_f[spac:]))

# %% [markdown]
# #### MORE STUFF TO SORT

# %%
import numpy as np
from spnc import spnc_anisotropy
import matplotlib.pyplot as plt
from spnc import calculate_energy_barriers

def NRMSE(Y,Y_pred):
    var = np.var(Y)
    return np.sqrt(np.square(Y_pred-Y).mean()/var)

def NRMSE_list(y,y_pred):
    Y = np.array(y)
    Y_pred = np.array(y_pred)
    return(NRMSE(Y,Y_pred))

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10

k_s_mag = 0.005

k_s = np.linspace(-k_s_mag,k_s_mag,100)
sp = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
sp.interdensity = 100
sp.minirestart()
plt.plot(k_s,sp.f_p1_eq(k_s))
p1s = np.zeros(np.shape(k_s))
for i,k in enumerate(k_s):
    sp.k_s = k
    calculate_energy_barriers(sp)
    sp.p1 = sp.get_p1_eq()
    sp.p2 = sp.get_p2_eq()
    p1s[i] = sp.p1
plt.plot(k_s,p1s)
plt.figure()
plt.plot(k_s,sp.f_p1_eq(k_s)-p1s)


k_s = (np.random.random(500)-0.5)*k_s_mag

params = {'theta': 1/3,'gamma' : .25,'delay_feedback' : 0,'Nvirt' : 400}

sps = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
ms = sps.gen_signal_slow_delayed_feedback(k_s,params)

spf = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
mf = spf.gen_signal_fast_delayed_feedback(k_s,params)

plt.figure()
plt.plot(ms[-500:])
plt.plot(mf[-500:])
plt.figure()
plt.plot(mf[-500:]-ms[-500:])
plt.figure()
plt.plot((mf[-500:]-ms[-500:])/np.std(ms[-500:]))

plt.figure()
plt.plot(ms[:500])
plt.plot(mf[:500])
plt.figure()
plt.plot(mf[:500]-ms[:500])
plt.figure()
plt.plot((mf[:500]-ms[:500])/np.std(ms[:500]))

print('NRMSE between fast and slow : ',NRMSE_list(mf,ms))

# %%
import spnc_ml as ml

# NARMA parameters
Ntrain = 200
Ntest = 100

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


spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transforms = spn.gen_signal_slow_delayed_feedback



# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)


spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transformf = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# %%
import spnc_ml as ml

# NARMA parameters
Ntrain = 200
Ntest = 100
spacer = 0

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

def get_ress(h=0.4,theta_H=90,k_s_0=0,phi=45,beta_prime=20):
    res = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
    transform = res.gen_signal_slow_delayed_feedback

    return res, transform

def get_resf(h=0.4,theta_H=90,k_s_0=0,phi=45,beta_prime=20):
    res = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
    transform = res.gen_signal_fast_delayed_feedback

    return res, transform

spns, transforms = get_ress(h,theta_H,k_s_0,phi,beta_prime)

spnf, transformf = get_resf(h,theta_H,k_s_0,phi,beta_prime)

# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True,spacer_NRMSE = spacer)

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True,spacer_NRMSE = spacer)

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True,spacer_NRMSE = spacer)



# %%

k_s = (np.random.random(50)-0.5)*k_s_mag

params = {'theta': 1/3,'gamma' : .25,'delay_feedback' : 0,'Nvirt' : 400}

spns, transforms = get_ress(h,theta_H,k_s_0,phi,beta_prime)

spnf, transformf = get_resf(h,theta_H,k_s_0,phi,beta_prime)

ms = transforms(k_s,params)

mf = transformf(k_s,params)

plt.figure()
plt.plot(ms[-500:])
plt.plot(mf[-500:])
plt.figure()
plt.plot(mf[-500:]-ms[-500:])
plt.figure()
plt.plot((mf[-500:]-ms[-500:])/np.std(ms[-500:]))

plt.figure()
plt.plot(ms[:50])
plt.plot(mf[:50])
plt.figure()
plt.plot(mf[:50]-ms[:50])
plt.figure()
plt.plot((mf[:50]-ms[:50])/np.std(ms[:50]))

print('NRMSE between fast and slow : ',NRMSE_list(mf,ms))

# %% [markdown]
# ## Do the ML in line

# %% [markdown]
# Imports

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

# %%

#tuple of Path variables
searchpaths = (Path.home() / 'repos', Path.home() / 'Desktop' / 'Stage_3A' / 'ML_library')
#tuple of repos
repos = ('machine_learning_library',)


# Add local modules and paths to local repos
from deterministic_mask import fixed_seed_mask, max_sequences_mask
import repo_tools
repo_tools.repos_path_finder(searchpaths, repos)
from single_node_res import single_node_reservoir
import ridge_regression as RR
from linear_layer import *
from mask import binary_mask
from utility import *
from NARMA10 import NARMA10
from datasets.load_TI46_digits import *
import datasets.load_TI46 as TI46
from sklearn.metrics import classification_report


# %% [markdown]
# Parameters and reservoirs

# %%
# NARMA parameters
Ntrain = 20
Ntest = 10

# Net Parameters
Nvirt = 4
m0 = 1e-2
bias = True

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
params = {'theta': 1/3,'gamma' : .25,'delay_feedback' : 0,'Nvirt' : Nvirt}

def get_ress(h=0.4,theta_H=90,k_s_0=0,phi=45,beta_prime=20):
    res = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
    transform = res.gen_signal_slow_delayed_feedback

    return res, transform

def get_resf(h=0.4,theta_H=90,k_s_0=0,phi=45,beta_prime=20):
    res = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
    transform = res.gen_signal_fast_delayed_feedback

    return res, transform

spns, transforms = get_ress(h,theta_H,k_s_0,phi,beta_prime)

spnf, transformf = get_resf(h,theta_H,k_s_0,phi,beta_prime)

seed_NARMA=1234
seed = seed_NARMA
seed_mask = seed_NARMA
seed_training =seed
fixed_mask=True
return_outputs=True




# %%
def NRMSE(Y,Y_pred):
    var = np.var(Y)
    return np.sqrt(np.square(Y_pred-Y).mean()/var)

def NRMSE_list(y,y_pred):
    Y = np.array(y)
    Y_pred = np.array(y_pred)
    return(NRMSE(Y,Y_pred))

k_s = (np.random.random(50)-0.5)*m0

print(params)

ms = transforms(k_s,params)

mf = transformf(k_s,params)

plt.figure()
plt.plot(ms)
plt.plot(mf)
plt.figure()
plt.plot(mf-ms)

print('NRMSE between fast and slow : ',NRMSE_list(mf,ms))

# %% [markdown]
# Narma data and net parameters

# %%
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

# %% [markdown]
# Make nets and SNRs

# %%
snrs = single_node_reservoir(Nin, Nout, Nvirt, m0, res = transforms)
nets = linear(Nin, Nout, bias = bias)

snrf = single_node_reservoir(Nin, Nout, Nvirt, m0, res = transformf)
netf = linear(Nin, Nout, bias = bias)

if fixed_mask==True:
    print("Deterministic mask will be used")
    if seed_mask>=0:
        print(seed_mask)
        snrs.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        snrf.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
    else:
        print("Max_sequences mask will be used")
        snr.M = max_sequences_mask(Nin, Nvirt, m0)

# %% [markdown]
# Training

# %%
# Training
S_trains, J_trains = snrs.transform(x_train,params)
np.size(S_trains)
RR.Kfold_train(nets,S_trains,y_train,10, quiet = False, seed_training=seed_training)

# # Training
# S_trainf, J_trainf = snrf.transform(x_train,params)
# np.size(S_trainf)
# RR.Kfold_train(netf,S_trainf,y_train,10, quiet = False, seed_training=seed_training)


# %% [markdown]
# Just testing f training - something changes when the transform is run twice.
# But, this is not a change in the output from the base reservoir...

# %%
# Training
S_trainf, J_trainf = snrf.transform(x_train,params)

np.size(S_trainf)

print(S_trainf)
S_trainf_1d = np.expand_dims(np.ravel(S_trainf, order='C'), axis = -1)
plt.figure()
plt.plot(S_trainf_1d)

# Training
S_trainf, J_trainf = snrf.transform(x_train,params)
np.size(S_trainf)

print(S_trainf)
S_trainf_1d = np.expand_dims(np.ravel(S_trainf, order='C'), axis = -1)
plt.plot(S_trainf_1d)


# %%
RR.Kfold_train(netf,S_trainf,y_train,10, quiet = False, seed_training=seed_training)

# %% [markdown]
# Check training is the same

# %%
S_trains_1d = np.expand_dims(np.ravel(S_trains, order='C'), axis = -1)
S_trainf_1d = np.expand_dims(np.ravel(S_trainf, order='C'), axis = -1)

start = 500
window = 50
end = start + window
plt.plot(S_trains_1d[start:end])
plt.plot(S_trainf_1d[start:end])
plt.figure()
plt.plot(S_trains_1d[start:end]-S_trainf_1d[start:end])

train_preds = nets.forward(S_trains)
train_predf = netf.forward(S_trainf)

plt.figure()
plt.plot(train_preds)
plt.plot(train_predf)
plt.figure()
plt.plot(train_preds-train_predf)

print(nets.W[0:10])
print(netf.W[0:10])

# %% [markdown]
# Set spacer

# %%
spacer = 10

# %% [markdown]
# Testing

# %%
print("Spacer NRMSE:"+str(spacer))

 # Testing
S_tests, J_tests = snrs.transform(x_test,params)
S_testf, J_testf = snrf.transform(x_test,params)

preds = nets.forward(S_tests)
predf = netf.forward(S_testf)

predNRMSEs = NRMSE(preds, y_test, spacer=spacer)
print( predNRMSEs)

predNRMSEf = NRMSE(predf, y_test, spacer=spacer)
print( predNRMSEf)

plt.figure()
plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0), 'k--')
plt.plot(y_test[spacer:], preds[spacer:], 'o')

plt.figure()
plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0), 'k--')
plt.plot(y_test[spacer:], predf[spacer:], 'o')

# %% [markdown]
# Compare to running in loaded code

# %%
# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True,spacer_NRMSE = spacer)

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True,spacer_NRMSE = spacer)

# %% [markdown]
# ### From Chen

# %%
import spnc_ml as ml

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
spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime,restart=True)
transforms = spn.gen_signal_slow_delayed_feedback
transformf = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test_s,y_pred_s)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transforms, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

# DO IT
(y_test_f,y_pred_f)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transformf, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

print(NRMSE_list(y_test_s[spac:],y_pred_s[spac:]))
print(NRMSE_list(y_test_f[spac:],y_pred_f[spac:]))

# %%
spac = 50
print(NRMSE_list(y_test_s[spac:],y_pred_s[spac:]))
print(NRMSE_list(y_test_f[spac:],y_pred_f[spac:]))
