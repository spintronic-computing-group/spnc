import spnc_ml as ml
from spnc import spnc_anisotropy
import numpy as np
import matplotlib.pyplot as plt

# NARMA parameters
Ntrain = 2000
Ntest = 1000

# Net Parameters
Nvirt = 256
m0 = .1
bias = True

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
params = {'theta': 10,'gamma' : .7,'delay_feedback' : 1,'Nvirt' : Nvirt}
spn = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transform = spn.gen_signal_fast_delayed_feedback_wo_SPN

#Difficult Task
y_test_w,pred_w = ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transform, params, seed_NARMA=1534, fixed_mask=True, return_NRMSE=True,return_outputs=True)
#Easy Task
y_test_b,pred_b = ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transform, params, seed_NARMA=1541, fixed_mask=True, return_NRMSE=True,return_outputs=True)

plt.figure(figsize=(10,6),dpi=200)
plt.plot(np.array(y_test_w)-np.array(pred_w))
plt.xlabel("Index of the output")
plt.ylabel("Distance between real and predicted outputs")
plt.show()

plt.figure(figsize=(10,6),dpi=200)
plt.plot(np.array(y_test_b)-np.array(pred_b))
plt.xlabel("Index of the output")
plt.ylabel("Distance between real and predicted outputs")
plt.show()