# Run narma10 for the SPNC basic class

import spnc_ml as ml
from spnc import spnc_anisotropy
import pandas as pd

# NARMA parameters
Ntrain = 2000
Ntest = 1000

# Net Parameters
Nvirt = 400
m0 = 2.7e-2
bias = True

# Resevoir parameters
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 20
params = {'theta': 10,'gamma' : .132,'delay_feedback' : 1,'Nvirt' : Nvirt}
spn = spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)
transform = spn.gen_signal_fast_delayed_feedback

# DO IT
(y_test,y_pred)=ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transform, params, seed_NARMA=1234,fixed_mask=True, return_outputs=True)

y_test_df = pd.DataFrame(y_test)
y_pred_df = pd.DataFrame(y_pred)

y_test_df.to_csv('../csv_files/y_test.csv')
y_pred_df.to_csv('../csv_files/y_pred.csv')