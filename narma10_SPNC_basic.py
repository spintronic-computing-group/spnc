# Run narma10 for the SPNC basic class

import spnc_ml as ml
from spnc import spnc_basic

# NARMA parameters
Ntrain = 10000
Ntest = 10000

# Resevoir parameters
params = {'theta': 0.2,'beta_prime' : 3}
basic = spnc_basic()
transform = basic.transform_sw

# Net Parameters
Nvirt = 100
m0 = 1
bias = False

# DO IT
ml.spnc_narma10(Ntrain, Ntest, Nvirt, m0, bias, transform, params)
