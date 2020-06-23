# Run TI46 spoken digit task for the SPNC basic class

import spnc_ml as ml
from spnc import spnc_basic

# spoken digit variables
speakers = ['f1', 'f2', 'f3', 'f4', 'f5'] # blank for all
# speakers = None # None for all

# Resevoir parameters
params = {'theta': 0.2,'beta_prime' : 3}
basic = spnc_basic()
transform = basic.transform_sw

# net parameters
Nvirt = 100
m0 = 1
bias = True

# DO IT
ml.spnc_spoken_digits(speakers,Nvirt,m0,bias,transform,params)
