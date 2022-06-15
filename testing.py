'''
A file for running tests on the class

'''


'''
spnc.py : testing for bugs when calling multiple reservoirs or chaning temps
          in spnc_anisotropy
Dependancies
------------


'''
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#sort out relative paths
import sys
from pathlib import Path
repodir = Path('../..').resolve()
print(repodir)
try:
    sys.path.index(str(repodir))
except ValueError:
    sys.path.append(str(repodir))

print(sys.path)
# local imports
from SPNC import spnc

res1 = spnc.spnc_anisotropy(0.4, 90, 0, 45, 10,f0=1.4e9)
res2 = spnc.spnc_anisotropy(0.4, 90, 0, 45, 20,f0=1.4e9)

baserateres1 = res1.f_om_tot(0)
baserateres2 = res2.f_om_tot(0)

print('relative rates = ', baserateres1/baserateres2)
