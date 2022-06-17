'''
A file for running tests on the class

'''


'''
spnc.py : testing for bugs when calling multiple reservoirs or chaning temps
          in spnc_anisotropy
Dependancies
------------
'''
# %% startup params and imports


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#sort out relative paths
import sys
from pathlib import Path
repodir = Path('..').resolve()
try:
    sys.path.index(str(repodir))
except ValueError:
    sys.path.append(str(repodir))

# local imports
from SPNC import spnc

outputdir = 'testing_output/'

# %% Two reservoir ops
'''
Two reservoir ops:
There is an issue where calling two versions of a reservoir does not produce entierly seperate instances.

This code so far shows that the rates are independant for different reservoirs
'''

print('Testing use of multiple reservoirs')

# %% some shared parameters

ks = np.linspace(-0.4,0.4,1000)

res1 = spnc.spnc_anisotropy(0.4, 90, 0, 45, 10,f0=1.4e9)
res2 = spnc.spnc_anisotropy(0.4, 90, 0, 45, 20,f0=1.4e9)
res3 = spnc.spnc_anisotropy(0.4, 90, 0, 45, 10,f0=1.4e9)
resnointerp = spnc.spnc_anisotropy(0.4, 90, 0, 45, 20,f0=1.4e9,compute_interpolation=False)

# %% rates


def rates(h,theta_H,k_s,phi,beta_prime,f0,*args,**kwargs):

    ks=np.linspace(-0.4,0.4,1000)

    res = spnc.spnc_anisotropy(h,theta_H,k_s,phi,beta_prime,f0=f0)

    res_p1s = res.f_p1_eq(ks)
    res_oms = res.f_om_tot(ks)

    res_om21s = res_p1s*res_oms
    res_om12s = res_oms - res_om21s
    
    return res_om21s,res_om12s



baserateres1 = res1.f_om_tot(0)
baserateres2 = res2.f_om_tot(0)

print('relative base rates between reservoirs = ', baserateres1/baserateres2)

#plotting rates between reservoirs

res1_p1s = res1.f_p1_eq(ks)
res1_oms = res1.f_om_tot(ks)

res1_om21s = res1_p1s*res1_oms
res1_om12s = res1_oms - res1_om21s

res2_p1s = res2.f_p1_eq(ks)
res2_oms = res2.f_om_tot(ks)

res2_om21s = res2_p1s*res2_oms
res2_om12s = res2_oms - res2_om21s


# fnres1_om21s, fnres1_om12s = rates(0.4, 90, 0, 45, 10,1.4e9)
# fnres2_om21s, fnres2_om12s = rates(0.4, 90, 0, 45, 20,1.4e9)

print('rates vs k applied with for reservoirs with different beta prime:')
fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight])
plt.plot(ks,res1_om21s,color='C0')
plt.plot(ks,res1_om12s,color='C0')
plt.plot(ks,res2_om21s,color='C1',linestyle='dotted')
plt.plot(ks,res2_om12s,color='C1',linestyle='dotted')
# plt.plot(ks,fnres2_om21s,color='C1',linestyle='dashed')
# plt.plot(ks,fnres2_om12s,color='C1',linestyle='dashed')

#plt.xlim(-0.45,0.45)
# plt.legend()
plt.ylabel(r'rate ($1/\tau$) / ns$^{-1} $',fontsize=fsz)
plt.xlabel(r'$K_{\sigma}/K$',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig(outputdir+'rates.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()


# %% energy barriers
#Get some random input (seeded so always the same)
rng = np.random.default_rng(12345)
inputs=rng.random(1000)
theta = 0.4
gamma = 0.132
delay_feedback = 0
Nvirt = 100
params = {'theta': theta, 'gamma' : gamma,'delay_feedback' : delay_feedback,'Nvirt' : Nvirt}
res1_outputs = res1.gen_signal_fast_delayed_feedback(inputs,params)
res2_outputs = res2.gen_signal_fast_delayed_feedback(inputs,params)
res3_outputs = res3.gen_signal_fast_delayed_feedback(inputs,params)

print('signal transformation for reservoirs with different beta prime:')
fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight])
plt.plot(res1_outputs,color='C0')
plt.plot(res2_outputs,color='C1',linestyle='dashed')
plt.plot(res3_outputs,color='C2',linestyle='dotted')
# plt.legend()
plt.ylabel(r'Outputs',fontsize=fsz)
plt.xlabel(r'Index',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig(outputdir+'input-transform.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

# %%
