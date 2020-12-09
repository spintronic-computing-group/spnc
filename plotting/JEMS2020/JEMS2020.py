#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 21:07:27 2020

@author: Alexander

When this code was written, these repos were used:
SPNC : v0.2.0
machine_learning_library : v0.1.2
"""

# Standard libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Sort out relative paths
import sys
from pathlib import Path
repodir = Path('../../..').resolve()
try:
    sys.path.index(str(repodir))
except ValueError:
    sys.path.append(str(repodir))

# local imports    
from SPNC import spnc

"""
Plotting response curves
"""
basic = spnc.spnc_basic()
h_primes = np.array([-0.20652659,  0.8896459,   0.55040415, -0.53700334,
                     0.47482908, -0.25440813, -0.19123128, -0.75899033,
                     -0.55614648,  0.04739548])
fields, thetas, times, mags = basic.plotter_sw(
    3,h_primes,0,0.2,100)

fsz =16

fig_res_response, ax = plt.subplots(dpi=200)
ax.plot(thetas,mags, color = 'C0')
ax.set_ylabel('Normalised response', color = 'C0',fontsize=fsz)
ax.tick_params(axis='y', labelsize=fsz*0.6 )
ax.set_ylim([-1.1,1.1])

axr = ax.twinx()
axr.plot(thetas,fields, color = 'C1')
axr.set_ylabel('Normalised input', color = 'C1',fontsize=fsz)
axr.tick_params(axis='y',labelsize=fsz*0.6 )
axr.set_ylim([-1.1,1.1])

#ax.set_title('Resevoir response to input')
ax.set_xlabel('Time / Base Time',fontsize=fsz)
ax.tick_params(axis='x', labelsize=fsz*0.6 )

plt.show()

"""
Plotting energy curves
"""

# Basic energy for 1D model with H along the anisotropy axis (per unit volume)
def basic_energy(theta,Ks_prime,alpha,h_prime,gamma):
    energy = (np.power(np.sin(theta*np.pi/180),2) + 
              Ks_prime*np.power(np.sin((theta-alpha)*np.pi/180),2) 
              - 2*h_prime * np.cos((theta-gamma)*np.pi/180)
             )
    return energy

figure, ax = plt.subplots(dpi=200)
'''
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
'''

ax.set_xlabel(r'Magnetisation angle / deg',fontsize=fsz)
ax.set_ylabel('Energy / AU',fontsize=fsz)

lower = 120
upper = 180 + (180-lower)
theta = np.arange(-lower,upper,1)
ax.set_xlim([-lower, upper,])

K = 0.5
Ks = 0
alpha = 0
muHMs = 0.3
gamma = 0

ax.plot(theta, basic_energy(theta,0.,45,0.3,90),'--',
        color='black', label = 'Input off',alpha = 0.5)
ax.plot(theta, basic_energy(theta,0.5,45,0.3,90),
        color='C3', label = 'Tensile strain',alpha = 0.8)
ax.plot(theta, basic_energy(theta,-0.5,45,0.3,90),
        color='C0', label = 'Compressive strain',alpha = 0.8)
ax.legend()
ax.tick_params(axis='x', labelsize=fsz*0.6 )
ax.tick_params(axis='y', labelsize=fsz*0.6 )
plt.show()