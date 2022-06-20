# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python (SPNC)
#     language: python
#     name: spnc
# ---

# %% [markdown]
# # SPNC plotting for advisory board - 2020/07/02

# %% [markdown]
# *This is a notebook is used for quick plotting of ideas* <br>
# *Running with updated code may break things...*

# %%
# Magic to choose matplotlib backend (used mostly)
%matplotlib widget

# Standard libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# %% [markdown]
# ## Plotting for advisory board

# %% [markdown]
# When this section was written, these repos were used: <br>
# SPNC : v0.2.0 <br>
# machine_learning_library : v0.1.2
#
# ---

# %%
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

# %%
# Remove previously open figure
if 'fig_res_basic_response' in locals():
    plt.close(fig_res_basic_response)

basic = spnc.spnc_basic()
h_primes = np.array([0.3, 0])
fields, thetas, times, mags = basic.plotter_sw(
    3,h_primes,0,7,100)

fig_res_basic_response, ax = plt.subplots()
ax.plot(thetas,mags, color = 'C0')
ax.set_ylabel('Normalised response', color = 'C0')
ax.tick_params(axis='y', labelcolor= 'C0')
ax.set_ylim([-0,1])

axr = ax.twinx()
axr.plot(thetas,fields, color = 'C1')
axr.set_ylabel('Normalised input', color = 'C1')
axr.tick_params(axis='y', labelcolor= 'C1')
axr.set_ylim([0,1])

#ax.set_title('Resevoir response to input')
ax.set_xlabel('Time / Base Time')

plt.show()

# %%
# Remove previously open figure
if 'fig_res_response' in locals():
    plt.close(fig_res_response)

basic = spnc.spnc_basic()
h_primes = np.array([-0.20652659,  0.8896459,   0.55040415, -0.53700334,
                     0.47482908, -0.25440813, -0.19123128, -0.75899033,
                     -0.55614648,  0.04739548])
fields, thetas, times, mags = basic.plotter_sw(
    3,h_primes,0,0.2,100)

fig_res_response, ax = plt.subplots()
ax.plot(thetas,mags, color = 'C0')
ax.set_ylabel('Normalised response', color = 'C0')
ax.tick_params(axis='y', labelcolor= 'C0')
ax.set_ylim([-1.1,1.1])

axr = ax.twinx()
axr.plot(thetas,fields, color = 'C1')
axr.set_ylabel('Normalised input', color = 'C1')
axr.tick_params(axis='y', labelcolor= 'C1')
axr.set_ylim([-1.1,1.1])

#ax.set_title('Resevoir response to input')
ax.set_xlabel('Time / Base Time')

plt.show()

# %%
# Remove previously open figure
if 'fig_res_response' in locals():
    plt.close(fig_res_response)

basic = spnc.spnc_basic()
h_primes = np.array([-0.20652659,  0.8896459,   0.55040415, -0.53700334,
                     0.47482908, -0.25440813, -0.19123128, -0.75899033,
                     -0.55614648,  0.04739548])
fields, thetas, times, mags = basic.plotter_sw(
    3,h_primes,0,1,100)

fig_res_response, ax = plt.subplots()
ax.plot(thetas,mags, color = 'C0')
ax.set_ylabel('Normalised response', color = 'C0')
ax.tick_params(axis='y', labelcolor= 'C0')
ax.set_ylim([-1.1,1.1])

axr = ax.twinx()
axr.plot(thetas,fields, color = 'C1')
axr.set_ylabel('Normalised input', color = 'C1')
axr.tick_params(axis='y', labelcolor= 'C1')
axr.set_ylim([-1.1,1.1])

#ax.set_title('Resevoir response to input')
ax.set_xlabel('Time / Base Time')

plt.show()

# %%
# Remove previously open figure
if 'figure' in locals():
    plt.close(figure)

# Basic energy for 1D model with H along the anisotropy axis (per unit volume)
def basic_energy(theta,Ks_prime,alpha,h_prime,gamma):
    energy = (np.power(np.sin(theta*np.pi/180),2) + 
              Ks_prime*np.power(np.sin((theta-alpha)*np.pi/180),2) 
              - 2*h_prime * np.cos((theta-gamma)*np.pi/180)
             )
    return energy

figure, ax = plt.subplots()
'''
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
'''
ax.set_title('Energy landscape for a single element')
ax.set_xlabel(r'$\theta$ / deg')
ax.set_ylabel('Energy / AU')

theta = np.arange(-180,180,1)
ax.set_xlim([-180, 180])

K = 0.5
Ks = 0
alpha = 0
muHMs = 0.3
gamma = 0

ax.plot(theta, basic_energy(theta,0,alpha,0,gamma), 
        color='C0', label = 'Input off')
plt.show()


# %% jupyter={"source_hidden": true}
# Remove previously open figure
if 'figure' in locals():
    plt.close(figure)

# Basic energy for 1D model with H along the anisotropy axis (per unit volume)
def basic_energy(theta,Ks_prime,alpha,h_prime,gamma):
    energy = (np.power(np.sin(theta*np.pi/180),2) + 
              Ks_prime*np.power(np.sin((theta-alpha)*np.pi/180),2) 
              - 2*h_prime * np.cos((theta-gamma)*np.pi/180)
             )
    return energy

figure, ax = plt.subplots()
'''
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
'''
ax.set_title('Changing landscape with field input')
ax.set_xlabel(r'$\theta$ / deg')
ax.set_ylabel('Energy / AU')

theta = np.arange(-180,180,1)
ax.set_xlim([-180, 180])

K = 0.5
Ks = 0
alpha = 0
muHMs = 0.3
gamma = 0

ax.plot(theta, basic_energy(theta,0,alpha,0,gamma), 
        color='C0',alpha = 0.3 , label = 'Input off')
ax.plot(theta, basic_energy(theta,0,alpha,0.3,gamma),
        color='C1', label = 'Input on')
ax.legend()
plt.show()

# %%
# Remove previously open figure
if 'figure' in locals():
    plt.close(figure)

# Basic energy for 1D model with H along the anisotropy axis (per unit volume)
def basic_energy(theta,Ks_prime,alpha,h_prime,gamma):
    energy = (np.power(np.sin(theta*np.pi/180),2) + 
              Ks_prime*np.power(np.sin((theta-alpha)*np.pi/180),2) 
              - 2*h_prime * np.cos((theta-gamma)*np.pi/180)
             )
    return energy

figure, ax = plt.subplots()
'''
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
'''
ax.set_title('Changing landscape with field rotation')
ax.set_xlabel(r'$\theta$ / deg')
ax.set_ylabel('Energy / AU')

theta = np.arange(-180,180,1)
ax.set_xlim([-180, 180])

K = 0.5
Ks = 0
alpha = 0
muHMs = 0.3
gamma = 0

ax.plot(theta, basic_energy(theta,0,alpha,0,gamma), 
        color='C0',alpha = 0.1 , label = 'No field')
ax.plot(theta, basic_energy(theta,0,alpha,0.3,gamma),
        color='C1',alpha = 0.4, label = 'Field on')
ax.plot(theta, basic_energy(theta,0,alpha,0.3,45),
        color='C2', label = 'Rotate field')
ax.legend()
plt.show()

# %%
# Remove previously open figure
if 'figure' in locals():
    plt.close(figure)

# Basic energy for 1D model with H along the anisotropy axis (per unit volume)
def basic_energy(theta,Ks_prime,alpha,h_prime,gamma):
    energy = (np.power(np.sin(theta*np.pi/180),2) + 
              Ks_prime*np.power(np.sin((theta-alpha)*np.pi/180),2) 
              - 2*h_prime * np.cos((theta-gamma)*np.pi/180)
             )
    return energy

figure, ax = plt.subplots()
'''
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
'''
ax.set_title('Energy landscape for zero voltage input')
ax.set_xlabel(r'$\theta$ / deg')
ax.set_ylabel('Energy / AU')

theta = np.arange(-180,180,1)
ax.set_xlim([-180, 180])

K = 0.5
Ks = 0
alpha = 0
muHMs = 0.3
gamma = 0

ax.plot(theta, basic_energy(theta,0.,45,0,0),
        color='C0', label = 'Input = 0.2',alpha = 0.3)
ax.plot(theta, basic_energy(theta,0.5,45,0,0),
        color='C1', label = 'Input = 0.2',alpha = 1)
plt.show()


# %%
# Remove previously open figure
if 'figure' in locals():
    plt.close(figure)

# Basic energy for 1D model with H along the anisotropy axis (per unit volume)
def basic_energy(theta,Ks_prime,alpha,h_prime,gamma):
    energy = (np.power(np.sin(theta*np.pi/180),2) + 
              Ks_prime*np.power(np.sin((theta-alpha)*np.pi/180),2) 
              - 2*h_prime * np.cos((theta-gamma)*np.pi/180)
             )
    return energy

figure, ax = plt.subplots()
'''
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
'''
ax.set_title('Energy landscape for zero voltage input')
ax.set_xlabel(r'$\theta$ / deg')
ax.set_ylabel('Energy / AU')

theta = np.arange(-180,180,1)
ax.set_xlim([-180, 180])

K = 0.5
Ks = 0
alpha = 0
muHMs = 0.3
gamma = 0

ax.plot(theta, basic_energy(theta,0.,45,0.3,90),
        color='C0', label = 'Input = 0.2',alpha = 1)
plt.show()


# %%
# Remove previously open figure
if 'figure' in locals():
    plt.close(figure)

# Basic energy for 1D model with H along the anisotropy axis (per unit volume)
def basic_energy(theta,Ks_prime,alpha,h_prime,gamma):
    energy = (np.power(np.sin(theta*np.pi/180),2) + 
              Ks_prime*np.power(np.sin((theta-alpha)*np.pi/180),2) 
              - 2*h_prime * np.cos((theta-gamma)*np.pi/180)
             )
    return energy

figure, ax = plt.subplots()
'''
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')
'''
ax.set_title('Energy change with voltage input')
ax.set_xlabel(r'$\theta$ / deg')
ax.set_ylabel('Energy / AU')

theta = np.arange(-180,180,1)
ax.set_xlim([-180, 180])

K = 0.5
Ks = 0
alpha = 0
muHMs = 0.3
gamma = 0

ax.plot(theta, basic_energy(theta,0.,45,0.3,90),
        color='C0', label = 'Input off',alpha = 0.3)
ax.plot(theta, basic_energy(theta,0.5,45,0.3,90),
        color='C1', label = 'Input on')
ax.legend()
plt.show()


# %%
