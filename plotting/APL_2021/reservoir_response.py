#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 08:55:36 2021

@author: Alexander

Local Dependancies
------------------
machine_learning_library  : v0.1.2
    This repository will need to be on your path in order to work.
    This is achieved with repo_tools module and a path find function
    Add to the searchpath and repos tuples if required


"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Sort out relative paths
import sys
from pathlib import Path
repodir = Path('../../..').resolve()
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



'''
Mumax and analytical rates
'''

# Add in rate data
kapplied = np.array([-0.4,-0.2,-0.1,0,0.1,0.2,0.4])
tauup = np.array([483,161,102,76,54,39,29])
tauuperror = np.array([70,14,8,6,4,4,4])
taudown = np.array([32,41,51,72,109,179,593])
taudownerror = np.array([5,4,4,6,9,17,94])

basetime = np.mean([76,72])
basetimeerror = np.sqrt(2*(6**2))
baserate = 1/basetime
baserateerror = basetimeerror/(np.square(basetime))

print('Mumax switching time vs k kapplied:')
plt.errorbar(kapplied,tauup,yerr=tauuperror,label = 'upwards')
plt.errorbar(kapplied,taudown,yerr=taudownerror,label = 'downwards')
plt.show()

print('Mumax rates vs k kapplied:')
plt.errorbar(kapplied,1/tauup,yerr=tauuperror/(np.square(tauup)), label = 'upwards')
plt.errorbar(kapplied,1/taudown,yerr=taudownerror/(np.square(taudown)),label = 'downwards')
plt.show()



res = spnc.spnc_anisotropy(0.4, 90, 0, 45, 10,f0=1.4e9)

baseratestoner = res.f_om_tot(0)/2
ratecorrection = baserate/baseratestoner
print('Attempt frequency to match mumax = ', ratecorrection,'e9')
#ratecorrection = 1e-9*1.4e9

ks = np.linspace(-0.4,0.4,1000)

p1s = res.f_p1_eq(ks)
oms = res.f_om_tot(ks)

om21s = p1s*oms
om12s = oms - om21s

print('analytical and mumax rates vs k applied with base rate matched:')
fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.errorbar(kapplied,1/tauup,yerr=tauuperror/(np.square(tauup)),
             marker = '.', linestyle = 'none', label = 'Left to right')
plt.errorbar(kapplied,1/taudown,yerr=taudownerror/(np.square(taudown)),
             marker = '.', linestyle = 'none',label = 'Right to left')
plt.plot(ks,om21s*ratecorrection,color='C0')
plt.plot(ks,om12s*ratecorrection,color='C1')
#plt.xlim(-0.45,0.45)
plt.legend()
plt.ylabel(r'rate ($1/\tau$) / ns$^{-1} $',fontsize=fsz)
plt.xlabel(r'$K_{\sigma}/K$',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'rates.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()


ks = np.linspace(-1,1,1000)

p1s = res.f_p1_eq(ks)
oms = res.f_om_tot(ks)

om21s = p1s*oms
om12s = oms - om21s

def eqiberror(tauup,taudown,tauuperror,taudownerror):
    return np.sqrt( ( (-2*taudown*tauuperror**2)/(tauup+taudown)**2 ) + ( (2*tauup*taudownerror**2)/(tauup+taudown)**2 ) )

plt.plot(ks,(om21s-om12s)/oms)
plt.scatter(kapplied,(1/tauup - 1/taudown)/(1/tauup+1/taudown))
#plt.errorbar(kapplied,(1/tauup - 1/taudown)/(1/tauup+1/taudown),yerr = eqiberror(tauup,taudown,tauuperror,taudownerror))


'''
Magnetisation plots
'''

def getmags(ks,beta):
    res = spnc.spnc_anisotropy(0.4, 90, 0, 45, beta,f0=1.4e9)


    p1s = res.f_p1_eq(ks)
    p2s = 1 - p1s
    #oms = res.f_om_tot(ks)

    #om21s = p1s*oms
    #om12s = oms - om21s

    #equibs = (om21s-om12s)/oms

    mags = np.cos(res.f_theta_1(ks)*np.pi/180)*p1s + np.cos(res.f_theta_2(ks)*np.pi/180)*p2s
    return mags

betas = np.array([10,20,30,50])
ks = np.linspace(-1,1,100)
colors = [plt.cm.viridis_r(x) for x in (betas-min(betas))/max(betas)]#np.linspace(0,1,len(betas))]

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
figureheight = figurewidth*figureaspect
# plt.plot(ks,eqibs)
for i, color in enumerate(colors):
    plt.plot(ks,getmags(ks,betas[i]),label=r'$KV/k_BT = $' + str(betas[i]),color = color)

#plt.xlim(-0.45,0.45)
plt.legend(fontsize=fsz*0.9)
plt.xlabel(r'$K_{\sigma} / K$',fontsize=fsz)
plt.ylabel(r'$m_x$',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'eq_mags.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()


'''
Timescale plots
'''

def getoms(ks,beta,f0):
    res = spnc.spnc_anisotropy(0.4, 90, 0, 45, beta,f0=f0)


    p1s = res.f_p1_eq(ks)
    p2s = 1 - p1s
    oms = res.f_om_tot(ks)*f0

    return oms

betas = np.array([10,20,30,50])
ks = np.linspace(-1,1,100)
f0 = 0.495e9
colors = [plt.cm.viridis_r(x) for x in (betas-min(betas))/max(betas)]#np.linspace(0,1,len(betas))]

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
figureheight = figurewidth*figureaspect
# plt.plot(ks,eqibs)
for i, color in enumerate(colors):
    plt.plot(ks,1/getoms(ks,betas[i],f0),label=r'$KV/k_BT = $' + str(betas[i]),color = color)

#plt.plot(ks,1/getoms(ks,20.198,f0),'--',color = 'k',alpha = 0.8) #20.198 for 1%, 21.818 for 10%
#plt.plot(ks,1/getoms(ks,19.798,f0),'--',color = 'k',alpha = 0.8) #1% -ve
#plt.xlim(-0.45,0.45)
plt.legend(fontsize=fsz*0.8)
plt.xlabel(r'$K_{\sigma} / K$',fontsize=fsz)
plt.ylabel(r'Internal timescale / s',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.yscale('log')
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'timescales.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

for beta in betas:
    print('KV/k_BT = ', beta, '   =>    tau0 =', 1/getoms(0,beta,f0), ' s')

'''
Timescale plot - timescale vs beta
'''

betas = np.linspace(10,50,100)
k = 0
f0 = 0.495e9

def getom0(beta,f0):
    taus = np.zeros(np.size(betas))
    res = spnc.spnc_anisotropy(0.4,90,0,45,beta,compute_interpolation=False)
    om0 = res.get_omega_prime()*f0
    return om0

taus = np.zeros(np.size(betas))
for i, beta in enumerate(betas):
    taus[i] = 1/getom0(beta,f0)

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
figureheight = figurewidth*figureaspect
plt.plot(betas,taus)
#plt.plot(betas,1/(f0*2*np.exp(-0.36*betas) + f0*2*np.exp(-1.96*betas)))
#plt.xlim(-0.45,0.45)
#plt.legend(fontsize=fsz*0.9)
plt.xlabel(r'Intrinsic anisotropy ($KV$)/ Thermal energy ($k_BT$)',fontsize=fsz)
plt.ylabel(r'Base timescale / s',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.yscale('log')
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'base_timescale.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

'''
Reservoir response plot
'''

# Reservoir behaviour plotting
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10

theta = 1.2
klist = np.array([0.4, 0.4,0.4,0.4,
                    -0.20652659,  0.8896459,   0.55040415, -0.53700334,
                     0.47482908, -0.25440813, -0.19123128, -0.75899033,
                     0.75614648,  0.54, 0.9,
                     0,0,0,0])
density = 100
# delayed feedback = 0 : feedback from the same node at the last time step
# delayed feedback = 1 : feedback from the previous node at the last time step
# gamma = 0 : no feedback
params = {'gamma' : 0,'delay_feedback' : 0,'Nvirt' : 400}

spn = spnc.spnc_anisotropy(h,theta_H,k_s_0,phi,beta_prime)

K_s, thetas, mags = spn.gen_trace_fast_delayed_feedback(klist, theta, density, params)

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 2/3
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.plot(thetas,K_s,linewidth=1,alpha=1,linestyle='--',label='input')
plt.plot(thetas,mags,linewidth=1,label='output')
plt.xlim(0,np.size(klist)*theta)
plt.hlines(0,0,np.size(klist)*theta,linestyles='dashed',linewidth=0.7,alpha=0.5)
plt.xlabel(r'Time / Base Time',fontsize=fsz)
plt.ylabel(r'Normalised input/output',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.legend(fontsize=fsz*0.8)
#plt.yscale('log')
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'reservoir_response.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()


# Reservoir properties for anisotropy axis plot
def getanisotropy(k_s,  h = 0.4, theta_H = 90, phi = 45, beta_prime = 10):

    spn = spnc.spnc_anisotropy(h, theta_H, k_s, phi, beta_prime, compute_interpolation=False)
    k_tilde = spnc.k_tilde(spn)
    psi = spnc.psi(spn)

    return k_tilde, psi

k_s = np.linspace(-5,5,100)
k_tildes = np.zeros(np.size(k_s))
psis = np.zeros(np.size(k_s))
for idx, k in enumerate(k_s):
    k_tildes[idx], psis[idx] = getanisotropy(k)

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.plot(k_s,psis)
plt.xlabel(r'$K_{\sigma} / K$',fontsize=fsz)
plt.ylabel(r'$\psi$ / deg',fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.legend(fontsize=fsz*0.9)
plt.savefig('output/'+'ea-rotation.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.plot(k_s,k_tildes)
plt.xlabel(r'$K_{\sigma} / K$',fontsize=fsz)
plt.ylabel(r'$\tilde{k}$ ',fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.legend(fontsize=fsz*0.9)
plt.savefig('output/'+'ea-magnitude.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()




def get_energy(theta, k_s,  h = 0.4, theta_H = 90, phi = 45, beta_prime = 10):
    spn = spnc.spnc_anisotropy(h, theta_H, k_s, phi, beta_prime, compute_interpolation=False)
    return spnc.energy(spn,theta)

k = 0.2
lower = 90
upper = 180 + (180-lower)
theta = np.linspace(-lower,upper,1000)

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.plot(theta, get_energy(theta,0.),'--',
        color='black', label = 'Input off',alpha = 0.5)
plt.plot(theta, get_energy(theta,k),
        color='C3', label = r'Tensile strain, ${K_{\sigma}}/{K} = 0.2 $',alpha = 0.8)
plt.plot(theta, get_energy(theta,-k),
        color='C0', label = r'Compressive strain, ${K_{\sigma}}/{K} = -0.2$',alpha = 0.8)
plt.xlabel(r'Magnetisation angle / deg',fontsize=fsz)
plt.ylabel('Energy / KV',fontsize=fsz)
plt.xlim([-lower, upper,])
plt.xticks(np.arange(-lower+30,upper,60), fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.legend(fontsize=fsz*0.7)
plt.savefig('output/'+'energy_plots.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()
