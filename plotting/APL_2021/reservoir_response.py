#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 08:55:36 2021

@author: Alexander
"""
import numpy as np
import matplotlib.pyplot as plt

# Sort out relative paths
import sys
from pathlib import Path
repodir = Path('../../../..').resolve()
try:
    sys.path.index(str(repodir))
except ValueError:
    sys.path.append(str(repodir))

# local imports    
from SPNC import spnc

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

plt.errorbar(kapplied,tauup,yerr=tauuperror,label = 'upwards')
plt.errorbar(kapplied,taudown,yerr=taudownerror,label = 'downwards')
plt.show()

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



fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.errorbar(kapplied,1/tauup,yerr=tauuperror/(np.square(tauup)),
             marker = '.', linestyle = 'none', label = 'Down to up')
plt.errorbar(kapplied,1/taudown,yerr=taudownerror/(np.square(taudown)),
             marker = '.', linestyle = 'none',label = 'Up to down')
plt.plot(ks,om21s*ratecorrection,color='C0')
plt.plot(ks,om12s*ratecorrection,color='C1')
#plt.xlim(-0.45,0.45)
plt.legend()
plt.ylabel(r'rate / ns$^{-1} $',fontsize=fsz)
plt.xlabel("Applied anisotropy / KV",fontsize=fsz)
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

plt.plot(ks,(om21s-om12s)/oms)
plt.scatter(kapplied,(1/tauup - 1/taudown)/(1/tauup+1/taudown))




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
    plt.plot(ks,getmags(ks,betas[i]),label=r'$KV/K_BT = $' + str(betas[i]),color = color)

#plt.xlim(-0.45,0.45)
plt.legend(fontsize=fsz*0.9)
plt.xlabel(r'Applied anisotropy / KV',fontsize=fsz)
plt.ylabel(r'$m_x$',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'eq_mags.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()


def getoms(ks,beta,f0):
    res = spnc.spnc_anisotropy(0.4, 90, 0, 45, beta,f0=f0)
    
    
    p1s = res.f_p1_eq(ks)
    p2s = 1 - p1s
    oms = res.f_om_tot(ks)*f0
    
    return oms

betas = np.array([10,20,30,50])
ks = np.linspace(-0.4,0.4,100)
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
    plt.plot(ks,1/getoms(ks,betas[i],f0),label=r'$KV/K_BT = $' + str(betas[i]),color = color)

#plt.xlim(-0.45,0.45)
plt.legend(fontsize=fsz*0.9)
plt.xlabel(r'Applied anisotropy / KV',fontsize=fsz)
plt.ylabel(r'Internal timescale / s',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.yscale('log')
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'timescales.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

betas = np.array([10])
ks = np.linspace(-0.4,0.4,100)
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
    plt.plot(ks,1e9/getoms(ks,betas[i],f0),label=r'$KV/K_BT = $' + str(betas[i]),color = color)

#plt.xlim(-0.45,0.45)
plt.legend(fontsize=fsz*0.9)
plt.xlabel(r'Applied anisotropy / KV',fontsize=fsz)
plt.ylabel(r'Internal timescale / ns',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'timescales_mumax.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()