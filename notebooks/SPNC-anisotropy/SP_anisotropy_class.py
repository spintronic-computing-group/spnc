# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Superparamagnetic Network - Control of magnetization through anisotropy (class only)

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy import interpolate


# %%
#Dimensionless equations
def k_tilde(spn):
    return(np.sqrt((1+spn.k_s*np.cos(2*spn.phi*np.pi/180))**2+(spn.k_s*np.sin(2*spn.phi*np.pi/180))**2))

def psi(spn):
    return(180*np.arctan2(spn.k_s*np.sin(2*spn.phi*np.pi/180),(1+spn.k_s*np.cos(2*spn.phi*np.pi/180)))/2/np.pi)
        
def energy(spn,theta):
    return(k_tilde(spn)*np.sin((theta-psi(spn))*np.pi/180)**2-2*spn.h*np.cos((theta-spn.theta_H)*np.pi/180))

#Computation of energy barriers
def calculate_energy_barriers(spn):
    theta = np.linspace(-180,180,10000)
    E = energy(spn,theta)
    
    #Localization of extrema
    id_max = argrelextrema(E, np.greater)[0]
    id_min = argrelextrema(E, np.less)[0]
    ind1 = 0
    ind2 = 1
    
    #Two-states case
    if(len(id_max)==2 and len(id_min)==2):
        if (theta[id_min[0]]<(-90)):
            ind1 = 1
            ind2 = 0
        theta_1 = theta[id_min[ind1]]
        theta_2 = theta[id_min[ind2]]
        e_12_big = max((E[id_max[0]]-E[id_min[ind1]]),(E[id_max[1]]-E[id_min[ind1]]))
        e_21_big = max((E[id_max[0]]-E[id_min[ind2]]),(E[id_max[1]]-E[id_min[ind2]]))
        e_12_small = min((E[id_max[0]]-E[id_min[ind1]]),(E[id_max[1]]-E[id_min[ind1]]))
        e_21_small = min((E[id_max[0]]-E[id_min[ind2]]),(E[id_max[1]]-E[id_min[ind2]]))
    
    #One minimum in 180°
    elif(len(id_min)==1 and len(id_max)==2):
        theta_1 = theta[id_min[0]]
        theta_2 = 180
        e_12_big = max((E[id_max[0]]-E[id_min[0]]),(E[id_max[1]]-E[id_min[0]]))
        e_21_big = max((E[id_max[0]]-energy(spn,180)),(E[id_max[1]]-energy(spn,180)))
        e_12_small = min((E[id_max[0]]-E[id_min[0]]),(E[id_max[1]]-E[id_min[0]]))
        e_21_small = min((E[id_max[0]]-energy(spn,180)),(E[id_max[1]]-energy(spn,180)))
        
    #One maximum in 180°
    elif(len(id_min)==2 and len(id_max)==1):
        if (theta[id_min[0]]<(-90)):
            ind1 = 1
            ind2 = 0
        theta_1 = theta[id_min[ind1]]
        theta_2 = theta[id_min[ind2]]
        e_12_big = max((E[id_max[0]]-E[id_min[ind1]]),(energy(spn,180)-E[id_min[ind1]]))
        e_21_big = max((E[id_max[0]]-E[id_min[ind2]]),(energy(spn,180)-E[id_min[ind2]]))
        e_12_small = min((E[id_max[0]]-E[id_min[ind1]]),(energy(spn,180)-E[id_min[ind1]]))
        e_21_small = min((E[id_max[0]]-E[id_min[ind2]]),(energy(spn,180)-E[id_min[ind2]]))
    
    #There might be only one minimum. In this case put nans for all parameters
    else:
        (theta_1,theta_2,e_12_big,e_21_big,e_12_small,e_21_small) = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
    
    #Check the condition e_b_min*beta_prime>=3
    e_b_min = min(e_12_small,e_21_small)
    if(e_b_min*spn.beta_prime<=3):
        (theta_1,theta_2,e_12_big,e_21_big,e_12_small,e_21_small) = (np.nan,np.nan,np.nan,np.nan,np.nan,np.nan)
    
    spn.theta_1 = theta_1
    spn.theta_2 = theta_2
    spn.e_12_small = e_12_small
    spn.e_21_small = e_21_small
    spn.e_12_big = e_12_big
    spn.e_21_big = e_21_big
    
    return()

#Interpolations to avoid calculating too many times the energy landscape
def functions_energy_barriers(spn,k_s_lim):
    #Computation on a sample
    k_s_list = np.linspace(-k_s_lim,k_s_lim,int(50*2*k_s_lim))
    #Make a copy of the network
    spn_copy = SP_Network(spn.h,spn.theta_H,spn.k_s,spn.phi,spn.beta_prime,compute_interpolation=False)
    Theta_1 = []
    Theta_2 = []
    E_12_small = []
    E_21_small = []
    E_12_big = []
    E_21_big = []
    for k_s in k_s_list:
        spn_copy.k_s = k_s
        (theta_1,theta_2,e_12_small,e_21_small,e_12_big,e_21_big) = spn_copy.get_energy_barriers()
        Theta_1.append(theta_1)
        Theta_2.append(theta_2)
        E_12_small.append(e_12_small)
        E_21_small.append(e_21_small)
        E_12_big.append(e_12_big)
        E_21_big.append(e_21_big)
        
    #Interpolation
    f_theta_1 = interpolate.interp1d(k_s_list, Theta_1)
    f_theta_2 = interpolate.interp1d(k_s_list, Theta_2)
    f_e_12_small = interpolate.interp1d(k_s_list, E_12_small)
    f_e_21_small = interpolate.interp1d(k_s_list, E_21_small)
    f_e_12_big = interpolate.interp1d(k_s_list, E_12_big)
    f_e_21_big = interpolate.interp1d(k_s_list, E_21_big)
    
    return(f_theta_1,f_theta_2,f_e_12_small,f_e_21_small,f_e_12_big,f_e_21_big)


# %%
#We define a superparamagnetic network as a class
class SP_Network:
    def __init__(self,h,theta_H,k_s,phi,beta_prime,k_s_lim=.5,compute_interpolation=True):
        #Parameters
        self.h = h
        self.theta_H = theta_H
        self.k_s = k_s
        self.phi = phi
        self.beta_prime = beta_prime
        #Computed
        self.e_12_small = np.nan
        self.e_21_small = np.nan
        self.e_12_big = np.nan
        self.e_21_big = np.nan
        self.theta_1 = np.nan
        self.theta_2 = np.nan
        #Dynamic
        calculate_energy_barriers(self)
        self.p1 = self.get_p1_eq()
        self.p2 = self.get_p2_eq()
        #Interpolations to fasten the code
        if compute_interpolation:
            (self.f_theta_1,self.f_theta_2,self.f_e_12_small,self.f_e_21_small,self.f_e_12_big,self.f_e_21_big) = functions_energy_barriers(self,k_s_lim)
            (self.f_p1_eq,self.f_om_tot) = self.calculate_f_p1_om(k_s_lim)
    
    def get_energy_barriers(self):
        calculate_energy_barriers(self)
        return(self.theta_1,self.theta_2,self.e_12_small,self.e_21_small,self.e_12_big,self.e_21_big)
    
    def get_m_eq(self):
        c1 = np.cos(self.theta_1*np.pi/180)
        c2 = np.cos(self.theta_2*np.pi/180)
        p1 = self.get_p1_eq()
        p2 = self.get_p2_eq()
        return(c1*p1+c2*p2)
    
    def get_e_b_min(self):
        return(min(self.e_12_small,self.e_21_small))
    
    def get_omega_prime_12(self):
        return(np.exp(-self.e_12_small*self.beta_prime)+np.exp(-self.e_12_big*self.beta_prime))
    
    def get_omega_prime_21(self):
        return(np.exp(-self.e_21_small*self.beta_prime)+np.exp(-self.e_21_big*self.beta_prime))
    
    def get_omega_prime(self):
        return(self.get_omega_prime_12()+self.get_omega_prime_21())
    
    def get_p1_eq(self):
        return(self.get_omega_prime_21()/self.get_omega_prime())
    
    def get_p2_eq(self):
        return(self.get_omega_prime_12()/self.get_omega_prime())
    
    def calculate_f_p1_om(self,k_s_lim):
        #Computation on a sample
        k_s_list = np.linspace(-k_s_lim,k_s_lim,int(50*2*k_s_lim))
        Om_12 = np.exp(-self.f_e_12_small(k_s_list)*self.beta_prime)+np.exp(-self.f_e_12_big(k_s_list)*self.beta_prime)
        Om_21 = np.exp(-self.f_e_21_small(k_s_list)*self.beta_prime)+np.exp(-self.f_e_21_big(k_s_list)*self.beta_prime)
        Om_tot = Om_12+Om_21
        P1_eq = Om_21/Om_tot
        
        #Interpolation
        f_p1_eq = interpolate.interp1d(k_s_list, P1_eq)
        f_om_tot = interpolate.interp1d(k_s_list, Om_tot)
        
        return(f_p1_eq,f_om_tot)
    
    #Dynamic
    
    def evolve(self,f0,t_step):
        self.p1 = self.get_p1_eq() + (self.p1 - self.get_p1_eq())*np.exp(-f0*self.get_omega_prime()*t_step)
        self.p2 = self.get_p2_eq() + (self.p2 - self.get_p2_eq())*np.exp(-f0*self.get_omega_prime()*t_step)
        return()
    
    def get_m(self):
        c1 = np.cos(self.theta_1*np.pi/180)
        c2 = np.cos(self.theta_2*np.pi/180)
        return(c1*self.p1+c2*self.p2)
    
    def evolve_fast(self,f0,tstep):
        om_tot = self.f_om_tot(self.k_s)
        p1_eq = self.f_p1_eq(self.k_s)
        self.p1 = p1_eq + (self.p1 - p1_eq)*np.exp(-f0*om_tot*tstep)
        self.p2 = 1 - self.p1
        return()
    
    def get_m_fast(self):
        #self.theta_1 and self.theta_2 are not up-to-date anymore, we use the interpolation functions
        theta_1 = self.f_theta_1(self.k_s)
        theta_2 = self.f_theta_2(self.k_s)
        c1 = np.cos(theta_1*np.pi/180)
        c2 = np.cos(theta_2*np.pi/180)
        return(c1*self.p1+c2*self.p2)

# %%
