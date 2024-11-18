# -*- coding: utf-8 -*-
"""
Superparamgnetic neuromorphic computing module

Functions
---------
rate(f0,ebarrier,temperature)
    Finds the transition rate based on an energy barrier relative to KbT

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
from scipy.signal import argrelextrema
from scipy import interpolate
import copy


class spnc_basic:
    """
    Simulate basic system with field along anisotropy axis

    Functions
    ---------
    ...

    """

#    def __init__(self,res_type = 'sw'):
#        self.res_tpye = res_type


    # Reduced magnetisation for two state system evolving with constant field
    # note: I am using the notation, w12: transition rate from state 1, to state 2
    # w12, w21 must be constant over the tstep!


    def magnetisation(self,w21,w12,m0,tstep):

        wdiff = w21 - w12
        w = w12 + w21
        wratio = wdiff/w
        m = wratio + (m0 - wratio)*np.exp(-w*tstep)

        return m

    # Stoner-wolfarth rate
        # plusminus = -1 gives rate 21, = +1 gives rate 12
    def rate_sw(self,beta_prime,h_prime,minusplus):

        w = np.exp( -beta_prime * np.power( (1 + minusplus * h_prime) ,2) )

        return w

    def evolve_sw(self,beta_prime,h_prime,m0,t_prime_step):

        w21 = self.rate_sw(beta_prime,h_prime,-1)
        w12 = self.rate_sw(beta_prime,h_prime,+1)

        return self.magnetisation(w21,w12,m0,t_prime_step)


    def plotter_sw(self,beta_prime,h_primes,m0,theta,plotpoints):

        baserate = (self.rate_sw(beta_prime, 0, 1) +
                    self.rate_sw(beta_prime, 0, -1))
        timestep = theta/baserate

        loopthetas = np.linspace(0,theta,plotpoints)
        looptimes = np.linspace(0,timestep,plotpoints)

        fields = np.array([0])
        thetas = np.array([0])
        times = np.array([0])
        mags = np.array([m0])

        for idx, h_prime in enumerate(h_primes):

            loopmags = self.evolve_sw(beta_prime,h_prime,mags[-1],looptimes)

            loopfields = np.full(plotpoints,h_prime)

            fields = np.concatenate([fields,loopfields],axis=0)
            thetas = np.concatenate([thetas, loopthetas+(theta*idx)],axis=0)
            times = np.concatenate([times,looptimes+(timestep*idx)],axis=0)
            mags = np.concatenate([mags,loopmags],axis=0)


        return fields, thetas, times, mags

    def transform_sw(self,h_primes,params,*args,**kwargs):

        # No feedback implemented yet!
        theta = params['theta']
        beta_prime = params['beta_prime']

        baserate = (self.rate_sw(beta_prime, 0, 1) +
                    self.rate_sw(beta_prime, 0, -1))
        t_prime = theta/baserate

        mag = np.zeros(h_primes.shape[0]+1)

        # How would you vectorize this?
        for idx, h_prime in enumerate(h_primes):

            mag[idx+1] = self.evolve_sw(beta_prime,h_prime, mag[idx],t_prime)

        return mag[1:]

# General rate equation
def rate(f0,ebarrier,temperature):
    """
    Finds the transition rate based on an energy barrier relative to KbT

    Parameters
    ---------
    f0 : float
        Attempt frequency
    ebarrier : float
        height of energy barrier
    temperature : float
        Temperature in kelvin
    """

    w = f0*np.exp(-ebarrier/(constants.k*temperature))

    return w

# ###########
# SP Network with a control on anisotropy
""" Externals function definitions for the spnc_anisotropy class

Functions
---------
k_tilde(spn)
    Finds the resultant anisotropy strength for given strain and intrinsic
     anisotropies (in a spnc_anisotropy object).

psi(spn)
    Finds the resultant anisotropy direction for given strain and intrinsic
    anisotropies (in a spnc_anisotropy object).

energy(spn,theta)
    Find the energy of the spnc_anisotropy object when the magnetisation is at
    an angle 'theta' (degrees).

calculate_energy_barriers(spn)
    Finds the energy barriers between states 'up' and 'down' by finding the
    extremas in the energy landscape. Contains some checks for if silly values are choosen.

functions_energy_barriers(spn,k_s_lim)
    Returns functions that interpolate the energy barrier parameters. This allows a speed up in performance.


"""

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
    k_s_list = np.linspace(-k_s_lim,k_s_lim,int(100*2*k_s_lim))
    #Make a copy of the network
    spn_copy = spnc_anisotropy(spn.h,spn.theta_H,spn.k_s,spn.phi,spn.beta_prime,compute_interpolation=False)
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
    f_theta_1 = interpolate.interp1d(k_s_list, Theta_1, fill_value="extrapolate")
    f_theta_2 = interpolate.interp1d(k_s_list, Theta_2, fill_value="extrapolate")
    f_e_12_small = interpolate.interp1d(k_s_list, E_12_small, fill_value="extrapolate")
    f_e_21_small = interpolate.interp1d(k_s_list, E_21_small, fill_value="extrapolate")
    f_e_12_big = interpolate.interp1d(k_s_list, E_12_big, fill_value="extrapolate")
    f_e_21_big = interpolate.interp1d(k_s_list, E_21_big, fill_value="extrapolate")

    return(f_theta_1,f_theta_2,f_e_12_small,f_e_21_small,f_e_12_big,f_e_21_big)

class spnc_anisotropy:
    """
    Simulate a SP netwotk with a control on anisotropy

    Bugs
    ----
    Nasty : If the user updates h, theta_H, phi or beta_prime after initalisation and is using the compute_interpolation=True (default), then the interpolation is not recalculated!

    Functions
    ---------
    __init__(self,h,theta_H,k_s,phi,beta_prime,k_s_lim=1.,compute_interpolation=True,f0=1e10)
        initialises the object.
    ...

    """

    def __init__(self,h,theta_H,k_s,phi,beta_prime,k_s_lim=1.,compute_interpolation=True,f0=1e10,**kwargs):
        # Meta parameters
        self.interdensity = kwargs.get('interdensity',100)
        self.restart = kwargs.get('restart',True)
        self.Primep1 = kwargs.get('Primep1', None)             
        # initialize = kwargs.get('initialize', False) 
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
        self.f0 = f0
        self.p1 = self.get_p1_eq()
        self.p2 = self.get_p2_eq()
        #Interpolations to fasten the code
        if compute_interpolation:
            (self.f_theta_1,self.f_theta_2,self.f_e_12_small,self.f_e_21_small,self.f_e_12_big,self.f_e_21_big) = functions_energy_barriers(self,k_s_lim)
            (self.f_p1_eq,self.f_om_tot) = self.calculate_f_p1_om(k_s_lim)

        self.k_s_lim = k_s_lim
        self.compute_interpolation = compute_interpolation
    
        # save the initial values
        # self._initial_state = copy.deepcopy(self.__dict__)
        # Initialize
        # if initialize:
            # self.initialize()

        # Initialisation
    # def initialize(self):

        # print("Initializing...")
        # print("Current state before initializing:", self.__dict__['p1'])
        # print("Initial state p1:", self._initial_state['p1'])
        
        # 1. save a copy of the initial state
        # initial_state_copy = copy.deepcopy(self._initial_state)

        # print("Initial state:", initial_state_copy)
        
        # 2. update the current state with the initial state
        # self.__dict__.update(initial_state_copy)
        
        # print("After initializing - current dict p1:", self.__dict__['p1'])
        # print("After initializing - initial state p1:", self._initial_state['p1'])
        # print('finished initializing..')

    def minirestart(self,k_s_lim=1.,compute_interpolation=True,f0=1e10):
        #Parameters
        self.k_s = 0

        #Computed
        self.e_12_small = np.nan
        self.e_21_small = np.nan
        self.e_12_big = np.nan
        self.e_21_big = np.nan
        self.theta_1 = np.nan
        self.theta_2 = np.nan

        #Dynamic
        calculate_energy_barriers(self)
        self.f0 = f0
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
        k_s_list = np.linspace(-k_s_lim,k_s_lim,int(100*2*k_s_lim))
        Om_12 = np.exp(-self.f_e_12_small(k_s_list)*self.beta_prime)+np.exp(-self.f_e_12_big(k_s_list)*self.beta_prime)
        Om_21 = np.exp(-self.f_e_21_small(k_s_list)*self.beta_prime)+np.exp(-self.f_e_21_big(k_s_list)*self.beta_prime)
        Om_tot = Om_12+Om_21
        P1_eq = Om_21/Om_tot

        #Interpolation
        f_p1_eq = interpolate.interp1d(k_s_list, P1_eq, fill_value="extrapolate")
        f_om_tot = interpolate.interp1d(k_s_list, Om_tot, fill_value="extrapolate")

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

    def get_f_m_eq(self):
        f_m = lambda x: self.f_p1_eq(x)*np.cos(self.f_theta_1(x)*np.pi/180)+(1-self.f_p1_eq(x))*np.cos(self.f_theta_2(x)*np.pi/180)
        return(f_m)

    '''
    add the noise function here

    here, the len(input) is chosen as the metrics for judging the phase of machine learning, and to decide if adding noise will be carried out

    17/11/24 by chen

    develop a new judge function to determine the phase of machine learning

    '''
    '''
    18/11/24 by chen

    add a new function to print out all parameters after changing the p1

    '''


    def gen_signal_fast_delayed_feedback(self, K_s,params, *args,**kwargs):

        # determine the phase of machine learning
        train_samples = params.get('train_sample', 2000)
        test_samples = params.get('test_sample', 1000)

        phase = 'train' if len(K_s) == train_samples else 'test'
        print('current phase:', phase)

        if phase == 'train':
            if self.Primep1 is not None:
                self.p1 = self.Primep1
            self.p2 = 1 - self.p1

            print('p1 in train & fast:', self.p1)

            print('==================== parameters after changing p1 ====================')

            calculate_energy_barriers(self)
            
            if self.compute_interpolation:
                (self.f_theta_1,self.f_theta_2,self.f_e_12_small,self.f_e_21_small,self.f_e_12_big,self.f_e_21_big) = functions_energy_barriers(self,self.k_s_lim)
                (self.f_p1_eq,self.f_om_tot) = self.calculate_f_p1_om(self.k_s_lim)



            import scipy.interpolate
            import matplotlib.pyplot as plt

            interpolations = []

            for attr, value in vars(self).items():
                if isinstance(value, scipy.interpolate.interpolate.interp1d):
                    # collect interpolation functions
                    x_data = value.x
                    y_data = value.y
                    interpolations.append((attr, x_data, y_data))
                elif isinstance(value, np.ndarray):
                    print(f"{attr}: shape: {value.shape}")
                    print(f"  value: {value}")
                elif callable(value):
                    print(f"{attr}: callable")
                else:
                    print(f"{attr}: {value}")

            # plot interpolation functions
            for attr, x_data, y_data in interpolations:
                plt.figure()
                plt.plot(x_data, y_data, label=f"{attr}")
                plt.title(f"{attr} interpolation")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.legend()
                plt.show()

            print('==================== parameters after changing p1 ====================')

            theta_T = params['theta']
            self.k_s = 0
            T = 1./(self.get_omega_prime()*self.f0)
            gamma = params['gamma']
            delay_fb = params['delay_feedback']
            Nvirt = params['Nvirt']

            # noise parameters

            noise_enable = params.get('noise_enable', 'none')
            noise_std = params.get('noise_std', 0.0)
        
            theta = theta_T*T

            N = K_s.shape[0]
            mag = np.zeros(N)

            # determine if the noise will be added

            add_noise = (noise_enable == 'both') or (noise_enable == 'train')

            print('noisy training output') if add_noise else print('noise-free training output')

            for idx, j in enumerate(K_s):
                self.k_s = j + gamma*mag[(idx-Nvirt-delay_fb)%N] #Delayed Feedback
                self.evolve_fast(self.f0,theta)
                mag[idx] = self.get_m_fast()

                if add_noise:
                    mag[idx] += np.random.normal(0.0001, noise_std)

            # if initialize:
            #     self.initialize()
            #     print('initialized')
            # else:
            #     print('skip initialization')

            if self.restart:
                self.minirestart()
                print('restarted')
            else:
                print('skip restarting..')

        else:
            print('p1 in test & fast:', self.p1)

            theta_T = params['theta']
            self.k_s = 0
            T = 1./(self.get_omega_prime()*self.f0)
            gamma = params['gamma']
            delay_fb = params['delay_feedback']
            Nvirt = params['Nvirt']

            # noise parameters

            noise_enable = params.get('noise_enable', 'none')
            noise_std = params.get('noise_std', 0.0)
        
            theta = theta_T*T

            N = K_s.shape[0]
            mag = np.zeros(N)

            # determine if the noise will be added

            add_noise = (noise_enable == 'both') or (noise_enable == 'test')

            print('noisy testing output') if add_noise else print('noise-free testing output')

            for idx, j in enumerate(K_s):
                self.k_s = j + gamma*mag[(idx-Nvirt-delay_fb)%N] #Delayed Feedback
                self.evolve_fast(self.f0,theta)
                mag[idx] = self.get_m_fast()

                if add_noise:
                    mag[idx] += np.random.normal(0.0001, noise_std)

            # if initialize:
            #     self.initialize()
            #     print('initialized')
            # else:
            #     print('skip initialization')

            if self.restart:
                self.minirestart()
                print('restarted')
            else:
                print('skip restarting..')
        
        return mag
    
    def gen_signal_slow_delayed_feedback(self, K_s, params, *args,**kwargs):  

        # determine the phase of machine learning
        train_samples = params.get('train_sample', 2000)
        test_samples = params.get('test_sample', 1000)

        phase = 'train' if len(K_s) == train_samples else 'test'
        print('current phase:', phase)

        if phase == 'train':
            if self.Primep1 is not None:
                self.p1 = self.Primep1
            self.p2 = 1 - self.p1

            print('p1 in train & slow:', self.p1)

            theta_T = params['theta']
            self.k_s = 0
            T = 1./(self.get_omega_prime()*self.f0)
            gamma = params['gamma']
            delay_fb = params['delay_feedback']
            Nvirt = params['Nvirt']

            # noise parameters
            noise_enable = params.get('noise_enable', 'none')
            noise_seed = params.get('noise_seed', None)
            print('noise_seed:', noise_seed)
            rng = np.random.default_rng(noise_seed)
            noise_mean = params.get('noise_mean', 0.0001)
            print('noise_mean:', noise_mean)
            noise_std = params.get('noise_std', 0.0)

            theta = theta_T*T

            N = K_s.shape[0]
            mag = np.zeros(N)

            # determine if the noise will be added

            add_noise = (noise_enable == 'both') or (noise_enable == 'train')
            print('noisy training output') if add_noise else print('noise-free training output')

            for idx, j in enumerate(K_s):
                self.k_s = j + gamma*mag[(idx-Nvirt-delay_fb)%N] #Delayed Feedback
                calculate_energy_barriers(self)
                self.evolve(self.f0,theta) # update the p1 and p2
                if add_noise:
                    mag[idx] = self.get_m() + rng.normal(noise_mean, noise_std,1)
                else:
                    mag[idx] = self.get_m() # depends on the updated p1, p2, theta_1, theta_2

            # if initialize:
            #     self.initialize()
            #     print('initialized')
            # else:
            #     print('skip initializing..')

            if self.restart:
                self.minirestart()
                print('restarted')
            else:
                print('skip restarting..')

        else:
            print('p1 in test & slow:', self.p1)

            theta_T = params['theta']
            self.k_s = 0
            T = 1./(self.get_omega_prime()*self.f0)
            gamma = params['gamma']
            delay_fb = params['delay_feedback']
            Nvirt = params['Nvirt']

            # noise parameters
            noise_enable = params.get('noise_enable', 'none')
            noise_seed = params.get('noise_seed', None)
            print('noise_seed:', noise_seed)
            rng = np.random.default_rng(noise_seed)
            noise_mean = params.get('noise_mean', 0.0001)
            print('noise_mean:', noise_mean)
            noise_std = params.get('noise_std', 0.0)

            theta = theta_T*T

            N = K_s.shape[0]
            mag = np.zeros(N)

            # determine if the noise will be added

            add_noise = (noise_enable == 'both') or (noise_enable == 'test')
            print('noisy testing output') if add_noise else print('noise-free testing output')

            for idx, j in enumerate(K_s):
                self.k_s = j + gamma*mag[(idx-Nvirt-delay_fb)%N] #Delayed Feedback
                calculate_energy_barriers(self)
                self.evolve(self.f0,theta) # update the p1 and p2
                if add_noise:
                    mag[idx] = self.get_m() + rng.normal(noise_mean, noise_std,1)
                else:
                    mag[idx] = self.get_m()

            # if initialize:
            #     self.initialize()
            #     print('initialized')
            # else:
            #     print('skip initializing..')
            
            if self.restart:
                self.minirestart()
                print('restarted')
            else:
                print('skip restarting..')

        return mag
        
    
    def gen_trace_fast_delayed_feedback(self,klist,theta,density,params,*args,**kwargs):

        theta_step = theta/density
        K_s_expanded = np.zeros(np.size(klist)*density)
        thetas = np.zeros(np.size(K_s_expanded))
        idx = 0
        for k in klist:
            for i in range(density):
                K_s_expanded[idx] = k
                thetas[idx] = (idx+1)*theta_step
                idx = idx +1

        params['theta'] = theta_step

        mags = self.gen_signal_fast_delayed_feedback(K_s_expanded, params)

        K_s = np.concatenate([np.array([0]),K_s_expanded],axis=0)
        thetas = np.concatenate([np.array([0]),thetas],axis=0)
        mags = np.concatenate([np.array([0]),mags],axis=0)

        return K_s, thetas, mags #fields, thetas, times, mags

    # def gen_trace_fast_delayed_feedback(self,K_s,density,params,*args,**kwargs):
    #     theta_T = params['theta']
    #     gamma = params['gamma']
    #     delay_fb = params['delay_feedback']
    #     Nvirt = params['Nvirt']
    #
    #     self.k_s = 0
    #     T = 1./(self.get_omega_prime()*self.f0)
    #     timestep = theta*T
    #
    #     loopthetas = np.linspace(0,theta,plotpoints)
    #     looptimes = np.linspace(0,timestep,plotpoints)
    #
    #     fields = np.array([0])
    #     thetas = np.array([0])
    #     times = np.array([0])
    #     mags = np.array([m0])
    #
    #     for idx, h_prime in enumerate(h_primes):
    #
    #         loopmags = self.evolve_sw(beta_prime,h_prime,mags[-1],looptimes)
    #
    #         loopfields = np.full(plotpoints,h_prime)
    #
    #         fields = np.concatenate([fields,loopfields],axis=0)
    #         thetas = np.concatenate([thetas, loopthetas+(theta*idx)],axis=0)
    #         times = np.concatenate([times,looptimes+(timestep*idx)],axis=0)
    #         mags = np.concatenate([mags,loopmags],axis=0)
    #
    #
    #     return fields, thetas, times, mags

    def gen_signal_fast_delayed_feedback_wo_SPN(self, K_s,params,*args,**kwargs):
        gamma = params['gamma']
        delay_fb = params['delay_feedback']
        Nvirt = params['Nvirt']

        N = K_s.shape[0]
        mag = np.zeros(N)

        f = np.tanh
        #f = lambda x: x

        for idx, j in enumerate(K_s):
            mag[idx] = f(j + gamma*mag[(idx-Nvirt-delay_fb)%N])

        return mag
