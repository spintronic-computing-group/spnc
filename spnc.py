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


    def transform_sw(self,h_primes,params,*args,**kwargs):

        # No feedback implemented yet!
        theta = params['theta']
        beta_prime = params['beta_prime']

        # Is this the right way round???
        baserate = self.rate_sw(beta_prime,0, 1)
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

""" Testing code below """

params ={'theta': 0.2,'beta_prime' : 3}
basic = spnc_basic()
transform = basic.transform_sw
x = np.array([0.4,0.5,0.6,-0.5,-.1,0.5,-.4])
x2 = np.random.rand(1000)
transform(x2,params)
