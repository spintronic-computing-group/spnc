"""
Superparamgnetic neuromorphic computing module

Functions
---------
...

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

    def __init__(self):
        self.test = []


    # Reduced magnetisation for two state system evolving with constant field
    # note: I am using the notation, w12: transition rate from state 1, to state 2
    # w12, w21 must be constant over the tstep!

    # General rate equation
    def rate(self,f0,ebarrier,temp):

        w = f0*np.exp(-ebarrier/(constants.k*temp))

        return w

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

    def magnetisation_sw(self,beta_prime,h_prime,m0,t_prime_step):

        w21 = self.rate_sw(beta_prime,h_prime,-1)
        w12 = self.rate_sw(beta_prime,h_prime,+1)

        return self.magnetisation(w21,w12,m0,t_prime_step)

    def transform_sw(self,h_primes,beta_prime,t_prime):

        time = np.zeros(h_primes.shape[0]+1)
        mag = np.zeros(h_primes.shape[0]+1)
        time[0] = 0
        mag[0] = 0

        for i in range(len(time)-1):

            time[i+1] =  time[i] + t_prime
            mag[i+1] = self.magnetisation_sw(beta_prime,h_primes[i],mag[i],t_prime)

        return time, mag

""" Testing code below """

basic = spnc_basic()
basic.transform_sw(np.array([0.4,0.5,0.6]),3,10)
