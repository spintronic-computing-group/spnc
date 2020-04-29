# %% [markdown]
# # SPNC development and testing

# %% [markdown]
# *This is a notebook for testing and development of ideas*

# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

# %% [markdown]
# ## Define our magnetic system

# %%
# Reduced magnetisation for two state system evolving with constant field 
# note: I am using the notation, w12: transition rate from state 1, to state 2
# w12, w21 must be constant over the tstep!
def SPNC_magnetisation_two_state(w21,w12,m0,tstep):

    wdiff = w21 - w12
    w = w12 + w21
    wratio = wdiff/w
    m = wratio + (m0 - wratio)*np.exp(-w*tstep)

    return m

# %%
# General rate equation
def SPNC_rate(f0,ebarrier,temp):

    w = f0*np.exp(-ebarrier/(constants.k*temp))

    return w

# %%
# Stoner-wolfarth rate
    # plusminus = -1 gives rate 21, = +1 gives rate 12 
def SPNC_rate_sw(beta_prime,h_prime,minusplus):
    
    w = np.exp( -beta_prime * np.power( (1 + minusplus * h_prime) ,2) )
    
    return w


# %%
def SPNC_magnetisation_sw(beta_prime,h_prime,m0,t_prime_step):
    
    w21 = SPNC_rate_sw(beta_prime,h_prime,-1)
    w12 = SPNC_rate_sw(beta_prime,h_prime,+1)
    
    return SPNC_magnetisation_two_state(w21,w12,m0,t_prime_step)


# %%
def SPNC_mag_evolver_sw(beta_prime,h_primes,t_prime):
    
    time = np.zeros(h_primes.shape[0]+1)
    mag = np.zeros(h_primes.shape[0]+1)
    time[0] = 0
    mag[0] = 0
    
    for i in range(len(time)-1):
        
        time[i+1] =  time[i] + t_prime
        mag[i+1] = SPNC_magnetisation_sw(beta_prime,h_primes[i],mag[i],t_prime)
        
    return time, mag    


# %% [markdown]
# ## Some testing

# %% [markdown]
# ### Excitation from zero m

# %%
time_prime = np.arange(0,50,0.1)

plt.plot(time_prime,SPNC_magnetisation_sw(3,0.5,0,time_prime))
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Single field long time: h' = 0.5")
plt.show()

# %%
plt.plot(time_prime,SPNC_magnetisation_sw(3,0.1,0,time_prime))
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Single field long time: h' = 0.1")
plt.show()

# %% [markdown]
# #### Long time excitation

# %%
h_primes = np.arange(-1,1,0.01)

plt.plot(h_primes,SPNC_magnetisation_sw(3,h_primes,0,1000))
plt.plot(h_primes,SPNC_magnetisation_sw(3,h_primes,0,5))
plt.plot(h_primes,np.tanh(2*3*h_primes),'k--')
plt.xlabel("h'")
plt.ylabel('m')
plt.legend(['Long time','Short time',"tanh(2b'h')"])
plt.title("magnetisation as a function of field at long and short times: t'short = 5, t'long = 1000")
plt.show()

# %% [markdown]
# We can see that the output field matches exactly the expected tanh function when in the limit of long time (dotted is tanh, blue is long time, orange is short time)

# %% [markdown]
# ### Decay from m

# %%
time_prime = np.arange(0,50,0.1)

plt.plot(time_prime,SPNC_magnetisation_sw(3,0,1,time_prime))
plt.xlabel("t'")
plt.ylabel('m')
plt.title("No field long time decay: m(0) = 1")
plt.show()

# %%
plt.plot(time_prime,SPNC_magnetisation_sw(3,0,0.5,time_prime))
plt.xlabel("t'")
plt.ylabel('m')
plt.title("No field long time decay: m(0) = 0.5")
plt.show()

# %% [markdown]
# The same decay length is seen regardless of the starting m(0) provided the applied field (here zero) is the same

# %% [markdown]
# #### Decay with field

# %%
plt.plot(time_prime,SPNC_magnetisation_sw(3,0.05,0.5,time_prime))
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Long time decay under field: m(0) = 0.5, h' = 0.05")
plt.show()

# %% [markdown]
# #### Excitation from starting m(0)

# %%
plt.plot(time_prime,SPNC_magnetisation_sw(3,0.1,0.5,time_prime))
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Long time excitation under field: m(0) = 0.5, h' = 0.1")
plt.show()

# %% [markdown]
# We see here that changing from $h' = 0.05$ to $h' = 0.1$ results in a change from decay down to a new value to exitation up to a new one

# %% [markdown]
# ### Successive field inputs (memory and non-linearity)

# %% [markdown]
# Here we use time_prime as a time step between inputs. A combination of beta and the time step sets the "memory"

# %% [markdown]
# #### First let's examine the case where $0 \le h' \le 1$

# %%
h_primes = np.random.rand(100)

plt.plot(h_primes)
plt.xlabel('index')
plt.ylabel("h'")
plt.title("random applied field sequence")
plt.show()

i = 0
h_avgs = []
while i < len(h_primes) - 11:
    this_window = h_primes[i:i+10]
    window_avg = sum(this_window)/10
    h_avgs.append(window_avg)
    i+=1
    
plt.plot(h_avgs)
plt.xlabel('index')
plt.ylabel("h' moving avg")
plt.title("moving avg of applied field sequence: window 10")
plt.show()

# %%
time, mag = SPNC_mag_evolver_sw(3,h_primes,0.2)
plt.plot(time,mag)
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Response to successive fields: h' is 100 random no's between 0 and 1; t'step = 0.2; b' = 3")
plt.show()

# %%
time, mag = SPNC_mag_evolver_sw(3,h_primes,10)
plt.plot(time,mag)
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Response to successive fields: h' is 100 random no's between 0 and 1; t'step = 10; b' = 3")
plt.show()

# %%
time, mag = SPNC_mag_evolver_sw(3,h_primes,0.01)
plt.plot(time,mag)
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Response to successive fields: h' is 100 random no's between 0 and 1; t'step = 0.01; b' = 3")
plt.show()

# %% [markdown]
# We can see that we run into problems of saturating our system, so let us change our bounds on $h'$

# %% [markdown]
# ####  $-1 \le h' \le 1$

# %%
h_primes = 2 * np.random.rand(100) - 1

plt.plot(h_primes)
plt.xlabel('index')
plt.ylabel("h'")
plt.title("random applied field sequence")
plt.show()

i = 0
h_avgs = []
while i < len(h_primes) - 11:
    this_window = h_primes[i:i+10]
    window_avg = sum(this_window)/10
    h_avgs.append(window_avg)
    i+=1
    
plt.plot(h_avgs)
plt.xlabel('index')
plt.ylabel("h' moving avg")
plt.title("moving avg of applied field sequence: window 10")
plt.show()

# %%
time, mag = SPNC_mag_evolver_sw(3,h_primes,0.2)
plt.plot(time,mag)
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Response to successive fields: h' is 100 random no's between -1 and 1; t'step = 0.2; b' = 3")
plt.show()

# %%
time, mag = SPNC_mag_evolver_sw(3,h_primes,10)
plt.plot(time,mag)
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Response to successive fields: h' is 100 random no's between -1 and 1; t'step = 10; b' = 3")
plt.show()

# %%
time, mag = SPNC_mag_evolver_sw(3,h_primes,0.01)
plt.plot(time,mag)
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Response to successive fields: h' is 100 random no's between -1 and 1; t'step = 0.01; b' = 3")
plt.show()

# %%
