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

# %% [markdown]
# #### Re-looking at the positive case, but $0 \le h' \le 0.25$ and longer time base

# %%
h_primes = np.random.rand(1000)/4

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
plt.title("Response to successive fields: h' is 1000 random no's between 0 and 0.25; t'step = 0.2; b' = 3")
plt.show()

# %%
time, mag = SPNC_mag_evolver_sw(3,h_primes,10)
plt.plot(time,mag)
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Response to successive fields: h' is 1000 random no's between 0 and 0.25; t'step = 10; b' = 3")
plt.show()

# %%
time, mag = SPNC_mag_evolver_sw(3,h_primes,0.01)
plt.plot(time,mag)
plt.xlabel("t'")
plt.ylabel('m')
plt.title("Response to successive fields: h' is 1000 random no's between 0 and 0.25; t'step = 0.01; b' = 3")
plt.show()

# %% [markdown]
# We see that the behaviour looks kind of like it is intergrating the average, with noise from the deviation. The time scale sets how much the noise effects it and how long the averaging takes. Let's take a look at this by finding the expected result from the average input:

# %%
h_prime_avg = np.mean(h_primes)
print("h_prime average =", h_prime_avg)
total_time = 0.2 * 1000 #time step times number of points

time_prime = np.arange(0,total_time,0.1)
plt.plot(time_prime,SPNC_magnetisation_sw(3,h_prime_avg,0,time_prime))
plt.xlabel("t'")
plt.ylabel('m')
plt.title("m for: h' = mean(h_primes)")
plt.show()

# %% [markdown]
# Based on this, it looks a little more complicated that straight up averaging. Perhaps it is averaging over a window? 

# %% [markdown]
# ## Looking towards machine learning

# %% [markdown]
# *This is based on on Matt's code for a single node dynamical resivour (originly for the Mackey-Glass equations)*

# %% [markdown]
# ### First Matt's code:

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def Ridge_regression(S, Y, l):
    '''
    For a linear layer we can solve the weights by a direct method
    If the error function is the mean square error given by
        E = |Y - S * W |^2 + \lambda |W|^2
    where the L2 norm is being applied and the variables are
        Y = [Nsamples x Noutputs] is the desired output
        S = [Nsamples x Nweights] is the input signal
        W = [Nweights x Noutputs] is the weight matrix
    To minimise E we need to solve:
        S^T * S = (S^T * Y  + \lambda I) * W
        W = (S^T*S + \lambda I)^-1 * S^T * Y
    '''
    STS = np.matmul(S.T, S)
    STY = np.matmul(S.T, Y)
    Sdag = np.linalg.pinv(STS + l*np.eye(len(STS)))
    return np.matmul(Sdag, STY)


# %%
def MG_func(x, J, gamma, eta, p):
    return eta*(x + gamma*J) / (1 + np.power( x + gamma*J, p))



# %%
def MSE (pred, desired):
    return np.mean(np.square(np.subtract(pred,desired)))


# %%
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


# %%
def inv_sigmoid(y):
    return - np.log((1.0/y) - 1.0)


# %%
class Mackey_Glass_SNR:
    def __init__(self, Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None):
        '''
        Nin = input size
        Nvirt = number of virtual nodes
        Nout = output size
        m0 = magnitude of the mask values
        mask_sparse = sparsity factor for mask matrix
        bias = bool flag for using bias
        act = pass an activation function to use
        inv_act = function which applies the inverse of act
        '''
        self.Nin = Nin
        self.Nvirt = Nvirt
        self.Nout = Nout
        self.m0 = m0
        
        # Mask is random matrix of -m0 and m0
        # mask_sparse defines the sparsity level of the input mask
        # i.e 1.0 = full, 0.0 = empty
        self.M = 2*self.m0*(np.random.randint(0,2, (Nvirt,Nin))-0.5)
        #self.M *= 1.0*(np.random.random(size=(Nvirt, Nin)) <= mask_sparse)
        # Empty weight matrix 
        self.W = np.zeros( (Nvirt + int(bias), Nout))
        
        self.use_bias=bias
        
        # Activation and inverse activation functions
        self.f_act = act
        self.f_inv_act = inv_act
        
    def transform(self, u, params):
        '''
        Function to generate the reservoir signal from an input u
        params = dict for various parameters
        '''
        Ns = len(u)
        
        # Unflattens input if it is 1d
        u = u.reshape((Ns, self.Nin))
        
        J = np.zeros((Ns, self.Nvirt))
        
        # expands the signal to include a bias column is req'd
        if self.use_bias:
            S = np.ones((Ns, self.Nvirt+1))
        else:
            S = np.zeros((Ns, self.Nvirt))
        
        # theta = temporal node spacing
        theta = params['theta']
        
        # parameters for the MG function
        Sigma = np.exp(-theta)
        gamma = 0.005
        eta = 0.5
        P = 1
        
        J = np.matmul(u, self.M.T)
        for k in range(Ns):              
            S[k,0] = S[k-1, self.Nvirt-1] * Sigma + (1.0 - Sigma)*MG_func( S[k-1,0], J[k,0], gamma, eta, P)
            for i in range(1,self.Nvirt):
                S[k,i] = S[k,i-1] * Sigma + (1.0 - Sigma)*MG_func( S[k-1,i], J[k,i], gamma, eta, P)   
        return S
    
    def forward(self, S):
        if self.f_act is not None:
            return self.f_act(np.matmul(S, self.W))
        else:
            return np.matmul(S, self.W)
    
    def train(self, u_train, d_train, u_valid, d_valid, params):
        
        S_train = self.transform(u_train, params)
        S_valid = self.transform(u_valid, params)
                
        if self.f_inv_act is not None:
            inv_act_d_train = self.f_inv_act(d_train)
            inv_act_d_valid = self.f_inv_act(d_valid)
        else:
            inv_act_d_train = d_train
            inv_act_d_valid = d_valid
        
        # regularisation parameters to validate over
        lambdas = np.exp(np.linspace(-6,0,num=20))
        lambdas[0] = 0.0
        
        errs = np.zeros(lambdas.shape)
        for i,l in enumerate(lambdas):
            self.W = Ridge_regression(S_train, inv_act_d_train, l)
            valid_pred = self.forward(S_valid)
            errs[i] = MSE(valid_pred, d_valid)
            print(l, MSE(valid_pred, d_valid))
    
        lopt = lambdas[np.argmin(errs)]
        print('Optimal lambda = ', lopt, 'with MSE = ', np.min(errs))
        self.W = Ridge_regression(S_train, d_train, lopt)
        
        

# %%
def NARMA10(N):
    u = np.random.random(N+50)*0.5
    y = np.zeros(N+50)
    for k in range(10,N+50):
        y[k] = 0.3*y[k-1] + 0.05*y[k-1]*np.sum(y[k-10:k]) + 1.5*u[k-1]*u[k-10] + 0.1
    return u[50:], y[50:]


# %%
Ntrain = 5000
Nvalid = 2000
Ntest = 2000

u, d = NARMA10(Ntrain + Nvalid + Ntest)

utrain = u[:Ntrain]
dtrain = d[:Ntrain]
uvalid = u[Ntrain:Ntrain+Nvalid]
dvalid = d[Ntrain:Ntrain+Nvalid]
utest = u[Ntrain+Nvalid:]
dtest = d[Ntrain+Nvalid:]

# %%
net = Mackey_Glass_SNR(1, 40, 1, m0=0.1, mask_sparse=0.5, bias=False)

params = {'theta':0.2}
net.train(utrain, dtrain, uvalid, dvalid, params)

# %%
Stest = net.transform(utest, params)
pred = net.forward(Stest)

# %%
plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

# %%
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

# %% [markdown]
# We can see it does a reasonable job of the NARMA10 task.

# %% [markdown]
# ### Now for our case

# %% [markdown]
# ***Need to tidy this up!***

# %% [markdown]
# #### Defining the net

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def Ridge_regression(S, Y, l):
    '''
    For a linear layer we can solve the weights by a direct method
    If the error function is the mean square error given by
        E = |Y - S * W |^2 + \lambda |W|^2
    where the L2 norm is being applied and the variables are
        Y = [Nsamples x Noutputs] is the desired output
        S = [Nsamples x Nweights] is the input signal
        W = [Nweights x Noutputs] is the weight matrix
    To minimise E we need to solve:
        S^T * S = (S^T * Y  + \lambda I) * W
        W = (S^T*S + \lambda I)^-1 * S^T * Y
    '''
    STS = np.matmul(S.T, S)
    STY = np.matmul(S.T, Y)
    Sdag = np.linalg.pinv(STS + l*np.eye(len(STS)))
    return np.matmul(Sdag, STY)


# %%
def MG_func(x, J, gamma, eta, p):
    return eta*(x + gamma*J) / (1 + np.power( x + gamma*J, p))



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
# Stoner-wolfarth rate
    # plusminus = -1 gives rate 21, = +1 gives rate 12 
def SPNC_rate_sw(beta_prime,h_prime,minusplus):
    
    w = np.exp( -beta_prime * np.power( (1 + minusplus * h_prime) ,2) )
    
    return w

def SPNC_magnetisation_sw(beta_prime,h_prime,m0,t_prime_step):
    
    w21 = SPNC_rate_sw(beta_prime,h_prime,-1)
    w12 = SPNC_rate_sw(beta_prime,h_prime,+1)
    
    return SPNC_magnetisation_two_state(w21,w12,m0,t_prime_step)


# %%
def MSE (pred, desired):
    return np.mean(np.square(np.subtract(pred,desired)))


# %%
class SPNC_SNR:
    def __init__(self, Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None):
        '''
        Nin = input size
        Nvirt = number of virtual nodes
        Nout = output size
        m0 = magnitude of the mask values
        mask_sparse = sparsity factor for mask matrix
        bias = bool flag for using bias
        act = pass an activation function to use
        inv_act = function which applies the inverse of act
        '''
        self.Nin = Nin
        self.Nvirt = Nvirt
        self.Nout = Nout
        self.m0 = m0
        
        # Mask is random matrix of -m0 and m0
        # mask_sparse defines the sparsity level of the input mask
        # i.e 1.0 = full, 0.0 = empty
        self.M = 2*self.m0*(np.random.randint(0,2, (Nvirt,Nin))-0.5)
        #self.M *= 1.0*(np.random.random(size=(Nvirt, Nin)) <= mask_sparse)
        # Empty weight matrix 
        self.W = np.zeros( (Nvirt + int(bias), Nout))
        
        self.use_bias=bias
        
        # Activation and inverse activation functions
        self.f_act = act
        self.f_inv_act = inv_act
        
    def transform(self, u, params):
        '''
        Function to generate the reservoir signal from an input u
        params = dict for various parameters
        '''
        Ns = len(u)
        
        # Unflattens input if it is 1d
        u = u.reshape((Ns, self.Nin))
        
        J = np.zeros((Ns, self.Nvirt))
        
        # expands the signal to include a bias column is req'd
        if self.use_bias:
            S = np.ones((Ns, self.Nvirt+1))
        else:
            S = np.zeros((Ns, self.Nvirt))
        
        #Getting parameters
        # theta = temporal node spacing
        theta = params['theta']
        
        # gamma = feedback term
        gamma = params['gamma']
        
        #beta_prime = KV/KbT (effective temperature)
        beta_prime = params['beta_prime']
        
        # parameters for the MG function
        #Sigma = np.exp(-theta)
        #gamma = 0.005
        #eta = 0.5
        #P = 1
        
        J = np.matmul(u, self.M.T)
        for k in range(Ns):              
            # S[k,0] = S[k-1, self.Nvirt-1] * Sigma + (1.0 - Sigma)*MG_func( S[k-1,0], J[k,0], gamma, eta, P)
            
            #First column is fed into from last column of previous row
            S[k,0] = SPNC_magnetisation_sw(beta_prime, (J[k,0] + gamma*S[k-1,0]) ,S[k-1,self.Nvirt-1],theta)
            
            for i in range(1,self.Nvirt):
                # S[k,i] = S[k,i-1] * Sigma + (1.0 - Sigma)*MG_func( S[k-1,i], J[k,i], gamma, eta, P)   
                
                #Moving along one row feeding in the values from the last column
                S[k,i] = SPNC_magnetisation_sw(beta_prime, (J[k,i] + gamma*S[k-1,i]), S[k,i-1], theta)
        return S
    
    def forward(self, S):
        if self.f_act is not None:
            return self.f_act(np.matmul(S, self.W))
        else:
            return np.matmul(S, self.W)
    
    def train(self, u_train, d_train, u_valid, d_valid, params):
        
        S_train = self.transform(u_train, params)
        S_valid = self.transform(u_valid, params)
                
        if self.f_inv_act is not None:
            inv_act_d_train = self.f_inv_act(d_train)
            inv_act_d_valid = self.f_inv_act(d_valid)
        else:
            inv_act_d_train = d_train
            inv_act_d_valid = d_valid
        
        # regularisation parameters to validate over
        lambdas = np.exp(np.linspace(-6,0,num=20))
        lambdas[0] = 0.0
        
        errs = np.zeros(lambdas.shape)
        for i,l in enumerate(lambdas):
            self.W = Ridge_regression(S_train, inv_act_d_train, l)
            valid_pred = self.forward(S_valid)
            errs[i] = MSE(valid_pred, d_valid)
            print(l, MSE(valid_pred, d_valid))
    
        lopt = lambdas[np.argmin(errs)]
        print('Optimal lambda = ', lopt, 'with MSE = ', np.min(errs))
        self.W = Ridge_regression(S_train, d_train, lopt)
        
        

# %% [markdown]
# #### Defining the task

# %%
def NARMA10(N):
    u = np.random.random(N+50)*0.5
    y = np.zeros(N+50)
    for k in range(10,N+50):
        y[k] = 0.3*y[k-1] + 0.05*y[k-1]*np.sum(y[k-10:k]) + 1.5*u[k-1]*u[k-10] + 0.1
    return u[50:], y[50:]


# %%
Ntrain = 5000
Nvalid = 2000
Ntest = 2000

u, d = NARMA10(Ntrain + Nvalid + Ntest)

utrain = u[:Ntrain]
dtrain = d[:Ntrain]
uvalid = u[Ntrain:Ntrain+Nvalid]
dvalid = d[Ntrain:Ntrain+Nvalid]
utest = u[Ntrain+Nvalid:]
dtest = d[Ntrain+Nvalid:]

# %% [markdown]
# #### Testing perfomance of different nets

# %% [markdown]
# *We can look at the NRMSE for some indication of performance. A shift register can't beat 0.4. Before Appeltant, 0.18 was the best. Appeltant achieves 0.15*

# %% [markdown]
# **No feedback: Theta = 0.2, gamma = 0, Nvirt = 40, m0 = 1, beta_prime = 3**

# %%
# Defining the net
# potential params : ( Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None)
net = SPNC_SNR(1, 40, 1, m0=1, mask_sparse=0.5, bias=False)
params = {'theta':0.2,'gamma':0.0,'beta_prime':3}


# Running the net
net.train(utrain, dtrain, uvalid, dvalid, params)

Stest = net.transform(utest, params)
pred = net.forward(Stest)

plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

#NRMSE
print('NRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

# %% [markdown]
# **Feedback: Theta = 0.3, gamma = 0.2, Nvirt = 40, m0 = 1, beta_prime = 3**

# %%
# Defining the net
# potential params : ( Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None)
net = SPNC_SNR(1, 40, 1, m0=1, mask_sparse=0.5, bias=False)
params = {'theta':0.3,'gamma':0.2,'beta_prime':3}


# Running the net
net.train(utrain, dtrain, uvalid, dvalid, params)

Stest = net.transform(utest, params)
pred = net.forward(Stest)

plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

#NRMSE
print('NRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

# %% [markdown]
# **More virtual nodes!: Theta = 0.3, gamma = 0.2, Nvirt = 100, m0 = 1, beta_prime = 3**

# %%
# Defining the net
# potential params : ( Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None)
net = SPNC_SNR(1, 100, 1, m0=1, mask_sparse=0.5, bias=False)
params = {'theta':0.3,'gamma':0.2,'beta_prime':3}


# Running the net
net.train(utrain, dtrain, uvalid, dvalid, params)

Stest = net.transform(utest, params)
pred = net.forward(Stest)

plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

#NRMSE
print('NRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

# %% [markdown]
# **As many nodes as Appeltant!: Theta = 0.3, gamma = 0.2, Nvirt = 400, m0 = 1, beta_prime = 3**

# %%
# Defining the net
# potential params : ( Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None)
net = SPNC_SNR(1, 400, 1, m0=1, mask_sparse=0.5, bias=False)
params = {'theta':0.3,'gamma':0.2,'beta_prime':3}


# Running the net
net.train(utrain, dtrain, uvalid, dvalid, params)

Stest = net.transform(utest, params)
pred = net.forward(Stest)

plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

#NRMSE
print('NRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))
print('NRMSE from std dev is ' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE (from variance) is' ,(MSE(pred,dtest))/np.power(np.std(dtest),2))

# %% [markdown]
# *Almost the Appeltant value!!*

# %% [markdown]
# **Same idea, but no feedback (with tweaks): Theta = 0.7, gamma = 0.0, Nvirt = 400, m0 = 1, beta_prime = 3**

# %%
# Defining the net
# potential params : ( Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None)
net = SPNC_SNR(1, 100, 1, m0=1, mask_sparse=0.5, bias=False)
params = {'theta':0.7,'gamma':0.0,'beta_prime':3}


# Running the net
net.train(utrain, dtrain, uvalid, dvalid, params)

Stest = net.transform(utest, params)
pred = net.forward(Stest)

plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

#NRMSE
print('NRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))


# %% [markdown]
# Not bad, but not amazing.

# %% [markdown]
# ***Look into it, it's not clear how exactly to calculate the NRMSE so these values can not be compared directly to the Appelton paper. Instead we will recompute using their optimum values:***

# %%
def MG_func(x, J, gamma, eta, p):
    return eta*(x + gamma*J) / (1 + np.power( x + gamma*J, p))



# %%
class Mackey_Glass_SNR:
    def __init__(self, Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None):
        '''
        Nin = input size
        Nvirt = number of virtual nodes
        Nout = output size
        m0 = magnitude of the mask values
        mask_sparse = sparsity factor for mask matrix
        bias = bool flag for using bias
        act = pass an activation function to use
        inv_act = function which applies the inverse of act
        '''
        self.Nin = Nin
        self.Nvirt = Nvirt
        self.Nout = Nout
        self.m0 = m0
        
        # Mask is random matrix of -m0 and m0
        # mask_sparse defines the sparsity level of the input mask
        # i.e 1.0 = full, 0.0 = empty
        self.M = 2*self.m0*(np.random.randint(0,2, (Nvirt,Nin))-0.5)
        #self.M *= 1.0*(np.random.random(size=(Nvirt, Nin)) <= mask_sparse)
        # Empty weight matrix 
        self.W = np.zeros( (Nvirt + int(bias), Nout))
        
        self.use_bias=bias
        
        # Activation and inverse activation functions
        self.f_act = act
        self.f_inv_act = inv_act
        
    def transform(self, u, params):
        '''
        Function to generate the reservoir signal from an input u
        params = dict for various parameters
        '''
        Ns = len(u)
        
        # Unflattens input if it is 1d
        u = u.reshape((Ns, self.Nin))
        
        J = np.zeros((Ns, self.Nvirt))
        
        # expands the signal to include a bias column is req'd
        if self.use_bias:
            S = np.ones((Ns, self.Nvirt+1))
        else:
            S = np.zeros((Ns, self.Nvirt))
        
        # theta = temporal node spacing
        theta = params['theta']
        
        # parameters for the MG function
        Sigma = np.exp(-theta)
        gamma = 0.01
        eta = 0.5
        P = 1
        
        J = np.matmul(u, self.M.T)
        for k in range(Ns):              
            S[k,0] = S[k-1, self.Nvirt-1] * Sigma + (1.0 - Sigma)*MG_func( S[k-1,0], J[k,0], gamma, eta, P)
            for i in range(1,self.Nvirt):
                S[k,i] = S[k,i-1] * Sigma + (1.0 - Sigma)*MG_func( S[k-1,i], J[k,i], gamma, eta, P)   
        return S
    
    def forward(self, S):
        if self.f_act is not None:
            return self.f_act(np.matmul(S, self.W))
        else:
            return np.matmul(S, self.W)
    
    def train(self, u_train, d_train, u_valid, d_valid, params):
        
        S_train = self.transform(u_train, params)
        S_valid = self.transform(u_valid, params)
                
        if self.f_inv_act is not None:
            inv_act_d_train = self.f_inv_act(d_train)
            inv_act_d_valid = self.f_inv_act(d_valid)
        else:
            inv_act_d_train = d_train
            inv_act_d_valid = d_valid
        
        # regularisation parameters to validate over
        lambdas = np.exp(np.linspace(-6,0,num=20))
        lambdas[0] = 0.0
        
        errs = np.zeros(lambdas.shape)
        for i,l in enumerate(lambdas):
            self.W = Ridge_regression(S_train, inv_act_d_train, l)
            valid_pred = self.forward(S_valid)
            errs[i] = MSE(valid_pred, d_valid)
            print(l, MSE(valid_pred, d_valid))
    
        lopt = lambdas[np.argmin(errs)]
        print('Optimal lambda = ', lopt, 'with MSE = ', np.min(errs))
        self.W = Ridge_regression(S_train, d_train, lopt)
        
        

# %%
net = Mackey_Glass_SNR(1, 400, 1, m0=0.1, mask_sparse=0.5, bias=False)

params = {'theta':0.2}
net.train(utrain, dtrain, uvalid, dvalid, params)

Stest = net.transform(utest, params)
pred = net.forward(Stest)

plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

#NRMSE
print('NRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))
print('NRMSE from std dev is ' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE (from variance) is' ,(MSE(pred,dtest))/np.power(np.std(dtest),2))


# %% [markdown]
# This implies that they aren't using NRMSE where the division is by standard deviation (as I thought was implied in the text) as the value is larger than they say you should get for a shift register (0.4). Not clear they are using my original one either though (they claim only 0.15)...At least this offers a comparison.

# %% [markdown]
# **Comparison to a shift register - not sure this is correct yet!!!**

# %%
def shift_func(j, s_old, gamma):
    return j + s_old*gamma



# %%
class shift_SNR:
    def __init__(self, Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None):
        '''
        Nin = input size
        Nvirt = number of virtual nodes
        Nout = output size
        m0 = magnitude of the mask values
        mask_sparse = sparsity factor for mask matrix
        bias = bool flag for using bias
        act = pass an activation function to use
        inv_act = function which applies the inverse of act
        '''
        self.Nin = Nin
        self.Nvirt = Nvirt
        self.Nout = Nout
        self.m0 = m0
        
        # Mask is random matrix of -m0 and m0
        # mask_sparse defines the sparsity level of the input mask
        # i.e 1.0 = full, 0.0 = empty
        self.M = 2*self.m0*(np.random.randint(0,2, (Nvirt,Nin))-0.5)
        #self.M *= 1.0*(np.random.random(size=(Nvirt, Nin)) <= mask_sparse)
        # Empty weight matrix 
        self.W = np.zeros( (Nvirt + int(bias), Nout))
        
        self.use_bias=bias
        
        # Activation and inverse activation functions
        self.f_act = act
        self.f_inv_act = inv_act
        
    def transform(self, u, params):
        '''
        Function to generate the reservoir signal from an input u
        params = dict for various parameters
        '''
        Ns = len(u)
        
        # Unflattens input if it is 1d
        u = u.reshape((Ns, self.Nin))
        
        J = np.zeros((Ns, self.Nvirt))
        
        # expands the signal to include a bias column is req'd
        if self.use_bias:
            S = np.ones((Ns, self.Nvirt+1))
        else:
            S = np.zeros((Ns, self.Nvirt))
        
        #Getting parameters
        # gamma = feedback term
        gamma = params['gamma']

        
        J = np.matmul(u, self.M.T)
        for k in range(Ns):              
            
            #First column is fed into from last column of previous row
            S[k,0] = shift_func(J[k,0],S[k-1,0], gamma)
            
            for i in range(1,self.Nvirt):
                #Moving along one row feeding in the values from the last column
                S[k,i] = shift_func(J[k,i],S[k-1,i],gamma)
                
        return S
    
    def forward(self, S):
        if self.f_act is not None:
            return self.f_act(np.matmul(S, self.W))
        else:
            return np.matmul(S, self.W)
    
    def train(self, u_train, d_train, u_valid, d_valid, params):
        
        S_train = self.transform(u_train, params)
        S_valid = self.transform(u_valid, params)
                
        if self.f_inv_act is not None:
            inv_act_d_train = self.f_inv_act(d_train)
            inv_act_d_valid = self.f_inv_act(d_valid)
        else:
            inv_act_d_train = d_train
            inv_act_d_valid = d_valid
        
        # regularisation parameters to validate over
        lambdas = np.exp(np.linspace(-6,0,num=20))
        lambdas[0] = 0.0
        
        errs = np.zeros(lambdas.shape)
        for i,l in enumerate(lambdas):
            self.W = Ridge_regression(S_train, inv_act_d_train, l)
            valid_pred = self.forward(S_valid)
            errs[i] = MSE(valid_pred, d_valid)
            print(l, MSE(valid_pred, d_valid))
    
        lopt = lambdas[np.argmin(errs)]
        print('Optimal lambda = ', lopt, 'with MSE = ', np.min(errs))
        self.W = Ridge_regression(S_train, d_train, lopt)
        
        

# %% [markdown]
# **Just a shift register - no feedback**

# %%
net = shift_SNR(1, 400, 1, m0=0.1, mask_sparse=0.5, bias=False)

params = {'gamma':0.0}
net.train(utrain, dtrain, uvalid, dvalid, params)

Stest = net.transform(utest, params)
pred = net.forward(Stest)

plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

#NRMSE
print('NRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))
print('NRMSE from std dev is ' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))

# %% [markdown]
# **Add feedback to shift register (gamma : 0.5)**

# %%
net = shift_SNR(1, 400, 1, m0=0.1, mask_sparse=0.5, bias=False)

params = {'gamma':0.5}
net.train(utrain, dtrain, uvalid, dvalid, params)

Stest = net.transform(utest, params)
pred = net.forward(Stest)

plt.plot(dtest[100:200], label='Desired Output')
plt.plot(pred[100:200], label='Model Output')
plt.legend(loc='lower left')
plt.xlabel('time')
plt.ylabel('NARMA10 output')
plt.show()

plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(dtest[:], pred[:], 'o')
plt.xlabel('Desired Output')
plt.ylabel('Model Output')
plt.show()

#NRMSE
print('NRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))
print('NRMSE from std dev is ' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))

# %% [markdown]
# Suprisingly good, although not as good as with the reservoir.

# %%
