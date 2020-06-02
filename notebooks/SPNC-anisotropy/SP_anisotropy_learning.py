# -*- coding: utf-8 -*-
# %% [markdown]
# # Superparamagnetic Network - Machine Learning Testing

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import random as rnd

import SP_anisotropy_class as SPN

#3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# %% [markdown]
# ## The SP Network
#
# $e = \frac{E}{KV}$, $H_K = \frac{2K}{\mu_0M_S}$, $h=\frac{H}{H_K}$, $k_\sigma=\frac{K_\sigma}{K}$, $\omega'=\frac{\omega}{f_0}$ and $\beta'=\frac{KV}{k_BT}$
#
# $h=0.4$
#
# $\theta_H=90°$
#
# $\phi=45°$
#
# $\beta'=10$
#
# The system is at equilibrium with $k_\sigma=0$ and we instantly set $k_\sigma=1$.
#
# $f_0=10^{10}Hz$
#
# We call the characteristic memory time scale $T$.

# %%
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
spn = SPN.SP_Network(h,theta_H,k_s_0,phi,beta_prime)

f0 = 1e10

# %%
spn.k_s = 1
SPN.calculate_energy_barriers(spn)
T = 1./(spn.get_omega_prime()*f0) #Characteristic memory time
t_step = T/100 #We take a t_step 100 times smaller than T
time = np.arange(0,5*T,t_step) #We want to see 5 T
m_t = [spn.get_m()]

for i in range(len(time)-1):
    spn.evolve(f0,t_step)
    m_t.append(spn.get_m())

# %%
plt.figure(figsize=(10,6))
plt.plot(time*1e9,m_t)
plt.grid(True)
plt.title("Response to excitation during 5"+r'$T$')
plt.ylabel(r'$m(t)$')
plt.xlabel("Time (ns)")
plt.show()


# %% [markdown]
# Now let's use a random signal as input.

# %%
def rnd_signal(n):
    signal = []
    for i in range(n):
        signal.append(2*rnd.random()-1)
    return(signal)


# %%
spn = SPN.SP_Network(h,theta_H,k_s_0,phi,beta_prime)
SPN.calculate_energy_barriers(spn)
T = 1./(spn.get_omega_prime()*f0)

n = 10 #Number of inputs
N = 100 #Number of steps per input
theta = T #Duration of each input
t_step = theta/N #We take a t_step 100 times smaller than theta
signal = rnd_signal(n) #Input signal
time_signal = np.arange(n)*theta
time = np.linspace(0,n*theta,n*N)

m_t = []

for i in range(n):
    m_t.append(spn.get_m())
    spn.k_s = signal[i]
    SPN.calculate_energy_barriers(spn)
    for j in range(N-1):
        spn.evolve(f0,t_step)
        m_t.append(spn.get_m())

# %%
plt.figure(figsize=(10,6))
plt.plot(time_signal*1e9,signal,'b-',drawstyle='steps-post',label="Input signal")
plt.plot(time*1e9,m_t,'r-',label="Output")
plt.grid(True)
plt.legend(loc="best")
plt.title("Response to random input with "+r'$\theta=T$')
plt.ylabel(r'$m(t)$')
plt.xlabel("Time (ns)")
plt.show()
plt.show()

# %%
spn = SPN.SP_Network(h,theta_H,k_s_0,phi,beta_prime)
SPN.calculate_energy_barriers(spn)
T = 1./(spn.get_omega_prime()*f0)

n = 10 #Number of inputs
N = 100 #Number of steps per input
theta = T/10 #Duration of each input
t_step = theta/N #We take a t_step 100 times smaller than theta
signal = rnd_signal(n) #Input signal
time_signal = np.arange(n)*theta
time = np.linspace(0,n*theta,n*N)

m_t = []

for i in range(n):
    m_t.append(spn.get_m())
    spn.k_s = signal[i]
    SPN.calculate_energy_barriers(spn)
    for j in range(N-1):
        spn.evolve(f0,t_step)
        m_t.append(spn.get_m())

# %%
plt.figure(figsize=(10,6))
plt.plot(time_signal*1e9,signal,'b-',drawstyle='steps-post',label="Input signal")
plt.plot(time*1e9,m_t,'r-',label="Output")
plt.grid(True)
plt.legend(loc="best")
plt.title("Response to random input with "+r'$\theta=T/10$')
plt.ylabel(r'$m(t)$')
plt.xlabel("Time (ns)")
plt.show()
plt.show()

# %%
spn = SPN.SP_Network(h,theta_H,k_s_0,phi,beta_prime)
SPN.calculate_energy_barriers(spn)
T = 1./(spn.get_omega_prime()*f0)

n = 10 #Number of inputs
N = 100 #Number of steps per input
theta = T*10 #Duration of each input
t_step = theta/N #We take a t_step 100 times smaller than tau_signal
signal = rnd_signal(n) #Input signal
time_signal = np.arange(n)*theta
time = np.linspace(0,n*theta,n*N)

m_t = []

for i in range(n):
    m_t.append(spn.get_m())
    spn.k_s = signal[i]
    SPN.calculate_energy_barriers(spn)
    for j in range(N-1):
        spn.evolve(f0,t_step)
        m_t.append(spn.get_m())

# %%
plt.figure(figsize=(10,6))
plt.plot(time_signal*1e9,signal,'b-',drawstyle='steps-post',label="Input signal")
plt.plot(time*1e9,m_t,'r-',label="Output")
plt.grid(True)
plt.legend(loc="best")
plt.title("Response to random input with "+r'$\theta=10T$')
plt.ylabel(r'$m(t)$')
plt.xlabel("Time (ns)")
plt.show()
plt.show()


# %% [markdown]
# ## Towards Machine Learning

# %%
def Ridge_regression(S, Y, alpha):
    '''
    For a linear layer we can solve the weights by a direct method
    If the error function is the mean square error given by
        E = |Y - S * W |^2 + \alpha |W|^2
    where the L2 norm is being applied and the variables are
        Y = [Nsamples x Noutputs] is the desired output
        S = [Nsamples x Nweights] is the input signal
        W = [Nweights x Noutputs] is the weight matrix
    To minimise E we need to solve:
        S^T * Y = (S^T * S  + \alpha I) * W
        W = (S^T*S + \alpha I)^-1 * S^T * Y
    '''
    STS = np.matmul(S.T, S)
    STY = np.matmul(S.T, Y)
    Sdag = np.linalg.pinv(STS + alpha*np.eye(len(STS)))
    return(np.matmul(Sdag, STY))


# %%
def NARMA10(Ns):
    # Ns is the number of samples
    u = np.random.random(Ns+50)*0.5
    y = np.zeros(Ns+50)
    for k in range(10,Ns+50):
        y[k] = 0.3*y[k-1] + 0.05*y[k-1]*np.sum(y[k-10:k]) + 1.5*u[k-1]*u[k-10] + 0.1
    return(u[50:],y[50:])


# %%
def mask_NARMA10(m0,Nvirt):
    # Nvirt is the number of virtual nodes
    mask = []
    for i in range(Nvirt):
        mask.append(rnd.choice([-1,1])*m0)
    mask = mask
    return(mask)


# %%
def NRMSE(Y,Y_pred):
    var = np.var(Y)
    return np.sqrt(np.square(Y_pred-Y).mean()/var)

def NRMSE_list(y,y_pred):
    Y = np.array(y)
    Y_pred = np.array(y_pred)
    return(NRMSE(Y,Y_pred))


# %%
h = 0.4
theta_H = 90
k_s_0 = 0
phi = 45
beta_prime = 10
f0 = 1e10
m0 = 2 #This gives approximately k_s between -1 and 1
class Single_Node_Reservoir_NARMA10:
    
    def __init__(self, Nvirt, T_theta_ratio):
        self.Nin = 1
        self.Nvirt = Nvirt
        self.Nout = 1
        
        self.spn = SPN.SP_Network(h,theta_H,k_s_0,phi,beta_prime)
        SPN.calculate_energy_barriers(spn)
        self.T = 1./(spn.get_omega_prime()*f0)
        self.theta = self.T/T_theta_ratio
        self.tau = self.Nvirt*self.theta
        
        self.M = mask_NARMA10(m0,Nvirt)
        self.W = np.zeros((Nvirt,1))
    
    def gen_signal(self, u):
        Ns = len(u)
        J = np.zeros((Ns,self.Nvirt))
        S = np.zeros((Ns,self.Nvirt))
        
        for k in range(Ns):
            if k%10==0:
                print(k)
            for i in range(self.Nvirt):
                S[k,i] = spn.get_m()
                j = self.M[i]*u[k]
                J[k,i] = j
                spn.k_s = j
                SPN.calculate_energy_barriers(spn)
                spn.evolve(f0,self.theta)
                
        return(J,S)
    
    def train(self, S, y, S_valid, y_valid):
        alphas = np.logspace(-10,-1,10)
        alphas[0] = 0.
        
        Ns = S.shape[0]
        Ns_valid = S_valid.shape[0]
        Y = y.reshape((Ns,1))
        Y_valid = y_valid.reshape((Ns_valid,1))
        
        errs = np.zeros(alphas.shape)
        for i in range(len(alphas)):
            self.W = Ridge_regression(S, Y, alphas[i])
            Y_pred_valid = np.array(self.predict(S_valid)).reshape(Ns_valid,1)
            errs[i] = NRMSE(Y_valid, Y_pred_valid)
            print(alphas[i], NRMSE(Y_valid, Y_pred_valid))
    
        alpha_opt = alphas[np.argmin(errs)]
        print('Optimal alpha = '+str(alpha_opt)+' with NRMSE = '+str(np.min(errs)))
        self.W = Ridge_regression(S, Y, alpha_opt)
    
    def predict(self, S):
        Ns = S.shape[0]
        return(np.matmul(S, self.W).reshape(1,Ns).tolist()[0])
    
    #Time lists (in ns)
    
    def get_time_list_u(self, u):
        #We need to make sure that time_u has Ns elements with a delay tau
        Ns = len(u)
        t_u = 0
        time_u = [t_u]
        while len(time_u)<Ns:
            t_u += self.tau
            time_u.append(t_u)
        return(np.array(time_u)*1e9)
    
    def get_time_list_S(self, S):
        Ns = S.shape[0]
        return(np.arange(0,Ns*self.tau,self.theta)*1e9)


# %%
Ntrain = 50
(u,y) = NARMA10(Ntrain)

net = Single_Node_Reservoir_NARMA10(400,5)
time_u = net.get_time_list_u(u)

# %%
plt.figure(figsize=(10,6))
plt.plot(time_u,u,drawstyle='steps-post',label="Input")
plt.plot(time_u,y,drawstyle='steps-post',label="Desired output")
plt.xlabel("Time (ns)")
plt.legend(loc="best")
plt.show()

# %%
(J,S) = net.gen_signal(u)
time_S = net.get_time_list_S(S)

# %%
plt.figure(figsize=(10,6))
L = 20
plt.grid(True)
plt.plot(time_u[-L:],u[-L:],drawstyle='steps-post',label="Input")
#plt.plot(time_S[-L*net.Nvirt:],J.flatten()[-L*net.Nvirt:],drawstyle='steps-post',label="Transformed input")
plt.plot(time_S[-L*net.Nvirt:],S.flatten()[-L*net.Nvirt:],'r-',label="Signal")
plt.legend(loc="best")
plt.xlabel("Time (ns)")
#plt.ylim(-0.6,0.6)
plt.show()

# %%
T_theta_list = np.logspace(-1.5,1.5,15)
amplitude = []
Ntrain = 50
(u,y) = NARMA10(Ntrain)
N_mean = 10
L = 20
for T_t in T_theta_list:
    print(T_t)
    amp_mean = 0
    for i in range(N_mean):
        net = Single_Node_Reservoir_NARMA10(20,T_t)
        (J,S) = net.gen_signal(u)
        #M = max(S.flatten()[-L*net.Nvirt:])
        #m = min(S.flatten()[-L*net.Nvirt:])
        #amp_mean += M-m
        amp_mean += np.std(S.flatten())
    amplitude.append(amp_mean/N_mean)

# %%
plt.figure(figsize=(10,6))
plt.plot(T_theta_list,amplitude,'r+')
plt.xscale("log")
plt.xlabel(r'$T/\theta$'+" ratio")
plt.ylabel("Standard deviation of the signal")
plt.title("20 virtual nodes")
plt.show()

# %%
Ntrain = 500
Nvalid = 100

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)

net = Single_Node_Reservoir_NARMA10(400,10)
(J,S) = net.gen_signal(u)
(J_valid,S_valid) = net.gen_signal(u_valid)

# %%
net.train(S,y,S_valid,y_valid)

# %%
y_pred_train = net.predict(S)
y_pred_valid = net.predict(S_valid)

Ntest = 100
(u_test,y_test) = NARMA10(Ntest)
(J_test,S_test) = net.gen_signal(u_test)
y_pred_test = net.predict(S_test)

# %%
time_u = net.get_time_list_u(u)
plt.figure(figsize=(10,6))
plt.plot(time_u,y_pred_train,drawstyle='steps-post',label="Predicted output (training)")
plt.plot(time_u,y,drawstyle='steps-post',label="Desired output (training)")
plt.xlabel("Time (ns)")
plt.legend(loc="best")
plt.show()

# %%
print("NRMSE (train) = "+str(NRMSE_list(y,y_pred_train)))
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y,y_pred_train,'ro')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.show()

# %%
print("NRMSE (validation) = "+str(NRMSE_list(y_valid,y_pred_valid)))
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_valid,y_pred_valid,'ro')
plt.xlabel("Desired output (validation)")
plt.ylabel("Predicted output (validation)")
plt.show()

# %%
print("NRMSE (test) = "+str(NRMSE_list(y_test,y_pred_test)))
plt.figure(figsize=(10,6))
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_test,y_pred_test,'ro')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.show()


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

# %% [markdown]
# We see that the behaviour looks kind of like it is intergrating the average, with noise from the deviation. The time scale sets how much the noise effects it and how long the averaging takes. Let's take a look at this by finding the expected result from the average input:

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
# ### An aside on errors

# %% [markdown]
# The subject of how to define the error is somewhat muddled in the literature. From here on, these definitions will be used: <br>
# **NRMSE** is $ \sqrt{\mbox{MSE}}/\sigma $ <br>
# **NMSE** is $ \mbox{MSE}/\sigma^2 $ <br>
# **Mean-NRMSE or MNRMSE** is $ \sqrt{\mbox{MSE}}/\mbox{mean} $ <br>
# $\sigma = $ standard deviation (although should it be n or n-1?) <br>
# MSE $= \frac{\sum{(\mbox{predicted}-\mbox{desired})^2}}{\mbox{total}} $ <br>
# *In my opinion, either of the measures based on standard deviation provide something which is meaningful - error from the data normalised to variance in the data. These seem to be good metrics to stick with.*

# %% [markdown]
# ***Looking into it, it's not clear how exactly to calculate the Appelton calculated NRMSE. Instead, here I recompute using their optimum values for the net using my definitions:***

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))
print('MNMSE (to check if Appelton was using this!) is ', (MSE(pred,dtest))/np.mean(dtest) )
print('Variance-NRMSE (to check if Appelton was using this!) is' , np.sqrt(MSE(pred,dtest))/np.power(np.std(dtest),2))

# %% [markdown]
# This implies that they aren't using NRMSE (as I thought was implied in the text) as the value is larger than they say you should get for a shift register (0.4). Not clear they are using mean normalised either though (they claim only 0.15, not 0.13)...In fact none of these numbers are the same as theirs. They do, at least offer a comparison though!

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
# *We can look at the NRMSE for some indication of performance. From above, Appeltant ahieved NRMSE = 0.50 or equivalently NMSE = 0.25.* 

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))


# %% [markdown]
# Not bad, but not amazing.

# %% [markdown]
# **Comparison to a shift register - not sure this is the correct definition of a shift register, but it is informative!!!**

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

# %% [markdown]
# Suprisingly good, although not as good as with the reservoir.

# %% [markdown]
# ## Examining characteristic timescales

# %% [markdown]
# We rewrote $w t' = w' [w_0 t'] $ where: $w' = w/w_0 = \cosh{(2 \beta' H')} \exp{(-\beta' H'^2)}$, and $w_0 = 2 \exp{(-\beta')}$

# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

#3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Stoner-wolfarth rate
    # plusminus = -1 gives rate 21, = +1 gives rate 12 
def SPNC_rate_sw(beta_prime,h_prime,minusplus):
    
    w = np.exp( -beta_prime * np.power( (1 + minusplus * h_prime) ,2) )
    
    return w


# %%
# Remove previously open figure
if 'w_prime_fig' in locals():
    plt.close(w_prime_fig)

def SPNC_total_rate_sw(beta_prime,h_prime):
    
    w = SPNC_rate_sw(beta_prime,h_prime,+1) + SPNC_rate_sw(beta_prime,h_prime,-1)
    return w
    
# Function that defines w_prime
def SPNC_w_prime_sw(beta_prime,h_prime):
    w_prime = SPNC_total_rate_sw(beta_prime,h_prime)/SPNC_total_rate_sw(beta_prime,0)
            
    return w_prime

# Set up plot
w_prime_fig = plt.figure()
ax = w_prime_fig.gca(projection='3d')
beta_prime = np.arange(3,30,0.1)
h_prime = np.arange(-1,1,0.05)
beta_prime, h_prime = np.meshgrid(beta_prime, h_prime)
ax.set_title("Sensitivity of rate to field w'")
ax.set_xlabel("beta_prime")
ax.set_ylabel('h_prime')
ax.set_zlabel('log(base rate multiplier)')
ax.view_init(azim=133, elev=45)
# Plot
surf = ax.plot_surface(beta_prime,h_prime,np.log(SPNC_w_prime_sw(beta_prime,h_prime)), 
                       cmap = cm.coolwarm, linewidth = 0, antialiased=False)
plt.show()

# %% [markdown]
# There is a dramatic increase in the sensitivity to field with increasing $\beta'$. <br>
#
# It is interesting to consider the absolute rate:

# %%
# Remove previously open figure
if 'w_prime_fig' in locals():
    plt.close(w_prime_fig)

def SPNC_total_rate_sw(beta_prime,h_prime):
    
    w = SPNC_rate_sw(beta_prime,h_prime,+1) + SPNC_rate_sw(beta_prime,h_prime,-1)
    return w
    

# Set up plot
w_prime_fig = plt.figure()
ax = w_prime_fig.gca(projection='3d')
beta_prime = np.arange(3,30,0.1)
h_prime = np.arange(-1,1,0.05)
beta_prime, h_prime = np.meshgrid(beta_prime, h_prime)
ax.set_title("Absolute rate")
ax.set_xlabel("beta_prime")
ax.set_ylabel('h_prime')
ax.set_zlabel('log(rate)')
ax.view_init(azim=133, elev=45)
# Plot
surf = ax.plot_surface(beta_prime,h_prime,np.log(SPNC_total_rate_sw(beta_prime,h_prime)), 
                       cmap = cm.coolwarm, linewidth = 0, antialiased=False)
plt.show()


# %% [markdown]
# From this we can see that whilst there is a dramatic drop in rate with increasing $\beta'$ at zero field, the rate for large field remains similar for all values. This makes sense as for large fields the energy barrier is always similar to the thermal energy. But for large $\beta'$ and low field the energy barrier becomes $\gg$ than the thermal energy.
#
# This field sensitivity might be a big problem for going to large values of $\beta'$ in ML. However, we should be able to stick to $\beta' < 30$ for any pratical device. 

# %% [markdown]
# ### ML at big beta

# %%
# Making sure net parameters are initilised

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

def MG_func(x, J, gamma, eta, p):
    return eta*(x + gamma*J) / (1 + np.power( x + gamma*J, p))

# Reduced magnetisation for two state system evolving with constant field 
# note: I am using the notation, w12: transition rate from state 1, to state 2
# w12, w21 must be constant over the tstep!
def SPNC_magnetisation_two_state(w21,w12,m0,tstep):

    wdiff = w21 - w12
    w = w12 + w21
    wratio = wdiff/w
    m = wratio + (m0 - wratio)*np.exp(-w*tstep)

    return m

# Stoner-wolfarth rate
    # plusminus = -1 gives rate 21, = +1 gives rate 12 
def SPNC_rate_sw(beta_prime,h_prime,minusplus):
    
    w = np.exp( -beta_prime * np.power( (1 + minusplus * h_prime) ,2) )
    
    return w

def SPNC_magnetisation_sw(beta_prime,h_prime,m0,t_prime_step):
    
    w21 = SPNC_rate_sw(beta_prime,h_prime,-1)
    w12 = SPNC_rate_sw(beta_prime,h_prime,+1)
    
    return SPNC_magnetisation_two_state(w21,w12,m0,t_prime_step)

def MSE (pred, desired):
    return np.mean(np.square(np.subtract(pred,desired)))

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
        
def NARMA10(N):
    u = np.random.random(N+50)*0.5
    y = np.zeros(N+50)
    for k in range(10,N+50):
        y[k] = 0.3*y[k-1] + 0.05*y[k-1]*np.sum(y[k-10:k]) + 1.5*u[k-1]*u[k-10] + 0.1
    return u[50:], y[50:]

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
# **Trying excessivley big beta: Theta = 0.005, gamma = 0.2, Nvirt = 40, m0 = 2, beta_prime = 300**

# %%
# Defining the net
# potential params : ( Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None)
net = SPNC_SNR(1, 40, 1, m0=2, mask_sparse=0.5, bias=False)
params = {'theta':0.005,'gamma':0.2,'beta_prime':300}


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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

# %% [markdown]
# **Reducing the field range: Theta = very big, gamma = 0.2, Nvirt = 40, m0 = 0.1, beta_prime = 300**

# %%
# Defining the net
# potential params : ( Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None)
net = SPNC_SNR(1, 40, 1, m0=0.1, mask_sparse=0.5, bias=False)
beta_prime = 300
theta = beta_prime/(2*np.exp(-beta_prime))
print('theta is ', theta, '\n')
params = {'theta': 0.0000000005*theta,'gamma':0.2,'beta_prime':beta_prime}

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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

# %% [markdown]
# This is clearly very problematic. It looks like when $\beta'$ is big then it becomes very hard to distinguish between the changes. Let's look at the limits of our more realistic range:

# %% [markdown]
# **beta at limit of practical range: Theta = 0.5, gamma = 0.2, Nvirt = 40, m0 = 2, beta_prime = 30**

# %%
# Defining the net
# potential params : ( Nin, Nvirt, Nout, m0=0.1, mask_sparse=1.0, bias=False, act=None, inv_act=None)
net = SPNC_SNR(1, 40, 1, m0=2, mask_sparse=0.5, bias=False)
params = {'theta': 0.5,'gamma':0.2,'beta_prime':30}
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

#Errors
print('NRMSE is' ,np.sqrt(MSE(pred,dtest))/np.std(dtest))
print('NMSE is' , (MSE(pred,dtest))/np.power(np.std(dtest),2) )
print('MNRMSE is ',np.sqrt(MSE(pred,dtest))/np.mean(dtest))

# %% [markdown]
# This is somewhat better, although definitely worse than the low $\beta'$ case. It at least (just) out performs our shift register with feedback.
# **One important factor might be scaling the narma input to be between -1 and 1 to make use of both sides of the resevoir or biasing it so we only need to deal with the oscilations, not the offset**
