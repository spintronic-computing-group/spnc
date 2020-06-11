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
plt.ylabel(r'$k_\sigma(t) ; m(t)$')
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
plt.ylabel(r'$k_\sigma(t) ; m(t)$')
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
#Ignore the first 50 elements of the output
spacer = 50


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
    u = np.random.random(Ns+50+spacer)*0.5
    y = np.zeros(Ns+50+spacer)
    for k in range(10,Ns+50+spacer):
        y[k] = 0.3*y[k-1] + 0.05*y[k-1]*np.sum(y[k-10:k]) + 1.5*u[k-1]*u[k-10] + 0.1
    return(u[50:],y[50+spacer:])


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
f0 = 1e10
class Single_Node_Reservoir_NARMA10:
    
    def __init__(self, Nvirt, T_theta_ratio, m0, gamma, beta_prime = 10, bias = True):
        self.Nin = 1
        self.Nvirt = Nvirt
        self.Nout = 1
        
        self.spn = SPN.SP_Network(h,theta_H,k_s_0,phi,beta_prime)
        SPN.calculate_energy_barriers(self.spn)
        self.T = 1./(self.spn.get_omega_prime()*f0)
        self.theta = self.T/T_theta_ratio
        self.tau = self.Nvirt*self.theta
        
        self.m0 = m0
        self.M = mask_NARMA10(m0,Nvirt)
        if bias:
            self.W = np.zeros((Nvirt+1,1))
        else:
            self.W = np.zeros((Nvirt,1))
        
        self.gamma = gamma
        
        self.use_bias = bias
    
    def gen_signal(self, u):
        Ns = len(u)
        if self.use_bias:
            print("Use bias")
            S = np.zeros((Ns,self.Nvirt+1))
        else:
            S = np.zeros((Ns,self.Nvirt))
        
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k]
                self.spn.k_s = j + self.gamma*S[k-1,i] #Feedback 
                SPN.calculate_energy_barriers(self.spn)
                self.spn.evolve(f0,self.theta)
                S[k,i] = self.spn.get_m()
        
        if self.use_bias:
            for k in range(Ns):
                S[k,self.Nvirt] = 1
        
        return(S[spacer:])
    
    def gen_signal_fast(self,u):
        Ns = len(u)
        if self.use_bias:
            print("Use bias")
            S = np.zeros((Ns,self.Nvirt+1))
        else:
            S = np.zeros((Ns,self.Nvirt))
        
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k]
                self.spn.k_s = j + self.gamma*S[k-1,i] #Feedback 
                self.spn.evolve_fast(f0,self.theta)
                S[k,i] = self.spn.get_m_fast()
        
        if self.use_bias:
            for k in range(Ns):
                S[k,self.Nvirt] = 1
        
        return(S[spacer:])
    
    def gen_signal_fast_2_inputs(self, u, back_input_ratio):
        Ns = len(u)
        Nin = int(self.Nvirt*back_input_ratio)
        if self.use_bias:
            print("Use bias")
            S = np.zeros((Ns,self.Nvirt+1))
        else:
            S = np.zeros((Ns,self.Nvirt))
            
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(Nin):
                #Input at k-1
                j = self.M[i]*u[k-1]
                self.spn.k_s = j + self.gamma*S[k-1,i] #Feedback
                self.spn.evolve_fast(f0,self.theta)
                S[k,i] = self.spn.get_m_fast()
            for i in range(Nin,self.Nvirt):
                #Input at k
                j = self.M[i]*u[k]
                self.spn.k_s = j + self.gamma*S[k-1,i] #Feedback 
                self.spn.evolve_fast(f0,self.theta)
                S[k,i] = self.spn.get_m_fast()
                
        if self.use_bias:
            for k in range(Ns):
                S[k,self.Nvirt] = 1
        
        return(S[spacer:])
    
    def gen_signal_fast_delayed_feedback(self, u, delay_fb):
        Ns = len(u)
        if self.use_bias:
            print("Use bias")
            S = np.zeros((Ns,self.Nvirt+1))
        else:
            S = np.zeros((Ns,self.Nvirt))
            
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k]
                self.spn.k_s = j + self.gamma*S[k-1,i-delay_fb] #Delayed Feedback 
                self.spn.evolve_fast(f0,self.theta)
                S[k,i] = self.spn.get_m_fast()
                
        if self.use_bias:
            for k in range(Ns):
                S[k,self.Nvirt] = 1
        
        return(S[spacer:])
    
    def gen_signal_delayed_feedback_without_SPN(self, u, delay_fb):
        Ns = len(u)
        if self.use_bias:
            print("Use bias")
            J = np.zeros((Ns,self.Nvirt+1))
        else:
            J = np.zeros((Ns,self.Nvirt))
            
        # Artificial nonlinearity
        #f = lambda x: x
        f=np.tanh
            
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k]
                J[k,i] = f(j + self.gamma*J[k-1,i-delay_fb]) #Delayed Feedback 
                
        if self.use_bias:
            for k in range(Ns):
                J[k,self.Nvirt] = 1
        
        return(J[spacer:])
    
    def gen_signal_without_SPN(self,u):
        Ns = len(u)
        if self.use_bias:
            print("Use bias")
            J = np.zeros((Ns,self.Nvirt+1))
        else:
            J = np.zeros((Ns,self.Nvirt))
        
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k]
                J[k,i] = j + self.gamma*J[k-1,i] #J will be useful to test the role of memory and nonlinearity
        
        if self.use_bias:
            for k in range(Ns):
                J[k,self.Nvirt] = 1
        
        return(J[spacer:])
    
    def train(self, S, y, S_valid, y_valid):
        alphas = np.logspace(-15,0,20)
        alphas[0] = 0.
        
        Ns = S.shape[0]
        Ns_valid = S_valid.shape[0]
        Y = y.reshape((Ns,1))
        Y_valid = y_valid.reshape((Ns_valid,1))
        
        errs = np.zeros(alphas.shape)
        for i in range(len(alphas)):
            self.W = Ridge_regression(S, Y, alphas[i])
            Y_pred = np.array(self.predict(S)).reshape(Ns,1)
            Y_pred_valid = np.array(self.predict(S_valid)).reshape(Ns_valid,1)
            errs[i] = NRMSE(Y_valid, Y_pred_valid)
            print("alpha = " + str(alphas[i]) + " ; NRMSE (train) = " + str(int(1000*NRMSE(Y,Y_pred))/1000) + " ; NRMSE (validation) = " + str(int(1000*NRMSE(Y_valid, Y_pred_valid))/1000))
    
        alpha_opt = alphas[np.argmin(errs)]
        print('Optimal alpha = ' + str(alpha_opt) + ' with NRMSE (validation) = ' + str(np.min(errs)))
        self.W = Ridge_regression(S, Y, alpha_opt)
    
    def train_without_SPN(self, J, y, J_valid, y_valid):
        alphas = np.logspace(-15,0,20)
        alphas[0] = 0.
        
        Ns = J.shape[0]
        Ns_valid = J_valid.shape[0]
        Y = y.reshape((Ns,1))
        Y_valid = y_valid.reshape((Ns_valid,1))
        
        errs = np.zeros(alphas.shape)
        for i in range(len(alphas)):
            self.W = Ridge_regression(J, Y, alphas[i])
            Y_pred_valid = np.array(self.predict(J_valid)).reshape(Ns_valid,1)
            errs[i] = NRMSE(Y_valid, Y_pred_valid)
            print(alphas[i], NRMSE(Y_valid, Y_pred_valid))
    
        alpha_opt = alphas[np.argmin(errs)]
        print('Optimal alpha = '+str(alpha_opt)+' with NRMSE = '+str(np.min(errs)))
        self.W = Ridge_regression(J, Y, alpha_opt)
    
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
    
    def get_time_list_y(self, y):
        #We need to make sure that time_y has Ns elements with a delay tau
        Ns = len(y)
        t_y = spacer*self.tau
        time_y = [t_y]
        while len(time_y)<Ns:
            t_y += self.tau
            time_y.append(t_y)
        return(np.array(time_y)*1e9)
    
    def get_time_list_S(self, S):
        Ns = S.shape[0]
        return(np.arange(spacer*self.tau,(Ns+spacer)*self.tau,self.theta)*1e9)


# %% [markdown]
# ### The task : NARMA10

# %%
Ntrain = 500
(u,y) = NARMA10(Ntrain)

net = Single_Node_Reservoir_NARMA10(40,1e-2,8e-2,0.26)
time_u = net.get_time_list_u(u)
time_y = net.get_time_list_y(y)

# %%
plt.figure(figsize=(10,6))
plt.plot(time_u[-100:],u[-100:],drawstyle='steps-post',label="Input")
plt.plot(time_y[-100:],y[-100:],drawstyle='steps-post',label="Desired output")
plt.xlabel("Time (ns)")
plt.legend(loc="best")
plt.show()

# %% [markdown]
# ### Aspect of the output

# %%
S = net.gen_signal(u)
time_S = net.get_time_list_S(S)

# %%
plt.figure(figsize=(10,6))
L = 5
plt.grid(True)
plt.plot(time_u[-L:],u[-L:],drawstyle='steps-post',label="Input")
#plt.plot(time_S[-L*net.Nvirt:],J.flatten()[-L*net.Nvirt:],drawstyle='steps-post',label="Transformed input")
plt.plot(time_S[-L*net.Nvirt:],S[:,:-1].flatten()[-L*net.Nvirt:],'r-',label="Signal")
plt.legend(loc="best")
plt.xlabel("Time (ns)")
#plt.ylim(-0.6,0.6)
plt.show()

# %%
T_theta_list = np.logspace(-3,2,15)
amplitude = []
Ntrain = 200
(u,y) = NARMA10(Ntrain)
N_mean = 10
L = 20
for T_t in T_theta_list:
    print(T_t)
    amp_mean = 0
    for i in range(N_mean):
        net = Single_Node_Reservoir_NARMA10(40,T_t,1,0.)
        S = net.gen_signal_fast(u)
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
plt.title("40 virtual nodes")
plt.show()

# %% [markdown]
# ### Training and fine-tuning

# %%
Ntrain = 5000
Nvalid = 5000

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)

net = Single_Node_Reservoir_NARMA10(400,3,1e-2,0.25)
S = net.gen_signal_fast(u)
S_valid = net.gen_signal_fast(u_valid)

# %%
net.train(S,y,S_valid,y_valid)

# %%
y_pred_train = net.predict(S)
y_pred_valid = net.predict(S_valid)

# %%
Ntest = 2500
(u_test,y_test) = NARMA10(Ntest)
S_test = net.gen_signal_fast(u_test)
y_pred_test = net.predict(S_test)

# %%
time_y = net.get_time_list_y(y_test)
plt.figure(figsize=(10,6))
xmin = 100
xmax = 200
plt.plot(time_y[xmin:xmax],y_test[xmin:xmax],drawstyle='steps-post',label="Desired output (test)")
plt.plot(time_y[xmin:xmax],y_pred_test[xmin:xmax],drawstyle='steps-post',label="Predicted output (test)")
plt.xlabel("Time (ns)")
plt.legend(loc="best")
plt.show()

# %%
print("NRMSE (train) = "+str(NRMSE_list(y,y_pred_train)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y,y_pred_train,'ro')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Ntrain))
H, xedges, yedges  = np.histogram2d(y,y_pred_train,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
print("NRMSE (validation) = "+str(NRMSE_list(y_valid,y_pred_valid)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_valid,y_pred_valid,'ro')
plt.xlabel("Desired output (validation)")
plt.ylabel("Predicted output (validation)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Nvalid))
H, xedges, yedges  = np.histogram2d(y_valid,y_pred_valid,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (validation)")
plt.ylabel("Predicted output (validation)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
print("NRMSE (test) = "+str(NRMSE_list(y_test,y_pred_test)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_test,y_pred_test,'ro')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Ntest))
H, xedges, yedges  = np.histogram2d(y_test,y_pred_test,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %% [markdown]
# #### 1. Influence of $T/\theta$

# %%
T_theta_list = np.logspace(-1,1.5,8)

Ntrain = 1500
Nvalid = 1500
Ntest = 750

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 5

for T_t in T_theta_list:
    print(T_t)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,T_t,1e-2,0.25)

        S = net.gen_signal_fast(u)
        S_valid = net.gen_signal_fast(u_valid)
        S_test = net.gen_signal_fast(u_test)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)

        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
    
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(T_theta_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(T_theta_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(T_theta_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$T/\theta$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %% [markdown]
# #### 2. Is the superparmagnetic network important?

# %%
Ntrain = 1500
Nvalid = 600
Ntest = 600

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

net = Single_Node_Reservoir_NARMA10(100,1e-1,3*8e-2,0.26,bias=False)

J = net.gen_signal_without_SPN(u)
J_valid = net.gen_signal_without_SPN(u_valid)
J_test = net.gen_signal_without_SPN(u_test)

net.train_without_SPN(J,y,J_valid,y_valid)

y_pred_train = net.predict(J)
y_pred_valid = net.predict(J_valid)
y_pred_test = net.predict(J_test)

print("NRMSE (train) = "+str(NRMSE_list(y,y_pred_train)))
print("NRMSE (validation) = "+str(NRMSE_list(y_valid,y_pred_valid)))
print("NRMSE (test) = "+str(NRMSE_list(y_test,y_pred_test)))

# %% [markdown]
# $T/\theta$ must be very low! Which means that the "memory" is useless when there is a feedback... The network doesn't work when we use J instead of S (without SPN) which tends to show that the crucial aspect of the SPN in nonlinearity (but not memory).

# %% [markdown]
# #### 3. Influence of $m_0$

# %%
m0_list = np.logspace(-2,0,8)

Ntrain = 1500
Nvalid = 1500
Ntest = 750

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 5

for m0 in m0_list:
    print(m0)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,3,m0,0.26)

        S = net.gen_signal_fast(u)
        S_valid = net.gen_signal_fast(u_valid)
        S_test = net.gen_signal_fast(u_test)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(m0_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(m0_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(m0_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$m_0$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %% [markdown]
# #### 4. Influence of $\gamma$

# %%
gamma_list = np.arange(0.1,0.5,0.05)

Ntrain = 1500
Nvalid = 1500
Ntest = 750

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 5

for gamma in gamma_list:
    print(gamma)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,3,1e-2,gamma)

        S = net.gen_signal_fast(u)
        S_valid = net.gen_signal_fast(u_valid)
        S_test = net.gen_signal_fast(u_test)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(gamma_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(gamma_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(gamma_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$\gamma$')
plt.ylabel("NRMSE")
#plt.xscale("log")
plt.show()

# %% [markdown]
# #### 4. Importance of the bias

# %%
bias_list = [True,False]

Ntrain = 1500
Nvalid = 1500
Ntest = 750

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 3

for use_bias in bias_list:
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,3,1e-2,0.25,bias=use_bias)

        S = net.gen_signal_fast(u)
        S_valid = net.gen_signal_fast(u_valid)
        S_test = net.gen_signal_fast(u_test)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))
    
print(NRMSE_train_mean)
print(NRMSE_valid_mean)
print(NRMSE_test_mean)

# %% [markdown]
# #### Influence of $\beta'$

# %%
beta_prime_list = np.logspace(1,2,5)

Ntrain = 500
Nvalid = 200
Ntest = 200

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_list = []
NRMSE_valid_list = []
NRMSE_test_list = []

N = 10

for bp in beta_prime_list:
    print(bp)
    NRMSE_train = 0
    NRMSE_valid = 0
    NRMSE_test = 0
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(400,1e-3,1e-1,0.25,beta_prime=bp)

        (J,S) = net.gen_signal(u)
        (J_valid,S_valid) = net.gen_signal(u_valid)
        (J_test,S_test) = net.gen_signal(u_test)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train += NRMSE_list(y,y_pred_train)
        NRMSE_valid += NRMSE_list(y_valid,y_pred_valid)
        NRMSE_test += NRMSE_list(y_test,y_pred_test)
        
    NRMSE_train_list.append(NRMSE_train/N)
    NRMSE_valid_list.append(NRMSE_valid/N)
    NRMSE_test_list.append(NRMSE_test/N)

# %%
plt.figure(figsize=(10,6))
plt.plot(beta_prime_list,NRMSE_train_list,marker='+',linestyle = '-',label="NRMSE (train)")
plt.plot(beta_prime_list,NRMSE_valid_list,marker='+',linestyle = '-',label="NRMSE (validation)")
plt.plot(beta_prime_list,NRMSE_test_list,marker='+',linestyle = '-',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$\beta^\prime$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %% [markdown]
# ### With 2 inputs (at $k$ and $k-1$)

# %%
Ntrain = 5000
Nvalid = 5000

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)

net = Single_Node_Reservoir_NARMA10(400,3,1e-2,0.25)
S = net.gen_signal_fast_2_inputs(u, 0.4)
S_valid = net.gen_signal_fast_2_inputs(u_valid, 0.4)

# %%
net.train(S,y,S_valid,y_valid)

# %%
y_pred_train = net.predict(S)
y_pred_valid = net.predict(S_valid)

# %%
Ntest = 2500
(u_test,y_test) = NARMA10(Ntest)
S_test = net.gen_signal_fast_2_inputs(u_test,0.4)
y_pred_test = net.predict(S_test)

# %%
time_y = net.get_time_list_y(y_test)
plt.figure(figsize=(10,6))
xmin = 100
xmax = 200
plt.plot(time_y[xmin:xmax],y_test[xmin:xmax],drawstyle='steps-post',label="Desired output (test)")
plt.plot(time_y[xmin:xmax],y_pred_test[xmin:xmax],drawstyle='steps-post',label="Predicted output (test)")
plt.xlabel("Time (ns)")
plt.legend(loc="best")
plt.show()

# %%
print("NRMSE (train) = "+str(NRMSE_list(y,y_pred_train)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y,y_pred_train,'ro')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Ntrain))
H, xedges, yedges  = np.histogram2d(y,y_pred_train,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
print("NRMSE (validation) = "+str(NRMSE_list(y_valid,y_pred_valid)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_valid,y_pred_valid,'ro')
plt.xlabel("Desired output (validation)")
plt.ylabel("Predicted output (validation)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Nvalid))
H, xedges, yedges  = np.histogram2d(y_valid,y_pred_valid,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (validation)")
plt.ylabel("Predicted output (validation)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
print("NRMSE (test) = "+str(NRMSE_list(y_test,y_pred_test)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_test,y_pred_test,'ro')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Ntest))
H, xedges, yedges  = np.histogram2d(y_test,y_pred_test,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
T_theta_list = np.logspace(0,1,5)

Ntrain = 500
Nvalid = 500
Ntest = 250

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for T_t in T_theta_list:
    print(T_t)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,T_t,1e-2,0.25)

        S = net.gen_signal_fast_2_inputs(u,0.5)
        S_valid = net.gen_signal_fast_2_inputs(u_valid,0.5)
        S_test = net.gen_signal_fast_2_inputs(u_test,0.5)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)

        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
    
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(T_theta_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(T_theta_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(T_theta_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$T/\theta$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %%
m0_list = np.logspace(-4,-2,5)

Ntrain = 500
Nvalid = 500
Ntest = 250

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for m0 in m0_list:
    print(m0)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,3,m0,0.25)

        S = net.gen_signal_fast_2_inputs(u,0.5)
        S_valid = net.gen_signal_fast_2_inputs(u_valid,0.5)
        S_test = net.gen_signal_fast_2_inputs(u_test,0.5)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(m0_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(m0_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(m0_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$m_0$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %%
input_ratio_list = np.linspace(0,0.5,6)

Ntrain = 500
Nvalid = 500
Ntest = 250

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for input_ratio in input_ratio_list:
    print(input_ratio)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,3,1e-2,0.25)

        S = net.gen_signal_fast_2_inputs(u,input_ratio)
        S_valid = net.gen_signal_fast_2_inputs(u_valid,input_ratio)
        S_test = net.gen_signal_fast_2_inputs(u_test,input_ratio)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(input_ratio_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(input_ratio_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(input_ratio_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel("Input ratio")
plt.ylabel("NRMSE")
#plt.xscale("log")
plt.show()

# %%
gamma_list = np.arange(0.1,0.5,0.05)

Ntrain = 500
Nvalid = 500
Ntest = 250

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for gamma in gamma_list:
    print(gamma)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,3,1e-2,gamma)

        S = net.gen_signal_fast_2_inputs(u,0.4)
        S_valid = net.gen_signal_fast_2_inputs(u_valid,0.4)
        S_test = net.gen_signal_fast_2_inputs(u_test,0.4)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(gamma_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(gamma_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(gamma_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$\gamma$')
plt.ylabel("NRMSE")
#plt.xscale("log")
plt.show()

# %% [markdown]
# ### With delayed feedback

# %%
Ntrain = 5000
Nvalid = 5000

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)

net = Single_Node_Reservoir_NARMA10(400,1e-2,1e-1/3.2,0.7/3.2)
S = net.gen_signal_fast_delayed_feedback(u, 1)
S_valid = net.gen_signal_fast_delayed_feedback(u_valid, 1)

# %%
net.train(S,y,S_valid,y_valid)

# %%
y_pred_train = net.predict(S)
y_pred_valid = net.predict(S_valid)

# %%
Ntest = 2500
(u_test,y_test) = NARMA10(Ntest)
S_test = net.gen_signal_fast_delayed_feedback(u_test,1)
y_pred_test = net.predict(S_test)

# %%
time_y = net.get_time_list_y(y_test)
plt.figure(figsize=(10,6))
xmin = 100
xmax = 200
plt.plot(time_y[xmin:xmax],y_test[xmin:xmax],drawstyle='steps-post',label="Desired output (test)")
plt.plot(time_y[xmin:xmax],y_pred_test[xmin:xmax],drawstyle='steps-post',label="Predicted output (test)")
plt.xlabel("Time (ns)")
plt.legend(loc="best")
plt.show()

# %%
print("NRMSE (train) = "+str(NRMSE_list(y,y_pred_train)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y,y_pred_train,'ro')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Ntrain))
H, xedges, yedges  = np.histogram2d(y,y_pred_train,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
print("NRMSE (test) = "+str(NRMSE_list(y_test,y_pred_test)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_test,y_pred_test,'ro')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Ntest))
H, xedges, yedges  = np.histogram2d(y_test,y_pred_test,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
T_theta_list = np.logspace(-4,0,5)

Ntrain = 500
Nvalid = 500
Ntest = 250

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for T_t in T_theta_list:
    print(T_t)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,T_t,1e-2,0.25)

        S = net.gen_signal_fast_delayed_feedback(u,1)
        S_valid = net.gen_signal_fast_delayed_feedback(u_valid,1)
        S_test = net.gen_signal_fast_delayed_feedback(u_test,1)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)

        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
    
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(T_theta_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(T_theta_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(T_theta_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$T/\theta$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %%
m0_list = np.logspace(-4,0,5)

Ntrain = 500
Nvalid = 500
Ntest = 250

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for m0 in m0_list:
    print(m0)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,1e-2,m0,0.25)

        S = net.gen_signal_fast_delayed_feedback(u,1)
        S_valid = net.gen_signal_fast_delayed_feedback(u_valid,1)
        S_test = net.gen_signal_fast_delayed_feedback(u_test,1)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(m0_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(m0_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(m0_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$m_0$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %%
delay_list = np.arange(1,10,2)

Ntrain = 500
Nvalid = 500
Ntest = 250

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for delay_fb in delay_list:
    print(delay_fb)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,1e-2,1e-2,0.25)

        S = net.gen_signal_fast_delayed_feedback(u,delay_fb)
        S_valid = net.gen_signal_fast_delayed_feedback(u_valid,delay_fb)
        S_test = net.gen_signal_fast_delayed_feedback(u_test,delay_fb)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(delay_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(delay_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(delay_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel("Delay on feedback")
plt.ylabel("NRMSE")
#plt.xscale("log")
plt.show()

# %%
gamma_list = np.arange(0.1,0.5,0.05)

Ntrain = 500
Nvalid = 500
Ntest = 250

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for gamma in gamma_list:
    print(gamma)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(100,1e-2,1e-2,gamma)

        S = net.gen_signal_fast_delayed_feedback(u,1)
        S_valid = net.gen_signal_fast_delayed_feedback(u_valid,1)
        S_test = net.gen_signal_fast_delayed_feedback(u_test,1)

        net.train(S,y,S_valid,y_valid)

        y_pred_train = net.predict(S)
        y_pred_valid = net.predict(S_valid)
        y_pred_test = net.predict(S_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(gamma_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(gamma_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(gamma_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$\gamma$')
plt.ylabel("NRMSE")
#plt.xscale("log")
plt.show()

# %% [markdown]
# #### Without the SPN?

# %%
Ntrain = 5000
Nvalid = 5000
Ntest = 2500

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

net = Single_Node_Reservoir_NARMA10(400,1e-2,1e-1,.7)

J = net.gen_signal_delayed_feedback_without_SPN(u,1)
J_valid = net.gen_signal_delayed_feedback_without_SPN(u_valid,1)
J_test = net.gen_signal_delayed_feedback_without_SPN(u_test,1)

net.train_without_SPN(J,y,J_valid,y_valid)

y_pred_train = net.predict(J)
y_pred_valid = net.predict(J_valid)
y_pred_test = net.predict(J_test)

print("NRMSE (train) = "+str(NRMSE_list(y,y_pred_train)))
print("NRMSE (validation) = "+str(NRMSE_list(y_valid,y_pred_valid)))
print("NRMSE (test) = "+str(NRMSE_list(y_test,y_pred_test)))

# %%
print("NRMSE (train) = "+str(NRMSE_list(y,y_pred_train)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y,y_pred_train,'ro')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Ntrain))
H, xedges, yedges  = np.histogram2d(y,y_pred_train,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (training)")
plt.ylabel("Predicted output (training)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
print("NRMSE (test) = "+str(NRMSE_list(y_test,y_pred_test)))
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(np.linspace(0,1.0),np.linspace(0,1.0), 'k--' )
plt.plot(y_test,y_pred_test,'ro')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.xlim(0,0.9)
plt.ylim(0,0.9)
plt.subplot(122)
nbins = int(2*np.sqrt(Ntest))
H, xedges, yedges  = np.histogram2d(y_test,y_pred_test,bins = nbins,range=[[0, 1], [0, 1]])
H = H.T
plt.imshow(H,origin='low',cmap='inferno')
plt.xlabel("Desired output (testing)")
plt.ylabel("Predicted output (testing)")
plt.xticks([],[''])
plt.yticks([],[''])
plt.xlim(0,0.9*nbins)
plt.ylim(0,0.9*nbins)
plt.show()

# %%
Y = J[:,:-2].flatten()
X = range(len(Y))
xmin=0
xmax=-1
plt.plot(X[xmin:xmax],Y[xmin:xmax],drawstyle='steps-post')
plt.show()

# %%
gamma_list = np.arange(0.6,.81,0.05)

Ntrain = 5000
Nvalid = 5000
Ntest = 2500

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for gamma in gamma_list:
    print(gamma)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(400,1e-2,1e-1,gamma)

        J = net.gen_signal_delayed_feedback_without_SPN(u,1)
        J_valid = net.gen_signal_delayed_feedback_without_SPN(u_valid,1)
        J_test = net.gen_signal_delayed_feedback_without_SPN(u_test,1)

        net.train_without_SPN(J,y,J_valid,y_valid)

        y_pred_train = net.predict(J)
        y_pred_valid = net.predict(J_valid)
        y_pred_test = net.predict(J_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(gamma_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(gamma_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(gamma_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$\gamma$')
plt.ylabel("NRMSE")
#plt.xscale("log")
plt.show()

# %%
m0_list = np.logspace(-1.5,-0.5,5)

Ntrain = 5000
Nvalid = 5000
Ntest = 2500

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for m0 in m0_list:
    print(m0)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(400,1e-2,m0,0.7)

        J = net.gen_signal_delayed_feedback_without_SPN(u,1)
        J_valid = net.gen_signal_delayed_feedback_without_SPN(u_valid,1)
        J_test = net.gen_signal_delayed_feedback_without_SPN(u_test,1)

        net.train_without_SPN(J,y,J_valid,y_valid)

        y_pred_train = net.predict(J)
        y_pred_valid = net.predict(J_valid)
        y_pred_test = net.predict(J_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(m0_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(m0_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(m0_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$m_0$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %%
#T_theta_list = np.logspace(-4,1,5)
T_theta_list = [1e-2]

Ntrain = 5000
Nvalid = 5000
Ntest = 2500

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 5

for T_t in T_theta_list:
    print(T_t)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(400,T_t,1e-1,0.8)

        J = net.gen_signal_delayed_feedback_without_SPN(u,1)
        J_valid = net.gen_signal_delayed_feedback_without_SPN(u_valid,1)
        J_test = net.gen_signal_delayed_feedback_without_SPN(u_test,1)

        net.train_without_SPN(J,y,J_valid,y_valid)

        y_pred_train = net.predict(J)
        y_pred_valid = net.predict(J_valid)
        y_pred_test = net.predict(J_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
    
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(T_theta_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(T_theta_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(T_theta_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$T/\theta$')
plt.ylabel("NRMSE")
plt.xscale("log")
plt.show()

# %%
print(NRMSE_test_mean)
print(NRMSE_test_std)

# %%
delay_list = np.arange(1,10)

Ntrain = 1500
Nvalid = 1500
Ntest = 750

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)
(u_test,y_test) = NARMA10(Ntest)

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []

N = 1

for delay_fb in delay_list:
    print(delay_fb)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        net = Single_Node_Reservoir_NARMA10(400,1e-2,1e-2,0.9)

        J = net.gen_signal_delayed_feedback_without_SPN(u,delay_fb)
        J_valid = net.gen_signal_delayed_feedback_without_SPN(u_valid,delay_fb)
        J_test = net.gen_signal_delayed_feedback_without_SPN(u_test,delay_fb)

        net.train_without_SPN(J,y,J_valid,y_valid)

        y_pred_train = net.predict(J)
        y_pred_valid = net.predict(J_valid)
        y_pred_test = net.predict(J_test)
        
        NRMSE_train.append(NRMSE_list(y,y_pred_train))
        NRMSE_valid.append(NRMSE_list(y_valid,y_pred_valid))
        NRMSE_test.append(NRMSE_list(y_test,y_pred_test))
        
    NRMSE_train_mean.append(np.mean(NRMSE_train))
    NRMSE_valid_mean.append(np.mean(NRMSE_valid))
    NRMSE_test_mean.append(np.mean(NRMSE_test))
    NRMSE_train_std.append(np.std(NRMSE_train,ddof=min(1,N-1)))
    NRMSE_valid_std.append(np.std(NRMSE_valid,ddof=min(1,N-1)))
    NRMSE_test_std.append(np.std(NRMSE_test,ddof=min(1,N-1)))

# %%
plt.figure(figsize=(10,6))
plt.errorbar(delay_list,NRMSE_train_mean,NRMSE_train_std,linestyle = '--',label="NRMSE (train)")
plt.errorbar(delay_list,NRMSE_valid_mean,NRMSE_valid_std,linestyle = '--',label="NRMSE (validation)")
plt.errorbar(delay_list,NRMSE_test_mean,NRMSE_test_std,linestyle = '--',label="NRMSE (test)")
plt.grid(True)
plt.legend(loc="best")
plt.xlabel("Delay on feedback")
plt.ylabel("NRMSE")
#plt.xscale("log")
plt.show()

# %% [markdown]
# ### With output feedback

# %%
