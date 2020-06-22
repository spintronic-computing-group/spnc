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

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import random as rnd
import collections

import SP_anisotropy_class as SPN

#3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# %%
#Ignore the first 50 elements of the output
spacer = 50


# %%
def build_optimal_list_rec(L,sequences,p):
    #print(L)
    if len(L)==2**p:
        if check_cyclic_seq(L,sequences,p):
            L_final = L
            return(True,L_final)
        else:
            return(False,[0])
    else:
        #Add a 0
        L.append(0)
        last_seq = get_last_sequence(L,p)
        if check_last_seq(last_seq,sequences):
            sequences.append(last_seq)
            (success,L_final) = build_optimal_list_rec(L,sequences,p)
            if success:
                return(True,L_final)
        L.pop()
        
        #Add a 1
        L.append(1)
        last_seq = get_last_sequence(L,p)
        if check_last_seq(last_seq,sequences):
            sequences.append(last_seq)
            (success,L_final) = build_optimal_list_rec(L,sequences,p)
            if success:
                return(True,L_final)
        L.pop()
        return(False,[0])
            
def get_last_sequence(L,p):
    key = 0
    for k in range(p):
        key += (L[-p+k])*10**(p-k-1)
    key = int(key)
    return(key)


def check_last_seq(last_seq,sequences):
    if last_seq in sequences:
        return False
    else:
        return True

def check_cyclic_seq(L,sequences,p):
    L_cyclic = L[-p+1:]+L[:p-1]
    for k in range(p-1):
        last_seq = get_last_sequence(L_cyclic[k:k+p],p)
        if check_last_seq(last_seq,sequences)==False:
            return False
    return True


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

def NARMA10(Ns):
    # Ns is the number of samples
    u = np.random.random(Ns+50+spacer)*0.5
    y = np.zeros(Ns+50+spacer)
    for k in range(10,Ns+50+spacer):
        y[k] = 0.3*y[k-1] + 0.05*y[k-1]*np.sum(y[k-10:k]) + 1.5*u[k-1]*u[k-10] + 0.1
    return(u[50:],y[50+spacer:])

def mask_NARMA10(m0,Nvirt):
    # Nvirt is the number of virtual nodes
    mask = []
    for i in range(Nvirt):
        mask.append(rnd.choice([-1,1])*m0)
    return(mask)

def mask_positive_ac(m0,Nvirt):
    #Mask with strong positive autocorrelation
    mask = []
    for i in range(Nvirt):
        mask.append(m0)
    return(mask)

def mask_negative_ac(m0,Nvirt):
    #Mask with strong negative autocorrelation
    mask = []
    for i in range(Nvirt):
        mask.append(2*((i%2)-0.5)*m0)
    return(mask)

def mask_random(m0,Nvirt):
    #Random mask
    mask = []
    for i in range(Nvirt):
        mask.append(2*(rnd.random()-0.5)*m0)
    return(mask)

def mask_switch(m0,Nvirt,K_switch_max):
    #Mask with as many "combinations of products" as possible
    mask = []
    K_switch = 1
    k_current = 0
    sign = 1
    while len(mask)<Nvirt:
        mask.append(sign*m0)
        k_current += 1
        if k_current==K_switch:
            k_current = 0
            if sign==1:
                sign=-1
            else:
                sign=1
                K_switch+=1
                if K_switch>K_switch_max:
                    K_switch=1
    return(mask)

def mask_max_sequences(m0,Nvirt):
    p = int(np.log2(Nvirt))
    L = [1]*p
    (success,L_final) = build_optimal_list_rec(L,[],p)
    mask = [0]*Nvirt
    N_L = len(L_final)
    for i in range(N_L):
        mask[i] = m0*2*(L_final[i]-0.5)
    for i in range(Nvirt-N_L):
        mask[-1-i] = m0*2*(L_final[-1-i]-0.5)
    return(mask)

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
    
    def __init__(self, Nvirt, T_theta_ratio, m0, gamma, beta_prime = 10, k_off = 0., bias = True, joined_states = False, mask_type = "Default", K_switch_max = 4):
        self.Nin = 1
        self.Nvirt = Nvirt
        self.Nout = 1
        
        self.spn = SPN.SP_Network(h,theta_H,k_s_0,phi,beta_prime)
        SPN.calculate_energy_barriers(self.spn)
        self.T = 1./(self.spn.get_omega_prime()*f0)
        self.theta = self.T/T_theta_ratio
        self.tau = self.Nvirt*self.theta
        
        self.m0 = m0
        if mask_type=="Positive":
            self.M = mask_positive_ac(m0,Nvirt)
        elif mask_type=="Negative":
            self.M = mask_negative_ac(m0,Nvirt)
        elif mask_type=="Random":
            self.M = mask_random(m0,Nvirt)
        elif mask_type=="Max_Product":
            self.M = mask_switch(m0,Nvirt,K_switch_max)
        elif mask_type=="Max_Sequences":
            self.M = mask_max_sequences(m0,Nvirt)
        else:
            self.M = mask_NARMA10(m0,Nvirt)
            
        if bias:
            if joined_states:
                self.W = np.zeros((2*Nvirt+1,1))
            else:
                self.W = np.zeros((Nvirt+1,1))
        else:
            if joined_states:
                self.W = np.zeros((2*Nvirt,1))
            else:
                self.W = np.zeros((Nvirt,1))
        
        self.gamma = gamma
        self.k_off = k_off
        
        self.use_bias = bias
    
    def gen_signal(self, u):
        Ns = len(u)
        if self.use_bias:
            S = np.zeros((Ns,self.Nvirt+1))
        else:
            S = np.zeros((Ns,self.Nvirt))
        
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k] + self.k_off #Offset
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
            S = np.zeros((Ns,self.Nvirt+1))
        else:
            S = np.zeros((Ns,self.Nvirt))
        
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k] + self.k_off #Offset
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
            S = np.zeros((Ns,self.Nvirt+1))
        else:
            S = np.zeros((Ns,self.Nvirt))
            
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(Nin):
                #Input at k-1
                j = self.M[i]*u[k-1] + self.k_off #Offset
                self.spn.k_s = j + self.gamma*S[k-1,i] #Feedback
                self.spn.evolve_fast(f0,self.theta)
                S[k,i] = self.spn.get_m_fast()
            for i in range(Nin,self.Nvirt):
                #Input at k
                j = self.M[i]*u[k] + self.k_off #Offset
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
            S = np.zeros((Ns,self.Nvirt+1))
        else:
            S = np.zeros((Ns,self.Nvirt))
            
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k] + self.k_off #Offset
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
                j = self.M[i]*u[k] + self.k_off #Offset
                J[k,i] = f(j + self.gamma*J[k-1,i-delay_fb]) #Delayed Feedback 
                
        if self.use_bias:
            for k in range(Ns):
                J[k,self.Nvirt] = 1
        
        return(J[spacer:])
    
    def gen_signal_without_SPN(self,u):
        Ns = len(u)
        if self.use_bias:
            J = np.zeros((Ns,self.Nvirt+1))
        else:
            J = np.zeros((Ns,self.Nvirt))
        
        for k in range(Ns):
            if k%100==0:
                print(k)
            for i in range(self.Nvirt):
                j = self.M[i]*u[k] + self.k_off #Offset
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
    
    #Joined states
    
    def get_joined_states_signal(self, S):
        Ns = S.shape[0]
        if self.use_bias:
            S_joined = np.zeros((Ns,2*self.Nvirt+1))
        else:
            S_joined = np.zeros((Ns,2*self.Nvirt))
            
        for k in range(Ns):
            S_joined[k,:self.Nvirt] = S[k-1,:self.Nvirt]
            S_joined[k,self.Nvirt:2*self.Nvirt] = S[k,:self.Nvirt]
            if self.use_bias:
                S_joined[k,-1] = 1
        
        return(S_joined)


# %%
mask_type_list = ["Default","Positive","Negative","Random"]

Nv = 400
T_theta = .3
m0 = 7e-2
gamma = .28

N = 5

Ntrain = 1000
Nvalid = 1000
Ntest = 500

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []
NRMSE_train_total = []
NRMSE_valid_total = []
NRMSE_test_total = []

for MT in mask_type_list:
    print(MT)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        (u,y) = NARMA10(Ntrain)
        (u_valid,y_valid) = NARMA10(Nvalid)
        (u_test,y_test) = NARMA10(Ntest)
        
        net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,mask_type=MT)

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
    NRMSE_train_total.append(NRMSE_train)
    NRMSE_valid_total.append(NRMSE_valid)
    NRMSE_test_total.append(NRMSE_test)

# %%
plt.figure(figsize=(10,6))
plt.errorbar(range(len(mask_type_list)),NRMSE_test_mean,NRMSE_test_std,linestyle='')
plt.xticks(range(len(mask_type_list)),mask_type_list)
plt.show()

# %%
K_switch_max_list = [-1,2,4,8,16]

Nv = 400
T_theta = .3
m0 = 7e-2
gamma = .28

N = 10

Ntrain = 1000
Nvalid = 1000
Ntest = 500

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []
NRMSE_train_total = []
NRMSE_valid_total = []
NRMSE_test_total = []

for K_s_m in K_switch_max_list:
    print(K_s_m)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    MT = "Max_Product"
    if K_s_m<1:
        MT = "Default"
        
    print(MT)
    
    for i in range(N):
        (u,y) = NARMA10(Ntrain)
        (u_valid,y_valid) = NARMA10(Nvalid)
        (u_test,y_test) = NARMA10(Ntest)
        
        net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,mask_type=MT,K_switch_max=K_s_m)

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
    NRMSE_train_total.append(NRMSE_train)
    NRMSE_valid_total.append(NRMSE_valid)
    NRMSE_test_total.append(NRMSE_test)

# %%
plt.figure(figsize=(10,6))
plt.errorbar(range(len(K_switch_max_list)),NRMSE_test_mean,NRMSE_test_std,linestyle='')
plt.xticks(range(len(K_switch_max_list)),["Default"]+K_switch_max_list[1:])
plt.show()

# %%
mask_type_list = ["Max_Sequences","Default"]

Nv = 512
T_theta = .3
m0 = 7e-2
gamma = .28

N = 5

Ntrain = 1000
Nvalid = 1000
Ntest = 500

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []
NRMSE_train_total = []
NRMSE_valid_total = []
NRMSE_test_total = []

for MT in mask_type_list:
    print(MT)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        (u,y) = NARMA10(Ntrain)
        (u_valid,y_valid) = NARMA10(Nvalid)
        (u_test,y_test) = NARMA10(Ntest)
        
        net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,mask_type=MT)

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
    NRMSE_train_total.append(NRMSE_train)
    NRMSE_valid_total.append(NRMSE_valid)
    NRMSE_test_total.append(NRMSE_test)

# %%
mask_type_list = ["Max_Sequences","Default"]

Nv = 512
T_theta = .3
m0 = 1e-1
gamma = .7

N = 10

Ntrain = 1000
Nvalid = 1000
Ntest = 500

NRMSE_train_mean = []
NRMSE_valid_mean = []
NRMSE_test_mean = []
NRMSE_train_std = []
NRMSE_valid_std = []
NRMSE_test_std = []
NRMSE_train_total = []
NRMSE_valid_total = []
NRMSE_test_total = []

for MT in mask_type_list:
    print(MT)
    NRMSE_train = []
    NRMSE_valid = []
    NRMSE_test = []
    
    for i in range(N):
        (u,y) = NARMA10(Ntrain)
        (u_valid,y_valid) = NARMA10(Nvalid)
        (u_test,y_test) = NARMA10(Ntest)
        
        net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,mask_type=MT)

        S = net.gen_signal_delayed_feedback_without_SPN(u,1)
        S_valid = net.gen_signal_delayed_feedback_without_SPN(u_valid,1)
        S_test = net.gen_signal_delayed_feedback_without_SPN(u_test,1)

        net.train_without_SPN(S,y,S_valid,y_valid)

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
    NRMSE_train_total.append(NRMSE_train)
    NRMSE_valid_total.append(NRMSE_valid)
    NRMSE_test_total.append(NRMSE_test)

# %%
p = 7
Nv = 2**p
net = Single_Node_Reservoir_NARMA10(Nv,.3,1,.28,mask_type="Max_Sequences")
counter = count_seq_net(net,p)
print(counter.keys())
print(counter.values())
print(2**p-len(counter.values()))

# %%
#Nvirt = 400
plt.figure(figsize=(10,6))
plt.errorbar(range(len(mask_type_list)),NRMSE_test_mean,NRMSE_test_std,linestyle='')
plt.xticks(range(len(mask_type_list)),mask_type_list)
plt.show()

# %%
#Nvirt = 128
plt.figure(figsize=(10,6))
plt.errorbar(range(len(mask_type_list)),NRMSE_test_mean,NRMSE_test_std,linestyle='')
plt.xticks(range(len(mask_type_list)),mask_type_list)
plt.show()

# %%
#Nvirt = 512
plt.figure(figsize=(10,6))
plt.errorbar(range(len(mask_type_list)),NRMSE_test_mean,NRMSE_test_std,linestyle='')
plt.xticks(range(len(mask_type_list)),mask_type_list)
plt.show()

# %%
#Nvirt = 512
#Without SPN
plt.figure(figsize=(10,6))
plt.errorbar(range(len(mask_type_list)),NRMSE_test_mean,NRMSE_test_std,linestyle='')
plt.xticks(range(len(mask_type_list)),mask_type_list)
plt.show()

# %%
Ntrain = 1000
Nvalid = 1000

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)

net = Single_Node_Reservoir_NARMA10(20,3e-1,7e-2,0.28,mask_type="Max_Product")
print(net.M)
S = net.gen_signal_fast_delayed_feedback(u, 1)
S_valid = net.gen_signal_fast_delayed_feedback(u_valid, 1)

net.train(S,y,S_valid,y_valid)

y_pred_train = net.predict(S)
y_pred_valid = net.predict(S_valid)

Ntest = 500
(u_test,y_test) = NARMA10(Ntest)
S_test = net.gen_signal_fast_delayed_feedback(u_test,1)
y_pred_test = net.predict(S_test)

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
Ntrain = 1000
Nvalid = 1000

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)

net = Single_Node_Reservoir_NARMA10(200,3e-1,7e-2,0.28,joined_states=True)
S = net.gen_signal_fast_delayed_feedback(u, 1)
S_valid = net.gen_signal_fast_delayed_feedback(u_valid, 1)
S_joined = net.get_joined_states_signal(S)
S_valid_joined = net.get_joined_states_signal(S_valid)

net.train(S_joined,y,S_valid_joined,y_valid)

y_pred_train = net.predict(S_joined)
y_pred_valid = net.predict(S_valid_joined)

Ntest = 500
(u_test,y_test) = NARMA10(Ntest)
S_test = net.gen_signal_fast_delayed_feedback(u_test,1)
S_test_joined = net.get_joined_states_signal(S_test)
y_pred_test = net.predict(S_test_joined)

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
def count_seq(mask_type,p,Nvirt,K_switch_max=4):
    mask = []
    if mask_type=="Positive":
        mask = mask_positive_ac(1,Nvirt)
    elif mask_type=="Negative":
        mask = mask_negative_ac(1,Nvirt)
    elif mask_type=="Max_Product":
        mask = mask_switch(1,Nvirt,K_switch_max)
    else:
        mask = mask_NARMA10(1,Nvirt)
    sequences = []
    for i in range(Nvirt-p+1):
        key = 0
        for k in range(p):
            key += ((mask[i+k]+1)/2)*10**(p-k-1)
        key = int(key)
        sequences.append(key)
    counter=collections.Counter(sequences)
    return(counter)

def count_seq_net(net,p):
    mask = net.M
    mask = mask+mask
    sequences = []
    for i in range(net.Nvirt):
        key = 0
        for k in range(p):
            key += ((mask[i+k]+1)/2)*10**(p-k-1)
        key = int(key)
        sequences.append(key)
    counter=collections.Counter(sequences)
    return(counter)


# %%
mask_type_1 = "Max_Product"
mask_type_2 = "Default"
p = 8
Nv = 300
K_s_m = 6
counter_1 = count_seq(mask_type_1,p,Nv,K_switch_max = K_s_m)
print(counter_1.keys())
print(counter_1.values())
print(2**p-len(counter_1.values()))
counter_2 = count_seq(mask_type_2,p,Nv,K_switch_max = K_s_m)
print(counter_2.keys())
print(counter_2.values())
print(2**p-len(counter_2.values()))
list_missed = []
for x in counter_2.keys():
    list_missed.append(x)
for x in counter_1.keys():
    if x in list_missed:
        list_missed.remove(x)
print(list_missed)

# %%
p = 3
L = [1]*p
(success,L_final) = build_optimal_list_rec(L,[],p)
print(L_final)

# %%
