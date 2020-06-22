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


import SP_anisotropy_class as SPN

#3D plotting
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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
    
    def __init__(self, Nvirt, T_theta_ratio, m0, gamma, beta_prime = 10, k_off = 0., bias = True, joined_states = False):
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
            print("Use bias")
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
            print("Use bias")
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
            print("Use bias")
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
            print("Use bias")
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
                j = self.M[i]*u[k] + self.k_off #Offset
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
            S_joined = np.zeros((2*self.Nvirt+1,Ns))
        else:
            S_joined = np.zeros((2*self.Nvirt,Ns))
            
        for k in range(Ns):
            S_joined[k,:self.Nvirt] = S[k-1,:self.Nvirt]
            S_joined[k,self.Nvirt:2*self.Nvirt] = S[k,:self.Nvirt]
            if self.use_bias:
                S_joined[k,-1] = 1
        
        return(S_joined)


# %% [markdown]
# ## No feedback

# %% [markdown]
# ### Role of memory ($T/\theta$)

# %% slideshow={"slide_type": "-"}
bp_list = [10]
T_theta_list = np.logspace(-.5,.5,5)

Ntrain = 1000
Nvalid = 1000
Ntest = 500

Nv = 800
m0 = 1e-2
gamma = 0.

NRMSE_train_mean_vs_bp = []
NRMSE_valid_mean_vs_bp = []
NRMSE_test_mean_vs_bp = []
NRMSE_train_std_vs_bp = []
NRMSE_valid_std_vs_bp = []
NRMSE_test_std_vs_bp = []

for bp in bp_list:
    

    NRMSE_train_mean = []
    NRMSE_valid_mean = []
    NRMSE_test_mean = []
    NRMSE_train_std = []
    NRMSE_valid_std = []
    NRMSE_test_std = []

    N = 1

    for T_theta in T_theta_list:
        print(T_theta)
        NRMSE_train = []
        NRMSE_valid = []
        NRMSE_test = []

        for i in range(N):
            (u,y) = NARMA10(Ntrain)
            (u_valid,y_valid) = NARMA10(Nvalid)
            (u_test,y_test) = NARMA10(Ntest)
            
            net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,beta_prime=bp)

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
    
    NRMSE_train_mean_vs_bp.append(NRMSE_train_mean)
    NRMSE_valid_mean_vs_bp.append(NRMSE_valid_mean)
    NRMSE_test_mean_vs_bp.append(NRMSE_test_mean)
    NRMSE_train_std_vs_bp.append(NRMSE_train_std)
    NRMSE_valid_std_vs_bp.append(NRMSE_valid_std)
    NRMSE_test_std_vs_bp.append(NRMSE_test_std)

# %%
plt.figure(figsize=(10,6))
for i in range(len(bp_list)):
    bp = bp_list[i]
    plt.errorbar(T_theta_list,NRMSE_test_mean_vs_bp[i],NRMSE_test_std_vs_bp[i],linestyle = '--',label=r'$\beta^\prime = $'+str(bp))
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$T/\theta$')
plt.ylabel("NRMSE (test)")
plt.xscale("log")
plt.show()

# %% [markdown]
# ### Role of nonlinearity ($m_0$)

# %%
bp_list = [10]
NL_list = np.logspace(-3,0,7)

Ntrain = 1000
Nvalid = 1000
Ntest = 500

Nv = 800
T_theta = .5
gamma = 0.

NRMSE_train_mean_vs_bp = []
NRMSE_valid_mean_vs_bp = []
NRMSE_test_mean_vs_bp = []
NRMSE_train_std_vs_bp = []
NRMSE_valid_std_vs_bp = []
NRMSE_test_std_vs_bp = []

for bp in bp_list:
    
    #m0_list
    (u,y) = NARMA10(Ntrain)
    spn = SPN.SP_Network(h,theta_H,k_s_0,phi,bp)
    f_m = spn.get_f_m_eq()
    dx = 5e-2
    fp0 = (f_m(dx/2)-f_m(-dx/2))/(dx)
    f_inf = f_m(1)
    U = max(u)-min(u)
    m0_list = NL_list*f_inf/(U*fp0)

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
            
            (u,y) = NARMA10(Ntrain)
            (u_valid,y_valid) = NARMA10(Nvalid)
            (u_test,y_test) = NARMA10(Ntest)
            
            net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,beta_prime=bp)

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
    
    NRMSE_train_mean_vs_bp.append(NRMSE_train_mean)
    NRMSE_valid_mean_vs_bp.append(NRMSE_valid_mean)
    NRMSE_test_mean_vs_bp.append(NRMSE_test_mean)
    NRMSE_train_std_vs_bp.append(NRMSE_train_std)
    NRMSE_valid_std_vs_bp.append(NRMSE_valid_std)
    NRMSE_test_std_vs_bp.append(NRMSE_test_std)

# %%
plt.figure(figsize=(10,6))
for i in range(len(bp_list)):
    bp = bp_list[i]
    plt.errorbar(NL_list,NRMSE_test_mean_vs_bp[i],NRMSE_test_std_vs_bp[i],linestyle = '--',label=r'$\beta^\prime = $'+str(bp))
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$NL$')
plt.ylabel("NRMSE (test)")
plt.xscale("log")
plt.show()

# %%
spn = SPN.SP_Network(h,theta_H,k_s_0,phi,10)
f_m = spn.get_f_m_eq()
dx = 5e-2
fp0 = (f_m(dx/2)-f_m(-dx/2))/(dx)
f_inf = f_m(1)
(u,y) = NARMA10(Ntrain)
U = max(u)-min(u)
print(0.005*U*fp0/f_inf)

# %% [markdown]
# ## Feedback

# %% [markdown]
# ### Role of memory ($T/\theta$)

# %%
bp_list = [10,20,30]
T_theta_list = np.logspace(-2,1,7)

Ntrain = 1000
Nvalid = 1000
Ntest = 500

Nv = 800
NL = 2e-1
gamma_fp0 = 0.85

N = 5

NRMSE_train_mean_vs_bp = []
NRMSE_valid_mean_vs_bp = []
NRMSE_test_mean_vs_bp = []
NRMSE_train_std_vs_bp = []
NRMSE_valid_std_vs_bp = []
NRMSE_test_std_vs_bp = []

for bp in bp_list:   

    NRMSE_train_mean = []
    NRMSE_valid_mean = []
    NRMSE_test_mean = []
    NRMSE_train_std = []
    NRMSE_valid_std = []
    NRMSE_test_std = []

    #m0 and gamma
    (u,y) = NARMA10(Ntrain)
    spn = SPN.SP_Network(h,theta_H,k_s_0,phi,bp)
    f_m = spn.get_f_m_eq()
    dx = 5e-2
    fp0 = (f_m(dx/2)-f_m(-dx/2))/(dx)
    f_inf = f_m(1)
    U = max(u)-min(u)
    m0 = NL*f_inf/(U*fp0)
    gamma = gamma_fp0/fp0

    for T_theta in T_theta_list:
        print(T_theta)
        NRMSE_train = []
        NRMSE_valid = []
        NRMSE_test = []

        for i in range(N):
            (u,y) = NARMA10(Ntrain)
            (u_valid,y_valid) = NARMA10(Nvalid)
            (u_test,y_test) = NARMA10(Ntest)

            net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,beta_prime=bp)

            S = net.gen_signal_fast_delayed_feedback(u,3)
            S_valid = net.gen_signal_fast_delayed_feedback(u_valid,3)
            S_test = net.gen_signal_fast_delayed_feedback(u_test,3)

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
    
    NRMSE_train_mean_vs_bp.append(NRMSE_train_mean)
    NRMSE_valid_mean_vs_bp.append(NRMSE_valid_mean)
    NRMSE_test_mean_vs_bp.append(NRMSE_test_mean)
    NRMSE_train_std_vs_bp.append(NRMSE_train_std)
    NRMSE_valid_std_vs_bp.append(NRMSE_valid_std)
    NRMSE_test_std_vs_bp.append(NRMSE_test_std)

# %%
plt.figure(figsize=(10,6))
for i in range(len(bp_list)):
    bp = bp_list[i]
    plt.errorbar(T_theta_list,NRMSE_test_mean_vs_bp[i],NRMSE_test_std_vs_bp[i],linestyle = '--',label=r'$\beta^\prime = $'+str(bp))
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$T/\theta$')
plt.ylabel("NRMSE (test)")
plt.xscale("log")
plt.show()

# %% [markdown]
# ### Role of nonlinearity ($m_0$)

# %%
bp_list = [10,20,30]
NL_list = np.logspace(-2,0,7)

Ntrain = 1000
Nvalid = 1000
Ntest = 500

Nv = 800
T_theta = .3
gamma_fp0 = 0.9

N = 5

NRMSE_train_mean_vs_bp = []
NRMSE_valid_mean_vs_bp = []
NRMSE_test_mean_vs_bp = []
NRMSE_train_std_vs_bp = []
NRMSE_valid_std_vs_bp = []
NRMSE_test_std_vs_bp = []

for bp in bp_list:
    
    #m0_list and gamma
    (u,y) = NARMA10(Ntrain)
    spn = SPN.SP_Network(h,theta_H,k_s_0,phi,bp)
    f_m = spn.get_f_m_eq()
    dx = 5e-2
    fp0 = (f_m(dx/2)-f_m(-dx/2))/(dx)
    f_inf = f_m(1)
    U = max(u)-min(u)
    m0_list = NL_list*f_inf/(U*fp0)
    gamma = gamma_fp0/fp0

    NRMSE_train_mean = []
    NRMSE_valid_mean = []
    NRMSE_test_mean = []
    NRMSE_train_std = []
    NRMSE_valid_std = []
    NRMSE_test_std = []

    for m0 in m0_list:
        print(m0)
        NRMSE_train = []
        NRMSE_valid = []
        NRMSE_test = []

        for i in range(N):
            
            (u,y) = NARMA10(Ntrain)
            (u_valid,y_valid) = NARMA10(Nvalid)
            (u_test,y_test) = NARMA10(Ntest)
            
            net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,beta_prime=bp)

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
    
    NRMSE_train_mean_vs_bp.append(NRMSE_train_mean)
    NRMSE_valid_mean_vs_bp.append(NRMSE_valid_mean)
    NRMSE_test_mean_vs_bp.append(NRMSE_test_mean)
    NRMSE_train_std_vs_bp.append(NRMSE_train_std)
    NRMSE_valid_std_vs_bp.append(NRMSE_valid_std)
    NRMSE_test_std_vs_bp.append(NRMSE_test_std)

# %%
plt.figure(figsize=(10,6))
for i in range(len(bp_list)):
    bp = bp_list[i]
    plt.errorbar(NL_list,NRMSE_test_mean_vs_bp[i],NRMSE_test_std_vs_bp[i],linestyle = '--',label=r'$\beta^\prime = $'+str(bp))
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$NL$')
plt.ylabel("NRMSE (test)")
plt.xscale("log")
plt.show()

# %%
bp_list = [10,20]
gamma_fp0_list = np.linspace(.7,0.95,5)

Ntrain = 1000
Nvalid = 1000
Ntest = 500

Nv = 800
T_theta = .3
NL = 2e-1

N = 5

NRMSE_train_mean_vs_bp = []
NRMSE_valid_mean_vs_bp = []
NRMSE_test_mean_vs_bp = []
NRMSE_train_std_vs_bp = []
NRMSE_valid_std_vs_bp = []
NRMSE_test_std_vs_bp = []

for bp in bp_list:
    
    #gamma_list and m0
    (u,y) = NARMA10(Ntrain)
    spn = SPN.SP_Network(h,theta_H,k_s_0,phi,bp)
    f_m = spn.get_f_m_eq()
    dx = 5e-2
    fp0 = (f_m(dx/2)-f_m(-dx/2))/(dx)
    f_inf = f_m(1)
    U = max(u)-min(u)
    m0 = NL*f_inf/(U*fp0)
    gamma_list = gamma_fp0_list/fp0

    NRMSE_train_mean = []
    NRMSE_valid_mean = []
    NRMSE_test_mean = []
    NRMSE_train_std = []
    NRMSE_valid_std = []
    NRMSE_test_std = []

    for gamma in gamma_list:
        print(gamma)
        NRMSE_train = []
        NRMSE_valid = []
        NRMSE_test = []

        for i in range(N):
            
            (u,y) = NARMA10(Ntrain)
            (u_valid,y_valid) = NARMA10(Nvalid)
            (u_test,y_test) = NARMA10(Ntest)
            
            net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,beta_prime=bp)

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
    
    NRMSE_train_mean_vs_bp.append(NRMSE_train_mean)
    NRMSE_valid_mean_vs_bp.append(NRMSE_valid_mean)
    NRMSE_test_mean_vs_bp.append(NRMSE_test_mean)
    NRMSE_train_std_vs_bp.append(NRMSE_train_std)
    NRMSE_valid_std_vs_bp.append(NRMSE_valid_std)
    NRMSE_test_std_vs_bp.append(NRMSE_test_std)

# %%
plt.figure(figsize=(10,6))
for i in range(len(bp_list)):
    bp = bp_list[i]
    plt.errorbar(gamma_fp0_list,NRMSE_test_mean_vs_bp[i],NRMSE_test_std_vs_bp[i],linestyle = '--',label=r'$\beta^\prime = $'+str(bp))
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$\gamma f^\prime(0)$')
plt.ylabel("NRMSE (test)")
#plt.xscale("log")
plt.show()

# %%
NL_list = [2e-1]
Offset_list = np.linspace(0,1,6)

Ntrain = 1000
Nvalid = 1000
Ntest = 500

Nv = 400
T_theta = .3
gamma_fp0 = .9
bp = 10

N = 3

NRMSE_train_mean = np.zeros((len(Offset_list),len(NL_list)))
NRMSE_valid_mean = np.zeros((len(Offset_list),len(NL_list)))
NRMSE_test_mean = np.zeros((len(Offset_list),len(NL_list)))
    
#Constants
(u,y) = NARMA10(Ntrain)
spn = SPN.SP_Network(h,theta_H,k_s_0,phi,bp)
f_m = spn.get_f_m_eq()
dx = 5e-2
fp0 = (f_m(dx/2)-f_m(-dx/2))/(dx)
f_inf = f_m(1)
U = max(u)-min(u)
gamma = gamma_fp0/fp0

for i in range(len(Offset_list)):
    Offset = Offset_list[i]
    k_offset = Offset*f_inf/fp0
    
    for j in range(len(NL_list)):
        NL = NL_list[j]
        m0 = NL*f_inf/(U*fp0)

        NRMSE_train = []
        NRMSE_valid = []
        NRMSE_test = []

        for k in range(N):
            
            (u,y) = NARMA10(Ntrain)
            (u_valid,y_valid) = NARMA10(Nvalid)
            (u_test,y_test) = NARMA10(Ntest)
            
            net = Single_Node_Reservoir_NARMA10(Nv,T_theta,m0,gamma,beta_prime=bp,k_off=k_offset)

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
        
        NRMSE_train_mean[i,j] = np.mean(NRMSE_train)
        NRMSE_valid_mean[i,j] = np.mean(NRMSE_valid)
        NRMSE_test_mean[i,j] = np.mean(NRMSE_test)

# %%
plt.figure(figsize=(10,6))
plt.errorbar(Offset_list,NRMSE_test_mean[:,0],NRMSE_test_std[:,0],linestyle = '--')
plt.grid(True)
plt.xlabel("Offset")
plt.ylabel("NRMSE (test)")
#plt.xscale("log")
plt.show()

# %%
(u,y) = NARMA10(Ntrain)
spn = SPN.SP_Network(h,theta_H,k_s_0,phi,10)
f_m = spn.get_f_m_eq()
dx = 5e-2
fp0 = (f_m(dx/2)-f_m(-dx/2))/(dx)
f_inf = f_m(1)
U = max(u)-min(u)
print(7e-2*fp0*U/f_inf)
print(fp0*0.28)

# %%
Ntrain = 1000
Nvalid = 1000

(u,y) = NARMA10(Ntrain)
(u_valid,y_valid) = NARMA10(Nvalid)

net = Single_Node_Reservoir_NARMA10(1600,1e-1,7e-2,0.28)
S = net.gen_signal_fast_delayed_feedback(u, 3)
S_valid = net.gen_signal_fast_delayed_feedback(u_valid, 3)

net.train(S,y,S_valid,y_valid)

y_pred_train = net.predict(S)
y_pred_valid = net.predict(S_valid)

Ntest = 500
(u_test,y_test) = NARMA10(Ntest)
S_test = net.gen_signal_fast_delayed_feedback(u_test,3)
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
