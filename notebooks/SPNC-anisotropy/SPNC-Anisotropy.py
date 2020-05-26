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
# # SPNC - Control of magnetization through anistropy

# %% [markdown]
# ## The Stoner-Wolfarth model - control of $\theta_H$
#
# We consider a superparamagnetic material in a field $H$, forming an angle $\theta_H$ with the magnet's easy axis. The energy of our system reads:
#
# $$E(\theta,\theta_H) = KV\sin^2{\theta} -\mu_0M_SVH\cos{(\theta-\theta_H)}$$
#
# where $K$ is the constant of anisotropy along the easy axis, $M_S$ is the satured magnetization, $V$ the volume of the magnet and $\theta$ the angle between the easy axis and the magnetization $m$. Here we are interested in the behaviour of the extrema of $E(\theta)$ when we change $\theta_H$. Knowing the extrema of $E(\theta)$, we will be able to calculate the energy barriers of this two-state system. Using Arhhenius equation, we will plot the evolution of the system's magnetization at equilibrium, according to $\theta_H$.

# %% jupyter={"outputs_hidden": false}
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

# %% [markdown]
# Let's fix the constant parameters and define the energy function.

# %% jupyter={"outputs_hidden": false}
K = 1
V = 1
mu_0 = 1
M_S = 2
H = 0.4
H_K = 2*K/(mu_0*M_S)


# %%
def energy(theta,theta_H):
    mu_H = mu_0*M_S*H
    return(K*V*np.sin(theta*np.pi/180)**2-mu_H*np.cos((theta-theta_H)*np.pi/180))


# %% [markdown]
# When $\theta_H = 30°$, the energy landscape looks like this:

# %% jupyter={"outputs_hidden": false}
theta = np.linspace(-180,180,100)
theta_H = 30
E = energy(theta,theta_H)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(theta, E, label = r'$\theta_H =$'+str(theta_H)+"°")
plt.grid(True)
plt.xlim(-180,180)
plt.legend(loc="best")
plt.xlabel(r'$\theta$')
plt.ylabel('Energy')
plt.title("Energy landscape with H/H_K = "+str(H/H_K))
plt.show()

# %% [markdown]
# There are 4 extrema in this landscape. The positions of minima will be called $\theta_1$ and $\theta_2$. From each minimum, there are two ways of going to the other minimum. One way is to go through the "high maxima". In this case the energy barriers are called $E_{12,+}$ and $E_{21,+}$. The other way is going through the "low maxima". In this case the energy barriers are called $E_{12,-}$ and $E_{21,-}$.

# %% jupyter={"source_hidden": true}
from IPython.display import Image
Image(filename="images/Energy_landscape_legend.png")


# %% [markdown]
# ### Energy barriers
#
# We can know evaluate the energy barriers of this landscape, according to the value of $\theta_H$. We both save the value of $\theta$ and the value $E(\theta)$ on the extrema.

# %% jupyter={"outputs_hidden": true}
def energy_barriers(theta_H):
    theta = np.linspace(-180,180,1000)
    E = energy(theta,theta_H)
    
    #Localization of extrema
    id_max = argrelextrema(E, np.greater)[0]
    id_min = argrelextrema(E, np.less)[0]
    
    #Two-state case
    if(len(id_max)==2 and len(id_min)==2):
        theta_1 = theta[id_min[0]]
        theta_2 = theta[id_min[1]]
        e_12_1 = E[id_max[0]]-E[id_min[0]]
        e_21_1 = E[id_max[0]]-E[id_min[1]]
        e_12_2 = E[id_max[1]]-E[id_min[0]]
        e_21_2 = E[id_max[1]]-E[id_min[1]]
        
    #Minimas in 0° and 180°
    elif(len(id_max)==2):
        print("Exception 1 "+str(theta_H))
        theta_1 = 0
        theta_2 = 180
        e_12_1 = E[id_max[0]]-energy(0,theta_H)
        e_21_1 = E[id_max[0]]-energy(180,theta_H)
        e_12_2 = E[id_max[1]]-energy(0,theta_H)
        e_21_2 = E[id_max[1]]-energy(180,theta_H)
        
    #The function argrelextrema fails for discrete values of theta_H, therefore we slightly change theta_H
    else:
        print("Exception 2 "+str(theta_H))
        E = energy(theta,theta_H-0.01)
        id_max = argrelextrema(E, np.greater)[0]
        id_min = argrelextrema(E, np.less)[0]
        theta_1 = theta[id_min[0]]
        theta_2 = theta[id_min[1]]
        e_12_1 = E[id_max[0]]-E[id_min[0]]
        e_21_1 = E[id_max[0]]-E[id_min[1]]
        e_12_2 = E[id_max[1]]-E[id_min[0]]
        e_21_2 = E[id_max[1]]-E[id_min[1]]
    
    return(theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2)


# %% jupyter={"outputs_hidden": false}
Theta_H = np.linspace(0,180,100)
E_12_1 = []
E_21_1 = []
E_12_2 = []
E_21_2 = []
Theta_1 = []
Theta_2 = []
for theta_H in Theta_H:
    (theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2) = energy_barriers(theta_H)
    Theta_1.append(theta_1)
    Theta_2.append(theta_2)
    E_12_1.append(e_12_1)
    E_21_1.append(e_21_1)
    E_12_2.append(e_12_2)
    E_21_2.append(e_21_2)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(Theta_H,E_12_1,'g--',label=r'$E_{12,+}$')
plt.plot(Theta_H,E_21_1,'r--',label=r'$E_{21,+}$')
plt.plot(Theta_H,E_12_2,'g-',label=r'$E_{12,-}$')
plt.plot(Theta_H,E_21_2,'r-',label=r'$E_{21,-}$')
plt.plot(Theta_H,np.array(E_12_2)-np.array(E_21_2),'b-',label=r'$E_{12,-}-E_{21,-}$')
plt.legend(loc = "best")
plt.xlim(0,180)
plt.grid(True)
plt.xlabel(r'$\theta_H$')
plt.ylabel(r'Energy')
plt.title("Energy barriers with H/H_K = "+str(H/H_K))
plt.show()

# %% [markdown]
# $(E_{12,-}-E_{21,-})$ changes with $\theta_H$, which is what we are looking for!
#
# *We may note that $E_{12,-}-E_{21,-}$ is almost linear in $\theta_H$ on a wide range around $\theta_H=90°$*

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(Theta_H,Theta_1,'g-',label=r'$\theta_1$')
plt.plot(Theta_H,Theta_2,'r-',label=r'$\theta_2$')
plt.legend(loc = "best")
plt.xlim(0,180)
plt.grid(True)
plt.xlabel(r'$\theta_H$')
plt.ylabel(r'$\theta_{eq}$')
plt.title("Angles of equilibrium with H/H_K = "+str(H/H_K))
plt.show()

# %% [markdown]
# ### Arrhenius equation
#
# According to Arrhenius equation, if the temperature is low enough ($E_b/(k_BT) = KV/(k_BT) > 3$), **and if there were only one barrier**, the transition rate from one state to the other would read:
#
# $$\omega = \frac{1}{\tau} = f_0\exp{\left(\frac{-E_b}{K_BT}\right)}$$
#
# where $E_b$ is the energy barrier, $k_B$ the Boltzmann constant and $T$ the temperature. However, we have two barriers of energy. For the moment, we will write de transition rate like this:
#
# $$\omega = \frac{1}{\tau} = f_{0,-}\exp{\left(\frac{-E_{b,-}}{K_BT}\right)}+f_{0,+}\exp{\left(\frac{-E_{b,+}}{K_BT}\right)}$$
#
# We will fix $f_{0,-}=f_{0,+}=1$. We will call $T_{max}$ the temperature verifying $KV/(k_BT) = 3$. Above $T_{max}$, the Arrhenius law cannot be used anymore and our simulation is incorrect.

# %% jupyter={"outputs_hidden": true}
f_0_1=1
f_0_2=2
def omega(e_b_1,e_b_2,k_BT):
    return(f_0_1*np.exp(-e_b_1/k_BT)+f_0_2*np.exp(-e_b_2/k_BT))


# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
k_BT = 0.1
plt.plot(Theta_H,omega(np.array(E_12_1),np.array(E_12_2),k_BT),'r-',label=r'$\omega_{12}$')
plt.plot(Theta_H,omega(np.array(E_21_1),np.array(E_21_2),k_BT),'g-',label=r'$\omega_{21}$')
plt.legend(loc = "best")
plt.xlim(0,180)
plt.grid(True)
plt.xlabel(r'$\theta_H$')
plt.ylabel(r'$\omega$')
plt.title("Transition rates with H/H_K = "+str(H/H_K)+" and k_BT = "+str(k_BT))
plt.show()

# %% [markdown]
# ### Two-states system
#
# In this two-states system, the probability of being in state $i$ follows the equation:
#
# $$p_i(t) =  \frac{\omega_{ji}}{\omega} + \left[p_i(0) - \frac{\omega_{ji}}{\omega} \right] \exp{(-\omega t)}$$
#
# where $\omega = \omega_{21}+\omega_{12}$. Let's first look at the shape of $\omega(\theta_H)$.

# %% jupyter={"outputs_hidden": false}
k_BT = 0.1
omega_12 = omega(np.array(E_12_1),np.array(E_12_2),k_BT)
omega_21 = omega(np.array(E_21_1),np.array(E_21_2),k_BT)
omega_tot = omega_12 + omega_21

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(Theta_H,omega_tot,color="magenta")
plt.xlim(0,180)
plt.grid(True)
plt.xlabel(r'$\theta_H$')
plt.ylabel(r'$\omega$')
plt.title("Transition rate "+r'$\omega$'+" with H/H_K = "+str(H/H_K)+" and k_BT = "+str(k_BT))
plt.show()


# %% [markdown]
# Let's see what happens when we start from equilibrium with $\theta_H=90°$ (so that $p_1(0)=p_2(0)=\frac{1}{2}$) and instaneously set $\theta_H=45°$.

# %% jupyter={"outputs_hidden": true}
def probability_state(t,p_0,omega,omega_ji):
    return(omega_ji/omega + (p_0-omega_ji/omega)*np.exp(-omega*t))


# %% jupyter={"outputs_hidden": false}
theta_H_input = 45
k_BT = 0.1
p_0_1 = 0.5
p_0_2 = 0.5
(theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2) = energy_barriers(theta_H_input)
omega_12 = omega(e_12_1,e_12_2,k_BT)
omega_21 = omega(e_21_1,e_21_2,k_BT)
omega_tot = omega_12 + omega_21
Time = np.linspace(0,10,100)
p_1 = probability_state(Time,p_0_1,omega_tot,omega_21)
p_2 = probability_state(Time,p_0_2,omega_tot,omega_12)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(Time,p_1,'g-',label=r'$p_1(t)$')
plt.plot(Time,p_2,'r-',label=r'$p_2(t)$')
plt.legend(loc="best")
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Probability')
plt.title("Evolution of "+r'$p(t)$'+" with H/H_K = "+str(H/H_K)+" and k_BT = "+str(k_BT))
plt.show()

# %% [markdown]
# We can also plot the evolution of $p_1(t=\infty)=p_{1,eq}$ and $p_2(t=\infty)=p_{2,eq}$ as functions of $\theta_H$.

# %% jupyter={"outputs_hidden": false}
k_BT = 0.1
omega_12 = omega(np.array(E_12_1),np.array(E_12_2),k_BT)
omega_21 = omega(np.array(E_21_1),np.array(E_21_2),k_BT)
omega_tot = omega_12 + omega_21

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(Theta_H,omega_21/omega_tot,color="green",label=r'$p_{1,eq}$')
plt.plot(Theta_H,omega_12/omega_tot,color="red",label=r'$p_{2,eq}$')
plt.legend(loc="best")
plt.xlim(0,180)
plt.grid(True)
plt.xlabel(r'$\theta_H$')
#plt.yscale("log")
plt.ylabel(r'$p_{eq}$')
plt.title("Probabilities for being in each state (at equilibrium) with H/H_K = "+str(H/H_K)+" and k_BT = "+str(k_BT))
plt.show()

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(Theta_H,omega_21/omega_12,color="orange")
plt.xlim(-0,180)
plt.grid(True)
plt.yscale("log")
plt.ylabel(r'$p_{1,eq}/p_{2,eq}$')
plt.xlabel(r'$\theta_H$')
plt.title("Ratio of probabilities for being in each state (at equilibrium)")
plt.show()


# %% [markdown]
# ### Magnetization at equilibrium
#
# Here, the states 1 and 2 are not necessarily aligned with the easy axis. Therefore, the projection of the magnetization along the easy axis (normalized) is:
#
# $$m(t) = \cos{\theta_1}p_1(t) + \cos{\theta_2}p_2(t)$$
#
# Which gives the following expression for $m(t)$:
#
# $$m(t) = m_{eq} + \left[m(0) - m_{eq} \right]\exp{(-\omega t)}$$
#
# where $m_{eq}$ is is the magnetization at equilibrium (projected on the east axis) and reads:
#
# $$m_{eq} = \frac{\cos{\theta_1}\omega_{21} + \cos{\theta_2}\omega_{12}}{\omega}$$

# %% jupyter={"outputs_hidden": true}
def mag_eq(theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2,k_BT):
    w_12 = omega(e_12_1,e_12_2,k_BT)
    w_21 = omega(e_21_1,e_21_2,k_BT)
    return((np.cos(theta_1*np.pi/180)*w_21+np.cos(theta_2*np.pi/180)*w_12)/(w_21+w_12))


# %% jupyter={"outputs_hidden": true}
Temperatures = [0.1, 0.3, 1, 3, 10]

# %% jupyter={"outputs_hidden": false}
Mag_eq_T = []
for k_BT in Temperatures:
    Mag_eq = []
    for i in range(len(Theta_1)):
        Mag_eq.append(mag_eq(Theta_1[i],Theta_2[i],E_12_1[i],E_21_1[i],E_12_2[i],E_21_2[i],k_BT))
    Mag_eq_T.append(Mag_eq)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
for i in range(len(Temperatures)):
    plt.plot(Theta_H,Mag_eq_T[i],label="k_BT = "+str(Temperatures[i]))
plt.xlim(0,180)
plt.legend(loc="best")
plt.grid(True)
plt.xlabel(r'$\theta_H$')
plt.ylabel(r'$m_{eq}$')
plt.title("Mean magnetization at equilibrium with H/H_K = "+str(H/H_K))
plt.show()


# %% [markdown]
# Since those curves look like hyperbolic tangent, let's try to interpolate those curves with the $\tanh$ function!
#
# $$m_{eq}=-M\tanh{\left(\alpha\left(\theta_H-\frac{\pi}{2}\right)\right)}$$

# %% jupyter={"outputs_hidden": true}
def interpolate_tanh(x,alpha,M):
    return(-M*np.tanh(alpha*(x-90)*np.pi/180))


# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
for i in range(len(Temperatures)):
    plt.plot(Theta_H,Mag_eq_T[i],label="k_BT = "+str(Temperatures[i]))
    popt, pcov = curve_fit(interpolate_tanh, Theta_H, Mag_eq_T[i])
    alpha = popt[0]
    M = popt[1]
    if (i==len(Temperatures)-1):
        plt.plot(Theta_H, interpolate_tanh(Theta_H,alpha,M),'k--',label="tanh fits")
    else:
        plt.plot(Theta_H, interpolate_tanh(Theta_H,alpha,M),'k--')
plt.xlim(0,180)
plt.legend(loc="best")
plt.grid(True)
plt.xlabel(r'$\theta_H$')
plt.ylabel(r'$M_{eq}$')
plt.title("Mean magnetization at equilibrium with H/H_K = "+str(H/H_K)+" and interpolations with hyperbolic tangents")
plt.show()

# %% [markdown]
# We can try to evaluate $\alpha$ and $M$ as functions of $k_BT$ in $m_{eq}=-M\tanh{(\alpha \theta_H)}$.

# %% jupyter={"outputs_hidden": false}
Temperatures_large = np.logspace(-2,1,30)
Mag_eq_T = []
for k_BT in Temperatures_large:
    Mag_eq = []
    for i in range(len(Theta_1)):
        Mag_eq.append(mag_eq(Theta_1[i],Theta_2[i],E_12_1[i],E_21_1[i],E_12_2[i],E_21_2[i],k_BT))
    Mag_eq_T.append(Mag_eq)

# %% jupyter={"outputs_hidden": false}
Alpha_vs_T = []
M_vs_T = []
for i in range(len(Temperatures_large)):
    popt, pcov = curve_fit(interpolate_tanh, Theta_H, Mag_eq_T[i])
    Alpha_vs_T.append(popt[0])
    M_vs_T.append(popt[1])

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (12,8))
k_BTmax = K*V/3
plt.subplot(211)
plt.axvline(x=k_BTmax,linestyle='--',color="red",label=r'$k_BT_{max}$')
plt.plot(Temperatures_large, Alpha_vs_T, color = "blue", marker = '+')
plt.grid(True)
plt.legend(loc="best")
plt.ylabel(r'$\alpha$')
plt.title("Alpha as a function of k_BT with H/H_K = "+str(H/H_K))
plt.xscale("log")
plt.yscale("log")
plt.subplot(212)
plt.axvline(x=k_BTmax,linestyle='--',color="red")
plt.plot(Temperatures_large, M_vs_T, color = "orange", marker='+')
plt.grid(True)
plt.xlabel(r'$k_BT$')
plt.ylabel(r'$M$')
plt.xscale("log")
plt.yscale("log")
plt.title("M as a function of k_BT with H/H_K = "+str(H/H_K))
plt.show()

# %% [markdown]
# We can distinguish two regimes for $\alpha$ and $M$, depending on $k_BT$. We can try to get an intuition of the threshold $k_BT_{lim}$ above which the behaviours of $\alpha$ and $M$ change.

# %% jupyter={"outputs_hidden": false}
H_list = [0.01,0.03,0.1,0.3]

Alpha_vs_H = []
M_vs_H = []

for h in H_list:
    H = h

    Theta_H = np.linspace(0,180,50)
    E_12_1 = []
    E_21_1 = []
    E_12_2 = []
    E_21_2 = []
    Theta_1 = []
    Theta_2 = []
    for theta_H in Theta_H:
        (theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2) = energy_barriers(theta_H)
        Theta_1.append(theta_1)
        Theta_2.append(theta_2)
        E_12_1.append(e_12_1)
        E_21_1.append(e_21_1)
        E_12_2.append(e_12_2)
        E_21_2.append(e_21_2)

    Temperatures_large = np.logspace(-2,1,30)
    Mag_eq_T = []
    for k_BT in Temperatures_large:
        Mag_eq = []
        for i in range(len(Theta_1)):
            Mag_eq.append(mag_eq(Theta_1[i],Theta_2[i],E_12_1[i],E_21_1[i],E_12_2[i],E_21_2[i],k_BT))
        Mag_eq_T.append(Mag_eq)

    Alpha_vs_T = []
    M_vs_T = []
    for i in range(len(Temperatures_large)):
        popt, pcov = curve_fit(interpolate_tanh, Theta_H, Mag_eq_T[i])
        Alpha_vs_T.append(popt[0])
        M_vs_T.append(popt[1])
    
    Alpha_vs_H.append(Alpha_vs_T)
    M_vs_H.append(M_vs_T)
        
H = 0.4

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (15,10))
k_BTmax = K*V/3
plt.subplot(211)
for i in range(len(H_list)):
    plt.plot(Temperatures_large, Alpha_vs_H[i], marker = '+', label = "mu_0*M_S*V*H = "+str(H_list[i]*mu_0*M_S*V))
    if (i==len(H_list)-1):
        plt.axvline(x=k_BTmax,linestyle='--',color="red",label=r'$k_BT_{max}$')
plt.grid(True)
plt.legend(loc="best")
plt.ylabel(r'$\alpha$')
plt.title("Alpha as a function of k_BT")
plt.xscale("log")
plt.yscale("log")
plt.subplot(212)
for i in range(len(H_list)):
    plt.plot(Temperatures_large, M_vs_H[i], marker = '+', label = "mu_0*M_S*V*H = "+str(H_list[i]*mu_0*M_S*V))
    if (i==len(H_list)-1):
        plt.axvline(x=k_BTmax,linestyle='--',color="red")
plt.grid(True)
plt.xlabel(r'$k_BT$')
plt.ylabel(r'$M$')
plt.xscale("log")
plt.yscale("log")
plt.title("M as a function of k_BT")
plt.show()

# %% [markdown]
# The threshold $k_BT_{lim}$ seems to depend on the characteristic energy $\mu_0M_SVH$.

# %% jupyter={"outputs_hidden": false}
K_list = [1,2,5]

Alpha_vs_K = []
M_vs_K = []

for k in K_list:
    K = k

    Theta_H = np.linspace(0,180,50)
    E_12_1 = []
    E_21_1 = []
    E_12_2 = []
    E_21_2 = []
    Theta_1 = []
    Theta_2 = []
    for theta_H in Theta_H:
        (theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2) = energy_barriers(theta_H)
        Theta_1.append(theta_1)
        Theta_2.append(theta_2)
        E_12_1.append(e_12_1)
        E_21_1.append(e_21_1)
        E_12_2.append(e_12_2)
        E_21_2.append(e_21_2)

    Temperatures_large = np.logspace(-2,1,30)
    Mag_eq_T = []
    for k_BT in Temperatures_large:
        Mag_eq = []
        for i in range(len(Theta_1)):
            Mag_eq.append(mag_eq(Theta_1[i],Theta_2[i],E_12_1[i],E_21_1[i],E_12_2[i],E_21_2[i],k_BT))
        Mag_eq_T.append(Mag_eq)

    Alpha_vs_T = []
    M_vs_T = []
    for i in range(len(Temperatures_large)):
        popt, pcov = curve_fit(interpolate_tanh, Theta_H, Mag_eq_T[i])
        Alpha_vs_T.append(popt[0])
        M_vs_T.append(popt[1])
    
    Alpha_vs_K.append(Alpha_vs_T)
    M_vs_K.append(M_vs_T)
        
K = 1

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (15,10))
colors = ["blue","orange","green"]
plt.subplot(211)
for i in range(len(K_list)):
    k_BTmax = K_list[i]*V/3
    plt.plot(Temperatures_large, Alpha_vs_K[i], color=colors[i], marker = '+', label = "KV = "+str(K_list[i]*V))
    plt.axvline(x=k_BTmax,linestyle='--',color=colors[i],label=r'$k_BT_{max}$')
plt.grid(True)
plt.legend(loc="best")
plt.ylabel(r'$\alpha$')
plt.title("Alpha as a function of k_BT")
plt.xscale("log")
plt.yscale("log")
plt.subplot(212)
for i in range(len(K_list)):
    k_BTmax = K_list[i]*V/3
    plt.plot(Temperatures_large, M_vs_K[i], color=colors[i], marker = '+', label = "KV = "+str(K_list[i]*V))
    plt.axvline(x=k_BTmax,linestyle='--',color=colors[i])
plt.grid(True)
plt.xlabel(r'$k_BT$')
plt.ylabel(r'$M$')
plt.xscale("log")
plt.yscale("log")
plt.title("M as a function of k_BT")
plt.show()

# %% [markdown]
# Surprinsingly, the threshold $k_BT_{lim}$ does not depend on the characteristic energy $KV$.
#
# Therefore, under the condition $k_BT\ll\mu_0M_SVH$, we have the following expressions for $M$ and $\alpha$:
#
# $$M = 1$$
#
# $$\alpha = c\frac{\mu_0M_SVH}{k_BT}$$
#
# Where $c$ is a constant we now want to determine.

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
for i in range(len(H_list)):
    plt.plot(Temperatures_large, Alpha_vs_H[i]*Temperatures_large/(mu_0*M_S*V*H_list[i]), marker = '+', label = "mu_0*M_S*V*H = "+str(H_list[i]*mu_0*M_S*V))
plt.grid(True)
plt.legend(loc="best")
plt.xlabel(r'$k_BT$')
plt.ylabel(r'$\alpha$')
plt.title("The constant c as a function of k_BT")
plt.xscale("log")
plt.yscale("log")
plt.show()

# %% [markdown]
# For $k_BT\ll\mu_0M_SVH$, the previous curves converge towards $c=1$.
#
# #### Conclusion
#
# In this simplistic system, we only looked at the influence $\theta_H$ on the energy barriers and on the magnetization at equilibrium (normalized and projected on the easy axis). The main result of this study is an expression of $m_{eq}$ as a function of $\theta_H$, when $k_BT \ll \mu_0M_SVH$ **and** $KV/(k_BT)>3$:
#
# $$m_{eq}(\theta_H)=-\tanh{\left(\frac{\mu_0M_SVH}{k_BT}\left(\theta_H-\frac{\pi}{2}\right)\right)}$$

# %% jupyter={"outputs_hidden": false}
Temperatures = [0.1, 0.3, 1, 3, 10]

Mag_eq_T = []
for k_BT in Temperatures:
    Mag_eq = []
    for i in range(len(Theta_1)):
        Mag_eq.append(mag_eq(Theta_1[i],Theta_2[i],E_12_1[i],E_21_1[i],E_12_2[i],E_21_2[i],k_BT))
    Mag_eq_T.append(Mag_eq)

plt.figure(figsize = (10,6))
for i in range(len(Temperatures)):
    plt.plot(Theta_H,Mag_eq_T[i],label="k_BT = "+str(Temperatures[i]))
    if(i==len(Temperatures)-1):
        plt.plot(Theta_H,-np.tanh(mu_0*M_S*V*H*(Theta_H-90)*np.pi/180/Temperatures[i]),'k--',label="Theory")
    else:
        plt.plot(Theta_H,-np.tanh(mu_0*M_S*V*H*(Theta_H-90)*np.pi/180/Temperatures[i]),'k--')
plt.xlim(0,180)
plt.legend(loc="best")
plt.grid(True)
plt.xlabel(r'$\theta_H$')
plt.ylabel(r'$M_{eq}$')
plt.title("Mean magnetization at equilibrium with H/H_K = "+str(H/H_K)+" and theoretical curves")
plt.show()

# %% [markdown]
# We see that when the temperature is low enough ($k_BT\ll mu_0M_SVH$), the theoretical curve is very close to the simulation!

# %% [markdown]
# ## The Stoner-Wolfarth model -  control of anisotropy
#
# In reality, the goal is to influence $m_{eq}$ by applying a strain on the magnet, which will change the anisotropy of the system. If the strain induces an anistropy $K_\sigma$ forming an angle $\phi$ with the easy axis, the energy will read:
#
# $$E(\theta, K_\sigma) = KV\sin^2{\theta} + K_\sigma V\sin^2{(\theta-\phi)} - \mu_0M_SVH\cos{(\theta-\theta_H)}$$
#
# Which can be wrote:
#
# $$E(\theta, K_\sigma) = \tilde{K}V\sin^2{(\theta-\psi)} - \mu_0M_SVH\cos{(\theta-\theta_H)}$$
#
# where
#
# $$\tilde{K} = \sqrt{\left(K+K_\sigma\cos{(2\phi)}\right)^2 +\left(K_\sigma\sin{(2\phi)}\right)^2}$$
#
# $$\psi = \frac{1}{2}\arctan{\left(\frac{K_\sigma\sin{(2\phi)}}{K+K_\sigma\cos{(2\phi)}}\right)}$$
#
# This form is close to the expression we had at the beginning. The control paramter is not $\theta_H$ anymore but $K_\sigma$, which influences the angle $\psi$. We should nonetheless keep in mind that $\tilde{K}$ depends on the control parameter $K_\sigma$.

# %% jupyter={"outputs_hidden": true}
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

# %% [markdown]
# Let's fix some constants and define $E(\theta, K_\sigma)$, $\tilde{K}$ and $\psi$.

# %% jupyter={"outputs_hidden": false}
K = 1
V = 1
mu_0 = 1
M_S = 2
H = 0.4 #H<0.5 so that there should always be two minima
H_K = 2*K/(mu_0*M_S)
phi = 45 #intermediate between 0 and 90
theta_H = 90 #Non-linearities are the greatest when theta_H=90


# %% jupyter={"outputs_hidden": true}
def K_tilde(K_sigma):
    return(np.sqrt((K+K_sigma*np.cos(2*phi*np.pi/180))**2+(K_sigma*np.sin(2*phi*np.pi/180))**2))

def psi(K_sigma):
    return(180*np.arctan2(K_sigma*np.sin(2*phi*np.pi/180),(K+K_sigma*np.cos(2*phi*np.pi/180)))/2/np.pi)


# %% jupyter={"outputs_hidden": false}
K_lim = 5
K_sigma_list = np.linspace(-K_lim,K_lim,100)
plt.figure(figsize = (10,8))
plt.subplot(211)
plt.plot(K_sigma_list,K_tilde(K_sigma_list),color="black")
plt.grid(True)
plt.xlim(-K_lim,K_lim)
plt.ylabel(r'$\tilde{K}$')
plt.title(r'$\tilde{K}$'+" as a function of "+r'$K_\sigma$')
plt.subplot(212)
plt.plot(K_sigma_list,psi(K_sigma_list),color="black")
plt.grid(True)
plt.xlim(-K_lim,K_lim)
plt.xlabel(r'$K_\sigma$')
plt.ylabel(r'$\psi$')
plt.title(r'$\psi$'+" as a function of "+r'$K_\sigma$')
plt.show()


# %% jupyter={"outputs_hidden": true}
def energy_ani(theta,K_sigma):
    return(K_tilde(K_sigma)*V*np.sin((theta-psi(K_sigma))*np.pi/180)**2-mu_0*M_S*V*H*np.cos((theta-theta_H)*np.pi/180))


# %% jupyter={"outputs_hidden": true}
theta = np.linspace(-180,180,100)
K_sigma = 10
E = energy_ani(theta,K_sigma)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(theta, E, label = r'$K_\sigma = $'+str(K_sigma))
plt.grid(True)
plt.xlim(-180,180)
plt.legend(loc="best")
plt.xlabel(r'$\theta$')
plt.ylabel('Energy')
plt.title("Energy landscape")
plt.show()


# %% [markdown]
# Again, there are 4 extrema. We will use the same notations as previously ($\theta_1$, $\theta_2$, $E_{12,-}$, $E_{12,+}$, $E_{21,-}$, $E_{21,+}$).

# %% [markdown]
# ### Energy barriers

# %% jupyter={"outputs_hidden": false}
def energy_barriers_ani(K_sigma):
    theta = np.linspace(-180,180,1000)
    E = energy_ani(theta,K_sigma)
    
    #Localization of extrema
    id_max = argrelextrema(E, np.greater)[0]
    id_min = argrelextrema(E, np.less)[0]
    ind1 = 0
    ind2 = 1
    
    #if theta_1 after theta_2, switch ind1 and ind2
    if (len(id_min)>=1 and theta[id_min[0]]>(-90)):
        ind1 = 1
        ind2 = 0
    
    #Two-states case (1 for + ; 2 for -)
    if(len(id_max)==2 and len(id_min)==2):
        theta_1 = theta[id_min[ind1]]
        theta_2 = theta[id_min[ind2]]
        e_12_1 = max((E[id_max[0]]-E[id_min[ind1]]),(E[id_max[1]]-E[id_min[ind1]]))
        e_21_1 = max((E[id_max[0]]-E[id_min[ind2]]),(E[id_max[1]]-E[id_min[ind2]]))
        e_12_2 = min((E[id_max[0]]-E[id_min[ind1]]),(E[id_max[1]]-E[id_min[ind1]]))
        e_21_2 = min((E[id_max[0]]-E[id_min[ind2]]),(E[id_max[1]]-E[id_min[ind2]]))
        
    #Minimas in 0° and 180°
    elif(len(id_min)<=1 and len(id_max)==2):
        print("Exception 1 ; K_sigma = "+str(K_sigma))
        theta_1 = 0
        theta_2 = 180
        e_12_1 = max((E[id_max[0]]-energy_ani(0,K_sigma)),(E[id_max[1]]-energy_ani(0,K_sigma)))
        e_21_1 = max((E[id_max[0]]-energy_ani(180,K_sigma)),(E[id_max[1]]-energy_ani(180,K_sigma)))
        e_12_2 = min((E[id_max[0]]-energy_ani(0,K_sigma)),(E[id_max[1]]-energy_ani(0,K_sigma)))
        e_21_2 = min((E[id_max[0]]-energy_ani(180,K_sigma)),(E[id_max[1]]-energy_ani(180,K_sigma)))
        
    #Maximas in 0° and 180°
    elif(len(id_min)==2 and len(id_max)<=1):
        print("Exception 2 ; K_sigma = "+str(K_sigma))
        theta_1 = theta[id_min[ind1]]
        theta_2 = theta[id_min[ind2]]
        e_12_1 = max((energy_ani(0,K_sigma)-E[id_min[ind1]]),(energy_ani(180,K_sigma)-E[id_min[ind1]]))
        e_21_1 = max((energy_ani(0,K_sigma)-E[id_min[ind2]]),(energy_ani(180,K_sigma)-E[id_min[ind2]]))
        e_12_2 = min((energy_ani(0,K_sigma)-E[id_min[ind1]]),(energy_ani(180,K_sigma)-E[id_min[ind1]]))
        e_21_2 = min((energy_ani(0,K_sigma)-E[id_min[ind2]]),(energy_ani(180,K_sigma)-E[id_min[ind2]]))
        
    #There might be only one minimum. In this case we take the arbitrary valu 0 for all parameters
    else:
        print("Exception 3 ; K_sigma = "+str(K_sigma))
        (theta_1,theta_2,e_12_1,e_12_2,e_21_1,e_21_2) = (0,0,0,0,0,0)
    
    return(theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2)


# %% jupyter={"outputs_hidden": false}
K_lim = 5
K_sigma_list = np.linspace(-K_lim,K_lim,100)
E_12_1 = []
E_21_1 = []
E_12_2 = []
E_21_2 = []
Theta_1 = []
Theta_2 = []
for K_sigma in K_sigma_list:
    (theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2) = energy_barriers_ani(K_sigma)
    Theta_1.append(theta_1)
    Theta_2.append(theta_2)
    E_12_1.append(e_12_1)
    E_21_1.append(e_21_1)
    E_12_2.append(e_12_2)
    E_21_2.append(e_21_2)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(K_sigma_list,E_12_1,'g--',label=r'$E_{12,+}$')
plt.plot(K_sigma_list,E_21_1,'r--',label=r'$E_{21,+}$')
plt.plot(K_sigma_list,E_12_2,'g-',label=r'$E_{12,-}$')
plt.plot(K_sigma_list,E_21_2,'r-',label=r'$E_{21,-}$')
plt.plot(K_sigma_list,np.array(E_21_2)-np.array(E_12_2),'b-',label=r'$E_{21,-}-E_{12,-}$')
plt.legend(loc = "best")
plt.xlim(-K_lim,K_lim)
plt.grid(True)
plt.xlabel(r'$K_\sigma$')
plt.ylabel(r'Energy')
plt.title("Energy barriers")
plt.show()

# %% [markdown]
# $(E_{21,-}-E_{12,-})$ changes with $K_\sigma$!

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(K_sigma_list,Theta_1,'g+',label=r'$\theta_1$')
plt.plot(K_sigma_list,Theta_2,'r+',label=r'$\theta_2$')
plt.legend(loc = "best")
plt.xlim(-K_lim,K_lim)
plt.grid(True)
plt.xlabel(r'$K_\sigma$')
plt.ylabel(r'$\theta_{eq}$')
plt.title("Angles of equilibrium")
plt.show()

# %% [markdown]
# ### Arrhenius equation
#
# We have two barriers of energy. For the moment, we will write de transition rate like this:
#
# $$\omega = \frac{1}{\tau} = f_{0,-}\exp{\left(\frac{-E_{b,-}}{K_BT}\right)}+f_{0,+}\exp{\left(\frac{-E_{b,+}}{K_BT}\right)}$$
#
# We will fix $f_{0,-}=f_{0,+}=1$. We will call $T_{max}$ the temperature verifying $KV/(k_BT) = 3$. Above $T_{max}$, the Arrhenius law cannot be used anymore and our simulation is incorrect.

# %% jupyter={"outputs_hidden": true}
f_0_1=1
f_0_2=1
def omega_ani(e_b_1,e_b_2,k_BT):
    return(f_0_1*np.exp(-e_b_1/k_BT)+f_0_2*np.exp(-e_b_2/k_BT))


# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
k_BT = 0.1
plt.plot(K_sigma_list,omega_ani(np.array(E_12_1),np.array(E_12_2),k_BT),'r-',label=r'$\omega_{12}$')
plt.plot(K_sigma_list,omega_ani(np.array(E_21_1),np.array(E_21_2),k_BT),'g-',label=r'$\omega_{21}$')
plt.legend(loc = "best")
plt.xlim(-K_lim,K_lim)
plt.yscale("log")
plt.grid(True)
plt.xlabel(r'$K_\sigma$')
plt.ylabel(r'$\omega$')
plt.title("Transition rates with "+r'$k_BT$'+" = "+str(k_BT))
plt.show()

# %% [markdown]
# ### Two states system
#
# As before, we can plot $\omega(K_\sigma)$, $p_{1,eq}(K_\sigma)$ and $p_{2,eq}(K_\sigma)$:

# %% jupyter={"outputs_hidden": true}
k_BT = 0.1
omega_12 = omega_ani(np.array(E_12_1),np.array(E_12_2),k_BT)
omega_21 = omega_ani(np.array(E_21_1),np.array(E_21_2),k_BT)
omega_tot = omega_12 + omega_21

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(K_sigma_list,omega_tot,color="magenta")
plt.xlim(-K_lim,K_lim)
plt.grid(True)
plt.xlabel(r'$K_\sigma$')
plt.ylabel(r'$\omega$')
plt.title("Transition rate "+r'$\omega$')
plt.show()

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,8))
plt.subplot(211)
plt.plot(K_sigma_list,omega_21/omega_tot,color="green")
plt.xlim(-K_lim,K_lim)
plt.grid(True)
#plt.yscale("log")
plt.ylabel(r'$p_{1,eq}$')
plt.title("Probability for being in state 1 (at equilibrium)")
plt.subplot(212)
plt.plot(K_sigma_list,omega_12/omega_tot,color="red")
plt.xlim(-K_lim,K_lim)
plt.grid(True)
plt.xlabel(r'$K_\sigma$')
#plt.yscale("log")
plt.ylabel(r'$p_{2,eq}$')
plt.title("Probability for being in state 2 (at equilibrium)")
plt.show()

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(K_sigma_list,omega_21/omega_tot,color="green",label=r'$p_{1,eq}$')
plt.plot(K_sigma_list,omega_12/omega_tot,color="red",label=r'$p_{2,eq}$')
plt.xlim(-K_lim,K_lim)
plt.legend(loc="best")
plt.grid(True)
#plt.yscale("log")
plt.ylabel(r'$p_{eq}$')
plt.xlabel(r'$K_\sigma$')
plt.title("Probabilities for being in each state (at equilibrium)")
plt.show()

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
plt.plot(K_sigma_list,omega_21/omega_12,color="orange")
plt.xlim(-K_lim,K_lim)
plt.grid(True)
plt.yscale("log")
plt.ylabel(r'$p_{1,eq}/p_{2,eq}$')
plt.xlabel(r'$K_\sigma$')
plt.title("Ratio of probabilities for being in each state (at equilibrium)")
plt.show()


# %% [markdown]
# ### Magnetization at equilibrium
#
# Here, the states 1 and 2 are not necessarily aligned with the easy axis. Therefore, the projection of the magnetization along the easy axis (normalized) is:
#
# $$m(t) = \cos{\theta_1}p_1(t) + \cos{\theta_2}p_2(t)$$
#
# Which gives the following expression for $m(t)$:
#
# $$m(t) = m_{eq} + \left[m(0) - m_{eq} \right]\exp{(-\omega t)}$$
#
# where $m_{eq}$ is is the magnetization at equilibrium (projected on the east axis) and reads:
#
# $$m_{eq} = \frac{\cos{\theta_1}\omega_{21} + \cos{\theta_2}\omega_{12}}{\omega}$$

# %% jupyter={"outputs_hidden": true}
def mag_eq_ani(theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2,k_BT):
    w_12 = omega_ani(e_12_1,e_12_2,k_BT)
    w_21 = omega_ani(e_21_1,e_21_2,k_BT)
    return((np.cos(theta_1*np.pi/180)*w_21+np.cos(theta_2*np.pi/180)*w_12)/(w_21+w_12))


# %% jupyter={"outputs_hidden": true}
Temperatures = [0.1, 0.3, 1, 3, 10]

# %% jupyter={"outputs_hidden": false}
Mag_eq_T = []
for k_BT in Temperatures:
    Mag_eq = []
    for i in range(len(Theta_1)):
        Mag_eq.append(mag_eq_ani(Theta_1[i],Theta_2[i],E_12_1[i],E_21_1[i],E_12_2[i],E_21_2[i],k_BT))
    Mag_eq_T.append(Mag_eq)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
for i in range(len(Temperatures)):
    plt.plot(K_sigma_list,Mag_eq_T[i],label="k_BT = "+str(Temperatures[i]))
plt.xlim(-K_lim,K_lim)
plt.legend(loc="best")
plt.grid(True)
plt.xlabel(r'$K_\sigma$')
plt.ylabel(r'$m_{eq}$')
plt.title("Mean magnetization at equilibrium")
plt.show()

# %% [markdown]
# ### Influence of $\theta_H$
#
# The goal is to maximize the influence of $K_\sigma$ on $p_{1,eq}$ and $p_{2,eq}$. The parameter $\theta_H$ seems to be crucial in this problem. Let's see what happens when $\theta_H$ varies. We will both look at $p_{1,eq}/p_{2,eq}$ and $m_{eq}$.

# %% jupyter={"outputs_hidden": false}
theta_H_list = np.linspace(0,90,5)
K_lim = 5
k_BT = 0.1

# %% jupyter={"outputs_hidden": false}
p1p2_vs_tH = []

for tH in theta_H_list:
    theta_H = tH
    
    K_sigma_list = np.linspace(-K_lim,K_lim,500)
    E_12_1 = []
    E_21_1 = []
    E_12_2 = []
    E_21_2 = []
    Theta_1 = []
    Theta_2 = []
    for K_sigma in K_sigma_list:
        (theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2) = energy_barriers_ani(K_sigma)
        Theta_1.append(theta_1)
        Theta_2.append(theta_2)
        E_12_1.append(e_12_1)
        E_21_1.append(e_21_1)
        E_12_2.append(e_12_2)
        E_21_2.append(e_21_2)
    
    omega_12 = omega_ani(np.array(E_12_1),np.array(E_12_2),k_BT)
    omega_21 = omega_ani(np.array(E_21_1),np.array(E_21_2),k_BT)
    
    p1p2_vs_tH.append(omega_21/omega_12)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
for i in range(len(theta_H_list)):
    plt.plot(K_sigma_list,p1p2_vs_tH[i],label=r'$\theta_H = $'+str(theta_H_list[i])+"°")
plt.xlim(-K_lim,K_lim)
plt.legend(loc="best")
plt.grid(True)
plt.yscale("log")
plt.ylabel(r'$p_{1,eq}/p_{2,eq}$')
plt.xlabel(r'$K_\sigma$')
plt.title("Ratio of probabilities for being in each state (at equilibrium)")
plt.show()

# %% jupyter={"outputs_hidden": false}
m_eq_vs_tH = []
k_BT = 0.1
K_sigma_list = np.linspace(-K_lim,K_lim,500)

for tH in theta_H_list:
    theta_H = tH
    
    E_12_1 = []
    E_21_1 = []
    E_12_2 = []
    E_21_2 = []
    Theta_1 = []
    Theta_2 = []
    for K_sigma in K_sigma_list:
        (theta_1,theta_2,e_12_1,e_21_1,e_12_2,e_21_2) = energy_barriers_ani(K_sigma)
        Theta_1.append(theta_1)
        Theta_2.append(theta_2)
        E_12_1.append(e_12_1)
        E_21_1.append(e_21_1)
        E_12_2.append(e_12_2)
        E_21_2.append(e_21_2)
    
    omega_12 = omega_ani(np.array(E_12_1),np.array(E_12_2),k_BT)
    omega_21 = omega_ani(np.array(E_21_1),np.array(E_21_2),k_BT)
    omega_tot = omega_12 + omega_21
    
    m_eq_vs_tH.append((omega_21*np.cos(np.array(Theta_1)*np.pi/180)+omega_12*np.cos(np.array(Theta_2)*np.pi/180))/omega_tot)

# %% jupyter={"outputs_hidden": false}
plt.figure(figsize = (10,6))
for i in range(len(theta_H_list)):
    plt.plot(K_sigma_list,m_eq_vs_tH[i], label=r'$\theta_H = $'+str(theta_H_list[i])+"°")
plt.xlim(-K_lim,K_lim)
plt.legend(loc="best")
plt.grid(True)
plt.ylabel(r'$m_{eq}$')
plt.xlabel(r'$K_\sigma$')
plt.title("Mean magnetization at equilibrium")
plt.show()

# %% [markdown]
# As we can see, when $\theta_H=90°$, we get the nonlinear response with the biggest amplitude.

# %% [markdown]
# #### Next steps
#
# 1. (done) Studying $p_1(t)$ and $p_2(t)$. Studying $p_{1,eq}$ and $p_{2,eq}$.
# 2. (done) Studying the influence of $K_\sigma$ on the energy barriers (and the magnetization at equilibrium), just like I did in the simple case.
# 3. (done) Studying $p_{1,eq}$ and $p_{2,eq}$.
# 4. Going deeper into studying the role of phi, theta_H and H. Studying $p_1(t)$ and $p_2(t)$.
# 5. Studying the link between the voltage imposed to the ferroelectric material and $K_\sigma$.
# 6. Considering the cinetic aspects of the system, and therefore study the "memory" of the magnet.
#
# ...

# %%
