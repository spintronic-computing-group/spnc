import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.kde import gaussian_kde

folder_name = 'single_nanomagnet_no_input_8'

#Read the data
df = pd.read_csv(folder_name+'/table.txt',sep='\t')
time = 1e9*df['# t (s)']
mx = df['mx ()']
my = df['my ()']
mz = df['mz ()']
N = len(mx)

bp = 10
h=0.4
thetas = []

for i in range(N):
    thetas.append(np.arctan2(my[i],mx[i])*180/np.pi)
    
density_function = gaussian_kde(thetas)
X=np.linspace(0,180,100)

empirical_energy = np.log(1/(density_function(X)))/bp
empirical_energy_normed = empirical_energy - empirical_energy[50]

K = 1
V = 1
mu_0 = 1
M_S = 2
H = h
H_K = 2*K/(mu_0*M_S)
phi = 45
theta_H = 90 

def K_tilde(K_sigma):
    return(np.sqrt((K+K_sigma*np.cos(2*phi*np.pi/180))**2+(K_sigma*np.sin(2*phi*np.pi/180))**2))

def psi(K_sigma):
    return(180*np.arctan2(K_sigma*np.sin(2*phi*np.pi/180),(K+K_sigma*np.cos(2*phi*np.pi/180)))/2/np.pi)

def energy_ani(theta,K_sigma):
    return(K_tilde(K_sigma)*V*np.sin((theta-psi(K_sigma))*np.pi/180)**2-mu_0*M_S*V*H*np.cos((theta-theta_H)*np.pi/180))

empirical_energy = np.log(1/(density_function(X)))/bp
empirical_energy_normed = empirical_energy - empirical_energy[50] + energy_ani(X[50],0)

plt.figure(figsize=(8,6),dpi=200)
plt.subplot(211)
plt.hist(thetas,ec="blue",alpha=.5,bins=100,density=True,label="Empirical Distribution")
plt.plot(X,density_function(X),'r-',label="Gaussian Kernel Density Estimation")
plt.legend(loc="best")
plt.ylabel("Density")
plt.yticks([0,0.005,0.01,0.015],['0','0.5e-2','1.0e-2','1.5e-2'])
plt.xlim(0,180)
plt.subplot(212)
plt.plot(X,empirical_energy_normed,'k-',label="Empirical Energy Landscape")
plt.plot(X,energy_ani(X,0),'k--',label="Stoner-Wolfarth Equation")
plt.legend(loc='best')
plt.xlabel(r'$\theta$'+" (°)")
plt.ylabel("Energy (normalized)")
plt.xlim(0,180)
#plt.polar(X,np.log(1/(density_function(X)))/bp)
plt.show()