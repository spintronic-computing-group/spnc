import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
from scipy.optimize import curve_fit

folder_name = 'single_nanomagnet_no_input_6'

#Read the data
df = pd.read_csv(folder_name+'/table.txt',sep='\t')
time = 1e9*df['# t (s)']
mx = df['mx ()']
my = df['my ()']
mz = df['mz ()']
N = len(mx)
dt = time[1]-time[0]

threshold = 0.5

switching_times = []

#Initial state
t_last_switch = dt
up = True
inversion_factor = 1
if mx[0]<0:
    up = False
    inversion_factor = -1
    
#Scan the data
for i in range(1,N):
    if mx[i]*inversion_factor < -threshold:
        #switch
        switching_times.append(t_last_switch)
        t_last_switch = 0
        up = not up
        inversion_factor = -inversion_factor
    t_last_switch += dt

P = ss.expon.fit(switching_times,floc=0)
print(P)
P_new=(0,P[1])
T = np.linspace(0,2000,500)
P_theo = ss.expon.pdf(T, *P_new)

plt.figure(figsize=(7,6),dpi=200)
plt.hist(switching_times,edgecolor="blue",alpha=.5,bins=20,density=True)
plt.plot(T,P_theo,color="red",lw=3,linestyle='--',label="Exponential Fit by MLE ("+r'$\tau$'+"=74ns)")
#plt.yscale("log")
plt.legend(loc="best",fontsize=14)
plt.ylabel("Density of switching times",fontsize=14)
plt.xlabel("Time (in ns)",fontsize=14)
plt.xlim(0,600)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.ylim(1e-6,1e-1)
plt.show()

sorted_switches = np.sort(switching_times)
total_switches = np.size(sorted_switches)
p_not_switching = np.arange(total_switches-1,-1,-1)/total_switches

def exp_func(t,tau):
    return np.exp(-t/tau)

popt, pcov = curve_fit(exp_func,sorted_switches,p_not_switching)
popt

fsz = 20
plt.figure(figsize=(7,6),dpi=200)
plt.scatter(sorted_switches,p_not_switching,color='black')
plt.plot(sorted_switches,exp_func(sorted_switches,*popt),color="red",lw=3,linestyle='--',label="Exponential Fit ("+r'$\tau$'+"=74ns)")
#plt.plot(T,P_theo*71.4,color="red",lw=3,linestyle='--',label="Exponential Fit by MLE ("+r'$\tau$'+"=74ns)")

plt.legend(loc="best",fontsize=fsz)
plt.ylabel("Probability of not switching",fontsize=fsz)
plt.xlabel("Time (in ns)",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.xlim(0,600)
plt.ylim(0,1)
plt.show()