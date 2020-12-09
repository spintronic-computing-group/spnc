import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss

folder_name = 'single_nanomagnet_no_input_8'

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
plt.plot(T,P_theo,color="red",lw=3,linestyle='--',label="Exponential Fit by MLE ("+r'$\tau$'+"=111ns)")
#plt.yscale("log")
plt.legend(loc="best",fontsize=14)
plt.ylabel("Density of switching times",fontsize=14)
plt.xlabel("Time (in ns)",fontsize=14)
plt.xlim(0,600)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#plt.ylim(1e-6,1e-1)
plt.show()