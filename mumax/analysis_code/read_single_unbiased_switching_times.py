import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
from scipy.optimize import curve_fit

folder_name = 'input'

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

# Sort switches and fit that way
sorted_switches = np.sort(switching_times)
total_switches = np.size(sorted_switches)
p_not_switching = np.arange(total_switches-1,-1,-1)/total_switches

def exp_func(t,tau):
    return np.exp(-t/tau)

popt, pcov = curve_fit(exp_func,sorted_switches,p_not_switching)
print(popt)
perr = np.sqrt(np.diag(pcov))
print(perr)

fsz = 20
plt.figure(figsize=(7,6),dpi=200)
plt.scatter(sorted_switches,p_not_switching,color='black')
P_theo2 = ss.expon.pdf(sorted_switches, *P_new)/ss.expon.pdf(0, *P_new)
plt.plot(sorted_switches,P_theo2,color="green",lw=3,linestyle='--',label="Exponential Fit ("+r'$\tau$'+"=74ns)")
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

# Does mean of time period converge with increasing number of points?
number_of_results = np.arange(10,np.size(switching_times),5)
means = np.zeros(np.shape(number_of_results))
errors = np.zeros(np.shape(number_of_results))
meanspdf = np.zeros(np.shape(number_of_results))
errorspdf = np.zeros(np.shape(number_of_results))
for idx, i in np.ndenumerate(number_of_results):

    switching_times_reduced = switching_times[0:i]
    sorted_switches_reduced = np.sort(switching_times_reduced)
    total_switches_reduced = np.size(sorted_switches_reduced)
    p_not_switching_reduced = np.arange(total_switches_reduced-1,-1,-1)/total_switches_reduced

    popt, pcov = curve_fit(exp_func,sorted_switches_reduced,p_not_switching_reduced)
    perr = np.sqrt(np.diag(pcov))
    means[idx] = popt
    errors[idx] = perr

    #alternate take
    tau = ss.expon.fit(switching_times_reduced,floc=0)
    tau = tau[1]
    tauerror = tau/(np.sqrt(i))

    meanspdf[idx]=tau
    errorspdf[idx]=tauerror

plt.figure(figsize=(7,6),dpi=200)
plt.errorbar(number_of_results, means, yerr =errors)
plt.ylabel(r'Observation of $\tau$ / ns',fontsize=14)
plt.xlabel("Number of switches observed",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

plt.figure(figsize=(7,6),dpi=200)
plt.errorbar(number_of_results, meanspdf, yerr =errorspdf)
plt.ylabel(r'PDF observation of $\tau$ / ns',fontsize=14)
plt.xlabel("Number of switches observed",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

'''
#playing with the idea of sampling many means
noofsamples = 20
switching_times_limited = switching_times[0:300]
chunk = int(np.size(switching_times_limited)/noofsamples)
print(chunk)
mean = np.zeros(10)
for i in range(0,noofsamples):
    mean[i] = np.mean(switching_times_limited[i*chunk:(i+1)*chunk])

print(np.std(mean,ddof=1)/np.sqrt(noofsamples))
'''
