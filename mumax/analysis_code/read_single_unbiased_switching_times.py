import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss

# #Filepath dialog
# import tkinter as tk
# from tkinter import filedialog
# root = tk.Tk()
# root.withdraw()
# folder_name = filedialog.askdirectory(title="Select the folder containing the data table")
# root.update()
# print(folder_name)
folder_name = 'input'
output = 'output'

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

#Find the error
total_switches = np.size(switching_times)
error = P[1]/np.sqrt(total_switches)

#display the results
print('Tau = ' + str(round(P[1])) +' +/- ' + str(int(round(error))) + " ns")

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.hist(switching_times,edgecolor="blue",alpha=.5,bins=20,density=True)
plt.plot(T,P_theo,color="red",lw=1.5,linestyle='--',label="Exponential Fit ("+r'$\tau = $' + str(round(P[1])) +r'$\pm$' + str(int(round(error))) + "ns)")
#plt.yscale("log")
plt.legend(loc="best",fontsize=fsz*0.9)
plt.ylabel("Density of switching times",fontsize=fsz)
plt.xlabel("Time (in ns)",fontsize=fsz)
plt.xlim(0,600)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.ylim(1e-6,1e-1)
plt.savefig('output/'+'switching_density_plot.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

# Sort switches and plot that way
sorted_switches = np.sort(switching_times)
p_not_switching = np.arange(total_switches-1,-1,-1)/total_switches

def exp_func(t,tau):
    return np.exp(-t/tau)


fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.scatter(sorted_switches,p_not_switching,s=1.5,color='black')
P_theo2 = ss.expon.pdf(sorted_switches, *P_new)/ss.expon.pdf(0, *P_new)
plt.plot(sorted_switches,P_theo2,color="red",lw=1.5,alpha=0.8,linestyle='--',
         label="Exponential Fit ("+r'$\tau = $' + str(round(P[1])) +r'$\pm$' + str(int(round(error))) + "ns)")

plt.legend(loc="best",fontsize=fsz*0.9)
plt.ylabel("Probability of not switching",fontsize=fsz)
plt.xlabel("Time (in ns)",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.xlim(0,600)
plt.ylim(0,1)
plt.savefig('output/'+'prob_not_switching.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
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

    #alternate take
    tau = ss.expon.fit(switching_times_reduced,floc=0)
    tau = tau[1]
    tauerror = tau/(np.sqrt(i))

    meanspdf[idx]=tau
    errorspdf[idx]=tauerror

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.errorbar(number_of_results, meanspdf, yerr =errorspdf,linewidth=1)
plt.ylabel(r'PDF observation of $\tau$ / ns',fontsize=fsz)
plt.xlabel("Number of switches observed",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.savefig('output/'+'estimates_of_tau.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()