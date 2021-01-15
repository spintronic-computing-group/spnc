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

switching_times_upwards = []
switching_times_downwards = []

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
        if up == True:
            switching_times_downwards.append(t_last_switch)
        else:
            switching_times_upwards.append(t_last_switch)
        t_last_switch = 0
        up = not up
        inversion_factor = -inversion_factor
    t_last_switch += dt

T = np.linspace(0,2000,500)

# Fit switches going upwards
P_upwards = ss.expon.fit(switching_times_upwards,floc=0)
print(P_upwards)
P_new_upwards=(0,P_upwards[1])
P_theo_upwards = ss.expon.pdf(T, *P_new_upwards)

#Find the error
total_switches_upwards = np.size(switching_times_upwards)
error_upwards = P_upwards[1]/np.sqrt(total_switches_upwards)

#display the results
print('Tau upwards = ' + str(round(P_upwards[1])) +' +/- ' + str(int(round(error_upwards))) + " ns")
print('Number of upwards switches = ' + str(total_switches_upwards))

# Fit switches going downwards
P_downwards = ss.expon.fit(switching_times_downwards,floc=0)
print(P_downwards)
P_new_downwards=(0,P_downwards[1])
P_theo_downwards = ss.expon.pdf(T, *P_new_downwards)

#Find the error
total_switches_downwards = np.size(switching_times_downwards)
error_downwards = P_downwards[1]/np.sqrt(total_switches_downwards)

#display the results
print('Tau downwards = ' + str(round(P_downwards[1])) +' +/- ' + str(int(round(error_downwards))) + " ns")
print('Number of downwards switches = ' + str(total_switches_downwards))


fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.hist(switching_times_upwards,edgecolor="blue",alpha=.5,bins=20,density=True)
plt.plot(T,P_theo_upwards,color="red",lw=1.5,linestyle='--',label="Exponential Fit ("+r'$\tau$ upwards $= $' + str(round(P_upwards[1])) +r'$\pm$' + str(int(round(error_upwards))) + "ns)")
#plt.yscale("log")
plt.legend(loc="best",fontsize=fsz*0.7)
plt.ylabel("Density of switching times",fontsize=fsz)
plt.xlabel("Time (in ns)",fontsize=fsz)
plt.xlim(0,2000)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.ylim(1e-6,1e-1)
plt.savefig('output/'+'switching_density_plot_upwards.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.hist(switching_times_downwards,edgecolor="blue",alpha=.5,bins=20,density=True)
plt.plot(T,P_theo_downwards,color="red",lw=1.5,linestyle='--',label="Exponential Fit ("+r'$\tau$ downwards $= $' + str(round(P_downwards[1])) +r'$\pm$' + str(int(round(error_downwards))) + "ns)")
#plt.yscale("log")
plt.legend(loc="best",fontsize=fsz*0.7)
plt.ylabel("Density of switching times",fontsize=fsz)
plt.xlabel("Time (in ns)",fontsize=fsz)
plt.xlim(0,2000)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.ylim(1e-6,1e-1)
plt.savefig('output/'+'switching_density_plot_downwards.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

# Sort switches and plot that way
sorted_switches_upwards = np.sort(switching_times_upwards)
p_not_switching_upwards = np.arange(total_switches_upwards-1,-1,-1)/total_switches_upwards

# Sort switches and plot that way
sorted_switches_downwards = np.sort(switching_times_downwards)
p_not_switching_downwards = np.arange(total_switches_downwards-1,-1,-1)/total_switches_downwards

def exp_func(t,tau):
    return np.exp(-t/tau)


fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.scatter(sorted_switches_upwards,p_not_switching_upwards,s=1.5,color='black')
P_theo2_upwards = ss.expon.pdf(T, *P_new_upwards)/ss.expon.pdf(0, *P_new_upwards)
plt.plot(T,P_theo2_upwards,color="red",lw=1.5,alpha=0.8,linestyle='--',
         label="Exponential Fit ("+r'$\tau$ upwards $ = $' + str(round(P_upwards[1])) +r'$\pm$' + str(int(round(error_upwards))) + "ns)")
plt.legend(loc="best",fontsize=fsz*0.7)
plt.ylabel("Probability of not switching",fontsize=fsz)
plt.xlabel("Time (in ns)",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.xlim(0,2000)
plt.ylim(0,1)
plt.savefig('output/'+'prob_not_switching_upwards.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.scatter(sorted_switches_downwards,p_not_switching_downwards,s=1.5,color='black')
P_theo2_downwards = ss.expon.pdf(T, *P_new_downwards)/ss.expon.pdf(0, *P_new_downwards)
plt.plot(T,P_theo2_downwards,color="red",lw=1.5,alpha=0.8,linestyle='--',
         label="Exponential Fit ("+r'$\tau$ downwards $ = $' + str(round(P_downwards[1])) +r'$\pm$' + str(int(round(error_downwards))) + "ns)")
plt.legend(loc="best",fontsize=fsz*0.7)
plt.ylabel("Probability of not switching",fontsize=fsz)
plt.xlabel("Time (in ns)",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.xlim(0,2000)
plt.ylim(0,1)
plt.savefig('output/'+'prob_not_switching_downwards.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

# Does mean of time period converge with increasing number of points?
number_of_results_upwards = np.arange(10,np.size(switching_times_upwards),5)
means_upwards = np.zeros(np.shape(number_of_results_upwards))
errors_upwards = np.zeros(np.shape(number_of_results_upwards))
meanspdf_upwards = np.zeros(np.shape(number_of_results_upwards))
errorspdf_upwards = np.zeros(np.shape(number_of_results_upwards))
for idx, i in np.ndenumerate(number_of_results_upwards):

    switching_times_reduced_upwards = switching_times_upwards[0:i]
    sorted_switches_reduced_upwards = np.sort(switching_times_reduced_upwards)
    total_switches_reduced_upwards = np.size(sorted_switches_reduced_upwards)
    p_not_switching_reduced_upwards = np.arange(total_switches_reduced_upwards-1,-1,-1)/total_switches_reduced_upwards

    #alternate take
    tau_upwards = ss.expon.fit(switching_times_reduced_upwards,floc=0)
    tau_upwards = tau_upwards[1]
    tauerror_upwards = tau_upwards/(np.sqrt(i))

    meanspdf_upwards[idx]=tau_upwards
    errorspdf_upwards[idx]=tauerror_upwards

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.errorbar(number_of_results_upwards, meanspdf_upwards, yerr =errorspdf_upwards,linewidth=1)
plt.ylabel(r'PDF observation of upwards $\tau$ / ns',fontsize=fsz)
plt.xlabel("Number of upwards switches observed",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.savefig('output/'+'estimates_of_tau_upwards.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()


# Does mean of time period converge with increasing number of points?
number_of_results_downwards = np.arange(10,np.size(switching_times_downwards),5)
means_downwards = np.zeros(np.shape(number_of_results_downwards))
errors_downwards = np.zeros(np.shape(number_of_results_downwards))
meanspdf_downwards = np.zeros(np.shape(number_of_results_downwards))
errorspdf_downwards = np.zeros(np.shape(number_of_results_downwards))
for idx, i in np.ndenumerate(number_of_results_downwards):

    switching_times_reduced_downwards = switching_times_downwards[0:i]
    sorted_switches_reduced_downwards = np.sort(switching_times_reduced_downwards)
    total_switches_reduced_downwards = np.size(sorted_switches_reduced_downwards)
    p_not_switching_reduced_downwards = np.arange(total_switches_reduced_downwards-1,-1,-1)/total_switches_reduced_downwards

    #alternate take
    tau_downwards = ss.expon.fit(switching_times_reduced_downwards,floc=0)
    tau_downwards = tau_downwards[1]
    tauerror_downwards = tau_downwards/(np.sqrt(i))

    meanspdf_downwards[idx]=tau_downwards
    errorspdf_downwards[idx]=tauerror_downwards

fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.errorbar(number_of_results_downwards, meanspdf_downwards, yerr =errorspdf_downwards,linewidth=1)
plt.ylabel(r'PDF observation of downwards $\tau$ / ns',fontsize=fsz)
plt.xlabel("Number of downwards switches observed",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.savefig('output/'+'estimates_of_tau_downwards.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()