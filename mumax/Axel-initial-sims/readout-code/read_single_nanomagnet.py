import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

folder_name = 'single_nanomagnet_no_input_8'

df = pd.read_csv(folder_name+'/table.txt',sep='\t')
time = 1e9*df['# t (s)']
mx = df['mx ()']
my = df['my ()']
mz = df['mz ()']

plt.figure(figsize=(8,6),dpi=200)
plt.grid(True)
#plt.ylim(0,1)
#plt.plot(time,mx,'k-')
beg=0
end=5000
plt.plot(time[beg:end],mx[beg:end],'k-')
plt.plot([0,500],[0.5,0.5],'b--',lw=3,label="Threshold when "+r'$m_x<0$')
plt.plot([0,500],[-0.5,-0.5],'r--',lw=3,label="Threshold when "+r'$m_x>0$')
#plt.plot(time[beg:end],my[beg:end],'b-')
#plt.plot(time[beg:end],mz[beg:end],'g-')
#plt.plot(time,mx**2+my**2+mz**2)
plt.legend(loc='best',fontsize=14)
plt.ylabel(r'$m_x$',fontsize=14)
plt.xlabel("Time (in ns)",fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(np.linspace(-1,1,5),fontsize=14)
plt.xlim(0,500)
#plt.title(folder_name)
plt.show()