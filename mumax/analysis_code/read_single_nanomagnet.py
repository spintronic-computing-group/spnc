import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

folder_name = 'input'

df = pd.read_csv(folder_name+'/table.txt',sep='\t')
time = 1e9*df['# t (s)']
mx = df['mx ()']
my = df['my ()']
mz = df['mz ()']


fsz = 10
figurewidth = 3.37 #inches (single column)
figureaspect = 1/3
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.grid(True)
#plt.ylim(0,1)
#plt.plot(time,mx,'k-')
beg=0
end=10000000
plt.plot(time[beg:end],mx[beg:end],'k-',linewidth=1)
#plt.plot([0,500],[0.5,0.5],'b--',lw=3,label="Threshold when "+r'$m_x<0$')
#plt.plot([0,500],[-0.5,-0.5],'r--',lw=3,label="Threshold when "+r'$m_x>0$')
#plt.plot(time[beg:end],my[beg:end],'b-')
#plt.plot(time[beg:end],mz[beg:end],'g-')
#plt.plot(time,mx**2+my**2+mz**2)
#plt.legend(loc='best',fontsize=14)
plt.ylabel(r'$m_x$',fontsize=fsz)
plt.xlabel("Time / ns",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(np.linspace(-1,1,5),fontsize=fsz)
plt.xlim(0, 2000)
plt.savefig('output/'+'telegraph-noise.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()