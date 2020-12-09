import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

folder_name = 'relaxation_bp_10_big_err_1640'
bp = 10
if bp==20:
    T = 6.8e-8
elif bp==10:
    T = 1.8e-9
# attempt freq from Hewen's project
T=T*10/3.02
print(T)

df = pd.read_csv(folder_name+'/table.txt',sep='\t')
time = df['# t (s)']
mx = df['mx ()']
X = np.linspace(min(time),max(time),100)
Y_theo = np.exp(-X/T)

plt.figure(figsize=(8,6),dpi=200)
plt.grid(True)
#plt.ylim(0,1)
plt.plot(time,mx,'ks',label='Simulation')
plt.plot(X,Y_theo,'k--',label='Theoretical relaxation')
plt.legend(loc='best')
plt.ylabel(r'$m_x$')
plt.xlabel("Time in sec")
plt.title(folder_name)
plt.show()