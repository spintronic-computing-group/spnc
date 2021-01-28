# -*- coding: utf-8 -*-
"""
@author: Alexander

This code runs some machine learning and saves the variables for plotting elsewhere

Local Dependancies
------------------
machine_learning_library  : v0.1.2
    This repository will need to be on your path in order to work.
    This is achieved with repo_tools module and a path find function
    Add to the searchpath and repos tuples if required


"""

import numpy as np
import matplotlib.pyplot as plt

# Sort out relative paths for this project
import sys
from pathlib import Path
repodir = Path('../../..').resolve()
try:
    sys.path.index(str(repodir))
except ValueError:
    sys.path.append(str(repodir))

# #tuple of Path variables
# searchpaths = (Path.home() / 'repos', )
# #tuple of repos
# repos = ('machine_learning_library',)

# # local imports    
# from SPNC import spnc
# #ML specific
# from SPNC.deterministic_mask import fixed_seed_mask, max_sequences_mask
# import SPNC.repo_tools
# SPNC.repo_tools.repos_path_finder(searchpaths, repos) #find ml library
# from single_node_res import single_node_reservoir
# import ridge_regression as RR
# from linear_layer import *
# from mask import binary_mask
# from utility import *
# from NARMA10 import NARMA10
# from sklearn.metrics import classification_report

'''
NARMA10 response
'''

#Load data from production elsewhere
NARMA10data = np.load('data/NARMA10_v1.npz')

Ntrain = NARMA10data['Ntrain']
Ntest = NARMA10data['Ntest']
Nvirt = NARMA10data['Nvirt']
gamma = NARMA10data['gamma']
delay_feedback = NARMA10data['delay_feedback']
spacer = NARMA10data['spacer']
x_train = NARMA10data['x_train']
y_train = NARMA10data['y_train']
x_test = NARMA10data['x_test']
y_test = NARMA10data['y_test']
Nin = NARMA10data['Nin']
Nout = NARMA10data['Nout']
S_train = NARMA10data['S_train']
J_train = NARMA10data['J_train']
S_test = NARMA10data['S_test']
J_test = NARMA10data['J_test']
pred = NARMA10data['pred']

'''
Some parameters for the next few plots
'''

width = 100 #How many points to plot
inputno = 57 #Which point is being expanded to show the internal input
figureaspect = 2/3


'''
Input
'''

fsz = 10
figurewidth = 3.37 #inches (single column)
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.plot(x_test[spacer:spacer+width],linewidth=1,zorder=0)
#plt.legend()
plt.xlabel(r'Time steps / $\tau$',fontsize=fsz)
plt.ylabel("Raw input",fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.ylim(auto=False)
ymin, ymax = plt.ylim()
plt.vlines(inputno-0.5,ymin,ymax,linestyles='--',linewidth=.6,alpha=0.7,zorder=1)
# plt.vlines(inputno+0.5,ymin,ymax,linestyles='--',linewidth=.7,alpha=0.7)
# plt.vlines(inputno-1.5,ymin,ymax,linestyles='--',linewidth=.7,alpha=0.7)
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'input.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()


'''
Plotting direct inputs and outputs of the reservoir
'''

breadth = 20 #How many thetas to show either side of the input change

J_1d = np.expand_dims(np.ravel(J_test,order='C'),axis=-1)
S_1d = np.expand_dims(np.ravel(S_test,order='C'),axis=-1)

N = J_1d.shape[0]
#I'm not quite sure the indexing is right here, especially for delayed case!
J_1d_true = np.zeros(np.shape(J_1d))
for idx, j in enumerate(J_1d):
    inidx = idx-Nvirt-delay_feedback
    if inidx < 0:
        J_1d_true[idx] = j
    else:
        J_1d_true[idx] = j + gamma*S_1d[inidx]

fsz = 10
figurewidth = 3.37 #inches (single column)
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
thetas = np.arange(-breadth,breadth,1)    
plt.plot(thetas,
         J_1d[(spacer+inputno)*Nvirt-breadth:(spacer+inputno)*Nvirt+breadth],
         linewidth=1,label='Masked input')
plt.plot(thetas,
         J_1d_true[(spacer+inputno)*Nvirt-breadth:(spacer+inputno)*Nvirt+breadth],
         linewidth=1,label='Masked input + feedback')
plt.plot(thetas,
         S_1d[(spacer+inputno)*Nvirt-breadth:(spacer+inputno)*Nvirt+breadth],
         linewidth=1,label='Raw output')
plt.ylim(auto=False)
ymin, ymax = plt.ylim()
plt.vlines(0,ymin,ymax,linestyles='--',linewidth=.6,alpha=1)
plt.legend(fontsize=fsz*0.7,loc='best')
plt.xlabel(r'Node time steps / $\theta$',fontsize=fsz)
plt.ylabel('Reservoir \n Input/Output ',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'internal.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()

'''
Demonstrate the correction of J_1d to get J_1d_true is working
'''

plt.plot(J_1d[(1)*400-20:(1)*400+20])
plt.plot(J_1d_true[(1)*400-20:(1)*400+20])
#plt.plot(gamma*S_1d[((1)*400-20):(0)*400+20])
plt.plot(np.arange(20,40,1),gamma*S_1d[0:20]+J_1d[(1)*400:(1)*400+20])
plt.show()


'''
Output vs predicted
'''

fsz = 10
figurewidth = 3.37 #inches (single column)
figureheight = figurewidth*figureaspect
plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
plt.plot(y_test[spacer:spacer+width],linewidth=1,label='Desired')
plt.plot(pred[spacer:spacer+width],linewidth=1,alpha=0.8,label='Predicted')
#plt.plot(y_test[spacer:spacer+width]-pred[spacer:spacer+width],linewidth=1,label='Error')
plt.legend(fontsize=fsz*0.8)
plt.ylabel("Final output",fontsize=fsz)
plt.xlabel(r'Time steps / $\tau$',fontsize=fsz)
plt.xticks(fontsize=fsz)
plt.yticks(fontsize=fsz)
#plt.xlim(0,2000)
#plt.ylim(0,1)
plt.savefig('output/'+'output.pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
plt.show()


# fsz = 10
# figurewidth = 3.37 #inches (single column)
# figureaspect = 1
# figureheight = figurewidth*figureaspect
# plt.figure(figsize=[figurewidth,figureheight],dpi=1200)
# #Plot something here...

# #plt.legend()
# plt.ylabel(r'',fontsize=fsz)
# plt.xlabel("",fontsize=fsz)
# plt.xticks(fontsize=fsz)
# plt.yticks(fontsize=fsz)
# #plt.xlim(0,2000)
# #plt.ylim(0,1)
# #plt.savefig('output/'+'....pdf',format='pdf',transparent=True,dpi=1200,bbox_inches='tight')
# plt.show()
