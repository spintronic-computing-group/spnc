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
