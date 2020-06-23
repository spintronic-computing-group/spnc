"""
Machine learning with SPNC resevoirs

Local Dependancies
------------------
machine_learning_library  : v0.1.2
    This repository will need to be on your path in order to work.
    This is achieved with repo_tools module and a path find function
    Add to the searchpath and repos tuples if required

Functions
---------
spnc_narma10(Ntrain,Ntest,Nvirt,m0, bias, transform,params,*args,**kwargs)
    Perform the NARMA10 task with a given resevoir

"""


# libraries
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt


# local repos and search paths. Each repo will be searched on every path

#tuple of Path variables
searchpaths = (Path.home() / 'repos', )
#tuple of repos
repos = ('machine_learning_library',)


# Add local modules and paths to local repos
import repo_tools
repo_tools.repos_path_finder(searchpaths, repos)
from machine_learning_library.single_node_res import single_node_reservoir
import ridge_regression as RR
from linear_layer import *
from mask import binary_mask
from utility import *
from NARMA10 import NARMA10


def spnc_narma10(Ntrain,Ntest,Nvirt,m0, bias,
                       transform,params,*args,**kwargs):
    """
    perform the NARMA10 task with a given resevoir

    Parameters
    ----------
    Ntrain : int
        Number of samples to train
    Ntest : int
        Number of sampels to test
    Nvirt : int
        Number of virtual nodes for the resevoir
    m0 : float
        input scaling, no scaling for value of 1
    bias : bool
        True - use bias, False - don't
    transform : function or class method
        transforms a 1D numpy array through the resevoir
    params : dict
        parameters for the resevoir
    """

    u, d = NARMA10(Ntrain + Ntest)

    x_train = u[:Ntrain]
    y_train = d[:Ntrain]
    x_test = u[Ntrain:]
    y_test = d[Ntrain:]

    print("Samples for training: ", len(x_train))
    print("Samples for test: ", len(x_test))

    # Net setup
    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(y_train))

    print( 'Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)

    snr = single_node_reservoir(Nin, Nout, Nvirt, m0, res = transform)
    net = linear(Nin, Nout, bias = bias)


    # Training
    S_train, J_train = snr.transform(x_train,params)
    np.size(S_train)
    RR.Kfold_train(net,S_train,y_train,10, quiet = False)


    # Testing
    S_test, J_test = snr.transform(x_test,params)

    pred = net.forward(S_test)
    np.size(pred)
    error = MSE(pred, y_test)
    predNRMSE = NRMSE(pred, y_test)
    print(error, predNRMSE)

    plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0), 'k--')
    plt.plot(y_test, pred, 'o')
    plt.show()
