"""
Machine learning with SPNC resevoirs

Local Dependancies
------------------
machine_learning_library  : v0.1.4.dev
    This repository will need to be on your path in order to work.
    This is achieved with repo_tools module and a path find function
    Add to the searchpath and repos tuples if required

Functions
---------
spnc_narma10(Ntrain,Ntest,Nvirt,m0, bias, transform,params,*args,**kwargs)
    Perform the NARMA10 task with a given resevoir

spnc_spoken_digits(speakers,Nvirt,m0,bias,transform,params,*args,**kwargs)
    Perform the TI46 spoken digit task with a given resevoir
"""


# libraries
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from single_node_heterogenous_reservoir import single_node_heterogenous_reservoir


# local repos and search paths. Each repo will be searched on every path

#tuple of Path variables 
searchpaths = (
    Path(r'C:\\icloud\\iCloudDrive\\Desktop\\Code'),  
    
)
#tuple of repos
repos = ('machine_learning_library',)


# Add local modules and paths to local repos
from deterministic_mask import fixed_seed_mask, max_sequences_mask
import repo_tools
repo_tools.repos_path_finder(searchpaths, repos)
from single_node_res import single_node_reservoir
import ridge_regression as RR
from linear_layer import *
from mask import binary_mask
from utility import *
from NARMA10 import NARMA10
from datasets.load_TI46_digits import *
import datasets.load_TI46 as TI46
from sklearn.metrics import classification_report


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

    seed_NARMA = kwargs.get('seed_NARMA', None)
    print("seed NARMA: "+str(seed_NARMA))
    u, d = NARMA10(Ntrain + Ntest,seed=seed_NARMA)

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

    fixed_mask = kwargs.get('fixed_mask', False)
    if fixed_mask==True:
        print("Deterministic mask will be used")
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask>=0:
            print(seed_mask)
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            print("Max_sequences mask will be used")
            snr.M = max_sequences_mask(Nin, Nvirt, m0)



    # Training
    S_train, J_train = snr.transform(x_train,params)
    np.size(S_train)
    seed_training = kwargs.get('seed_training', 1234)
    RR.Kfold_train(net,S_train,y_train,10, quiet = True, seed_training=seed_training)


    # Testing
    S_test, J_test = snr.transform(x_test,params)

    spacer = kwargs.get('spacer_NRMSE', 0) # avoid the problem of dividing by zero
    print("Spacer NRMSE:"+str(spacer))
    pred = net.forward(S_test)
    np.size(pred)
    error = MSE(pred, y_test)
    predNRMSE = NRMSE(pred, y_test, spacer=spacer)
    print(error, predNRMSE)

    plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0), 'k--')
    plt.plot(y_test, pred, 'o')
    plt.show()

    return_outputs = kwargs.get('return_outputs', False)
    if return_outputs:
        return(y_test,pred)

    return_NRMSE = kwargs.get('return_NRMSE', False)
    if return_NRMSE:
        return(predNRMSE)
    

# create a new function for the narma10 task that use heterogenous reservoirs
def spnc_narma10_heterogenous(Ntrain,Ntest,Nvirt,gamma, beta_prime, beta_ref,deltabeta_list,h,theta,m0,step,beta_left,beta_right,*weights,bias
                       ,params,**kwargs):
    """
    perform the NARMA10 task with a given heterogenous resevoir

    Parameters
    ----------
    Ntrain : int
        Number of samples to train
    Ntest : int
        Number of sampels to test
    Nvirt : int
        Number of virtual nodes for the heterogenous resevoir
    m0 : float
        input scaling, no scaling for value of 1
    bias : bool
        True - use bias, False - don't
    transform : function or class method
        transforms a 1D numpy array through the heterogenous resevoir
    params : dict
        parameters for the heterogenous resevoir

    adjust it by comparing with the wide_temperature_testing.py
    """

    # params = {
    #     'theta': theta,
    #     'gamma': gamma,
    #     'beta_prime': beta_prime,
    #     'delay_feedback': 0,
    #     'Nvirt': Nvirt
    # }

    seed_NARMA = kwargs.get('seed_NARMA', None)
    print("seed NARMA: "+str(seed_NARMA))
    u, d = NARMA10(Ntrain + Ntest,seed=seed_NARMA)

    x_train = u[:Ntrain]
    y_train = d[:Ntrain]
    x_test = u[Ntrain:]
    y_test = d[Ntrain:]

    print("Samples for training: ", len(x_train))
    print("Samples for test: ", len(x_test))

    # Net setup
    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(y_train))

    # not sure if the shape of input is suitable for the heterogenous resevoir

    print( 'Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)

    # create a heterogenous reservoir
    snr = single_node_heterogenous_reservoir(Nin, Nvirt, Nout,gamma, beta_prime, beta_ref, deltabeta_list, h, theta, m0)
    print('snr_beta_prime:', snr.beta_prime)
    # print('Mask matrix M_initial:', snr.M.M)
    net = linear(Nin, Nout, bias = bias)
    # print('net_w_initial',net.W)

    fixed_mask = kwargs.get('fixed_mask', False)
    if fixed_mask==True:
        print("Deterministic mask will be used")
        seed_mask = kwargs.get('seed_mask', 1234)
        if seed_mask>=0:
            print(seed_mask)
            snr.M = fixed_seed_mask(Nin, Nvirt, m0, seed=seed_mask)
        else:
            print("Max_sequences mask will be used")
            snr.M = max_sequences_mask(Nin, Nvirt, m0)
    # print('Mask matrix M_after:', snr.M.M)

    # save the trained mask matrix
    trained_mask = snr.M


    # Training
    S_train, J_train = snr.transform(x_train,params,beta_ref, *weights)
    np.size(S_train)
    seed_training = kwargs.get('seed_training', 1234)
    RR.Kfold_train(net,S_train,y_train,10, quiet = True, seed_training=seed_training)
    # print('net_w_aftertrain',net.W)

    # set the testing temperature range
    start_beta_prime = beta_prime
    step_beta_prime = step

    left_sequence = np.arange(start_beta_prime, beta_left, -step_beta_prime)[1:]  
    right_sequence = np.arange(start_beta_prime, beta_right + step_beta_prime, step_beta_prime) 
    beta_prime_list = np.sort(np.concatenate((left_sequence, right_sequence)))

    beta_primes_temp = []
    nrmse_temp = []


    # Testing in a wide temperature range
    for i in tqdm(beta_prime_list, desc="Processing beta_prime"):
        params['beta_prime'] = i
        print("beta_prime in single_node_heterogenous_reservoir:", i)
        snr_test = single_node_heterogenous_reservoir(Nin, Nvirt,Nout, gamma, i, beta_ref, deltabeta_list,h, theta, m0)
        print('snr_test_beta_prime:', snr_test.beta_prime)
        snr_test.M = trained_mask
        # print('Mask matrix M_test:', snr_test.M.M)
        S_test, J_test = snr_test.transform(x_test,params,beta_ref, *weights)
        spacer = kwargs.get('spacer_NRMSE', 0) # avoid the problem of dividing by zero
        print("Spacer NRMSE:"+str(spacer))
        pred = net.forward(S_test)
        # print('net_w_test',net.W)
        np.size(pred)
        error = MSE(pred, y_test)
        predNRMSE = NRMSE(pred, y_test, spacer=spacer)
        print(error, predNRMSE)

        plt.plot( np.linspace(0.0,1.0), np.linspace(0.0,1.0), 'k--')
        plt.plot(y_test, pred, 'o')
        plt.show()

        beta_primes_temp.append(i)
        nrmse_temp.append(predNRMSE)

    return beta_primes_temp, nrmse_temp

    # return_outputs = kwargs.get('return_outputs', False)
    # if return_outputs:
    #     return(y_test,pred)

    # return_NRMSE = kwargs.get('return_NRMSE', False)
    # if return_NRMSE:
    #     return(predNRMSE)


def spnc_spoken_digits(speakers,Nvirt,m0,bias,transform,params,*args,**kwargs):
    """
    perfoms the spoken digit task with a given resevoirs

    This code is in draft, it is subject to change and error!

    Parameters
    ----------
    speakers : list of str
        Leave as None for all, otherwise list, e.g: speakers = ['f1', 'f2',...]
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

    # Specifying digits_only=True and train=True returns only
    # the spoken digits part of TI20 training set
    # It returns the signal, label, sampling rate and speaker of the data
    train_signal, train_label, train_rate, train_speaker = TI46.load_TI20(
        speakers, digits_only=True, train=True)

    def stratified_split( labels, N, seed=1234):
        '''
        keys = tuple
        N = int, number of each key for the first split
        seed = int, seed for RNG
        '''

        indexes = tuple(np.unique(x) for x in labels)
        sizes = tuple(len(x) for x in indexes)
        label_ids = list( np.array([], dtype=int) for i in range(np.prod(sizes)))

        for i,label in enumerate(zip(*labels)):
            ids = np.array( [np.where(l==index)[0][0] for l,index in zip(label, indexes)])
            idx = np.ravel_multi_index(ids, sizes)
            label_ids[idx] = np.append(label_ids[idx], i)

        rng = np.random.default_rng(seed)
        split1 = np.array([], dtype=int)
        split2 = np.array([], dtype=int)
        for idxs in label_ids:
            rng.shuffle(idxs)
            split1 = np.append( split1, idxs[:N])
            split2 = np.append( split2, idxs[N:])
        rng.shuffle(split1)
        rng.shuffle(split2)
        return split1, split2

    split1, split2 = stratified_split((train_speaker, train_label), 9)


    # To load the test data, specify train=False
    test_signal, test_label, test_rate, test_speaker = TI46.load_TI20(
        speakers, digits_only=True, train=False)

    print("Samples for training: ", len(train_signal))
    print("Samples for test: ", len(test_signal))


    # Pre-processing
    from audio_preprocess import mfcc_func
    pre_process = mfcc_func

    nf = kwargs.get('nfft', 512)
    x_train = pre_process(train_signal, train_rate, nfft=nf)

    print(x_train[0].shape)

    #Normalise the input into the range 0 - 1
    xn = normalise(x_train)

    Nin = x_train[0].shape[-1]
    Nout = len(np.unique(train_label))

    print( 'Nin =', Nin, ', Nout = ', Nout, ', Nvirt = ', Nvirt)

    SNR = single_node_reservoir(Nin, Nout, Nvirt, m0, res = transform)

    fixed_mask = kwargs.get('fixed_mask', False)
    if fixed_mask:
        print("Deterministic mask will be used")
        SNR.M = fixed_seed_mask(Nin, Nvirt, m0)

    S_train, J_train = SNR.transform(xn, params)
    
    post_process = block_process
    post_process = lambda S, *args, **kwargs : np.copy(S)

    Nblocks = 4
    z_train = post_process(S_train, Nblocks, plot=False)

    y_train_1h = create_1hot_like(Nout, z_train, train_label)


    # Instantiate a linear output layer
    # act and inv_act are the activation function and it's inverse
    # either leave blank or set to linear to not apply activation fn
    net = linear(Nin, Nout, bias=bias)

    z_train_flat = np.vstack(z_train[split1])
    y_train_1h_flat = np.vstack(y_train_1h[split1])
    # Use the ridge regression training routine
    RR.Kfold_train(net, z_train_flat, y_train_1h_flat, 5, quiet=True)


    print('Weight matrix size = ', net.W.shape)

    conf_mat = np.zeros((Nout,Nout))
    pred_labels = np.zeros(len(split2), dtype=int)
    Ncorrect = 0
    for i, (zi, li) in enumerate(zip(z_train[split2], train_label[split2])):
        pi = net.forward(zi)
        pl = np.argmax(np.mean(pi, axis=0))
        pred_labels[i] = pl
        #print(y_test[i], pl, np.mean(pi,axis=0))
        conf_mat[li,pl] += 1.0
        if pl == li:
            Ncorrect += 1

    print(classification_report(train_label[split2], pred_labels,))
    valid_report = classification_report(train_label[split2], pred_labels, output_dict=True)
    print(valid_report.keys())
    print(Ncorrect/len(split2), Ncorrect, len(split2))

    plt.imshow(conf_mat)
    plt.show()

    plt.imshow(net.W)
    plt.show()

    x_test = pre_process(test_signal, test_rate, nfft=nf)
    xn_test = normalise(x_test)
    S_test, J_test = SNR.transform(xn_test, params)
    z_test = post_process(S_test, Nblocks, plot=False)

    conf_mat = np.zeros((Nout,Nout))

    Ncorrect = 0
    for i in range(len(z_test)):
        pi = net.forward(z_test[i])
        pl = np.argmax(np.mean(pi, axis=0))
        #print(y_test[i], pl, np.mean(pi,axis=0))
        conf_mat[test_label[i],pl] += 1.0
        if pl == test_label[i]:
            Ncorrect += 1

    print(Ncorrect/len(S_test), Ncorrect, len(z_test))

    plt.imshow(conf_mat)
    plt.show()


    plt.imshow(net.W)
    plt.show()

    return_accuracy = kwargs.get('return_accuracy', False)
    if return_accuracy:
        return(Ncorrect/len(S_test))

    # np.savetxt('W', net.W)
    # np.savetxt('M', SNR.M.M)
    # np.savetxt('V', np.matmul(net.W.T[:,:-1], SNR.M.M).T)


# ################# THIS LINE IS LEFT INTENTIONALLY COMMENTED ###############
