import numpy as np
from numpy import inf
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras import metrics, regularizers

#from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, gStyle
#from ROOT import gErrorIgnoreLevel, kInfo, kWarning, kError

import math
import pickle
import sys
import os
from functions import *
from constants import *
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import h5py
import pandas as pd

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def _prior_normal_fn(sigma, dtype, shape, name, trainable, add_variable_fn):
    """Normal prior with mu=0 and sigma=sigma. Can be passed as an argument to                                                                                               
    the tpf.layers                                                                                                                                                           
    """
    del name, trainable, add_variable_fn
    dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(sigma))
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)


def PlotPerformance(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print 'Now plotting the performance'
    #gErrorIgnoreLevel = kWarning

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Get model and its history
    model = keras.models.load_model(outputfolder+'/model.h5')
    with open(outputfolder+'/model_history.pkl', 'r') as f:
        model_history = pickle.load(f)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix) 



    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)


    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder)
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf')
    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)


    pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test)
    plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    # plot_outputs_2d(parameters=parameters, plotfolder=plotfolder, pred_vals=pred_vals, lumiweights_vals=lumiweights_vals, use_best_model=use_best_model)
    # best_cuts = cut_iteratively(parameters=parameters, outputfolder=outputfolder, pred_val=pred_val, labels_val=labels_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals)
    # plot_cuts(parameters=parameters, outputfolder=outputfolder, plotfolder=plotfolder, best_cuts=best_cuts, pred_vals=pred_vals, labels_val=labels_val, lumiweights_vals=lumiweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    # apply_cuts(parameters=parameters, outputfolder=outputfolder, best_cuts=best_cuts, input_train=input_train, input_val=input_val, input_test=input_test, labels_train=labels_train, labels_val=labels_val, labels_test=labels_test, sample_weights_train=sample_weights_train, sample_weights_val=sample_weights_val, sample_weights_test=sample_weights_test, eventweights_train=eventweights_train, eventweights_val=eventweights_val, eventweights_test=eventweights_test, pred_train=pred_train, pred_val=pred_val, pred_test=pred_test, signals=signals, eventweight_signals=eventweight_signals, pred_signals=pred_signals, signal_identifiers=signal_identifiers, use_best_model=use_best_model)


    # for cl in range(labels_train.shape[1]):
    #     # 'cl' is the output node number
    #     nbins = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001])
    #     y_trains = {}
    #     y_vals = {}
    #     y_tests = {}
    #     ytots = {}
    #
    #
    #
    #     for i in range(labels_train.shape[1]):
    #         # 'i' is the true class (always the first index)
    #         y_trains[i], dummy = np.histogram(pred_trains[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
    #         y_vals[i], dummy = np.histogram(pred_vals[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
    #         y_tests[i], dummy = np.histogram(pred_tests[i][cl], bins=nbins, weights=lumiweights_tests[i][cl])
    #         ytots[i] = y_trains[i] + y_vals[i] + y_tests[i]
    #         print "node %i, class %i" % (cl, i)
    #         print ytots[i]
    #         print 'sum: %f' % (ytots[i].sum())
    #Store model as JSON file for usage in UHH2
    arch = model.to_json()
    # save the architecture string to a file somehow, the below will work
    with open(outputfolder+'/architecture.json', 'w') as arch_file:
        arch_file.write(arch)
    # now save the weights as an HDF5 file
    model.save_weights(outputfolder+'/weights.h5')
    print "--- END of DNN Plotting ---"

def PlotBayesianPerformance(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print ' ---- Now plotting the performance (PlotBayesianPerformance) ---- '
    gErrorIgnoreLevel = kWarning

    # Get parameters
    # runonfullsample = parameters['runonfullsample']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    # classes = parameters['classes']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)
    layers=parameters['layers']

    # Get model and its history
    regrate=parameters['regrate']
    #sigma=1.0 #FixME, read https://arxiv.org/pdf/1904.10004.pdf to choose right value           
    sigma=parameters['sigma']         
    prior = lambda x, y, z, w, k: _prior_normal_fn(sigma, x, y, z, w, k)                                                                                                     
#    method = lambda d: d.mean()
    method = lambda d: d.sample()
    layer_hd = []
    batchnorm_hd = []
    dropout_hd = []
    inputs = tf.keras.layers.Input(shape=(input_train.shape[1],))


    #### Without DropOut
    #layer_hd.append(tf.layers.Dense(layers[0], activation=tf.nn.relu)(inputs))
    layer_hd.append(tfp.layers.DenseFlipout(layers[0], activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method)(inputs))
    k=1
    for i in layers[1:len(layers)+1]:
        print("current k:",k)
        label=str(k+1)
        #layer_hd.append(tfp.layers.DenseFlipout(i, activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method)(layer_hd[k-1]))
        #layer_hd.append(tf.layers.Dense(i, activation=tf.nn.relu)(layer_hd[k-1]))
        layer_hd.append(tfp.layers.DenseFlipout(i, activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method)(layer_hd[k-1]))
        k = k+1
    print("total number of hidden layers:",k)
    last_layer = tfp.layers.DenseFlipout(labels_train.shape[1], activation='softmax', kernel_prior_fn=prior, kernel_posterior_tensor_fn=method)(layer_hd[k-1])
    #last_layer = tf.layers.Dense(labels_train.shape[1], activation='softmax')(layer_hd[k-1])

    print 'Number of output classes: %i' % (labels_train.shape[1])
    model = tf.keras.models.Model(inputs=inputs,outputs=last_layer)
    opt  = tf.train.AdamOptimizer()
    mymetrics = [metrics.categorical_accuracy]
    file = h5py.File(outputfolder+'/model_weights.h5', 'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight' + str(i)][:])
    model.set_weights(weight)
    print model.summary()

    model_history = pd.read_hdf(outputfolder+'/summary.h5')

    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train_all, pred_test_all, pred_val_all, pred_signals_all = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix) #load predictions sampled N times


    pred_signals = {}
    pred_signals_std = {}
#    for i in range(len(signal_identifiers)):
#        pred_signals[i] = np.median(pred_signals_all[i],axis=0)
#        pred_signals_std[i] = np.std(pred_signals_all[i],axis=0)

    pred_train = np.median(pred_train_all,axis=0)
    pred_test = np.median(pred_test_all,axis=0)
    pred_val = np.median(pred_val_all,axis=0)

    pred_train_std = np.std(pred_train_all,axis=0)
    pred_test_std = np.std(pred_test_all,axis=0)
    pred_val_std = np.std(pred_val_all,axis=0)
 
    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)
    for eventid in range(100):
        plot_prediction_samples(parameters=parameters,plotfolder=plotfolder+'/ResponseSamples',pred_train_all=pred_train_all,labels_train=labels_train,eventID=eventid)

    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder) #OK
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history) #OK
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history) #OK
#    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf') #OK
    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)

    #print "--- mean ---"
    pred_trains, weights_trains, normweights_trains, lumiweights_trains = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_train, sample_weights=sample_weights_train, pred=pred_train, labels=labels_train)
    pred_vals, weights_vals, normweights_vals, lumiweights_vals = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_val, sample_weights=sample_weights_val, pred=pred_val, labels=labels_val)
    pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_test, sample_weights=sample_weights_test, pred=pred_test, labels=labels_test)

   # plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    #print "--- std ---"
    pred_trains_std, weights_trains_std, normweights_trains_std, lumiweights_trains_std = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_train, sample_weights=sample_weights_train, pred=pred_train_std, labels=labels_train)
    pred_vals_std, weights_vals_std, normweights_vals_std, lumiweights_vals_std = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_val, sample_weights=sample_weights_val, pred=pred_val_std, labels=labels_val)
    pred_tests_std, weights_tests_std, normweights_tests_std, lumiweights_tests_std = get_data_dictionaries_onesample(parameters=parameters, eventweights=eventweights_test, sample_weights=sample_weights_test, pred=pred_test_std, labels=labels_test)
    # #plot classifier output with error from std of the output
    # plot_outputs_1d_nodes_with_stderror(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains,pred_trains_std=pred_train_std, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals,pred_vals_std=pred_vals_std, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, pred_signals_std=pred_signals_std,eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    # #plot std of classifier output
    #print("aaaaa pred_trains_std[0] = ",pred_trains_std[0])
    plot_outputs_1d_nodes_std(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains,pred_trains_std=pred_trains_std, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals,pred_vals_std=pred_vals_std, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, pred_signals_std=pred_signals_std,eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)

    # # #FixME:
    # # #TypeError: ('Not JSON Serializable:', <function _fn at 0x2b8475af1668>)
    # print("... Store model architecture in PlotBayesianPerformance ...")
    # #Store model as JSON file for usage in UHH2
    # arch = model.to_json()
    # # save the architecture string to a file somehow, the below will work
    # with open(outputfolder+'/architecture.json', 'w') as arch_file:
    #     arch_file.write(arch)
    # # now save the weights as an HDF5 file
    # model.save_weights(outputfolder+'/weights.h5')
    print("End of PlotBayesianPerformance")
   
def PlotDeepPerformance(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print ' ---- Now plotting the performance (PlotDeepPerformance) ---- '
    gErrorIgnoreLevel = kWarning

    # Get parameters
    # runonfullsample = parameters['runonfullsample']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    # classes = parameters['classes']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)
    layers=parameters['layers']

    # Get model and its history
    regrate=parameters['regrate']
    #sigma=1 #FixME, read https://arxiv.org/pdf/1904.10004.pdf to choose right value   
    sigma=parameters['sigma']                    
    prior = lambda x, y, z, w, k: _prior_normal_fn(sigma, x, y, z, w, k)                                                                                                     
    method = lambda d: d.mean()
    layer_hd = []
    batchnorm_hd = []
    dropout_hd = []
    inputs = tf.keras.layers.Input(shape=(input_train.shape[1],))
    layer_hd.append(tf.layers.Dense(layers[0], activation=tf.nn.relu)(inputs))
    dropout_hd.append(tf.layers.dropout(layer_hd[0], rate=regrate)) #FixME: regmethod might be different
    #batchnorm_hd.append(tf.layers.batch_normalization(dropout_hd[0]))
    k=1
    for i in layers[1:len(layers)+1]:
        print("current k:",k)
        label=str(k+1)
        #layer_hd.append(tfp.layers.DenseFlipout(i, activation=tf.nn.relu)(batchnorm_hd[k-1])) 
        layer_hd.append(tf.layers.Dense(i, activation=tf.nn.relu)(dropout_hd[k-1]))
        dropout_hd.append(tf.layers.dropout(layer_hd[k], rate=regrate))
        #batchnorm_hd.append(tf.layers.batch_normalization(dropout_hd[k]))
        k = k+1
    print("total number of hidden layers:",k)
    #last_layer = tf.layers.Dense(labels_train.shape[1], activation=tf.nn.relu)(dropout_hd[k-1])
    last_layer = tf.layers.Dense(labels_train.shape[1], activation='softmax')(dropout_hd[k-1])
    print 'Number of output classes: %i' % (labels_train.shape[1])
    model = tf.keras.models.Model(inputs=inputs,outputs=last_layer) 
    file = h5py.File('output/DNN_'+tag+'/model.h5', 'w')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight' + str(i)][:])
    model.set_weights(weight)                                                                                                                             
    model.summary()

    
    with open(outputfolder+'/model_history.pkl', 'r') as f:
        model_history = pickle.load(f)


    predpostfix = ''
    if use_best_model:
        predpostfix = '_best'
    pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=outputfolder, filepostfix=predpostfix)

    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)

    log_model_performance(parameters=parameters, model_history=model_history, outputfolder=outputfolder)
    plot_loss(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_accuracy(parameters=parameters, plotfolder=plotfolder, model_history=model_history)
    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.pdf')
    plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)


    pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test)
    plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)

    plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    



def PlotInputs(parameters, inputfolder, filepostfix, plotfolder):

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    classtag = get_classes_tag(parameters)
    tag = dict_to_str(parameters)
    eqweight=parameters['eqweight']

    if not os.path.isdir(plotfolder):
        os.makedirs(plotfolder)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    with open(inputfolder+'/variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)

    print "GET for plotting: input_train[0] = ", input_train[0]

    # Divide into classes
    input_train_classes = {}
    input_test_classes = {}
    input_val_classes = {}
    weights_train_classes = {}
    weights_test_classes = {}
    weights_val_classes = {}
    for i in range(labels_train.shape[1]):
        input_train_classes[i] = input_train[labels_train[:,i] == 1]
        input_test_classes[i] = input_test[labels_test[:,i] == 1]
        input_val_classes[i] = input_val[labels_val[:,i] == 1]
        weights_train_classes[i] = sample_weights_train[labels_train[:,i] == 1]
        weights_test_classes[i] = sample_weights_test[labels_test[:,i] == 1]
        weights_val_classes[i] = sample_weights_val[labels_val[:,i] == 1]

        if not eqweight: 
          weights_train_classes[i] = eventweights_train[labels_train[:,i] == 1]
          weights_test_classes[i] =  eventweights_test[labels_test[:,i] == 1]
          weights_val_classes[i] =   eventweights_val[labels_val[:,i] == 1]

    # Create class-title dictionary
    classes = parameters['classes']
    classtitles = {}
    for key in classes.keys():
        list = classes[key]
        title = ''
        for i in range(len(list)):
            title = title + list[i]
            if i < len(list)-1:
                title = title + '+'
        classtitles[key] = title

    matplotlib.style.use('default')
    # print input_train_classes
    nbins = 100
    idx = 0
    for varname in variable_names:
        xmax = max([max(input_train_classes[i][:,idx]) for i in range(len(input_train_classes))])
        xmin = min([min(input_train_classes[i][:,idx]) for i in range(len(input_train_classes))])
        if xmax == xmin: xmax = xmin + 1.
        xmin = min([0,xmin])
        binwidth = (xmax - xmin) / float(nbins)
        bins = np.arange(xmin, xmax + binwidth, binwidth)

        plt.clf()
        fig = plt.figure()
        for i in range(len(input_train_classes)):
            mycolor = 'C'+str(i)
            #print input_train_classes[i][:,idx], len(input_train_classes[i][:,idx]), str(i), colorstr[i], varname
            plt.hist(input_train_classes[i][:,idx], weights=weights_train_classes[i], bins=bins, histtype='step', label='Training sample, '+classtitles[i], color=colorstr[i])
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel(varname)
        plt.ylabel('Number of events / bin')
        fig.savefig(plotfolder + '/' + varname + '_'+fraction+'.pdf')
        # if runonfullsample: fig.savefig('Plots/InputDistributions/' + classtag+  '/' + varname + '_full.pdf')
        # else: fig.savefig('Plots/InputDistributions/' + classtag+  '/' + varname + '_part.pdf')
        idx += 1

        sys.stdout.write( '{0:d} of {1:d} plots done.\r'.format(idx, len(variable_names)))
        if not i == len(variable_names): sys.stdout.flush()
