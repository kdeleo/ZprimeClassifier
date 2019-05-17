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

from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, gStyle
from ROOT import gErrorIgnoreLevel, kInfo, kWarning, kError

import math
import pickle
import sys
import os
from functions import *
from constants import *

def PlotPerformance(parameters, inputfolder, outputfolder, filepostfix, plotfolder, use_best_model=False, usesignals=[0]):
    print 'Now plotting the performance'
    gErrorIgnoreLevel = kWarning

    # Get parameters
    # runonfullsample = parameters['runonfullsample']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight = parameters['eqweight']
    # classes = parameters['classes']
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
#    plot_rocs(parameters=parameters, plotfolder=plotfolder, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, pred_signals=pred_signals, eventweight_signals=eventweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    plot_model(model, show_shapes=True, to_file=plotfolder+'/Model.ps')
    #plot_confusion_matrices(parameters=parameters, plotfolder=plotfolder, pred_train=pred_train, labels_train=labels_train, sample_weights_train=sample_weights_train, eventweights_train=eventweights_train, pred_val=pred_val, labels_val=labels_val, sample_weights_val=sample_weights_val, eventweights_val=eventweights_val, use_best_model=use_best_model)
    #
    #
    # pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests = get_data_dictionaries(parameters=parameters, eventweights_train=eventweights_train, sample_weights_train=sample_weights_train, pred_train=pred_train, labels_train=labels_train, eventweights_val=eventweights_val, sample_weights_val=sample_weights_val, pred_val=pred_val, labels_val=labels_val, eventweights_test=eventweights_test, sample_weights_test=sample_weights_test, pred_test=pred_test, labels_test=labels_test)
    # plot_outputs_1d_nodes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, pred_signals=pred_signals, eventweight_signals=eventweight_signals, normweight_signals=normweight_signals, usesignals=usesignals, use_best_model=use_best_model)
    #
    # plot_outputs_1d_classes(parameters=parameters, plotfolder=plotfolder, pred_trains=pred_trains, labels_train=labels_train, weights_trains=weights_trains, lumiweights_trains=lumiweights_trains, normweights_trains=normweights_trains, pred_vals=pred_vals, labels_val=labels_val, weights_vals=weights_vals, lumiweights_vals=lumiweights_vals, normweights_vals=normweights_vals, use_best_model=use_best_model)
    # # plot_outputs_2d(parameters=parameters, plotfolder=plotfolder, pred_vals=pred_vals, lumiweights_vals=lumiweights_vals, use_best_model=use_best_model)
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



def PlotInputs(parameters, inputfolder, filepostfix, plotfolder):

    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    classtag = get_classes_tag(parameters)
    tag = dict_to_str(parameters)

    if not os.path.isdir(plotfolder):
        os.makedirs(plotfolder)

    # Get inputs
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)
    with open('input/'+classtag+'/variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)

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
    nbins = 50
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
        plt.close()
