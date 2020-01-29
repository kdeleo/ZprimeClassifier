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

from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, gStyle
from ROOT import gErrorIgnoreLevel, kInfo, kWarning, kError

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
