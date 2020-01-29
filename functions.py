import numpy as np
from numpy import inf
import itertools
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils import check_consistent_length, assert_all_finite, column_or_1d, check_array
import scipy.optimize as opt
from scipy.optimize import fsolve
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from copy import deepcopy

from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, kOrange, gStyle
from ROOT import gErrorIgnoreLevel
from ROOT import kInfo, kWarning, kError

from constants import *

import math
import pickle
import sys
import os

# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def dict_to_str(parameters):
    layers_str = [str(parameters['layers'][i]) for i in range(len(parameters['layers']))]
    tag = 'layers_'
    for i in range(len(layers_str)):
        tag = tag + layers_str[i]
        if i < len(layers_str)-1: tag = tag + '_'
    tag = tag + '__batchsize_' + str(parameters['batchsize'])
    tag = tag + '__classes_' + str(len(parameters['classes'])) + '_'
    for i in range(len(parameters['classes'])):
        for j in range(len(parameters['classes'][i])):
            tag = tag + parameters['classes'][i][j]
            if j < len(parameters['classes'][i]) - 1:
                tag = tag + '+'
        if i < len(parameters['classes']) - 1:
            tag = tag + '_'

    tag = tag + '__regmethod_' + parameters['regmethod']
    tag = tag + '__regrate_' + '{num:06d}'.format(num=int(parameters['regrate']*100000.))
    tag = tag + '__batchnorm_' + str(parameters['batchnorm'])
    tag = tag + '__epochs_' + str(parameters['epochs'])
    tag = tag + '__learningrate_' + '{num:06d}'.format(num=int(parameters['learningrate']*100000.))
    tag = tag + '__runonfraction_' + '{num:03d}'.format(num=int(parameters['runonfraction']*100.))
    tag = tag + '__eqweight_' + str(parameters['eqweight'])
    tag = tag + '__preprocess_' + str(parameters['preprocess'])
    tag = tag + '__priorSigma_' + '{num:03d}'.format(num=int(parameters['sigma']*100.))
    #print("------ Sigma in TAG: ",parameters['sigma'])
    #if len(tag.split('__')) != len(parameters): raise ValueError('in dict_to_str: Number of parameters given in the dictionary does no longer match the prescription how to build the tag out of it.')
    return tag

def get_classes_tag(parameters):
    tag = 'classes_' + str(len(parameters['classes'])) + '_'
    for i in range(len(parameters['classes'])):
        for j in range(len(parameters['classes'][i])):
            tag = tag + parameters['classes'][i][j]
            if j < len(parameters['classes'][i]) - 1:
                tag = tag + '+'
        if i < len(parameters['classes']) - 1:
            tag = tag + '_'
    return tag

def get_classtitles(parameters):
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
    return classtitles

def get_fraction(parameters):
    runonfraction = parameters['runonfraction']
    string = str('{num:03d}'.format(num=int(parameters['runonfraction']*100.)))
    return string




def load_data(parameters, inputfolder, filepostfix):

    print 'Loading data...'
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)


    input_train = np.load(inputfolder+'/input_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    input_test = np.load(inputfolder+'/input_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    input_val = np.load(inputfolder+'/input_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)
    labels_train = np.load(inputfolder+'/labels_'+fraction+'_train'+filepostfix+'.npy')
    labels_test = np.load(inputfolder+'/labels_'+fraction+'_test'+filepostfix+'.npy')
    labels_val = np.load(inputfolder+'/labels_'+fraction+'_val'+filepostfix+'.npy')
    sample_weights_train = np.load(inputfolder+'/sample_weights_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    eventweights_train = np.load(inputfolder+'/eventweights_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    sample_weights_test = np.load(inputfolder+'/sample_weights_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    eventweights_test = np.load(inputfolder+'/eventweights_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    sample_weights_val = np.load(inputfolder+'/sample_weights_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)
    eventweights_val = np.load(inputfolder+'/eventweights_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)
  
    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']
    signals = {}
    signal_eventweights = {}
    signal_normweights = {}
    return input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights
    

