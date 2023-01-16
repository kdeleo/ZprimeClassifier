#!/usr/bin/env python

import numpy as np
from numpy import inf
import keras
import matplotlib
matplotlib.use('Agg')
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
import math
import pickle
from Training import *
from Plotting import *
from GetInputs import *
from RankNetworks import *
from PredictExternal import *
from functions import *
#from TrainModelOnPredictions import *
#from TrainSecondNetwork import *
#from TrainThirdNetwork import *
from ExportModel import *

# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

#variations = ['NOMINAL','JEC_up','JEC_down','JER_up','JER_down']
variations = ['NOMINAL']
merged_str = 'Merged'
parameters = {
    'layers':[512,512],
    'batchsize': 32768,
    'classes':{0: ['TTbar'], 1: ['ST'], 2:['WJets','DY']},
    'regmethod': 'dropout',
    'regrate':0.50,
    'batchnorm': False,
    'epochs':500,
    'learningrate': 0.00050,
    'runonfraction': 1.0,
    'eqweight': False,
    'preprocess': 'StandardScaler',
    'sigma': 1.0, #sigma for Gaussian prior (BNN only)
    #'inputdir': '../Inputs_UL18_muon/',
    'inputdir': '../Inputs_UL17_muon_lite/',
    #        'systvar': variations[ivars],
    'inputsubdir': 'MLInput/', #path to input files: inputdir + systvar + inputsubdir
    'prepreprocess': 'RAW' #for inputs with systematics don't do preprocessing before merging all inputs on one,     #FixME: add prepreprocessing in case one does not need to merge inputs
}

tag = dict_to_str(parameters)
classtag = get_classes_tag(parameters)

########## GetInputs ########
for ivars in range(len(variations)):
     merged_str = merged_str+'__'+variations[ivars]
     parameters['systvar'] = variations[ivars]
     # # # # # # Get all the inputs
     # # # # # # # # # ==================
     inputfolder = parameters['inputdir']+parameters['inputsubdir']+parameters['systvar']+'/'+parameters['prepreprocess']+'/'+ classtag
     GetInputs(parameters)
     PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder='Plots/'+parameters['prepreprocess']+'/InputDistributions/'+parameters['systvar']+'/' + classtag)
  
MixInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, variations=variations, filepostfix='')
SplitInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
FitPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='train')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='test')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='val')
#ApplySignalPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')

inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag
outputfolder='output/'+parameters['preprocess']+'/'+merged_str+'/' + classtag+'/DNN_'+tag
plotfolder = 'Plots/'+parameters['preprocess']
PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder=plotfolder+'/InputDistributions/'+merged_str+'/' + classtag)

#####

# DNN 
TrainNetwork(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag)
PredictExternal(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='')
PlotPerformance(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='', plotfolder='Plots/'+parameters['preprocess']+'/DNN_'+tag, use_best_model=True, usesignals=[2,4])
#ExportModel(parameters, inputfolder='input/', outputfolder='output/', use_best_model=True)
