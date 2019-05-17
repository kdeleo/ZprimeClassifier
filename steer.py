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
from TrainModelOnPredictions import *
from TrainSecondNetwork import *
from TrainThirdNetwork import *
from ExportModel import *





parameters = {'layers':[512, 512],
              'batchsize': 8192,
              'classes':{0: ['QCD_Mu'], 1: ['TTbar'], 2:['DYJets'], 3:['WJets'], 4:['ST']},
              # 'classes':{0: ['QCD_Mu'], 1: ['TTbar', 'DYJets', 'WJets', 'ST']},
              # 'classes':{0: ['TTbar'], 1:['DYJets'], 2:['WJets'], 3:['ST']},
              # 'classes':{0: ['TTbar'], 1:['ST']},
              # 'classes':{0: ['TTbar'], 1: ['DYJets', 'WJets', 'ST']},
              'regmethod': 'dropout',
              'regrate':0.40000,
              'batchnorm': True,
#              'epochs':500,
              'epochs':100,
              'learningrate': 0.00100,
#              'runonfraction': 0.99,
              'runonfraction': 0.01,
              'eqweight':False}

parameters_onpredictions ={'layers':[20, 20],
                            'batchsize': 8192,
                            'classes':parameters['classes'],
                            'regmethod': 'dropout',
                            'regrate':0.01000,
                            'batchnorm': True,
                            'epochs':499,
                            'learningrate': 0.00100,
                            'runonfraction': parameters['runonfraction'],
                            'eqweight':False}

tag = dict_to_str(parameters)
classtag = get_classes_tag(parameters)
tag_onpredictions = dict_to_str(parameters_onpredictions)
classtag_onpredictions = get_classes_tag(parameters_onpredictions)


# # # Get all the inputs
# # # ==================
GetInputs(parameters)
PlotInputs(parameters, inputfolder='input/'+classtag, filepostfix='', plotfolder='Plots/InputDistributions/' + classtag)

# # First network
# # =============
TrainNetwork(parameters)
# PredictExternal(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='')
# PlotPerformance(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='', use_best_model=False, usesignals=[2,4])
PlotPerformance(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='', plotfolder='Plots/'+tag, use_best_model=True, usesignals=[2,4])
# PlotInputs(parameters, inputfolder='output/'+tag+'/cut', filepostfix='_pass_best', plotfolder='Plots/'+tag+'/InputDistributions/pass')
# PlotInputs(parameters, inputfolder='output/'+tag+'/cut', filepostfix='_fail_best', plotfolder='Plots/'+tag+'/InputDistributions/fail')
# ExportModel(parameters, use_best_model=True)
# RankNetworks(outputfolder='output/')

# # Network trained on outputs
# # ==========================
# TrainModelOnPredictions(parameters, parameters_onpredictions, use_best_model=True)
# PredictExternalOnPredictions(parameters, inputfolder='input/'+classtag, inputfolder_predictions='output/'+tag, filepostfix='_best', outputfolder='output/'+tag+'/ModelsOnPredictions/output/'+tag_onpredictions)
# PlotPerformance(parameters_onpredictions, inputfolder='input/'+classtag, outputfolder='output/'+tag+'/ModelsOnPredictions/output/'+tag_onpredictions, filepostfix='', plotfolder='output/'+tag+'/ModelsOnPredictions/Plots/'+tag_onpredictions, use_best_model=True, usesignals=[2,4])

# # Second network
# # ==============
# TrainSecondNetwork(parameters, parameters2, use_best_model=True)
# PredictExternal(parameters, inputfolder='output/'+tag+'/cut', filepostfix='_fail_best', outputfolder='output/'+tag+'/SecondModels/output/'+tag2)
# PlotPerformance(parameters2, inputfolder='output/'+tag+'/cut', outputfolder='output/'+tag+'/SecondModels/output/'+tag2, filepostfix='_fail_best', plotfolder='output/'+tag+'/SecondModels/Plots/'+tag2, use_best_model=True, usesignals=[2,4])
# RankNetworks(outputfolder='output/'+tag+'/SecondModels/output/')

# # Third network
# # ==============
# TrainThirdNetwork(parameters, parameters2, parameters3, use_best_model=True)
# PredictExternal(parameters, inputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/cut', filepostfix='_fail_best', outputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/ThirdModels/output/'+tag3)
# PlotPerformance(parameters3, inputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/cut', outputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/ThirdModels/output/'+tag3, filepostfix='_fail_best', plotfolder='output/'+tag+'/SecondModels/output/'+tag2+'/ThirdModels/Plots/'+tag3, use_best_model=True, usesignals=[2,4])
# RankNetworks(outputfolder='output/'+tag+'/SecondModels/output/'+tag2+'/ThirdModels/output/')]



parameters2 ={'layers':[512, 512],
              'batchsize': 8192,
              'classes':parameters['classes'],
              'regmethod': 'dropout',
              'regrate':0.20000,
              'batchnorm': True,
              'epochs':500,
              'learningrate': 0.00100,
              'runonfraction': parameters['runonfraction'],
              'eqweight':False}

parameters3 ={'layers':[512, 512],
              'batchsize': 8192,
              'classes':parameters['classes'],
              'regmethod': 'dropout',
              'regrate':0.20000,
              'batchnorm': True,
              'epochs':500,
              'learningrate': 0.00100,
              'runonfraction': parameters['runonfraction'],
              'eqweight':False}

tag2 = dict_to_str(parameters2)
classtag2 = get_classes_tag(parameters2)
tag3= dict_to_str(parameters3)
classtag3 = get_classes_tag(parameters3)
