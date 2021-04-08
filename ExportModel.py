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


def WriteArray(fout, array):
    array = array.tolist()
    str_ = ""
    for el in array:
        str_ += " " if el>0 else ""
        str_ += str(round(el,6)) + ",\t"
    fout.write(str_+"\n")


def WriteWeights(fout, weights):
    if len(weights.shape) == 1:
        WriteArray(fout, weights)
    elif len(weights.shape) == 2:
        for w in weights:
            WriteArray(fout, w)

def ExportModel(parameters, inputfolder='input/', outputfolder='output/', use_best_model=False):
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    postfix = ''
    if use_best_model: postfix += '_best'
    model = keras.models.load_model(outputfolder+tag+'/model'+postfix+'.h5')
    filename1 = 'mymodel'
    if use_best_model: filename1 += '_best'
    filename1 += '.txt'
    with open(outputfolder+tag+'/'+filename1, 'w') as fout:
        for ind, l in enumerate(model.get_config()['layers']):
            if l['class_name'] == "Dropout":
                continue
            fout.write("New Layer\n")
            fout.write(l["class_name"]+"\n")
            if l['class_name'] == "Dense":
                fout.write("weights"+"\n")
                WriteWeights(fout, model.layers[ind].get_weights()[0])
                fout.write("bias"+"\n")
                WriteWeights(fout, np.expand_dims(model.layers[ind].get_weights()[1],axis=0) )
                fout.write("activation\t"+l["config"]["activation"]+"\n")
            if l['class_name'] == "BatchNormalization":
                fout.write("epsilon"+"\n")
                fout.write(str(model.layers[ind].epsilon)+"\n")
                fout.write("gamma"+"\n")
                WriteWeights(fout, np.expand_dims(model.layers[ind].get_weights()[0], axis=0) )
                fout.write("beta"+"\n")
                WriteWeights(fout, np.expand_dims(model.layers[ind].get_weights()[1], axis=0) )
                fout.write("moving_mean"+"\n")
                WriteWeights(fout, np.expand_dims(model.layers[ind].get_weights()[2], axis=0) )
                fout.write("moving_variance"+"\n")
                WriteWeights(fout, np.expand_dims(model.layers[ind].get_weights()[3], axis=0) )

#    # Second file: General model information
#    # 1) layers
#    filename2 = 'mymodelinfo'
#    if use_best_model: filename2 += '_best'
#    filename2 += '.txt'
#    layers_str = '['
#    for i in range(len(parameters['layers'])):
#        layers_str += str(parameters['layers'][i])
#        if i < len(parameters['layers']) - 1:
#            layers_str += ', '
#    layers_str += ']'
#    print layers_str
#
#    # 2) output classes
#    classes_str = '['
#    for i in range(len(parameters['classes'])):
#        for j in range(len(parameters['classes'][i])):
#            classes_str += parameters['classes'][i][j]
#            if j < len(parameters['classes'][i]) - 1:
#                classes_str += '+'
#        if i < len(parameters['classes']) - 1:
#            classes_str += ', '
#    classes_str += ']'
#    print classes_str
#
#    # 3) variable names
#    with open(inputfolder+classtag+'/variable_names.pkl', 'r') as f:
#        variable_names = pickle.load(f)
#    variables_str = '['
#    for i in range(len(variable_names)):
#        variables_str += str(variable_names[i])
#        if i < len(variable_names) - 1:
#            variables_str += ', '
#    variables_str += ']'
#
#    with open(outputfolder+tag+'/'+filename2, 'w') as fout:
#        fout.write('layers %s\n' % (layers_str))
#        fout.write('classes %s\n' % (classes_str))
#        fout.write('variables %s\n' % (variables_str))
