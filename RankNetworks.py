import numpy as np
from numpy import inf
import keras
import matplotlib
import math
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
from keras import metrics
import pickle
import os
from functions import *

def RankNetworks(outputfolder):

    # Get all the model folders
    modellist = os.listdir(outputfolder)

    # Loop over them, open the correct file
    modelresults_lumiweighted = {}
    modelresults_equallyweighted = {}
    for tag in modellist:
        if not os.path.isdir(outputfolder+tag): continue
        try:
            infile = open(outputfolder+tag+'/ModelPerformance.txt','r')
        except:
            print 'couldn\'t open file %s, skipping this one.' % (outputfolder+tag+'/ModelPerformance.txt')
            continue
        # print 'found file "%s"' % (outputfolder+tag+'/ModelPerformance.txt')
        if not 'runonfraction_100' in tag: continue
        if not 'epochs_500' in tag: continue
        if not 'layers_512_512__' in tag: continue
        # if not 'regrate_080000' in tag: continue
        lines = infile.readlines()
        # vallossclosest = -1
        for line in lines:
            """if 'Tag:' == line[0:4]:
                tag = line.split(' ')[1]
            el"""
            if '    ' == line[0:4]:
                line = line.replace(' ','')
                vallosses = line.split('--')[1]
                vallossmin = vallosses.split(',')[0].replace('(','')
                vallossfin = vallosses.split(',')[1].replace(')','')
                # print line
            elif 'Validation loss in point of closest approach' in line:
                line = line.split(':')[1]
                line = line.replace(' ','')
                vallossclosest = line.split(',')[0]
        if vallossclosest == -1:
            continue
        if ('eqweight_False' in tag) or ('equallyweighted_False' in tag):
            print tag
            # print vallossmin, vallossfin
            modelresults_lumiweighted[tag] = [float(vallossclosest), float(vallossmin), float(vallossfin)]
            # modelresults_lumiweighted[tag] = [float(vallossmin), float(vallossfin)]
            print modelresults_lumiweighted[tag]
            print len(modelresults_lumiweighted)
        elif ('eqweight_True') in tag or ('equallyweighted_True' in tag):
            modelresults_equallyweighted[tag] = [float(vallossclosest), float(vallossmin), float(vallossfin)]
            # modelresults_equallyweighted[tag] = [float(vallossmin), float(vallossfin)]
        else:
            raise ValueError('the part "eqweight_(True|False)" is not in the tag.')
        infile.close()

    for tag in modelresults_lumiweighted.keys():
        print modelresults_lumiweighted[tag]
    outfile = open(outputfolder + 'ModelRanking_lumiweighted.txt', 'w')
    idx = 0
    print len(modelresults_lumiweighted)
    for tag in sorted(modelresults_lumiweighted, key=modelresults_lumiweighted.get, reverse=False):
        text = 'Rank: ' + str(idx) + '\nTag: ' + tag + '\nLosses: ' + str(modelresults_lumiweighted[tag]) + '\n\n'
        print text
        outfile.write(text)
        idx=idx+1
    outfile.close()
    outfile = open(outputfolder + 'ModelRanking_equallyweighted.txt', 'w')
    idx = 0
    for tag in sorted(modelresults_equallyweighted, key=modelresults_equallyweighted.get, reverse=False):
        text = 'Rank: ' + str(idx) + '\nTag: ' + tag + 'Losses: ' + str(modelresults_equallyweighted[tag]) + '\n\n'
        outfile.write(text)
        idx=idx+1
    outfile.close()



def RankNetworksReduced(outputfolder):

    # Get all the model folders
    modellist = os.listdir(outputfolder)

    # Loop over them, open the correct file
    modelresults_lumiweighted = {}
    modelresults_equallyweighted = {}
    for tag in modellist:
        if not os.path.isdir(outputfolder+tag): continue
        try:
            infile = open(outputfolder+tag+'/ModelPerformance.txt','r')
        except:
            print 'couldn\'t open file %s, skipping this one.' % (outputfolder+tag+'/ModelPerformance.txt')
            continue
        # print 'found file "%s"' % (outputfolder+tag+'/ModelPerformance.txt')
        if not 'runonfraction_100' in tag: continue
        # if not 'regrate_080000' in tag: continue
        lines = infile.readlines()
        # vallossclosest = -1
        for line in lines:
            """if 'Tag:' == line[0:4]:
                tag = line.split(' ')[1]
            el"""
            if '    ' == line[0:4]:
                line = line.replace(' ','')
                vallosses = line.split('--')[1]
                vallossmin = vallosses.split(',')[0].replace('(','')
                vallossfin = vallosses.split(',')[1].replace(')','')
                # print line
            elif 'Validation loss in point of closest approach' in line:
                line = line.split(':')[1]
                line = line.replace(' ','')
                vallossclosest = line.split(',')[0]
        if vallossclosest == -1:
            continue
        if ('eqweight_False' in tag) or ('equallyweighted_False' in tag):
            print tag
            # print vallossmin, vallossfin
            modelresults_lumiweighted[tag] = [float(vallossclosest), float(vallossmin), float(vallossfin)]
            # modelresults_lumiweighted[tag] = [float(vallossmin), float(vallossfin)]
            print modelresults_lumiweighted[tag]
            print len(modelresults_lumiweighted)
        elif ('eqweight_True') in tag or ('equallyweighted_True' in tag):
            modelresults_equallyweighted[tag] = [float(vallossclosest), float(vallossmin), float(vallossfin)]
            # modelresults_equallyweighted[tag] = [float(vallossmin), float(vallossfin)]
        else:
            raise ValueError('the part "eqweight_(True|False)" is not in the tag.')
        infile.close()

    for tag in modelresults_lumiweighted.keys():
        print modelresults_lumiweighted[tag]
    outfile = open(outputfolder + 'ModelRanking_lumiweighted.txt', 'w')
    idx = 0
    print len(modelresults_lumiweighted)
    for tag in sorted(modelresults_lumiweighted, key=modelresults_lumiweighted.get, reverse=False):
        text = 'Rank: ' + str(idx) + '\nTag: ' + tag + '\nLosses: ' + str(modelresults_lumiweighted[tag]) + '\n\n'
        print text
        outfile.write(text)
        idx=idx+1
    outfile.close()
    outfile = open(outputfolder + 'ModelRanking_equallyweighted.txt', 'w')
    idx = 0
    for tag in sorted(modelresults_equallyweighted, key=modelresults_equallyweighted.get, reverse=False):
        text = 'Rank: ' + str(idx) + '\nTag: ' + tag + 'Losses: ' + str(modelresults_equallyweighted[tag]) + '\n\n'
        outfile.write(text)
        idx=idx+1
    outfile.close()
