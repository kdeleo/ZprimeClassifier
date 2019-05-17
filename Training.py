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
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras import metrics, regularizers
import pickle
from copy import deepcopy
import os
from functions import *
from PredictExternal import *

def TrainNetwork(parameters):

    # Get parameters
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    regmethod=parameters['regmethod']
    regrate=parameters['regrate']
    batchnorm=parameters['batchnorm']
    epochs=parameters['epochs']
    learningrate = parameters['learningrate']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight=parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    train_new_model = True
    try:
        model = keras.models.load_model('output/'+tag+'/model.h5')
        train_new_model = False
    except:
        pass
    if train_new_model: print 'Couldn\'t find the model "%s", a new one will be trained!' % (tag)
    else:
        print 'Found the model "%s", not training a new one, go on to next function.' % (tag)
        return
        # print 'Found model, but I will retrain it!'
    if not os.path.isdir('output/' + tag): os.makedirs('output/'+tag)

    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder='input/'+classtag, filepostfix='')


    # Define the network
    model = Sequential()
    kernel_regularizer = None
    if regmethod == 'L1':
        kernel_regularizer=regularizers.l1(regrate)
    elif regmethod == 'L2':
        kernel_regularizer=regularizers.l2(regrate)


    print 'Number of input variables: %i' % (input_train.shape[1])
    model.add(Dense(layers[0], activation='relu', input_shape=(input_train.shape[1],), kernel_regularizer=kernel_regularizer))
    if regmethod == 'dropout': model.add(Dropout(regrate))
    if batchnorm: model.add(BatchNormalization())

    for i in layers[1:len(layers)+1]:
        model.add(Dense(i, activation='relu', kernel_regularizer=kernel_regularizer))
        if batchnorm: model.add(BatchNormalization())
        if regmethod == 'dropout': model.add(Dropout(regrate))

    model.add(Dense(labels_train.shape[1], activation='softmax', kernel_regularizer=kernel_regularizer))
    # model.add(Dense(labels_train.shape[1], activation='sigmoid', kernel_regularizer=kernel_regularizer))
    print 'Number of output classes: %i' % (labels_train.shape[1])



    # Train the network
    opt = keras.optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # mymetrics = [metrics.categorical_accuracy]
    mymetrics = [metrics.categorical_accuracy, metrics.mean_squared_error, metrics.categorical_crossentropy, metrics.kullback_leibler_divergence, metrics.cosine_proximity]
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=mymetrics)
    print model.summary()

    period = epochs / 5
    checkpointer = ModelCheckpoint(filepath='output/'+tag+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=period)
    checkpointer_everymodel = ModelCheckpoint(filepath='output/'+tag+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=1)
    checkpoint_bestmodel = ModelCheckpoint(filepath='output/'+tag+'/model_best.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=20, verbose=0, mode='min', baseline=None, restore_best_weights=True)

    LRreducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_delta=0.001, mode='min')
    weights_train, weights_test = sample_weights_train, sample_weights_test
    if not eqweight:
        weights_train, weights_test = eventweights_train, eventweights_test



    # model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer, checkpoint_bestmodel, earlystopping], verbose=1)
    # model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer, checkpoint_bestmodel, LRreducer], verbose=2)
    # model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer, checkpoint_bestmodel], verbose=1)

#    tensorboard = TensorBoard(log_dir='output/logs', histogram_freq=0, update_freq='epoch') #FixME
    model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer_everymodel, checkpoint_bestmodel], verbose=1)


    model.save('output/'+tag+'/model.h5')
    with open('output/'+tag+'/model_history.pkl', 'w') as f:
        pickle.dump(model.history.history, f)


    PredictExternal(parameters, inputfolder='input/'+classtag, outputfolder='output/'+tag, filepostfix='')


def TrainForMoreEpochs(parameters, nepochs):
    layers=parameters['layers']
    batch_size=parameters['batchsize']
    regmethod=parameters['regmethod']
    regrate=parameters['regrate']
    batchnorm=parameters['batchnorm']
    epochs=parameters['epochs']
    learningrate = parameters['learningrate']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    eqweight=parameters['eqweight']
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)

    new_parameters = parameters
    new_parameters['epochs'] = new_parameters['epochs'] + nepochs
    new_tag = dict_to_str(new_parameters)

    new_model_already_exists = False
    # Find out if the new model would already exist
    try:
        model = keras.models.load_model('output/'+new_tag+'/model.h5')
        new_model_already_exists = True
    except:
        pass
    if new_model_already_exists:
        print 'The new model would already exists'
        return new_parameters
    else:
        print 'Going to train for %i more epochs.' % nepochs

    if not os.path.isdir('output/' + new_tag): os.makedirs('output/'+new_tag)



    # Get inputs
    input_train = np.load('input/'+classtag+'/input_'+fraction+'_train.npy')
    input_test = np.load('input/'+classtag+'/input_'+fraction+'_test.npy')
    input_val = np.load('input/'+classtag+'/input_'+fraction+'_val.npy')
    labels_train = np.load('input/'+classtag+'/labels_'+fraction+'_train.npy')
    labels_test = np.load('input/'+classtag+'/labels_'+fraction+'_test.npy')
    labels_val = np.load('input/'+classtag+'/labels_'+fraction+'_val.npy')
    with open('input/'+classtag+'/sample_weights_'+fraction+'_train.pkl', 'r') as f:
        sample_weights_train = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_train.pkl', 'r') as f:
        eventweights_train = pickle.load(f)
    with open('input/'+classtag+'/sample_weights_'+fraction+'_test.pkl', 'r') as f:
        sample_weights_test = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_test.pkl', 'r') as f:
        eventweights_test = pickle.load(f)
    with open('input/'+classtag+'/sample_weights_'+fraction+'_val.pkl', 'r') as f:
        sample_weights_val = pickle.load(f)
    with open('input/'+classtag+'/eventweights_'+fraction+'_val.pkl', 'r') as f:
        eventweights_val = pickle.load(f)


    model = keras.models.load_model('output/'+tag+'/model.h5')
    with open('output/'+tag+'/model_history.pkl', 'r') as f:
        model_history = pickle.load(f)

    # Train the network
    opt = keras.optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    mymetrics = [metrics.categorical_accuracy]
    # mymetrics = [metrics.categorical_accuracy, metrics.mean_squared_error, metrics.categorical_crossentropy, metrics.kullback_leibler_divergence]
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=mymetrics)
    print model.summary()

    period = epochs / 5
    checkpointer = ModelCheckpoint(filepath='output/'+tag+'/model_epoch{epoch:02d}.h5', verbose=1, save_best_only=False, period=period)
    weights_train, weights_test = sample_weights_train, sample_weights_test
    if not eqweight:
        weights_train, weights_test = eventweights_train, eventweights_test
    model.fit(input_train, labels_train, sample_weight=weights_train, batch_size=batch_size, epochs=nepochs, shuffle=True, validation_data=(input_test, labels_test, weights_test), callbacks=[checkpointer], verbose=1)


    model.save('output/'+new_tag+'/model.h5')
    with open('output/'+new_tag+'/model_history.pkl', 'w') as f:
        pickle.dump(model.history.history, f)



    # Do the predictions
    print 'Now that the model is trained, we\'re going to predict the labels of all 3 sets. '
    print 'predicting for training set'
    pred_train = model.predict(input_train)
    np.save('output/'+new_tag+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save('output/'+new_tag+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for test set'
    pred_test = model.predict(input_test)
    np.save('output/'+new_tag+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save('output/'+new_tag+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for val set'
    pred_val = model.predict(input_val)
    np.save('output/'+new_tag+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save('output/'+new_tag+'/prediction_val_class'+str(cl)+'.npy'  , tmp)

    return new_parameters
