import numpy as np
from numpy import inf
import keras
import matplotlib
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
#from tensorflow_probability.layers import DenseFlipout
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras import metrics, regularizers
import pickle
from copy import deepcopy
import os
from functions import *
import tensorflow as tf
import tensorflow_probability as tfp
import h5py
from tensorflow_probability import distributions as tfd

def _prior_normal_fn(sigma, dtype, shape, name, trainable, add_variable_fn):
    """Normal prior with mu=0 and sigma=sigma. Can be passed as an argument to 
    the tpf.layers
    """
    del name, trainable, add_variable_fn
    dist = tfd.Normal(loc=tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(sigma))
    batch_ndims = tf.size(input=dist.batch_shape_tensor())
    return tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)

def PredictExternal(parameters, inputfolder, outputfolder, filepostfix):
    print 'Making predictions now'
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)

    # Get inputs
    model = keras.models.load_model(outputfolder+'/model.h5')
    model_best = keras.models.load_model(outputfolder+'/model_best.h5')
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

#    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']

    # Do the predictions
    print 'Now that the model is trained, we\'re going to predict the labels of all 3 sets. '
    print 'predicting for training set'

    pred_train = model.predict(input_train)
    #print("input_train[0]",input_train[0])

    # print("input_train[1]",input_train[1])
    # print ("Keras pred_train[1] =",pred_train[1])
    # print("input_train[2]",input_train[2])
    # print ("Keras pred_train[2] =",pred_train[2])
    np.save(outputfolder+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for test set'
    print input_test.shape
    print labels_test.shape
    pred_test = model.predict(input_test)
    print pred_test.shape
    print labels_test.shape
    np.save(outputfolder+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for val set'
    pred_val = model.predict(input_val)
    np.save(outputfolder+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'.npy'  , tmp)

    # Do predictions with best model instead of last
    print 'predicting for training set, best model'
    pred_train = model_best.predict(input_train)
    np.save(outputfolder+'/prediction_train_best.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, best model, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'_best.npy'  , tmp)
    print 'predicting for test set, best model'
    pred_test = model_best.predict(input_test)
    np.save(outputfolder+'/prediction_test_best.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, best model, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'_best.npy'  , tmp)
    print 'predicting for val set, best model'
    pred_val = model_best.predict(input_val)
    np.save(outputfolder+'/prediction_val_best.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, best model, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'_best.npy'  , tmp)

#    print 'predicting for signals'
#    for i in range(len(signal_identifiers)):
#        pred_signal= model.predict(signals[i])
#        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'.npy'  , pred_signal)
#        pred_signal = model_best.predict(signals[i])
#        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'_best.npy'  , pred_signal)

    print "Keras pred_train[0] =",pred_train[0]
    print "Keras pred_val[0] =",pred_val[0]
    print "Keras pred_test[0] =",pred_test[0]
    print " --- END of PredictExternal for DNN ---"


def PredictExternalOnPredictions(parameters, inputfolder, inputfolder_predictions, outputfolder, filepostfix):
    print 'Making predictions now'
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)
    # Get inputs
    model = keras.models.load_model(outputfolder+'/model.h5')
    model_best = keras.models.load_model(outputfolder+'/model_best.h5')
    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix='')

    input_train, input_test, input_val, signals = load_predictions(outputfolder=inputfolder_predictions, filepostfix=filepostfix)
    # pred_train, pred_test, pred_val, pred_signals = load_predictions(outputfolder=inputfolder_predictions, filepostfix=filepostfix)
    # input_train = np.concatenate((input_train, pred_train), axis=1)
    # input_test = np.concatenate((input_test, pred_test), axis=1)
    # input_val = np.concatenate((input_val, pred_val), axis=1)
    # for i in signals.keys():
    #     signals[i] = np.concatenate((signals[i], pred_signals[i]), axis=1)

#    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']

    # Do the predictions
    print 'Now that the model is trained, we\'re going to predict the labels of all 3 sets. '
    print 'predicting for training set'

    pred_train = model.predict(input_train)
    np.save(outputfolder+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for test set'
    pred_test = model.predict(input_test)
    np.save(outputfolder+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for val set'
    pred_val = model.predict(input_val)
    np.save(outputfolder+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'.npy'  , tmp)

    # Do predictions with best model instead of last
    print 'predicting for training set, best model'
    pred_train = model_best.predict(input_train)
    np.save(outputfolder+'/prediction_train_best.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, best model, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'_best.npy'  , tmp)
    print 'predicting for test set, best model'
    pred_test = model_best.predict(input_test)
    np.save(outputfolder+'/prediction_test_best.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, best model, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'_best.npy'  , tmp)
    print 'predicting for val set, best model'
    pred_val = model_best.predict(input_val)
    np.save(outputfolder+'/prediction_val_best.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, best model, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'_best.npy'  , tmp)

#    print 'predicting for signals'
#    for i in range(len(signal_identifiers)):
#        pred_signal= model.predict(signals[i])
#        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'.npy'  , pred_signal)
#        pred_signal = model_best.predict(signals[i])
#        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'_best.npy'  , pred_signal)



def PredictExternalBayesianNetwork(parameters, inputfolder, outputfolder, filepostfix, nsamples):
    print 'Making predictions now'
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)

    # Get parameters
    layers=parameters['layers']
    batch_size=parameters['batchsize']

    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

 # Begin defining calculation graph:
    # wrapping graph with graph.as_default() is better for the case of starting
    # main() several times in same python code -> each time new graph and new Session
    graph = tf.Graph()
    with graph.as_default():
        # just for dense deterministic dropout layer! Not used for BNNs
        is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

        handle = tf.placeholder(tf.string, shape=[], name="handle")
        # # # iterator for train and validation dataset
        # Use placeholders to be more flexiable to handle big size of the input
        # See https://www.tensorflow.org/guide/datasets
        input_train_placeholder = tf.placeholder(input_train.dtype,input_train.shape)
        labels_train_placeholder = tf.placeholder(labels_train.dtype,labels_train.shape)
        sample_weights_train_placeholder = tf.placeholder(sample_weights_train.dtype,sample_weights_train.shape)

        print("- input_train.shape = ",input_train.shape)
        print("- input_train.dtype = ",input_train.dtype)
        print("labels_train.shape = ",labels_train.shape)
        dataset_train = tf.data.Dataset.from_tensor_slices((input_train_placeholder,labels_train_placeholder,sample_weights_train_placeholder))
#        dataset_train = tf.data.Dataset.from_tensor_slices((input_train_placeholder,labels_train_placeholder))
        batched_dataset_train = dataset_train.batch(batch_size)


        input_val_placeholder = tf.placeholder(input_val.dtype,input_val.shape)
        labels_val_placeholder = tf.placeholder(labels_val.dtype,labels_val.shape)
        sample_weights_val_placeholder = tf.placeholder(sample_weights_val.dtype,sample_weights_val.shape)

        print("-- input_val.shape = ",input_val.shape)
        print("-- input_val.dtype = ",input_val.dtype)

        dataset_val = tf.data.Dataset.from_tensor_slices((input_val_placeholder,labels_val_placeholder,sample_weights_val_placeholder))
#        dataset_val = tf.data.Dataset.from_tensor_slices((input_val_placeholder,labels_val_placeholder))
        batched_dataset_val = dataset_val.batch(batch_size)

        input_test_placeholder = tf.placeholder(input_test.dtype,input_test.shape)
        labels_test_placeholder = tf.placeholder(labels_test.dtype,labels_test.shape)
        sample_weights_test_placeholder = tf.placeholder(sample_weights_test.dtype,sample_weights_test.shape)

        print("-- input_test.shape = ",input_test.shape)
        print("-- input_test.dtype = ",input_test.dtype)

        dataset_test = tf.data.Dataset.from_tensor_slices((input_test_placeholder,labels_test_placeholder,sample_weights_test_placeholder))
#        dataset_test = tf.data.Dataset.from_tensor_slices((input_test_placeholder,labels_test_placeholder))
        batched_dataset_test = dataset_test.batch(batch_size)

        iterator_train = batched_dataset_train.make_initializable_iterator()
        iterator_test = batched_dataset_test.make_initializable_iterator()
        iterator_val = batched_dataset_val.make_initializable_iterator()
        #  iterator_signal = batched_dataset_signal.make_initializable_iterator()

        # feedable iterator, for switching between train/test/val
        iterator = tf.data.Iterator.from_string_handle(
            handle, batched_dataset_train.output_types, batched_dataset_train.output_shapes)
        print("--- iterator ---",iterator)
        print("--- batched_dataset_train.output_shapes: ",batched_dataset_train.output_shapes)
        print("--- dataset_val.shapes: ",dataset_val.output_shapes)
        next_element, next_labels, next_sample_weights = iterator.get_next()
#        next_element, next_labels = iterator.get_next()

#        sigma=1.0 #FixME, read https://arxiv.org/pdf/1904.10004.pdf to choose right value
        sigma=parameters['sigma']
        prior = lambda x, y, z, w, k: _prior_normal_fn(sigma, x, y, z, w, k)
#        method = lambda d: d.mean() #deterministic?
        method = lambda d: d.sample()
        layer_hd = []
        batchnorm_hd = []
        dropout_hd = []
        inputs = tf.keras.layers.Input(shape=(input_train.shape[1],),name="Input")
#        layer_hd.append(tf.layers.Dense(layers[0], activation=tf.nn.relu)(inputs))
        layer_hd.append(tfp.layers.DenseFlipout(layers[0], activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method, name="DenseFlipout_1")(inputs))
        k=1
        for i in layers[1:len(layers)+1]:
            label=str(k+1)
#            layer_hd.append(tf.layers.Dense(i, activation=tf.nn.relu)(layer_hd[k-1]))
            layer_hd.append(tfp.layers.DenseFlipout(i, activation=tf.nn.relu, kernel_prior_fn=prior, kernel_posterior_tensor_fn=method,name="DenseFlipout_"+label)(layer_hd[k-1]))
            k = k+1
        print("total number of hidden layers:",k)
        label=str(k)
        last_layer = tfp.layers.DenseFlipout(labels_train.shape[1], kernel_prior_fn=prior, kernel_posterior_tensor_fn=method,name="DenseFlipout_last")(layer_hd[k-1]) #not normilised output
#        last_layer = tf.layers.Dense(labels_train.shape[1], activation='softmax')(layer_hd[k-1])
        print 'Number of output classes: %i' % (labels_train.shape[1])
        model = tf.keras.models.Model(inputs=inputs,outputs=last_layer)
        print model.summary() 

       
#        signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']

        # num_thresholds = 5000 # default is 200
        # auc, auc_op = tf.metrics.auc(labels, mean_op, num_thresholds=num_thresholds)
        # roc_curve, roc_op = tf.contrib.metrics.streaming_curve_points(labels, mean_op, num_thresholds=num_thresholds)
        saver = tf.train.Saver()

    #Allow growing memory for GPU, see https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        file = h5py.File(outputfolder+'/model_weights.h5'.format(tag), 'r')
        print("File with weights:",outputfolder+'/model_weights.h5'.format(tag))
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)][:])
        model.set_weights(weight)

#        print("Loaded model, layer 1:", model.layers[1].get_weights()[0][0])
        print("Loaded model, layer 3:", model.layers[3].get_weights()[0][0])
#        print("Loaded model, layer 0:", model.layers[0].get_weights()[0][0])


        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))

#        logits = graph.get_tensor_by_name("DenseFlipout_last:0")
        logits = model(next_element)
        label_distribution = tf.nn.softmax(logits, name="label_distribution")
#        print "Model restored with status: ",status
        print "Model restored with weights from input file"
        #status.assert_consumed()

#        label_distribution = graph.get_tensor_by_name("label_distribution:0")
        # Do the predictions
        print 'Now that the BNN model is trained, we\'re going to predict the labels of all 3 sets. '
#        print 'predicting for test set'
        # sess.run(iterator_test.initializer, feed_dict={input_test_placeholder: input_test, labels_test_placeholder: labels_test,
        #                                                input_train_placeholder: input_train, labels_train_placeholder: labels_train,
        #                                                input_val_placeholder: input_val, labels_val_placeholder: labels_val})
        n_iterations_until_new_data_batch = nsamples-1
        
        #print("input_train.shape",input_train.shape)
        pred_train = np.ones((nsamples, batch_size,labels_train.shape[1]))
        pred_val = np.ones((nsamples, batch_size,labels_train.shape[1]))
        pred_test = np.ones((nsamples, batch_size,labels_train.shape[1]))
        pred_train = 2*pred_train
        pred_val = 2*pred_val
        pred_test = 2*pred_test

        # inp_train = np.ones((nsamples, batch_size,input_train.shape[1]))
        # sample_weight_train = np.ones((nsamples, batch_size))
        # logits_train = np.ones((nsamples, batch_size,labels_train.shape[1]))

        #sess.run(tf.global_variables_initializer()) # Never use it with loaded model! trained parameters will be replaced with the initial value 
        #sess.run(tf.local_variables_initializer()) # Never use it with loaded model! trained parameters will be replaced with the initial value 
        train_handle = sess.run(iterator_train.string_handle())
        test_handle = sess.run(iterator_test.string_handle())
        val_handle = sess.run(iterator_val.string_handle())
        
        # loop size: ~ n_MonteCarlo * number_of_batches
        prob_std, prob_mean, image_list, label_dist_list, label_list, logits_dist_list = [], [], [], [], [], []
        mean_logits_list, stddev_logits_list = [], []

        #print ("labels_train.shape[1] = ",labels_train.shape[1])
        for i in range(nsamples):
            #print("AAA Loaded model, layer 3:", model.layers[3].get_weights()[0][0])
            pred_train_1d = np.ones((batch_size,labels_train.shape[1]))
            pred_val_1d = np.ones((batch_size,labels_train.shape[1]))
            pred_test_1d = np.ones((batch_size,labels_train.shape[1]))
            pred_train_1d = 2*pred_train_1d
            pred_val_1d = 2*pred_val_1d
            pred_test_1d = 2*pred_test_1d

            # inp_train_1d = np.ones((batch_size,input_train.shape[1]))
            # sample_weight_train_1d = np.ones((batch_size))
            # logits_train_1d = np.ones((batch_size,labels_train.shape[1]))

#            print("Init Values: pred_train_1d[0,:]", pred_train_1d[0,:])
#            print("Init Values: pred_train_1d[0,0]", pred_train_1d[0,0])
#            print("pred_train_1d.shape: ", pred_train_1d.shape)

            sess.run(iterator_train.initializer, 
                     feed_dict={input_train_placeholder: input_train, 
                                labels_train_placeholder: labels_train,
                                sample_weights_train_placeholder: sample_weights_train,

                            })
            # loop over all batches                                                                                                                           
            try:   
                while True:                                                                                                                                           
                    print("!!!### Now do it for TRAIN, i#",i)
#                    predic_train, next_element_value, next_sample_weights_value,logits_value = sess.run([label_distribution,next_element,next_sample_weights,logits],feed_dict={handle: train_handle,is_training: False})

                    predic_train, __ = sess.run([label_distribution, next_element],feed_dict={handle: train_handle,is_training: False})
                    #print 'next_element_value[0] = ',next_element_value[0]
                    #print 'predic_train[0] = ',predic_train[0]
                    #print("predic_train.shape: ",predic_train.shape)
                    #print("np.asarray(predic_train)[0] = ",np.asarray(predic_train)[0])
#                    predic_train = sess.run([true_predicted_label],feed_dict={handle: train_handle,is_training: False})
                    #            predic_train, next_element_value, logits_value = sess.run([true_predicted_label, next_element, logits],feed_dict={handle: train_handle, is_training: False})
                    #            print("predic_train.shape: ",predic_train.shape)
                    #            print("logits_value: ",logits_value[0,:])
                    #            print("predic_train: ",predic_train[0,:])
                    #            print("correct_labels:",correct_labels[0,:])
                    #            print("next_element_value:",next_element_value[0,:])
                    #            pred_train[:,:,i] = next_labels
                    #if pred_train[i,0,0]!=2:
                    if pred_train_1d[0,0]!=2:
                        tmp_pred_train = np.append(pred_train_1d, np.asarray(predic_train),axis=0)
                        pred_train_1d.resize(tmp_pred_train.shape[0],tmp_pred_train.shape[1])
                        pred_train_1d = tmp_pred_train.copy()
                        # tmp_inp_train = np.append(inp_train_1d, np.asarray(next_element_value),axis=0)
                        # inp_train_1d.resize(tmp_inp_train.shape[0],tmp_inp_train.shape[1])
                        # inp_train_1d = tmp_inp_train.copy()

                        # tmp_sample_weight_train = np.append(sample_weight_train_1d, np.asarray(next_sample_weights_value),axis=0)
                        # sample_weight_train_1d.resize(tmp_sample_weight_train.shape[0])
                        # sample_weight_train_1d = tmp_sample_weight_train.copy()

                        # tmp_logits_train = np.append(logits_train_1d, np.asarray(logits_value),axis=0)
                        # logits_train_1d.resize(tmp_logits_train.shape[0],tmp_logits_train.shape[1])
                        # logits_train_1d = tmp_logits_train.copy()

                    else:
                        pred_train_1d = np.asarray(predic_train).copy()
                        # inp_train_1d = np.asarray(next_element_value).copy()
                        # sample_weight_train_1d = np.asarray(next_sample_weights_value).copy()
                        # logits_train_1d = np.asarray(logits_value).copy()

            except tf.errors.OutOfRangeError:
                pass
                
            pred_train.resize(nsamples,pred_train_1d.shape[0],pred_train_1d.shape[1])
            pred_train[i] = pred_train_1d.copy()

            # inp_train.resize(nsamples,inp_train_1d.shape[0],inp_train_1d.shape[1])
            # inp_train[i] = inp_train_1d.copy()

            # sample_weight_train.resize(nsamples,sample_weight_train_1d.shape[0])
            # sample_weight_train[i] = sample_weight_train_1d.copy()
            # #print("Store Values: i, pred_train[i,0,:]",i, pred_train[i,0,:])

            # logits_train.resize(nsamples,logits_train_1d.shape[0],logits_train_1d.shape[1])
            # logits_train[i] = logits_train_1d.copy()

            sess.run(iterator_val.initializer, 
                     feed_dict={input_val_placeholder: input_val, 
                                labels_val_placeholder: labels_val,
                                sample_weights_val_placeholder: sample_weights_val,
                            })
            # loop over all batches  
            try:                                                                                                                            
                while True:                                                                                                                                           
                    print("!!!!##### Now do it for VAL, i#",i)
#                    predic_val, next_element_value = sess.run([label_distribution,next_element],feed_dict={handle: val_handle, is_training: False})
                    predic_val, __ = sess.run([label_distribution,next_element],feed_dict={handle: val_handle, is_training: False})
                    if pred_val_1d[0,0]!=2:
                        tmp_pred_val = np.append(pred_val_1d, np.asarray(predic_val),axis=0)
                        pred_val_1d.resize(tmp_pred_val.shape[0],tmp_pred_val.shape[1])
                        pred_val_1d = tmp_pred_val.copy()
                    else:
                        pred_val_1d = np.asarray(predic_val).copy()
            except tf.errors.OutOfRangeError:
                pass

            pred_val.resize(nsamples,pred_val_1d.shape[0],pred_val_1d.shape[1])
            pred_val[i] = pred_val_1d.copy()


            sess.run(iterator_test.initializer, 
                     feed_dict={input_test_placeholder: input_test, 
                                labels_test_placeholder: labels_test,
                                sample_weights_test_placeholder: sample_weights_test,

                            })
            # loop over all batches                                                                                                                           
            try:   
                while True:           
                    print("!!!#### Now do it for TEST, i#",i)                                                                                                                                
#                    predic_test, next_element_value = sess.run([label_distribution,next_element],feed_dict={handle: test_handle,is_training: False})
                    predic_test, __ = sess.run([label_distribution,next_element],feed_dict={handle: test_handle,is_training: False})
                    if pred_test_1d[0,0]!=2:
                        tmp_pred_test = np.append(pred_test_1d, np.asarray(predic_test),axis=0)
                        pred_test_1d.resize(tmp_pred_test.shape[0],tmp_pred_test.shape[1])
                        pred_test_1d = tmp_pred_test.copy()
                    else:
                        pred_test_1d = np.asarray(predic_test).copy()
            except tf.errors.OutOfRangeError:
                pass

            pred_test.resize(nsamples,pred_test_1d.shape[0],pred_test_1d.shape[1])
            pred_test[i] = pred_test_1d.copy()

        print("Actual Values: pred_train[0,0,:]", pred_train[0,0,:])
        print("Actual Values: pred_train[1,0,:]", pred_train[1,0,:])

        # https://ipython-books.github.io/48-processing-large-numpy-arrays-with-memory-mapping/
        # f = np.memmap(outputfolder+'/prediction_train_memmapped.dat', dtype=np.float32,mode='w+', shape=(pred_train.shape[0], pred_train.shape[1],pred_train.shape[2]))
        # for x in xrange(pred_train.shape[0]):
        #     for y in xrange(pred_train.shape[1]):
        #         for z in xrange(pred_train.shape[2]):
        #             f[x,y,z] = pred_train[x,y,z]
        # del f

        np.save(outputfolder+'/prediction_train.npy'  , pred_train)
        for cl in range(len(parameters['classes'])):
            print 'predicting for training set, class ' + str(cl)
            print("labels_train[:,cl]:",labels_train[:,cl])
            print("labels_train[:,cl].shape:",labels_train[:,cl].shape)
            tmp = pred_train[:,labels_train[:,cl] == 1]
            np.save(outputfolder+'/prediction_train_class'+str(cl)+'.npy'  , tmp)

        np.save(outputfolder+'/prediction_val.npy'  , pred_val)
        for cl in range(len(parameters['classes'])):
            print 'predicting for val set, class ' + str(cl)
            tmp = pred_val[:,labels_val[:,cl] == 1]
            np.save(outputfolder+'/prediction_val_class'+str(cl)+'.npy'  , tmp)

        np.save(outputfolder+'/prediction_test.npy'  , pred_test)
        for cl in range(len(parameters['classes'])):
            print 'predicting for test set, class ' + str(cl)
            tmp = pred_test[:,labels_test[:,cl] == 1]
            np.save(outputfolder+'/prediction_test_class'+str(cl)+'.npy'  , tmp)


#        print 'predicting for signals'
#
#        for i in range(len(signal_identifiers)):
#            pred_signal = np.ones((nsamples, signals[i].shape[0],labels_train.shape[1]))
#            for isamp in range(nsamples):
#
#                #pred_signal_1d = np.asarray((model.predict(signals[i]))
#                pred_signal_1d = tf.nn.softmax(model.predict(signals[i])).eval(session=sess)
#                print("pred_signal_1d.shape = ",pred_signal_1d.shape)
#                pred_signal[isamp] = pred_signal_1d
#            np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'.npy'  , pred_signal)

        # print 'predicting for signals'
        # for i in range(len(signal_identifiers)):
        #     pred_signal= np.asarray(model.predict(signals[i]))
        #     print("pred_signal.shape = ",pred_signal.shape)
        #     np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'.npy'  , pred_signal)




def PredictExternalDeepNetwork(parameters, inputfolder, outputfolder, filepostfix):
    print 'Making predictions now'
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)

    # Get parameters
    layers=parameters['layers']

    input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)

    regrate=parameters['regrate']
#    sigma=1 #FixME, read https://arxiv.org/pdf/1904.10004.pdf to choose right value
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
        #layer_hd.append(tf.layers.Dense(i, activation=tf.nn.relu)(batchnorm_hd[k-1]))
        layer_hd.append(tf.layers.Dense(i, activation=tf.nn.relu)(dropout_hd[k-1]))
        dropout_hd.append(tf.layers.dropout(layer_hd[k], rate=regrate))
        #batchnorm_hd.append(tf.layers.batch_normalization(dropout_hd[k]))  
        k = k+1
#    print("total number of hidden layers:",k)
    #last_layer = tf.layers.Dense(labels_train.shape[1], activation=tf.nn.relu)(batchnorm_hd[k-1])
    last_layer = tf.layers.Dense(labels_train.shape[1], activation='softmax')(dropout_hd[k-1])
    print 'Number of output classes: %i' % (labels_train.shape[1])
    model = tf.keras.models.Model(inputs=inputs,outputs=last_layer)
    print model.summary()
    file = h5py.File('output/DNN_'+tag+'/model.h5', 'r')
    weight = []
    for i in range(len(file.keys())):
        weight.append(file['weight' + str(i)][:])
    model.set_weights(weight)


#    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']

    # Do the predictions
    print 'Now that the model is trained, we\'re going to predict the labels of all 3 sets. '
    print 'predicting for training set'

    pred_train = model.predict(input_train)
    print("input_train[0]",input_train[0])
    print ("DNN pred_train[0] =",pred_train[0])
    print("input_train[1]",input_train[1])
    print ("DNN pred_train[1] =",pred_train[1])
    print("input_train[2]",input_train[2])
    print ("DNN pred_train[2] =",pred_train[2])

    np.save(outputfolder+'/prediction_train.npy'  , pred_train)
    for cl in range(len(parameters['classes'])):
        print 'predicting for training set, class ' + str(cl)
        tmp = pred_train[labels_train[:,cl] == 1]
        np.save(outputfolder+'/prediction_train_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for test set'
    print input_test.shape
    print labels_test.shape
    pred_test = model.predict(input_test)
    print pred_test.shape
    print labels_test.shape
    np.save(outputfolder+'/prediction_test.npy'  , pred_test)
    for cl in range(len(parameters['classes'])):
        print 'predicting for test set, class ' + str(cl)
        tmp = pred_test[labels_test[:,cl] == 1]
        np.save(outputfolder+'/prediction_test_class'+str(cl)+'.npy'  , tmp)
    print 'predicting for val set'
    pred_val = model.predict(input_val)
    np.save(outputfolder+'/prediction_val.npy'  , pred_val)
    for cl in range(len(parameters['classes'])):
        print 'predicting for val set, class ' + str(cl)
        tmp = pred_val[labels_val[:,cl] == 1]
        np.save(outputfolder+'/prediction_val_class'+str(cl)+'.npy'  , tmp)

#    print 'predicting for signals'
#    for i in range(len(signal_identifiers)):
#        pred_signal= model.predict(signals[i])
#        np.save(outputfolder+'/prediction_'+signal_identifiers[i]+'.npy'  , pred_signal)

