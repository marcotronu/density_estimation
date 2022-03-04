import scipy.io
import os 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import argparse

import scipy.io as sio
import math as mt
from scipy.fftpack import ifft, fft, fftshift
from scipy.signal.windows import hann, gaussian
import pickle
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


import tensorflow as tf 
import tarfile 
import xlrd
from functools import reduce
import time
import cv2
import datetime as dt
import seaborn as sns
from glob import glob
import re
import pydot
import scipy.misc
from tensorflow.keras.preprocessing import image
from keras import backend as K
from google.colab.patches import cv2_imshow
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython.display import SVG
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import mean_absolute_error
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras import Sequential
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, accuracy_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from tensorflow.python.client import device_lib
from sklearn import preprocessing

################################################################################
def decision_fusion(model=None,predictions=None,data_test=None,n_antenna=None):
    '''
    Apply the decision fusion approach to the model predictions.
    This function was thought for a model trained on 4 links/antennas with a 
    classification approach.
    For each prediction (which is a group of 4 vectors, 1 for each antenna, and
    each one being n_persone+1 dimensional), you check whether at least 3 out of
    4 predictions are the same (i.e. 3 antennas agree on the label), otherwise
    you sum the vectors (which resulted from the softmax layer) and take the
    argmax.
    ----------------------------------------------------------------------------
    Parameters:
        - model: if None, must provide predictions
        - predictions: dictionary containing the array of predictions for each 
        key, being each key a different antenna.
        - data_test: dictionary containing the test from which compute the predictions
        - n_antennas: number of separate links. Must be equal to the number of keys
        of predictions

    Note: if no predictions are provided, but the model is, make sure that the model
    takes one link at a time.
    ----------------------------------------------------------------------------
    '''


    if not model and not predictions:
        raise ValueError('Please provide either a model with the test data or its predictions!')
    elif model and data_test and not predictions:
        predictions = {antenna:[] for antenna in range(4)}
        
        for antenna in range(n_antenna):
            preds = model(data_test[antenna],training=False)
            predictions[antenna] = np.array(preds)

    if n_antenna != len(list(predictions.keys())):
        raise ValueError(f'Please make sure that predictions has a number of keys equal to {n_antenna}!')
    

    new_predictions = []

    for j in range(len(predictions[0])):
        preds = [predictions[antenna][j] for antenna in range(n_antenna)]
        # print(preds)
        preds_s = [np.argmax(pred) for pred in preds] 
        if len(set(preds_s)) <= 2: #2 because if 3 out of 4 antennas agree then len(se(preds)) = 2
            pred = max(set(preds_s), key = preds_s.count)
        else:
            pred = np.array([0.0] * 11)
            # print(len(pred))
    
            for ls in preds:
                # print(len(ls))
                pred += ls
            pred = np.argmax(pred)
        new_predictions.append(pred)
    return new_predictions
##########################################################################

##########################################################################
def decision_fusion_regression(predictions,n_antenna):
    '''
    Apply the decision fusion approach to the model predictions.
    This function was thought for a model trained on n_antenna links/antennas with a 
    regression approach.
    For each prediction (which is a group of n_antenna numbers, 1 for each antenna), you check whether at least 3 out of
    4 predictions are the same (i.e. 3 antennas agree on the label), otherwise
    you compute the average and you round up.
    ----------------------------------------------------------------------------
    Parameters:
        - predictions: dictionary containing the array of predictions for each 
        key, being each key a different antenna.
        - n_antenna: number of separate links. Must be equal to the number of keys
        of predictions
    ----------------------------------------------------------------------------

    '''
    if n_antenna != len(list(predictions.keys())):
        raise ValueError(f'Please make sure that predictions has a number of keys equal to {n_antenna}!')
    
    new_predictions = []
    for j in range(len(predictions[0])):
        preds = [int(round(predictions[antenna][j][0])) for antenna in range(n_antennas)]

        if n_antennas - len(set(preds)) >= n_antennas - 2: #-2 because if 3 out of 4 antennas agree then len(se(preds)) = 2
            preds = max(set(preds), key = preds.count)
        else:
            preds = int(round(np.mean(preds)))

        new_predictions.append(preds)
        # print(np.mean(preds))
    return new_predictions
######################################################


######################################################
def decision_fusion_permutation(model=None,X_test=None,allowed_perms=None,len_inputs=None,threshold=None):
    '''
    Apply the idea of decision fusion to the permutations models
    ------------------------------------------------------------
    Parameters:
        - model
        - X_test
        - allowed_perms: list of allowed permutations, can be None
        - len_inputs: integer, it's the number of inputs you pass to the model
        - threshold: float between 0 and 1. It defines the lim inf for the tolerated disagreement
          For example, if threshold = 0.6, and we have 12 permutations, it means that 
          if f_max/12 >= threshold, where f_max is how many times out of the same prediction occurs,
          then we take the predictions as the one generating f_max
    ------------------------------------------------------------
    '''
    import itertools


    n_links = len(list(X_test.keys()))

    if not allowed_perms:
        allowed_perms = []
        for perm in list(itertools.combinations(range(n_links),len_inputs)):
            allowed_perms.extend(itertools.permutations(perm))
    
    if threshold == None:
        raise ValueError('Please insert a valid threshold')
    # else:
    #     threshold = np.int(np.round(len(allowed_perms) * (1 - threshold)))

    predictions = {k:[] for k in range(len(allowed_perms))}

    '''
    First, compute the predictions for all the permutations of the links:
    '''

    for n,perm in enumerate(allowed_perms):
        # print(perm)
        predictions[n] = model([X_test[perm[k]] for k in range(len_inputs)],training = False)
        # preds_s = [np.argmax(pred) for pred in preds] 

        # preds = [np.argmax(pred) for pred in preds]
        # predictions.append(preds)
    # print(len(predictions[0]))
    '''
    Then, apply the same idea of the standard decision fusion:
    '''
    new_predictions = []
    for j in range(len(predictions[0])):
        preds = [predictions[antenna][j] for antenna in range(len(predictions))] #build array of length n-permutations, each element is a 11-long vector
        # print(preds)
        preds_s = [np.argmax(pred) for pred in preds] 

        unique, counts = np.unique(preds_s, return_counts=True)
        freq = dict(zip(unique, counts))
        # print(freq)
        max_freq = np.max([freq[m] for m in freq.keys()])
        if max_freq/len(preds) >= threshold:
            # print(max_freq/len(preds))
            # print(preds_s)
            pred = max(preds_s, key = preds_s.count)
        else:
            pred = np.array([0.0] * 11)
            # print(len(pred))
            for ls in preds:
                # print(len(ls))
                pred += ls
            pred = np.argmax(pred)
        new_predictions.append(pred)
    return new_predictions
#########################################################


################################################################################
def model_metrics(predictions,test_labels):
    '''
    Compute the accuracy, mean absolute error, P(E<=1), P(E<=2) and the 
    confusion matrix for the given predictions and labels.
    ----------------------------------------------------------------------------
    Parameters:
        - predictions: array of the predictions;
        - labels: array containing the corresponding labels.
    ----------------------------------------------------------------------------      
    '''
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    acc = np.mean(predictions == test_labels)
    mae = np.mean(np.abs(predictions-test_labels))
    print(f'The accuracy of the model is {acc} \nThe Mean-Absolute-Error is {mae}')

    print(f'Probability of prediction being at most 2 persons away: {np.mean(np.abs(predictions - test_labels)<=2)}') 
    print(f'Probability of prediction being at most 1 person away: {np.mean(np.abs(predictions - test_labels)<=1)}')

    plt.style.use('ggplot')
    cm = confusion_matrix(test_labels,predictions)
    plt.figure(figsize = (10,7))
    sns.heatmap(cm, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()
################################################################################
