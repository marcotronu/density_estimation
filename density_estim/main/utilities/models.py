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
from tensorflow.keras.regularizers import l2
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l1_l2

#################################################
def paper_model(delta_t=None,amplitude=False,max_people=10):
    '''
    Create the model of the main Paper
    '''
    if amplitude:
        input = Input(shape = (delta_t,245,1))
    else:
        input = Input(shape = (delta_t,100,1))
    max1 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='MaxPooling1')(input)
    conv1 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',name='Conv2D_1')(input)
    conv2 = Conv2D(3, (1, 1),name='Conv2D_2_1')(input)
    conv2 = Conv2D(6, (2, 2),name='Conv2D_2_2')(conv2)
    conv2 = Conv2D(9, (4, 4), strides=(2,2),padding='same',name='Conv2D_2_3')(conv2)

    concat = Concatenate()([max1,conv1,conv2])

    out = Conv2D(3, (1, 1), activation = 'relu',name='Conv2D_3')(concat)
    out = Flatten()(out)
    out = Dropout(0.2)(out)
    out = Dense(100)(out)
    out = Dropout(0.2)(out)
    
    out = Dense(max_people+1, activation = 'softmax')(out)

    model = Model(input,out)

    optim = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # mse = tf.keras.losses.MeanSquaredError()
    # rmse = tf.keras.metrics.RootMeanSquaredError()
    # mae = tf.keras.metrics.MeanAbsoluteError()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='False')

    model.compile(loss=loss,optimizer=optim,metrics='sparse_categorical_accuracy')

    return model
########################################################

##############################################################################
def phase_amp_dop_model(max_people=10,delta_t=None):
    '''
    Create new model based on the paper architecture
    Receives as input phase + amplitude + doppler in parallel
    '''
    input_amp = Input(shape = (delta_t,245,1))

    max1_amp = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1')(input_amp)
    conv1_amp = Conv2D(5, (2, 2), strides=(2,2), padding='valid',name='Amp_Conv2D_1')(input_amp)
    conv2_amp = Conv2D(3, (1, 1),name='Amp_Conv2D_2_1')(input_amp)
    conv2_amp = Conv2D(6, (2, 2),name='Amp_Conv2D_2_2')(conv2_amp)
    conv2_amp = Conv2D(9, (4, 4), strides=(2,2),padding='same',name='Amp_Conv2D_2_3')(conv2_amp)
    concat_amp =  Concatenate()([max1_amp,conv1_amp,conv2_amp])
    out_amp = Conv2D(3, (1, 1), activation = 'swish',name='Amp_Conv2D_3')(concat_amp)
    out_amp = Flatten()(out_amp)
    out_amp = Dropout(0.2)(out_amp)

    input_phase = Input(shape = (delta_t,245,1))

    max1_phase = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Phase_MaxPooling1')(input_phase)
    conv1_phase = Conv2D(5, (2, 2), strides=(2,2), padding='valid',name='Phase_Conv2D_1')(input_phase)
    conv2_phase = Conv2D(3, (1, 1),name='Phase_Conv2D_2_1')(input_phase)
    conv2_phase = Conv2D(6, (2, 2),name='Phase_Conv2D_2_2')(conv2_phase)
    conv2_phase = Conv2D(9, (4, 4), strides=(2,2),padding='same',name='Phase_Conv2D_2_3')(conv2_phase)
    concat_phase = Concatenate()([max1_phase,conv1_phase,conv2_phase])
    out_phase = Conv2D(3, (1, 1), activation = 'swish',name='Phase_Conv2D_3')(concat_phase)
    out_phase = Flatten()(out_phase)
    out_phase = Dropout(0.2)(out_phase)

    input_dop = Input(shape = (delta_t,100,1))

    max1_dop = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='dop_MaxPooling1')(input_dop)
    conv1_dop = Conv2D(5, (2, 2), strides=(2,2), padding='valid',name='dop_Conv2D_1')(input_dop)
    conv2_dop = Conv2D(3, (1, 1),name='dop_Conv2D_2_1')(input_dop)
    conv2_dop = Conv2D(6, (2, 2),name='dop_Conv2D_2_2')(conv2_dop)
    conv2_dop = Conv2D(9, (4, 4), strides=(2,2),padding='same',name='dop_Conv2D_2_3')(conv2_dop)
    concat_dop = Concatenate()([max1_dop,conv1_dop,conv2_dop])
    out_dop = Conv2D(3, (1, 1), activation = 'swish',name='dop_Conv2D_3')(concat_dop)
    out_dop = Flatten()(out_dop)
    out_dop = Dropout(0.2)(out_dop)


    out = Concatenate()([out_amp, out_phase])
    out = Dense(50, activation = 'swish')(out)
    out = Concatenate()([out,out_dop])
    out = Dropout(0.2)(out)
    out = Dense(50, activation = 'swish')(out)
    out = Dropout(0.2)(out)
    out = Dense(max_people+1, activation = 'softmax')(out)

    model = Model([input_amp,input_phase,input_dop],out)
    
    optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='False')

    model.compile(loss=loss,optimizer=optim,metrics='sparse_categorical_accuracy')

    return model
##############################################################################


##############################################################################
def phase_amp_model(max_people=10,delta_t=None):
    '''
    Create new model based on the paper architecture
    Receives as input amplitude + phase in parallel
    '''
    input_amp = Input(shape = (delta_t,245,1))

    max1_amp = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1')(input_amp)
    conv1_amp = Conv2D(5, (2, 2), strides=(2,2), padding='valid',name='Amp_Conv2D_1')(input_amp)
    conv2_amp = Conv2D(3, (1, 1),name='Amp_Conv2D_2_1')(input_amp)
    conv2_amp = Conv2D(6, (2, 2),name='Amp_Conv2D_2_2')(conv2_amp)
    conv2_amp = Conv2D(9, (4, 4), strides=(2,2),padding='same',name='Amp_Conv2D_2_3')(conv2_amp)
    concat_amp =  Concatenate()([max1_amp,conv1_amp,conv2_amp])
    out_amp = Conv2D(3, (1, 1), activation = 'swish',name='Amp_Conv2D_3')(concat_amp)
    out_amp = Flatten()(out_amp)
    out_amp = Dropout(0.2)(out_amp)

    input_phase = Input(shape = (delta_t,245,1))

    max1_phase = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Phase_MaxPooling1')(input_phase)
    conv1_phase = Conv2D(5, (2, 2), strides=(2,2), padding='valid',name='Phase_Conv2D_1')(input_phase)
    conv2_phase = Conv2D(3, (1, 1),name='Phase_Conv2D_2_1')(input_phase)
    conv2_phase = Conv2D(6, (2, 2),name='Phase_Conv2D_2_2')(conv2_phase)
    conv2_phase = Conv2D(9, (4, 4), strides=(2,2),padding='same',name='Phase_Conv2D_2_3')(conv2_phase)
    concat_phase = Concatenate()([max1_phase,conv1_phase,conv2_phase])
    out_phase = Conv2D(3, (1, 1), activation = 'swish',name='Phase_Conv2D_3')(concat_phase)
    out_phase = Flatten()(out_phase)
    out_phase = Dropout(0.2)(out_phase)

    out = Concatenate()([out_amp, out_phase])
    out = Dense(100, activation = 'swish')(out)
    out = Dropout(0.2)(out)
    out = Dense(1, activation = None)(out)

    model = Model([input_amp,input_phase],out)
    
    optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    
    loss = tf.keras.losses.MeanAbsoluteError()

    model.compile(loss=loss,optimizer=optim,metrics='mean_absolute_error')

    return model
##############################################################################

##############################################################################
def split_antennas_model(max_people=10,classification = True,delta_t=None,doppler=False,wd1=None,wd2=None,act=None,dense1 = None, dense2 = None):
    '''
    Create new model based on the paper architecture
    Receives as input the 4 streams of antenna separately (but in parallel)
    Inputs:
        - max_people: if N, and if the classification method is chosen, the final
            layer of the network will be a N+1 softmax layer
        - delta_t: is the integer defining the first dimension (the time window)
        - doppler: if true, the second dimension is 100, else, it's equal to 245
        - wd1: the weight decay for the l1 weight regularization (0<=wd1<=1)
        - wd1: the weight decay for the l2 weight regularization (0<=wd1<=1)
        - act: the activity regularization parameter
    '''
    if doppler == False:
        dim_in = 245
    else:
        dim_in = 100
    input_amp_antenna_0 = Input(shape = (delta_t,dim_in,1))

    max1_amp_antenna_0 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1_antenna_0')(input_amp_antenna_0)
    conv1_amp_antenna_0 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_1_antenna_0')(input_amp_antenna_0)
    conv2_amp_antenna_0 = Conv2D(3, (1, 1),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_1_antenna_0')(input_amp_antenna_0)
    conv2_amp_antenna_0 = Conv2D(6, (2, 2),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_2_antenna_0')(conv2_amp_antenna_0)
    conv2_amp_antenna_0 = Conv2D(9, (4, 4), strides=(2,2),padding='same',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_3_antenna_0')(conv2_amp_antenna_0)
    concat_amp_antenna_0 =  Concatenate()([max1_amp_antenna_0,conv1_amp_antenna_0,conv2_amp_antenna_0])
    out_amp_antenna_0 = Conv2D(3, (1, 1), activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_3_antenna_0')(concat_amp_antenna_0)
    out_amp_antenna_0 = Flatten()(out_amp_antenna_0)
    out_amp_antenna_0 = Dropout(0.2)(out_amp_antenna_0)
    out_amp_antenna_0 = Dense(dense1, activation = 'swish')(out_amp_antenna_0)

    input_amp_antenna_1 = Input(shape = (delta_t,dim_in,1))

    max1_amp_antenna_1 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1_antenna_1')(input_amp_antenna_1)
    conv1_amp_antenna_1 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_1_antenna_1')(input_amp_antenna_1)
    conv2_amp_antenna_1 = Conv2D(3, (1, 1),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_1_antenna_1')(input_amp_antenna_1)
    conv2_amp_antenna_1 = Conv2D(6, (2, 2),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_2_antenna_1')(conv2_amp_antenna_1)
    conv2_amp_antenna_1 = Conv2D(9, (4, 4), strides=(2,2),padding='same',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_3_antenna_1')(conv2_amp_antenna_1)
    concat_amp_antenna_1 =  Concatenate()([max1_amp_antenna_1,conv1_amp_antenna_1,conv2_amp_antenna_1])
    out_amp_antenna_1 = Conv2D(3, (1, 1), activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_3_antenna_1')(concat_amp_antenna_1)
    out_amp_antenna_1 = Flatten()(out_amp_antenna_1)
    out_amp_antenna_1 = Dropout(0.2)(out_amp_antenna_1)
    out_amp_antenna_1 = Dense(dense1, activation = 'swish')(out_amp_antenna_1)


    input_amp_antenna_2 = Input(shape = (delta_t,dim_in,1))

    max1_amp_antenna_2 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1_antenna_2')(input_amp_antenna_2)
    conv1_amp_antenna_2 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_1_antenna_2')(input_amp_antenna_2)
    conv2_amp_antenna_2 = Conv2D(3, (1, 1),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_1_antenna_2')(input_amp_antenna_2)
    conv2_amp_antenna_2 = Conv2D(6, (2, 2),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_2_antenna_2')(conv2_amp_antenna_2)
    conv2_amp_antenna_2 = Conv2D(9, (4, 4), strides=(2,2),padding='same',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_3_antenna_2')(conv2_amp_antenna_2)
    concat_amp_antenna_2 =  Concatenate()([max1_amp_antenna_2,conv1_amp_antenna_2,conv2_amp_antenna_2])
    out_amp_antenna_2 = Conv2D(3, (1, 1), activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_3_antenna_2')(concat_amp_antenna_2)
    out_amp_antenna_2 = Flatten()(out_amp_antenna_2)
    out_amp_antenna_2 = Dropout(0.2)(out_amp_antenna_2)
    out_amp_antenna_2 = Dense(dense1, activation = 'swish')(out_amp_antenna_2)


    input_amp_antenna_3 = Input(shape = (delta_t,dim_in,1))

    max1_amp_antenna_3 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1_antenna_3')(input_amp_antenna_3)
    conv1_amp_antenna_3 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_1_antenna_3')(input_amp_antenna_3)
    conv2_amp_antenna_3 = Conv2D(3, (1, 1),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_1_antenna_3')(input_amp_antenna_3)
    conv2_amp_antenna_3 = Conv2D(6, (2, 2),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_2_antenna_3')(conv2_amp_antenna_3)
    conv2_amp_antenna_3 = Conv2D(9, (4, 4), strides=(2,2),padding='same',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_3_antenna_3')(conv2_amp_antenna_3)
    concat_amp_antenna_3 =  Concatenate()([max1_amp_antenna_3,conv1_amp_antenna_3,conv2_amp_antenna_3])
    out_amp_antenna_3 = Conv2D(3, (1, 1), activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_3_antenna_3')(concat_amp_antenna_3)
    out_amp_antenna_3 = Flatten()(out_amp_antenna_3)
    out_amp_antenna_3 = Dropout(0.2)(out_amp_antenna_3)
    out_amp_antenna_3 = Dense(dense1, activation = 'swish')(out_amp_antenna_3)


    out = Concatenate()([out_amp_antenna_0,out_amp_antenna_1,out_amp_antenna_2, out_amp_antenna_3])
    # out = Dense(dense1, activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act))(out)
    out = Dropout(0.2)(out)
    out = Dense(dense2, activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act))(out)
    out = Dropout(0.2)(out)
    if classification:
        out = Dense(max_people+1, activation = 'softmax')(out)

        model = Model([input_amp_antenna_0,input_amp_antenna_1,input_amp_antenna_2,input_amp_antenna_3],out)
        
        # for layer in model.layers:
        # if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
        #     layer.add_loss(keras.regularizers.l2(alpha)(layer.kernel))
        # if hasattr(layer, 'bias_regularizer') and layer.use_bias:
        #     layer.add_loss(keras.regularizers.l2(alpha)(layer.bias))

        
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='False')
        optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)

        model.compile(loss=loss,optimizer=optim,metrics='sparse_categorical_accuracy')
    else:
        out = Dense(1)(out)

        model = Model([input_amp_antenna_0,input_amp_antenna_1,input_amp_antenna_2,input_amp_antenna_3],out)
        
        # for layer in model.layers:
        # if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
        #     layer.add_loss(keras.regularizers.l2(alpha)(layer.kernel))
        # if hasattr(layer, 'bias_regularizer') and layer.use_bias:
        #     layer.add_loss(keras.regularizers.l2(alpha)(layer.bias))

        
        loss = tf.keras.losses.MeanAbsoluteError()
        optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)

        model.compile(loss=loss,optimizer=optim,metrics=tf.keras.metrics.MeanAbsoluteError())

    return model
##############################################################################


##############################################################################
def split_antennas_modelv2(max_people=10,classification = True,delta_t=None,doppler=False,wd1=None,wd2=None,act=None,dense1 = None, dense2 = None):
    '''
    Create new model based on the paper architecture
    Receives as input the 4 streams of antenna separately (but in parallel)
    Inputs:
        - max_people: if N, and if the classification method is chosen, the final
            layer of the network will be a N+1 softmax layer
        - delta_t: is the integer defining the first dimension (the time window)
        - doppler: if true, the second dimension is 100, else, it's equal to 245
        - wd1: the weight decay for the l1 weight regularization (0<=wd1<=1)
        - wd1: the weight decay for the l2 weight regularization (0<=wd1<=1)
        - act: the activity regularization parameter
    '''
    if doppler == False:
        dim_in = 245
    else:
        dim_in = 100
    input_amp_antenna_0 = Input(shape = (delta_t,dim_in,1))

    max1_amp_antenna_0 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1_antenna_0')(input_amp_antenna_0)
    conv1_amp_antenna_0 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_1_antenna_0')(input_amp_antenna_0)
    conv2_amp_antenna_0 = Conv2D(3, (1, 1),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_1_antenna_0')(input_amp_antenna_0)
    conv2_amp_antenna_0 = Conv2D(6, (2, 2),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_2_antenna_0')(conv2_amp_antenna_0)
    conv2_amp_antenna_0 = Conv2D(9, (4, 4), strides=(2,2),padding='same',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_3_antenna_0')(conv2_amp_antenna_0)
    concat_amp_antenna_0 =  Concatenate()([max1_amp_antenna_0,conv1_amp_antenna_0,conv2_amp_antenna_0])
    out_amp_antenna_0 = Conv2D(3, (1, 1), activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_3_antenna_0')(concat_amp_antenna_0)
    out_amp_antenna_0 = Flatten()(out_amp_antenna_0)
    out_amp_antenna_0 = Dropout(0.2)(out_amp_antenna_0)
    # out_amp_antenna_0 = Dense(dense1, activation = 'swish')(out_amp_antenna_0)

    input_amp_antenna_1 = Input(shape = (delta_t,dim_in,1))

    max1_amp_antenna_1 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1_antenna_1')(input_amp_antenna_1)
    conv1_amp_antenna_1 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_1_antenna_1')(input_amp_antenna_1)
    conv2_amp_antenna_1 = Conv2D(3, (1, 1),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_1_antenna_1')(input_amp_antenna_1)
    conv2_amp_antenna_1 = Conv2D(6, (2, 2),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_2_antenna_1')(conv2_amp_antenna_1)
    conv2_amp_antenna_1 = Conv2D(9, (4, 4), strides=(2,2),padding='same',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_3_antenna_1')(conv2_amp_antenna_1)
    concat_amp_antenna_1 =  Concatenate()([max1_amp_antenna_1,conv1_amp_antenna_1,conv2_amp_antenna_1])
    out_amp_antenna_1 = Conv2D(3, (1, 1), activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_3_antenna_1')(concat_amp_antenna_1)
    out_amp_antenna_1 = Flatten()(out_amp_antenna_1)
    out_amp_antenna_1 = Dropout(0.2)(out_amp_antenna_1)
    # out_amp_antenna_1 = Dense(dense1, activation = 'swish')(out_amp_antenna_1)


    input_amp_antenna_2 = Input(shape = (delta_t,dim_in,1))

    max1_amp_antenna_2 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1_antenna_2')(input_amp_antenna_2)
    conv1_amp_antenna_2 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_1_antenna_2')(input_amp_antenna_2)
    conv2_amp_antenna_2 = Conv2D(3, (1, 1),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_1_antenna_2')(input_amp_antenna_2)
    conv2_amp_antenna_2 = Conv2D(6, (2, 2),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_2_antenna_2')(conv2_amp_antenna_2)
    conv2_amp_antenna_2 = Conv2D(9, (4, 4), strides=(2,2),padding='same',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_3_antenna_2')(conv2_amp_antenna_2)
    concat_amp_antenna_2 =  Concatenate()([max1_amp_antenna_2,conv1_amp_antenna_2,conv2_amp_antenna_2])
    out_amp_antenna_2 = Conv2D(3, (1, 1), activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_3_antenna_2')(concat_amp_antenna_2)
    out_amp_antenna_2 = Flatten()(out_amp_antenna_2)
    out_amp_antenna_2 = Dropout(0.2)(out_amp_antenna_2)
    # out_amp_antenna_2 = Dense(dense1, activation = 'swish')(out_amp_antenna_2)


    input_amp_antenna_3 = Input(shape = (delta_t,dim_in,1))

    max1_amp_antenna_3 = MaxPool2D((2,2), strides=(2,2),padding = 'valid',name='Amp_MaxPooling1_antenna_3')(input_amp_antenna_3)
    conv1_amp_antenna_3 = Conv2D(5, (2, 2), strides=(2,2), padding='valid',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_1_antenna_3')(input_amp_antenna_3)
    conv2_amp_antenna_3 = Conv2D(3, (1, 1),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_1_antenna_3')(input_amp_antenna_3)
    conv2_amp_antenna_3 = Conv2D(6, (2, 2),kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_2_antenna_3')(conv2_amp_antenna_3)
    conv2_amp_antenna_3 = Conv2D(9, (4, 4), strides=(2,2),padding='same',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_2_3_antenna_3')(conv2_amp_antenna_3)
    concat_amp_antenna_3 =  Concatenate()([max1_amp_antenna_3,conv1_amp_antenna_3,conv2_amp_antenna_3])
    out_amp_antenna_3 = Conv2D(3, (1, 1), activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act),name='Amp_Conv2D_3_antenna_3')(concat_amp_antenna_3)
    out_amp_antenna_3 = Flatten()(out_amp_antenna_3)
    out_amp_antenna_3 = Dropout(0.2)(out_amp_antenna_3)
    # out_amp_antenna_3 = Dense(dense1, activation = 'swish')(out_amp_antenna_3)


    out = Concatenate()([out_amp_antenna_0,out_amp_antenna_1,out_amp_antenna_2, out_amp_antenna_3])
    out = Dense(dense1, activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act))(out)
    out = Dropout(0.2)(out)
    out = Dense(dense2, activation = 'swish',kernel_regularizer=l1_l2(l1=wd1,l2=wd2), bias_regularizer=l1_l2(l1=wd1,l2=wd2),activity_regularizer=l1(act))(out)
    out = Dropout(0.2)(out)
    if classification:
        out = Dense(max_people+1, activation = 'softmax')(out)

        model = Model([input_amp_antenna_0,input_amp_antenna_1,input_amp_antenna_2,input_amp_antenna_3],out)
        
        # for layer in model.layers:
        # if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
        #     layer.add_loss(keras.regularizers.l2(alpha)(layer.kernel))
        # if hasattr(layer, 'bias_regularizer') and layer.use_bias:
        #     layer.add_loss(keras.regularizers.l2(alpha)(layer.bias))

        
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='False')
        optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)

        model.compile(loss=loss,optimizer=optim,metrics='sparse_categorical_accuracy')
    else:
        out = Dense(1)(out)

        model = Model([input_amp_antenna_0,input_amp_antenna_1,input_amp_antenna_2,input_amp_antenna_3],out)
        
        # for layer in model.layers:
        # if isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.Dense):
        #     layer.add_loss(keras.regularizers.l2(alpha)(layer.kernel))
        # if hasattr(layer, 'bias_regularizer') and layer.use_bias:
        #     layer.add_loss(keras.regularizers.l2(alpha)(layer.bias))

        
        loss = tf.keras.losses.MeanAbsoluteError()
        optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)

        model.compile(loss=loss,optimizer=optim,metrics=tf.keras.metrics.MeanAbsoluteError())

    return model
##############################################################################


##############################################################################
def CreateCallbacks(ckp = None,training_log = None):
    """
    Create callbacks for training
    Parameters:
        - ckp: (string) path in which the weights will be stored (.h5 format)
        - training_log: (string) path in which the training log will be stored each epoch
        - batch_size: (int) size of the batch
    """
    if ckp == None:
        raise Exception('Please select a valid checkpoint path!')   
    if training_log == None:
        raise Exception('Please select a valid training log path!')
    
        # early_stopping = EarlyStopping(monitor='val_loss',
        #                             min_delta=0,
        #                             patience=10,
        #                             verbose=0,
        #                             mode='auto')
        # reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', verbose = 1, factor=0.7, patience=10,mode='min', min_delta=0.0001, cooldown=5, min_lr=0.000001)

    my_cps = ModelCheckpoint('/content/drive/MyDrive/UNI/second_year/TESI/NEWDATA/marco/models/'+ckp,monitor='val_loss',mode='min',save_best_only=True)
    csv_logger = tf.keras.callbacks.CSVLogger('/content/drive/MyDrive/UNI/second_year/TESI/NEWDATA/marco/models/'+training_log,append=True) #append True is mandatory otherwise it overwrites everytime
    # lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule,verbose=1)
    return [my_cps,csv_logger]#[reduceLROnPlat, my_cps,csv_logger,early_stopping]
##############################################################################

####################################################
def pem_model(af = 'swish'):
    input = Input(shape = (245))
    out = Dense(1000,activation = af)(input)
    out = Dropout(0.2)(out)
    out = Dense(500,activation = af)(out)
    out = Dropout(0.2)(out)
    out = Dense(500, activation = af)(out)
    out = Dropout(0.2)(out)
    out = Dense(500,activation = af)(out)
    out = Dropout(0.2)(out)
    out = Dense(500,activation = af)(out)
    out = Dropout(0.2)(out)
    out = Dense(500, activation = af)(out)
    out = Dropout(0.2)(out)

    # out = Dense(1)(out)
    out = Dense(11, activation = 'softmax')(out)
   
    model = Model(input,out)
    
    optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    
    # loss = tf.keras.losses.MeanAbsoluteError()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='False')
    # metrics = 'mean_absolute_error'
    metrics = 'accuracy'
    model.compile(loss=loss,optimizer=optim,metrics=metrics)
    return model
####################################################

####################################################
def pem_split_model(af = 'swish'):
    input1 = Input(shape=(245,))
    out1 = Dense(1000,activation = af)(input1)
    out1 = Dropout(0.2)(out1)
    out1 = Dense(1000,activation = af)(out1)
    out1 = Dropout(0.2)(out1)

    input2 = Input(shape=(245,))
    out2 = Dense(1000,activation = af)(input2)
    out2 = Dropout(0.2)(out2)
    out2 = Dense(1000,activation = af)(out2)
    out2 = Dropout(0.2)(out2)  

    out = Concatenate()([out1,out2])
    out = Dense(500,activation=af)(out)
    out = Dropout(0.2)(out)
    out = Dense(500,activation=af)(out)
    out = Dropout(0.2)(out)
    out = Dense(500,activation=af)(out)
    out = Dropout(0.2)(out)
    out = Dense(500,activation=af)(out)

    out = Dense(11, activation = 'softmax')(out)   

    model = Model([input1,input2],out)
    
    optim = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    
    # loss = tf.keras.losses.MeanAbsoluteError()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits='False')
    # metrics = 'mean_absolute_error'
    metrics = 'accuracy'
    model.compile(loss=loss,optimizer=optim,metrics=metrics)
    return model
####################################################
