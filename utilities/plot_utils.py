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
from keras.applications.inception_v3 import preprocess_input
from sklearn import preprocessing


############################################################################
def plot_amp_and_phase(densities,n_persone=10,fig_width=25,fig_height=5):
    '''
    Takes as input the dictionary containing all the csi matrices
    and plot the amplitudes and the phases ordered for the number 
    persons.
    It assumes 4 antennas at the RX
    Inputs:
        - densities: dictionary containing the CSI matrices
        - n_persone: integer determining the max amount of person
        - fig_width / fig_length: fig dimensions (integer)
    '''
    for n_persone in range(n_persone + 1):
        these_keys = []
        for key in densities.keys():
            if key.split('_')[1] == str(n_persone):
                these_keys.append(key)



        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(25,5))
        fig.suptitle('Amplitude for {} people in the room'.format(n_persone))

        ax1.pcolormesh(densities[these_keys[0]][:,:,0].T)


        ax2.pcolormesh(densities[these_keys[1]][:,:,0].T)

        ax3.pcolormesh(densities[these_keys[2]][:,:,0].T)

        ax4.pcolormesh(densities[these_keys[3]][:,:,0].T)

        plt.show()



        fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4,figsize=(25,5))
        fig1.suptitle('Phase for {} people in the room'.format(n_persone))

        ax1.pcolormesh(densities[these_keys[0]][:,:,1].T)


        ax2.pcolormesh(densities[these_keys[1]][:,:,1].T)

        ax3.pcolormesh(densities[these_keys[2]][:,:,1].T)

        ax4.pcolormesh(densities[these_keys[3]][:,:,1].T)

        plt.show()
############################################################################

#########################################################################################
def plt_fft_doppler_antennas(doppler_spectrum_list = None, sliding_lenght=1, delta_v=10, name_plot = None, Tc = 0.006):
    '''
    Take a list of doppler spectrums, plots them and saves them in an image called name_plot, inside the images folder 
    Parameters:
        - dopplers_spectrum_list --> list of the doppler spectrums
        - sliding_lenght --> it's the variable sliding you put in the function create_doppler_files (in my case 1)
        - delta_v --> ?
        - name_plot --> the name of the figure in which you plot all the doppler_spectrums
    '''
    if len(doppler_spectrum_list) > 0:
        fig = plt.figure()
        gs = gridspec.GridSpec(len(doppler_spectrum_list), 1, figure=fig)
        step = 15
        length_v = mt.floor(doppler_spectrum_list[0].shape[1] / 2)
        factor_v = step * (mt.floor(length_v / step))
        ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
        ticks_x = np.arange(0, doppler_spectrum_list[0].shape[0], int(doppler_spectrum_list[0].shape[0]/20))
        ax = []

        for p_i in range(len(doppler_spectrum_list)):
            ax1 = fig.add_subplot(gs[(p_i, 0)])
            plt1 = ax1.pcolormesh(doppler_spectrum_list[p_i].T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
            plt1.set_edgecolor('face')
            cbar1 = fig.colorbar(plt1)
            cbar1.ax.set_ylabel('power [dB]', rotation=270, labelpad=14)
            ax1.set_ylabel(r'velocity [m/s]')
            ax1.set_xlabel(r'time [s]')
            ax1.set_yticks(ticks_y + 0.5)
            ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
            ax1.set_xticks(ticks_x)
            ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * Tc, 2))
            ax.append(ax1)

        for axi in ax:
            axi.label_outer()
        fig.set_size_inches(20, 10)
        plt.plot()
        # plt.savefig('images/'+name_plot, bbox_inches='tight')
        # plt.close()

###################################################################################################


#########################################################################################
def plot_doppler(dopplers = None, n_persone = 10,sliding_lenght=1, num_symbols=None,Tc=6e-3,fig_width=200,fig_height=20):
    '''
    Take a dictionary containing the doppler spectrums, plots them and saves them in an image called name_plot, inside the images folder 
    Parameters:
        - dopplers_spectrum_list --> list of the doppler spectrums
        - sliding_lenght --> it's the variable sliding you put in the function create_doppler_files (in my case 1)
        - delta_v --> ?
        - name_plot --> the name of the figure in which you plot all the doppler_spectrums
    '''
    fc = 5e9
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * num_symbols), 3)
    to_esclude = []
    p_i = 0
    for persone in range(n_persone + 1):
        for key in list(dopplers.keys()):
            if key.split('_')[1] == str(persone) and key not in to_esclude:
                to_esclude.append(key)
                fig = plt.figure(figsize = (fig_width,fig_height))
                gs = gridspec.GridSpec(len(list(dopplers.keys())), 1, figure=fig)
                step = 15
                length_v = mt.floor(dopplers[key].shape[1] / 2)
                factor_v = step * (mt.floor(length_v / step))
                ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
                ticks_x = np.arange(0, dopplers[key].shape[0], int(dopplers[key].shape[0]/20))
                ax1 = fig.add_subplot(gs[(p_i, 0)])
                plt1 = ax1.pcolormesh(dopplers[key].T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
                plt1.set_edgecolor('face')
                cbar1 = fig.colorbar(plt1)
                cbar1.ax.set_ylabel('power [dB]', rotation=270, labelpad=14)
                ax1.set_ylabel(r'velocity [m/s]')
                ax1.set_xlabel(r'time [s]')
                ax1.set_yticks(ticks_y + 0.5)
                ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
                ax1.set_xticks(ticks_x)
                ax1.set_xticklabels(np.round(ticks_x * sliding_lenght * Tc, 2))
                plt.title('{}'.format(key))
                plt.show()

                p_i+=1


###################################################################################################



############################################################################
def plot_pem_DMc_fixed(data,data_labels,persone,max_count,n_antenna,style='seaborn-deep'):
    '''
    Plot the PEMs for count random intervals and a chosen number of persone
    -----------------------------------------------------------------------
    Parameters:
        - data: must be a dictionary having as keys the different antennas
        - data_labels: must be the corresponding array having the labels for each pem
        - persone: integer, the number of people you're interested in;
        - max_count: how many random intervals for each antenna
        - n_antenna: how many antennas you want to consider
        - style: string, must be a seaborn style
    '''
    plt.style.use(f'{style}')


    for antenna in range(n_antenna):
        # print('Antenna: {} Persone: {}'.format(antenna,persone))
        count = 0
        to_exclude = []
        plt.figure(figsize=(10,8))
        while count < max_count:
            idx = np.random.randint(0,len(data[0])-1)
            if data_labels[idx] == persone and idx not in to_exclude:
                to_exclude.append(idx)
                plt.plot(data[antenna][idx])
                count+=1
        plt.title('Antenna: {} Persone: {}'.format(antenna,persone),fontsize = 12)
        plt.ylabel('% of non-zero elements',fontsize = 10)
        plt.xlabel('Subcarriers',fontsize = 10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.show()
############################################################################


###################################################################################################
def plot_pem_DMc_range(path,cut,persone,antenna,D_range,Mc_range,W,style='seaborn-deep'):
    '''
    Plot the PEMs for the n_persone requested and values of D and Mc
    in the range of D_range and Mc_range respectively
    ----------------------------------------------------------------
    Parameters:
        - path: string indicating the path from which to load the csi
        - cut: integer, the interval to cut at the end and start of the csi
        - persone: integer
        - antenna: integer
        - D_range: list of values of D
        - Mc_range: list of values of Mc
        - W: integer,the window over which to copute the PEM
        - style: must be a seaborn style
    ----------------------------------------------------------------
    '''
    plt.style.use(f'{style}')

    if path and persone and antenna:
        if path[-1] != '/':
            path+='/'
        path = path + f'persone_{persone}_1_stream_{antenna}'
    else:
        raise ValueError('Please specify legit values for the path, persone and antenna!')
 
    # w = len(data[0][0])

    data = scipy.io.loadmat(path)
    data = data['csi_matrix_processed']

    data = data[cut:-cut,:,:]       
    '******************************'
    'Divide for the mean of all the subcarriers'
    data[:,:,0] = data[:,:,0]/np.mean(data[:, :, 0], axis=1,  keepdims=True) 
    '******************************'
    'Normalize the amp '
    data[:, :, 0] = data[:, :, 0] / np.max(data[:, :, 0], axis=1, keepdims=True)

    data = data[:,:,0]


    for Mc in Mc_range:
        for D in D_range:
            if D/Mc < 0.4:
                for k in range(10):
                    start = random.randint(0,np.shape(data)[0]-W)
                    # print(start)

                    plt.plot(compute_PEM(data[start:start+W,:],D=D,Mc=Mc))
                plt.title(f'Antenna: {antenna} Persone: {persone}, D: {D}, Mc: {Mc}, W: {W}',fontsize = 12)
                plt.ylabel('% of non-zero elements',fontsize = 10)
                plt.xlabel('Subcarriers',fontsize = 10)
                plt.xticks(fontsize=8)
                plt.yticks(fontsize=8)
                plt.show()
###################################################################################################


