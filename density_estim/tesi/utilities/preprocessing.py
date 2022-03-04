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
import numba
from numba import njit
from numpy import random


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
def import_csi(path=None,cut=None,preprocess = None):
    '''
    Returns dictionaries containing the csi for each stream of measurement
    Note that the files containing the CSI must be written with the format "persone_N_....mat" 
    where N is the number of the people in the measurement
    Inputs:
        - path: directory from which the csi matrices will be loaded;
        - cut: start and index to cut the csi matrices
    '''

    density_path = path
    densities = {}
    if preprocess == None:
        raise ValueError('Please choose if preprocess or not!')
    if path == None or cut == None:
        raise ValueError('Insert a valid path and cut index!')

    for name in os.listdir(density_path):
        if name.split('_')[0] == 'persone':
            dens = scipy.io.loadmat(density_path + name)
            dens = dens['csi_matrix_processed']
            if preprocess:
                '******************************'
                'Divide for the mean of all the subcarriers'
                dens[:,:,0] = dens[:,:,0]/np.mean(dens[:, :, 0], axis=1,  keepdims=True) 
                '******************************'
                'Normalize the amp and phase:'
                dens[:, :, 0] = dens[:, :, 0] / np.max(dens[:, :, 0], axis=1, keepdims=True)
                dens[:, :, 1] = dens[:, :, 1] / np.max(dens[:, :, 1], axis=1, keepdims=True)
            
            'Cut around 1 second at the start and the end of the traces:'
            dens = dens[cut:-cut,:,:]

            densities[name.split('.')[0]] = dens
    
    return densities
############################################################################

############################################################################
def average_packet_length(densities,do_print,n_antennas=4,n_persone=10):
    '''
    Return a float being the average packet length among all observations
    Inputs:
        - densities is the dictionaty containing the different csi observations
        - do_print is boolean, if true prints the avg length for each observation
    '''
    avg_tc = 0
    for antenna in range(n_antennas):
        for persone in range(1,n_persone+1):
            if persone == 1:
                time = 6*60
            else:
                time = 2*60
            if persone == 5:
                prova = 3
            elif persone == 7:
                prova = 1
            else:
                prova = 1
            if do_print:
                print('Tc pacchetto {} persone: {}'.format(persone,time/(np.shape(densities['persone_{}_{}_stream_{}'.format(persone,prova,antenna)])[0]+100)))
            avg_tc+= time/(np.shape(densities['persone_{}_{}_stream_0'.format(persone,prova)])[0]+100)
    avg_tc /= 40

    return avg_tc
############################################################################


########################################################################
# @numba.njit
def create_doppler_files(save=False,plot=True,cut=None,num_persone=None,zero_pad=None,path_doppler=None,path_density=None,noise_lev=-2.0,num_symbols=51,sliding=1,Tc=6e-3):

    '''
    Taken from Signet GitHub.
    Compute doppler spectrum from the CSI matrices.
    ----------------------------------------------------------------------------
    Parameters:
        - noise_lev --> noise level (below which data is hard thresholded to 0)
        - num_symbols --> time window in which we consider the data to make the 
        fourier transform 
        - sliding --> time shift 
    ----------------------------------------------------------------------------
    '''    

    if path_doppler == None and save:
        raise ValueError('Specity a path in which to save the doppler files.')
    if path_density == None:
        raise ValueError('Specify a path from which to load the csi matrices.')

    middle = int(mt.floor(num_symbols / 2))

    Tc = Tc
    fc = 5e9
    v_light = 3e8
    delta_v = round(v_light / (Tc * fc * num_symbols), 3)
    # v_max = 2.5
    # idx_max = mt.ceil(2.5 / delta_v)
    # start_idx = middle - idx_max
    # stop_idx = middle + idx_max + 1

    sliding = sliding #should be enough to consider the velocity in this window constant
    noise_lev = noise_lev  #this looks to be the minimum value below which there's too much noise in the doppler plots

    csi = {}
    dopplers = {}
    for name in os.listdir(path_density):
        if name.split('_')[0] == 'persone' or name.split('_')[0] == 'density':
            print(name)
            this_csi = scipy.io.loadmat(path_density+name)
            this_csi = this_csi['csi_matrix_processed']
            this_csi = this_csi[cut:-cut,:,:]       
            this_csi[:, :, 0] = this_csi[:, :, 0] / np.mean(this_csi[:, :, 0], axis=1,  keepdims=True)
            csi[name.split('.')[0]] = this_csi
                    

    # path_doppler = '/content/drive/MyDrive/UNI/second_year/TESI/NEWDATA/marco/doppler/'
    if save and not os.path.exists(path_doppler):
        os.mkdir(path)

    to_exclude_keys = []
    for n_persone in range(num_persone + 1):                
        for name in list(csi.keys()):
            if (name.split('_')[1] == str(n_persone)) and (name not in to_exclude_keys):
                to_exclude_keys.append(name)
                csi_matrix_processed = csi[name]

                # csi_matrix_processed[:, :, 0] = csi_matrix_processed[:, :, 0] / np.mean(csi_matrix_processed[:, :, 0],
                #                                                                         axis=1,  keepdims=True)

                csi_matrix_complete = csi_matrix_processed[:, :, 0]*np.exp(1j*csi_matrix_processed[:, :, 1])

                csi_d_profile_list = []

                for i in range(0, csi_matrix_complete.shape[0]-num_symbols, sliding):

                    csi_matrix_cut = csi_matrix_complete[i:i+num_symbols, :]
                    csi_matrix_cut = np.nan_to_num(csi_matrix_cut) #Replace NaN with zero and infinity with large finite numbers

                    hann_window = np.expand_dims(hann(num_symbols), axis=-1) #hann calculates the hann window https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html
                    csi_matrix_wind = np.multiply(csi_matrix_cut, hann_window)
                    # print('csi_matrix_wind shape: {}'.format(np.shape(csi_matrix_wind)))
                    
                    'Transpose the matrix so to have the subcarriers in the first axis and the window in the second axis'
                    # csi_matrix_cut = csi_matrix_wind.T
                    
                    csi_doppler_prof = fft(csi_matrix_wind, n=zero_pad, axis=0) #makes discrete fourier transform
                    csi_doppler_prof = fftshift(csi_doppler_prof, axes=0) #order the vector of frequencies putting the 0 frequency one in the center of the spectrum
            
                    csi_d_map = np.abs(csi_doppler_prof * np.conj(csi_doppler_prof)) #calculate the module of the complex numbers inside csi_doppler_prof
                    csi_d_map = np.sum(csi_d_map, axis=1)
                    # print('csi_d_map shape: {}'.format(np.shape(csi_d_map)))
                    # plt.plot(csi_d_map)
                    # plt.show()
                    csi_d_profile_list.append(csi_d_map)

                csi_d_profile_array = np.asarray(csi_d_profile_list)
                csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
                # print('csi_d_profile_array and max shapes: {}'.format(np.shape(csi_d_profile_array),np.shape(csi_d_profile_array_max)))

                csi_d_profile_array = csi_d_profile_array/csi_d_profile_array_max
                # csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_lev)] = mt.pow(10, noise_lev)


                # csi_d_profile_array = np.asarray(csi_d_profile_array[:])
                # csi_d_profile_array = np.flip(np.sum(csi_d_profile_array, axis=2), axis=1)

                '''
                Save the doppler spectrum into the ditionary
                '''
                dopplers[name] = csi_d_profile_array

                if save:
                    base = path_doppler
                    path_doppler_name =  base + name + '.txt'

                    with open(path_doppler_name, "wb") as fp:  # Pickling
                        pickle.dump(csi_d_profile_array, fp)

                if plot:
                    """
                    Plot the doppler matrix:
                    """

                    fig = plt.figure(figsize = (20,5))
                    gs = gridspec.GridSpec(1, 1, figure=fig)
                    step = 15
                    length_v = mt.floor(csi_d_profile_array.shape[1] / 2)
                    factor_v = step * (mt.floor(length_v / step))
                    ticks_y = np.arange(length_v - factor_v, length_v + factor_v + 1, step)
                    ticks_x = np.arange(0, csi_d_profile_array.shape[0], int(csi_d_profile_array.shape[0]/20))
                    ax1 = fig.add_subplot(gs[(0, 0)])
                    plt1 = ax1.pcolormesh(csi_d_profile_array.T, cmap='viridis', linewidth=0, rasterized=True)  # , shading='gouraud')
                    plt1.set_edgecolor('face')
                    cbar1 = fig.colorbar(plt1)
                    cbar1.ax.set_ylabel('power [dB]', rotation=270, labelpad=14)
                    ax1.set_ylabel(r'velocity [m/s]')
                    ax1.set_xlabel(r'time [s]')
                    ax1.set_yticks(ticks_y + 0.5)
                    ax1.set_yticklabels(np.round((ticks_y - length_v) * delta_v, 2))
                    ax1.set_xticks(ticks_x)
                    ax1.set_xticklabels(np.round(ticks_x * sliding * Tc, 2))
                    plt.title('{}'.format(name))
                    plt.show()
                
    return dopplers
########################################################################

########################################################################
def load_doppler_spectrum(path_doppler=None,densities=None,show=False):
    """
    Return dopplers into dopplers dictionary.
    You can provide the name of the files to be loaded either directly
    with the path or by giving the dictionary containing the CSI matrices.
    """
    if densities != None:
        keys = list(densities.keys())
    elif densities == None and path_doppler != None:
        keys = [fil.split('.')[0] for fil in os.listdir(path_doppler)]
    elif densities == None and path_doppler == None:
        raise ValueError('You must insert either the dictionary containing the CSI or the path from which to load the doppler files.')

    dopplers = {}
    count =1

    to_exclude = []
    for k in range(4):
        to_exclude.append(f'indoor_1_lab_stream_{k}.txt') #A lot of noise in these streams

    for density in sorted(list(keys)):#densities.keys():
        if density+'.txt' in os.listdir(path_doppler) and density+'.txt'not in to_exclude: 
            if show:
                print(density, count)
            count+=1
            # with open(path_doppler+density+'.txt','rb') as f:
            #     x = pickle.load(f)
            # dopplers[density] = x
            dopplers[density] = pickle.load(open(path_doppler + density + '.txt', "rb"), encoding="latin1")

    print(f'Overall number of measurements: {count//4}')
    return dopplers
########################################################################


############################################################################
'''
Compute various doppler features
--------------------------------
Note: in all the cases below, dop_window is a vector of dimension (W,1) with W 
being the length of the sliding window.
Therefore all the functions below return a number.
'''

# @numba.njit 
def spectral_energy(dop_window):
    '''
    Computes the total spectral energy of the doppler spectrum:
    '''

    return np.mean(np.sum(dop_window**2,axis=-1))

# @numba.njit
# def mean_spectrum(doppler):
#     '''
#     Takes a doppler spectrum (in our case a (Window,100) matrix), and return a (Window,1) 
#     vector in which the spectrum has been averaged on all the subcarriers
#     NOTE however, that you could in theory also compute the features for the doppler 
#     spectrum (not it's average). But as of now it's better to stick to Domenico's paper
#     '''
#     return np.mean(doppler,axis=1)

# @numba.njit
def mean_dop(dop_window):
    'Computes mean from a window of a doppler spectrum of shape (window,100)'
    return np.mean(np.sum(dop_window,axis=-1)/np.shape(dop_window)[-1])


# @numba.njit
def sd_dop(dop_window):
    'Computes standard deviation from a window of a doppler spectrum of shape (window,100)'
    

    mu_dop = mean_dop(dop_window)

    
    return np.mean(1/(np.shape(dop_window)[-1]-1) * np.sum((dop_window-mu_dop)**2,axis=-1))

# @numba.njit
def spectral_centroid(dop_window):
    '''
    Computes the spectral centroid ("center of mass of the doppler spectrum")
    '''
    W = np.shape(dop_window)[-1]
    den = np.sum(dop_window**2,axis=-1)
    fact = (np.arange(W)+1)*25/W
    num = np.dot(dop_window**2,fact)

    # num = np.sum(np.array([dop_window[k]**2 * (k+1)*25/W for k in range(W)]),axis=-1)
    return np.mean(num/den)

# @numba.njit
def decay_factor(dop_window):
    '''
    Computes the decay factor for the doppler window
    '''
    W = np.shape(dop_window)[-1]

    alpha_n = np.sum(np.array([dop_window[k] * (k+1)*25 for k in range(W)]),axis=-1) / W
    lambda_n = - 1/(W - 1) * np.sum(np.array([ (dop_window[k] - dop_window[k-1])/(25/W) * 2 / (dop_window[k]+dop_window[k-1]) for k in range(1,W)]),axis=-1)

    return alpha_n,lambda_n

# @numba.njit
def spectral_entropy(dop_window,S=1):
    '''
    Computes the amount of information contained in the doppler spectrum
    '''
    E_tot = np.sum(dop_window**2,axis=-1)

    SEn = np.zeros(shape=np.shape(dop_window)[0])
    N_blk = int(np.shape(dop_window)[-1]/S)
    # print(N_blk)
    for t in range(1,N_blk+1):
        Et_blk = np.zeros(shape=np.shape(dop_window)[0])
        for k in range(int((t-1)*S+1),int(t*S)+1):
            Et_blk += dop_window[:,k-1]**2
            # print(Et_blk)
    
        SEn += Et_blk/E_tot * np.log2(Et_blk/E_tot)

    return np.mean(-SEn)

# @numba.njit
def spectral_flatness(dop_window):
    '''
    Spectral flatness quantifies the noise in the signal
    '''
    den = np.mean(dop_window**2,axis=-1)
    # print(den)
    num = np.mean(dop_window**2,axis=-1)**(1.0/np.shape(dop_window)[-1])
    # print(num)
    return np.mean(num/den)

# @numba.njit
def spectral_rolloff(dop_window):
    '''
    It's the frequency below which the 90% of the accumulated magnitude of the spectrum is concentrated
    '''
    E_k = 0
    E_tot = np.sum(dop_window**2,axis=-1)

    k = 0
    while E_k < 0.9 * E_tot:
        E_k += dop_window[k]**2
        k+=1

    return k

# @numba.njit
def spectral_slope(dop_window):
    '''
    It's a measure of the lsope of the spectral shape
    '''
    W = len(dop_window)
    fk = lambda k: (k+1) * 25/ W
    fk_bar = lambda k: 1/W**2 * 25 * np.sum(np.arange(1,k+1))
    Hk_bar = lambda k: 1/W * np.sum(dop_window[:k])

    den = np.sum(np.array([(fk(k) - fk_bar(k))**2 for k in range(W)]))

    num = np.sum(np.array([(fk(k) - fk_bar(k))*(dop_window[k]-Hk_bar(k)) for k in range(W)]))

    return num/den

# @numba.njit
def spectral_spread(dop_window):
    '''
    The spectral spread is defined as the second central moment of the 
    log-frequency spectrum and gives indications about how the spetrum is 
    distributed around its centroid
    '''
    W = np.shape(dop_window)[-1]
    fk = (np.arange(W)+1)*25/W 
    SCn = spectral_centroid(dop_window)
    num = np.dot(dop_window**2 , (fk - SCn)**2)
    den = np.sum(dop_window**2,axis=-1)

    return np.mean(np.sqrt(num/den))


# @numba.njit
def order_n_spectral_moment(dop_window,n):
    '''
    Computes the order n-th spectral moment
    In the paper 2<=n<=4
    '''
    W = np.shape(dop_window)[-1]
    # fk = lambda k: (k+1) * 25/ W
    # SCn = spectral_centroid(dop_window)
    fact = ((np.arange(W)+1)*25/W)**n
    # print(np.shape(np.dot(dop_window**2,fact)))
    return np.mean(np.dot(dop_window**2,fact)/np.sum(dop_window**2,axis=-1))
    # return np.mean(np.sum(np.array([dop_window[k]**2 * fk(k)**n for k in range(W)]),axis=-1)/np.sum(dop_window**2,axis=-1))


# @numba.njit
def order_n_spectral_central_moment(dop_window,n):
    '''
    Computes the order n-th spectral central moment
    In the paper 2<=2<=4
    '''
    W = np.shape(dop_window)[-1]
    SCn = spectral_centroid(dop_window)
    # fk = lambda k: (k+1) * 25/ W
    fact = ((np.arange(W)+1)*25/W - SCn)**n
    num = np.dot(dop_window**2,fact)
    # num = np.sum(np.array([dop_window[k]**2 * (fk(k) - SCn)**n for k in range(W)]),axis=-1)
    den = np.sum(dop_window**2,axis=-1)

    return np.mean(num/den)

# @numba.njit
def spectral_skweness(dop_window):
    '''
    The spectarl skewness is a measure of the asymmetry of the doppler spectrum about its mean
    '''
    W = np.shape(dop_window)[-1]
    SCn = spectral_centroid(dop_window)
    # fk = lambda k: (k+1) * 25/ W
    fk = (np.arange(W)+1) * 25/W
    csi_2 = order_n_spectral_central_moment(dop_window,2)
    # Tkn = lambda k: (fk(k) - SCn)/np.sqrt(csi_2)
    Tkn = np.abs((fk - SCn)/np.sqrt(csi_2))

    return np.mean(np.dot(dop_window**2 ,Tkn**3))

# @numba.njit
def spectral_kurtosis(dop_window):
    '''
    It's a measure of the "tailedness" of the doppler spectrum
    '''
    W = np.shape(dop_window)[-1]
    SCn = spectral_centroid(dop_window)
    # fk = lambda k: (k+1) * 25/ W
    fk = (np.arange(W)+1)*25/W
    csi_2 = order_n_spectral_central_moment(dop_window,2)
    # Tkn = lambda k: (fk(k) - SCn)/np.sqrt(csi_2)
    Tkn = (fk - SCn)/np.sqrt(csi_2)
    return np.mean(np.dot(dop_window**2,Tkn**4 ))


# @numba.njit
def mean_sigma_ratio(dop_window):
    '''
    It's the ratio betweeen the arithmetic mean of the doppler spectrum and its standard deviation
    '''

    mu = mean_dop(dop_window)
    sigma = sd_dop(dop_window)

    return mu/sigma
############################################################################



#############################################################
# @numba.njit
def compute_feature_vector(dop_window,W):
    '''
    This function returns, for each doppler window and each link a vector of 18 features
    ------------------------------------------------------------------------------------
    Parameters:
        - dop_window: vector of shape (W,1), with length(W)
    '''
    if np.shape(dop_window)[-1] != W:
        raise ValueError('Check the shape of the doppler window! Did you want to use a single doppler window? In such case input the doppler window as a shape (1,window) (for ex. (1,100))')

    feature_vector = np.array([spectral_energy(dop_window),
                               mean_dop(dop_window),
                               sd_dop(dop_window),
                            #    spectral_centroid(dop_window),
                            #    spectral_entropy(dop_window),
                            #    spectral_flatness(dop_window),
                            #    spectral_slope(dop_window),
                               spectral_spread(dop_window),
                               order_n_spectral_moment(dop_window,2),
                               order_n_spectral_central_moment(dop_window,2),
                            #    order_n_spectral_moment(dop_window,3),
                            #    order_n_spectral_central_moment(dop_window,3),
                               order_n_spectral_moment(dop_window,4),
                               order_n_spectral_central_moment(dop_window,4),
                               spectral_skweness(dop_window),
                               spectral_kurtosis(dop_window),
                               mean_sigma_ratio(dop_window)])
    
    # feature_vector = np.array([mean_dop(dop_window),
    #                            sd_dop(dop_window),
    #                            mean_sigma_ratio(dop_window),
    #                            spectral_energy(dop_window),
    #                            spectral_centroid(dop_window),
    #                            order_n_spectral_moment(dop_window,2),
    #                            order_n_spectral_central_moment(dop_window,2),
    #                            spectral_kurtosis(dop_window)
    #                            ])
    
    return feature_vector
##########################################################################

#######################################################################
# @numba.njit
def feature_extraction_train_flow(data,is_doppler,n_antennas,W):
    '''
    Returns, for each doppler window, a vector of 18 (number of features as mean between the links) 
    + 18 * 4 (number of features considering the 4 link(antennas) separate) = 90 total features, with 
    the respective labels
    ----------------------------------------------------------------------------
    Parameters:
        - data: dictionary, each key being a different antenna
        - is_doppler: boolean, if true take first element
        - n_anteannas: integer, the number of separate links to be considered
        - W: integer, the window for which the doppler has been computated
    '''
    # print(len(data),n_antennas)
    # if len(data) != n_antennas:
    #     raise ValueError(f'You are not passing a dictionary having the {n_antennas} antennas as keys!')
    # return None
    adjust = len(compute_feature_vector(data[0][0],W))

    # train_features = np.zeros(shape=(len(data[0]), int(adjust*4 + adjust)))
    train_features = []
    progress = 1
    for idx in range(len(data[0])):
        if int(idx/len(data[0])*100) == int(10 * progress):
            
            print(f'{10*progress}')
            progress+=1
        # temp_features = np.zeros(shape=(int(adjust*4 + adjust)))
        temp_features = []
        # print(np.shape(data[0][idx]))
        # if is_doppler:
        #     fvk = np.asarray([compute_feature_vector(data[k][idx],W) for k in range(n_antennas)])
        # else:
        # fvk = np.asarray([compute_feature_vector(data[k][idx],W) for k in range(n_antennas)])
        
        # fvk = np.zeros(shape=(n_antennas,adjust))
        fvk = []
        mean_links = np.zeros(shape = adjust)

        for ant in range(n_antennas):
            # fvk[ant] = np.asarray(compute_feature_vector(data[ant][idx],W))
            fvk.append(np.asarray(compute_feature_vector(data[ant][idx],W)))
            mean_links += fvk[ant]
        # fv1 = compute_feature_vector(data[0][idx])
        # fv2 = compute_feature_vector(data[1][idx])
        # fv3 = compute_feature_vector(data[2][idx])
        # fv4 = compute_feature_vector(data[3][idx])
        # print(len(fv1))
        # for k in range(len(fv1)):
        #     temp_features.extend([(fv1[k]+fv2[k]+fv3[k]+fv4[k])/4])
        # for l in range(n_antennas):
        #     mean_links += fvk[l]
        mean_links/=n_antennas 

        # np.mean(fvk,axis = 0)
        # for m in range(adjust):
        #     temp_features[m] = mean_links[m]
        temp_features.extend(mean_links)

        # count = adjust
        for k in range(n_antennas):
            # for m in range(adjust):
            #     temp_features[count] = fvk[k][m]
            #     count+=1
            temp_features.extend(fvk[k])
        # print(len(temp_features))

        # for l in range(len(temp_features)):
        #     train_features[idx][l] = temp_features[l]
        train_features.append(temp_features)
    return train_features
#######################################################################


##################################################
def create_train_val_test(data=None,path_data=None,cut=None,delta_t=None,overlap=None):
    '''
    Cretes train, val and test and saves them inside doppler folder 
    Parameters:
        - dopplers: dictionary containing the doppler spectrums
        - delta_t: time interval to consider during training and testing
    Return: train, val and test where each of them has been segmented into time frames of delta_t length
    '''

    if overlap == 1:
        raise ValueError('The overlap must be strictly smaller than 1!')
    elif delta_t == None:
        raise ValueError('Please insert an integer value for the delta_t')
    elif data==None and path_data==None:
        raise ValueError('If you don\'t provide a dataset, you must at least indicate where to load it from.')

    if data==None and path_data!=None:
            data = {}
            for name in os.listdir(path_data):
                if name.split('_')[0] == 'persone':
                    print(name)
                    this_csi = scipy.io.loadmat(path_data+name)
                    this_csi = this_csi['csi_matrix_processed']

                    this_csi = this_csi[cut:-cut,:,:]       
                    '******************************'
                    'Divide for the mean of all the subcarriers'
                    this_csi[:,:,0] = this_csi[:,:,0]/np.mean(this_csi[:, :, 0], axis=1,  keepdims=True) 
                    '******************************'
                    'Normalize the amp and phase:'
                    this_csi[:, :, 0] = this_csi[:, :, 0] / np.max(this_csi[:, :, 0], axis=1, keepdims=True)
                    this_csi[:, :, 1] = this_csi[:, :, 1] / np.max(this_csi[:, :, 1], axis=1, keepdims=True)
                    


                    data[name.split('.')[0]] = this_csi

    validation = {}
    test = {}
    train = {}

    k_i = lambda i: int(delta_t * i * (1-overlap))
    for this_data in data:

        dopp_spect = data[this_data]
        num_delta = int(round((len(dopp_spect) - delta_t)/(delta_t * (1 - overlap))))

        train_length_delta = int(round(0.6  * ((len(dopp_spect) - delta_t)/(delta_t * (1 - overlap)))))
        val_length_delta = int(round(0.2 * ((len(dopp_spect) - delta_t)/(delta_t * (1 - overlap)))))
        test_length_delta = val_length_delta

        print('Length {}: {}'.format(this_data,len(data[this_data])))

        validation[this_data] = np.asarray([dopp_spect[k_i(i): k_i(i) + delta_t] for i in np.arange(0,val_length_delta)])
        val_end = int(delta_t * (val_length_delta-1)*(1-overlap)) + delta_t
        print('Validation start: {} Validation end: {}'.format(0,val_end))
        print(len(validation[this_data]))

        test[this_data] = np.asarray([dopp_spect[val_end + k_i(i): val_end + k_i(i) + delta_t] for i in np.arange(0,val_length_delta)])
        test_end = int(val_end + delta_t * (val_length_delta-1)*(1-overlap)) + delta_t
        print('Test start: {} Test end: {}'.format(val_end + 1,test_end))
        print(len(test[this_data]))


        train_array = []
        for i in np.arange(0,len(data[this_data])):
            if test_end + k_i(i) + delta_t < len(dopp_spect):
                train_array.append(dopp_spect[test_end + k_i(i): test_end + k_i(i) + delta_t])
            else:
                break
        # train[this_data] = np.array([dopp_spect[int(test_end + delta_t * i*(1-overlap)): int(test_end + delta_t * i*(1-overlap) + delta_t)] if int(delta_t * i*(1-overlap) + delta_t) <= len(dopp_spect) else 0 for i in np.arange(0,len(validation[this_data]) + len(test[this_data])) ])
        train[this_data] = np.asarray(train_array)

        print('Train start: {}'.format(test_end))
        print(len(train[this_data]))
        print('Summed length of train,val,test: {}'.format(len(validation[this_data]) + len(test[this_data]) + len(train[this_data])))

        # train[this_data] = [dopp_spect[val_length_delta * delta_t * 2 + delta_t*i: val_length_delta * delta_t * 2 + delta_t * (i + 1)] for i in np.arange(0, train_length_delta) if val_length_delta * delta_t * 2 + delta_t * i < len(dopp_spect) ]
    return train, validation, test

#############################################################

# import re
def data_flow(data,amplitude=False,phase=False,split_antennas = False, features = False):
    """
    Create train,val and test flow. Since we have 4 different antennas I follow the approach of SIGNET paper --> obtain 4 independent prediction
    Parameters:
        - data -->  dataset (dictionary)
        - validation --> validation dataset (dictionary)
    """
    values = []
    labels = []

    if split_antennas == False:
        data_values = np.array([])
        data_labels = np.array([])

        # val_values = np.array([])
        # val_labels = np.array([])
    else:
        data_values = {0:np.array([]),1:np.array([]),2:np.array([]),3:np.array([])}

        data_labels = np.array([])

        # val_values = {0:np.array([]),1:np.array([]),2:np.array([]),3:np.array([])}

        # val_labels = np.array([])
    brute_force = 0
    if split_antennas == False:
        for antenna in range(4):
        # print('antenna: {}'.format(antenna))
            for n_persone in range(11):
                # print('n persone:{}'.format(n_persone))
                for key in data.keys():
                    if key.split('_')[-1] == str(antenna):#len(re.findall('stream_{}'.format(antenna),key)) == 1:
                            if key.split('_')[1] == str(n_persone): #len(re.findall('persone_{}'.format(n_persone),key)) == 1:
                                for value in data[key]:  
                                    if amplitude:
                                        values.append(value[:,:,0])
                                        # data_values = np.append(data_values,value[:,:,0])
                                    elif phase:
                                        values.append(value[:,:,0])
                                        # data_values = np.append(data_values,value[:,:,1])
                                    else: 
                                        values.append(value)
                                        # data_values = np.append(data_values,value)
                                    labels.append(int(key.split('_')[1]))
                                    # data_labels = np.append(data_labels,int(key.split('_')[1]))
                                    # if brute_force == 0:
                                    #     data_values = np.asarray(values)
                                    #     brute_force = 666

            
        data_values = np.asarray(values)
        data_labels = np.asarray(labels)


    else:
        for antenna in range(4):
            for n_persone in range(11):
                # print('n persone:{}'.format(n_persone))
                for key in data.keys():
                    if key.split('_')[-1] == str(antenna):#len(re.findall('stream_{}'.format(antenna),key)) == 1:
                            if key.split('_')[1] == str(n_persone):
                                for value in data[key]:  
                                    if amplitude:
                                        values.append(value[:,:,0])
                                        # data_values[antenna] = np.append(data_values[antenna],value[:,:,0])
                                    elif phase:
                                        values.append(value[:,:,1])
                                        # data_values[antenna] = np.append(data_values[antenna],value[:,:,1])
                                    else: 
                                        values.append(value)
                                        # data_values[antenna] = np.append(data_values[antenna],value)

                                    if antenna == 0:
                                        data_labels = np.append(data_labels,int(key.split('_')[1]))

        # print(len(values))
        # print(np.shape(values[0]))
            data_values[antenna] = np.asarray(values)
            values = []
            labels = []


    return data_values, data_labels
########################################


###########################################
def test_flow(test,amplitude=False,phase=False):
    '''
    Generates test to make predictions
    '''
    test_values = {}
    test_labels = {}
    for antenna in range(4):
        test_values[antenna] = []
        test_labels[antenna] = []
        for n_person in range(11):
            for key in test.keys():
                if len(re.findall('stream_{}'.format(antenna),key)) == 1:
                    for value in test[key]:
                        if int(key.split('_')[1]) == n_person:
                            if amplitude:
                                test_values[antenna].append(value[:,:,0])
                            elif phase:
                                test_values[antenna].append(value[:,:,1])
                            else:
                                test_values[antenna].append(value)

                            test_labels[antenna].append(int(key.split('_')[1]))
                        # print(key)
                        # print(key.split('_')[1])
        test_values[antenna] = np.array(test_values[antenna])
        test_labels[antenna] = np.array(test_labels[antenna])
    return test_values, test_labels
############################################

################################################
def get_obj_size(obj):
    '''
    Get size of an object.
    ----------------------------------------------------------------------------
    Parameters:
        - obj: object
    ----------------------------------------------------------------------------   
    '''
    import gc
    import sys  
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz
###########################################################

################################################################################
def permutation_augmentation(values=None, len_inputs = None, labels=None, mod_n = 1,verbose = 0):
    '''
    Augment training dataset by considering the antennas permutations
    To use only if you plan on building the split antennas model.
    ----------------------------------------------------------------------------
    Parameters:
        - values: dictionary containing the data 
        - len_inputs: integer indicating the amount of links you pass 
        simoultaneously to the model
        - labels: the array containing the labels for the values
        - mod_n: integers, Take a permutation once every mod_n step;
        - verbose: if True print size of the data and the permutation 
        throughtout the process
    ----------------------------------------------------------------------------
    '''
    import itertools


    if len_inputs in [None,0,1]:
        raise ValueError('Len inputs (the number of links you feed into the model) needs to be at least 2!')
    # else:
    #     new_values = {k:[] for k in range(len_inputs)}

    perms = []
    # print(list(itertools.combinations(np.arange(len(values)), len_inputs)))
    for term in list(itertools.combinations(np.arange(len(values)), len_inputs)):
        perms.extend(list(itertools.permutations(term)))
    # print(perms)
    
    len_init = len(values[0])

    for count,perm in enumerate(perms):
        object = values 
        if verbose:
            print('Size of dataset: {} Bytes'.format(get_obj_size(object)))
        if count%mod_n == 0 and perm != tuple(range(len_inputs)): 
            if verbose:
                print('Permutation {} ({} of {})'.format(perm,count,len(perms)))

            labels = np.append(labels,labels[0:len_init])

            for k in range(len_inputs):
                values[k] = np.append(values[k],values[perm[k]][0:len_init],axis=0)
            
    for k in range(len(values)):
        if k >= len_inputs:
            del values[k]


    return values,labels  
###################################################
    

#######################################################################################
def rearrange_train(values_amp,labels_amp,values_phase,labels_phase,labels_dop):
    '''
    Use this function only if you want to use doppler with phase and amplitude.
    Since doppler calculation will drop some packets (last interval in densities), with this 
    function you remove these packets also in train_amp and phase so that you can 
    feed them together to a model
    '''
    # values_amp,labels_amp,values_phase,labels_phase = list(values_amp),list(labels_amp),list(values_phase),list(labels_phase)
    start_idx = 0
    idx_to_delete = []
    redo = False
    while len(labels_amp) != len(labels_dop):
        # print(len(labels_amp) - len(labels_dop))

        for i in np.arange(start_idx,len(labels_dop)):
            if labels_amp[i] != labels_dop[i]:
                # print('INdex:',i)
                # print( labels_amp[i+len(idx_to_delete)] , labels_dop[i])
                # idx_to_delete.append(i+len(idx_to_delete))    
                # labels_amp = np.delete(labels_amp,i,axis=0)
                # values_amp = np.delete(values_amp, i,axis=0)

                # labels_phase = np.delete(labels_phase, i,axis=0)
                # values_phase = np.delete(values_phase, i,axis=0)

                labels_amp.pop(i)
                values_amp.pop(i)
                labels_phase.pop(i)
                values_phase.pop(i)

                start_idx = i
                break
            else:
                end_idx = i
                
        if (len(labels_amp) - len(labels_dop)  >= 1) and (end_idx == len(labels_dop)-1):
                # idx_to_delete.append(-1)
                # labels_amp = np.delete(labels_amp,-1,axis=0)
                # values_amp = np.delete(values_amp, -1,axis=0)

                # labels_phase = np.delete(labels_phase, -1,axis=0)
                # values_phase = np.delete(values_phase, -1,axis=0)
                for k in np.arange(0,len(labels_amp)-len(labels_dop)):
                    labels_amp.pop(-1)
                    values_amp.pop(-1)
                    labels_phase.pop(-1)
                    values_phase.pop(-1)
    # values_amp = np.delete(values_amp,idx_to_delete,axis=0)
    # labels_amp = np.delete(labels_amp,idx_to_delete,axis=0),
    # values_phase = np.delete(values_phase,idx_to_delete,axis=0)
    # labels_phase = np.delete(labels_phase,idx_to_delete,axis=0)
    # return #np.delete(values_amp,idx_to_delete,axis=0),np.delete(labels_amp,idx_to_delete,axis=0),np.delete(values_phase,idx_to_delete,axis=0),np.delete(labels_phase,idx_to_delete,axis=0)
    return np.array(values_amp),np.array(labels_amp),np.array(values_phase),np.array(labels_phase)
    # return idx_to_delete
#######################################################################################


##############################################################################
def balance_data(data_values,data_labels,include = None):
    ''' 
    Balance the dataset based on the list of classes include.
    ----------------------------------------------------------------------------
    Parameters:
        - data_values: list/array
        - data_labels: list/array
        - include: list/array: set of labels to include in the balancing count
    '''

    count = []

    new_labels = []
    new_values = []

    # if include == None:
    #     include = [1,2,3,4,6,8,9]

    exclude = list(set(list(range(len(list(set(data_labels)))))) - set(include))

    for l in include:
        count.append(np.sum(np.array([data_labels[k] == l for k in range(len(data_labels))])))
    
    n = np.min(count)

    np.random.seed(42)

    mask = np.hstack([np.random.choice(np.where(data_labels == l)[0], n, replace=False)
                      for l in include])

    for k in range(len(data_labels)):
        if ((k not in mask) and (data_labels[k] not in include)) or ((k in mask) and (data_labels[k] in include)):
            new_labels.append(data_labels[k])
            new_values.append(data_values[k])
    
    return np.asarray(new_values),np.asarray(new_labels)
##############################################################################


################################################################################3
@numba.njit
def compute_PEM(csi,D,Mc):
    '''
    Compute the PEM from the CSI
    Inputs: 
        - csi: it's the np.array csi matrix of dimensions (S,P) (P is the number
         of packets in the window, S the number of subcarriers)
        - D (int) is the dilatation coefficient 
        - Mc (int) is the matrix resolution
    '''
    csi = csi.T 
    P = np.shape(csi)[1]
    S = np.shape(csi)[0]
    Cu = np.max(csi)
    Cl = np.min(csi)
    P_array = np.zeros(S)
    for i in range(S):
        M = np.zeros(shape = (Mc,P))
        for j in range(P):
            k = int(np.round((csi[i,j] - Cl) * (Mc - 1)/(Cu - Cl) ))            
            for u in range(max([-k,-D]),min([Mc - k,D+1])):

                for v in range(max([-j,-D]),min([P - j,D+1])):
                    M[k+u,j+v] = 1
        P_array[i] = np.sum(M)/(P * Mc)
    return np.asarray(P_array)
################################################################################

################################################################################
def helper_pem(csi_ij,j,D,Mc,P,Cl,Cu):
    k = int(np.round((csi_ij - Cl) * (Mc - 1)/(Cu - Cl) )) 
    
    row = np.arange(-k,Mc-k)
    to_col = np.multiply(row,row) <= D**2

    col = np.arange(-j,P-j) 
    to_row = np.multiply(col,col) <= D**2

    return np.outer(to_col,to_row)
################################################################################

def helper_pem2(csi_i,D,Mc,P,Cl,Cu):

    M = np.asarray([helper_pem(csi_i[j],j,D,Mc,P,Cl,Cu) for j in np.arange(P)])
    M = np.sum(M,axis=0)
    M = M>=1

    return np.sum(M)/(P * Mc)
################################################################################
def compute_PEM_slower(csi,D,Mc):
    '''
    Compute the PEM from the CSI.
    Note: technically, this would be a better algorithm, that's why I kept him.
    However, numba.njit makes the other one (which is just nested loops) 
    much much faster.
    ----------------------------------------------------------------------------
    Params:
        - csi: it's the np.array csi matrix of dimensions (S,P) (P is the number 
        of packets in the window, S the number of subcarriers)
        - D (int) is the dilatation coefficient 
        - Mc (int) is the matrix resolution
    '''

    csi = csi.T
    P = np.shape(csi)[1]
    S = np.shape(csi)[0]
    Cu = np.max(csi)
    Cl = np.min(csi)
    return [helper_pem2(csi[i],D,Mc,P,Cl,Cu) for i in np.arange(S)]
################################################################################


################################################################################
@numba.njit
def train_ready_pem(data=None,path=None,name=None,save=None,D=None,Mc=None):
    '''
    Transform training into pem vectors.
    Note: it's vital for having PEMs that make sense to have a normalized
    csi (which you do already when creating the train,validation,test)
    ----------------------------------------------------------------------------
    Parameters:
        - data: np.ndarray containing the data for a single antenna/link;
        - path: string containing the path in which to save the computed PEM array;
        - name: the name of the newly crated file;
        - save: boolean, if True it saves the newly generated data;
        - D: integer, the dilatation parameter;
        - Mc: integer, the 'resolution' of the dilated matrix;
    '''
    if not path and save:
        raise ValueError('Please specify a proper path in which to save the list')
    if path and path[-1] != '/':
        path += '/'

    if path and not save:
        raise ValueError('Please specify the name of the file .npy extension')
    if path and save:
        if not os.path.exists(path):
            os.mkdir(path)


    import time
    from console_progressbar import ProgressBar

    pb = ProgressBar(total=100,prefix='Done:', suffix='Now', decimals=0, length=50, fill='#', zfill='-')

    new_data = []

    counter = 0
    for k in range(np.shape(data)[0]):
        # print(int(round(k/np.shape(data)[0] * 100)))
        progress = int(round(k/np.shape(data)[0] * 100))
        pb.print_progress_bar(progress)

        new_data.append(compute_PEM(data[k],D=D,Mc=Mc))

        if progress % 10 == 0 and save:
            np.save(path+name, new_data)       
    if save:
        np.save(path+name, new_data)
################################################################################     


def load_pem(path,split_links,split_antennas):
    '''
    Load the computed PEMs from path and return either array or dictionary
    depending whether split_link and split_antennas are True/False.
    The files inside path must be inside the same folder and organized in the 
    following way:

    - train_0.npy |
    - ...         | --> if split_links is False else train.npy
    - train_4.npy |

    - train_labels.npy

    - validation_0.npy |
    - ...              | --> if split_links is False else validation.npy
    - validation_4.npy |

    - validation_labels.npy

    - test_0.npy  |
    - ...         | --> if split_links is False else test.npy
    - test_4.npy  |

    - test_labels.npy

    ----------------------------------------------------------------------
    Parameters:
        - path: string, path from which to load
        - split_link: True if the PEMs have been saved in different files for 
        each antenna, else give False;
        - split_antennas: either True or False depending on whether you want
            to build a model with parallel links as input or not.
    ----------------------------------------------------------------------
    '''
    '''
    Load the computed pem vectors
    '''
    # split_links = True
    if path[-1] == '/':
        path = path[:-1]

    if split_links == False:
        train = np.load(path + '/' + 'train.npy')
        validation = np.load(path + '/'+ 'validation.npy')

    elif split_links == True:
        train = {k:[] for k in range(4)}
        validation = {k:[] for k in range(4)}
        for k in range(4):
            train[k] = np.load(path + '/' + 'train_{}.npy'.format(k))
            validation[k] = np.load(path + '/' + 'validation_{}.npy'.format(k))
        


    train_labels = np.load(path + '/' + 'train_labels.npy')
    val_labels = np.load(path + '/'+ 'val_labels.npy')
    test_labels = np.load(path + '/' + 'test_labels.npy')

    test = {0:np.load(path + '/'+ 'test_0.npy'),
            1:np.load(path + '/'+ 'test_1.npy'),
            2:np.load(path + '/'+ 'test_2.npy'),
            3:np.load(path + '/'+ 'test_3.npy')}

    if not split_antennas:
        '''
        Transform back the train and val into arrays ready to be given to the model:
        '''
        train_r = []
        train_l = []
        validation_r = []
        val_l = []
        for k in range(4):
            train_r.extend(train[k])
            validation_r.extend(validation[k])
            train_l.extend(train_labels)
            val_l.extend(val_labels)

        train = np.asarray(train_r)
        train_labels = np.asarray(train_l)
        validation = np.asarray(validation_r)
        val_labels = np.asarray(val_l)

    return train,train_labels,validation,val_labels,test,test_labels

################################################################################


