#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 08:11:18 2020

@author: leixinma
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:41:33 2020

@author: leixin
"""
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
import scipy.io as sio
from matplotlib import cm
#import plotly
import math
#from scipy.stats import halfnorm
import random
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from numpy.random import seed 
seed(1) 
from tensorflow import keras
import keras.backend as K
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten,concatenate

#from mpl_toolkits.mplot3d import Axes3D
import os
from random import randint
from matplotlib.ticker import MaxNLocator
#from matplotlib import *
import matplotlib.ticker as ticker
from numpy import linalg as LA

from keras import regularizers
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.regularizers import Regularizer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import set_random_seed 
import change as c
import entropy_estimators as ee
import sklearn.preprocessing
from sklearn.neighbors import KernelDensity, KDTree, NearestNeighbors
from sklearn.linear_model import Ridge, lars_path
import lime
import lime.lime_tabular              
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

def dice_loss(mapee,thres,Y,ww):
    def dice(y_true, y_pred):
#        print('oooo',str(K.eval(oo)))
        return dice_coef(y_true, y_pred,mapee,thres,Y,ww)
    return dice
def dice_coef(y_true, y_pred,mapee,thres,Y,ww):
    y_true= K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    Y=K.flatten(Y)## REF     
    mask1=K.abs(K.tanh(Y+thres/10)-K.tanh(Y-thres/10))/2## 0 or 2//2
    mask2=(-K.tanh(Y+thres/10)+1)/2##0 or 1
    mask3=(K.tanh(Y-thres/10)+1)/2##0 or 1
    mask4=K.abs(mask1-1)
#    error= (K.mean((y_true - y_pred)**2 ))  # K.mean(K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf)))
    y_true1=y_true*mask1
    y_pred1=y_pred*mask1

    y_true2=y_true*mask2
    y_pred2=y_pred*mask2
    
    y_true3=y_true*mask3
    y_pred3=y_pred*mask3
    if mapee==2:
        error=K.mean(K.abs(y_true1-y_pred1)/y_true*mask1)*(1-ww)+K.mean((K.sigmoid((y_pred2-y_true2)/y_true*10000))*mask2)*ww  ## decrease trend
#    if mapee==20:
        error=K.mean(K.abs(y_true1-y_pred1)*K.abs(y_true1-y_pred1)*mask1)*(1-ww)+K.mean((K.sigmoid((y_pred2-y_true2)/y_true*10000))*mask2)*ww  ## decrease trend
    elif mapee==3:
        error=K.mean(K.abs(y_true1-y_pred1)/y_true*mask1)*(1-ww)+K.mean((K.sigmoid((y_pred2-y_true2)/y_true))*mask2)*ww/2+K.mean((K.sigmoid((y_true3-y_pred3)/y_true))*mask3)*ww/2
#        error=K.sum(K.abs(y_true1-y_pred1)/y_true*mask1)/K.sum(mask1)*(1-ww)+K.sum((K.sigmoid((y_true2-y_pred2)/y_true*10000*Y/abs(Y+100)))*mask4)/K.sum(mask4)*ww
    elif mapee==4:
        error=K.mean(K.abs(y_true1-y_pred1)/y_true*mask1)*(1-ww)+K.mean((K.sigmoid((y_true3-y_pred3)/y_true*100))*mask3)*ww #/K.sum(mask1)
    elif mapee==5:
        error=K.sum(K.abs(y_true1-y_pred1)/y_true*mask1)/K.sum(mask1)*(1-ww)
    elif mapee>=6:
        rrtt=1 # K.mean(K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), np.inf)))                                    K.mean((K.sigmoid((y_true-y_pred)/y_true*10000))*mask3)*1/2*rrtt
        error=K.mean(K.abs(y_true1-y_pred1)/y_true*mask1)*(1-ww)+K.mean((K.sigmoid((y_pred-y_true)/y_true*1000))*mask2)*ww/2+K.mean((K.sigmoid((y_true-y_pred)/y_true*1000))*mask3)*ww/2

    return error


def build_modelmapee(noutput,nnnum,paramstotune,l1r,pruning,train_data,mapee,thres,wwee,traininit):
  l1r0=l1r+0
  droprat=paramstotune[0]
  if wwee>0:
      regurat=paramstotune[1]*(l1r/.0001)## for shear ##FEB 23 SEEMS TO WORK try reduce further
  else:
      regurat=paramstotune[1]#*(l1r/.001)## for shear ##FEB 23 SEEMS TO WORK try reduce further
  dtt=paramstotune[3]
  strings='sigmoid'
  print('dtt',dtt,'droprat',droprat,'regurat',regurat)

  if wwee==0:
      l1r=0.001
  else:
      l1r=1e-5
  if wwee>0 and paramstotune[0]<1:
      l1rr=l1r/3
  else:
      l1rr=l1r
  layernumsall=[10,10,10,20,20,20,40,40,40,80,80,80,60,60,60,90,90,90,200,200,200,400,400, 400,30,30,30,70,70,70,100,100,100,10,10,10,8,8,8,15,15,15]
  if traininit<=1 and paramstotune[0]<1:
      layernums=[layernumsall[nnnum*3-3]]*int(layernumsall[nnnum*3-3]/10+1)
      layernums=[layernumsall[nnnum*3-3],layernumsall[nnnum*3-2],layernumsall[nnnum*3-2],layernumsall[nnnum*3-1]]#,layernumsall[nnnum*3-1],layernumsall[nnnum*3-1]]       
  main_input= Input(shape=(train_data.shape[1],), name='main_input')
  ref_input = Input(shape=(1,), name='ref_input')
  inputs=concatenate([main_input, ref_input])
  x = Dense(units =layernums[0], activation=strings,kernel_regularizer=regularizers.l2(regurat))(main_input)
  for ii in range(len(layernums)-1):
        x = Dense(layernums[ii+1], activation=strings,kernel_regularizer=regularizers.l2(regurat))(x)
        x=Dropout(droprat)(x)
        
  predictions = Dense(noutput,kernel_regularizer=regularizers.l2(regurat), activation='linear')(x)
    
  model = Model(inputs=[main_input, ref_input], outputs=predictions)
  if mapee==1 or traininit==1 or wwee==0:
      l2st=0.0001#1e-1
  elif mapee>=2:
      l2st=0.0001
  else:
      l2st=l1r*100
  if mapee<2 or traininit==1 or wwee==0:
      if pruning==0:
          optimizer=tf.optimizers.Ftrl(dtt, l1_regularization_strength=l1r, l2_regularization_strength=l2st,l2_shrinkage_regularization_strength=l1r) ## OPTIMIZATION STRIGNTH BUGOU HAHA 
      else:
          optimizer=tf.optimizers.Ftrl(dtt)
          optimizer = tf.keras.optimizers.Adam(lr=dtt)
          print('run')
  elif paramstotune[0]<1:
      optimizer = tf.keras.optimizers.Adam(lr=dtt)
  else:
      optimizer=tf.optimizers.Ftrl(dtt, l1_regularization_strength=l1r, l2_regularization_strength=l1r)
  if mapee==1 or traininit==1 or wwee==0:
      model.compile(loss='MAPE',
                    optimizer=optimizer,
                    metrics=['mape'])
  elif mapee>=2:
      model_dice = dice_loss(mapee,thres,ref_input,wwee)
      model.compile(loss=model_dice,
                    optimizer=optimizer)
  return  model, droprat ,regurat, layernums,l1r0


def plot_history(histories,splitt,mapee):
  plt.figure()
  plt.xlabel('Epoch')
  if mapee==1:
      plt.ylabel('Mean Abs Percentage Error')
      abserror=np.array(histories.history['mape'])
      plt.plot(histories.epoch, abserror/100,
               label='Train Loss')
      if splitt>0:
          plt.plot(histories.epoch, np.array(histories.history['val_mape'])/100,
                   label = 'Validation loss')
      plt.legend()
 
  elif mapee>=2 :
      abserror=np.array(histories.history['loss'])
#      abserrorval=np.array(histories.history['val_loss'])
      if np.max(abserror)>1:
          abserror=abserror/100
      plt.ylabel('Mean Abs Percentage Error')
      plt.plot(histories.epoch, abserror,
               label='Train Loss')
      plt.legend()
  mapetrain=abserror[-1].tolist()
  mapetrainout=[mapetrain]
#  mapetrainoutval=[abserrorval[-1].tolist()]
  print("Training loss/error",mapetrain)
  return mapetrainout