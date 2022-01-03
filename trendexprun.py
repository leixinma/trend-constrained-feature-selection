#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 17:08:17 2019

@author: leixinma
"""
from __future__ import absolute_import, division,print_function       
#import pandas as pd
import tensorflow as tf
from keras.regularizers import Regularizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import scipy.io as sio
from matplotlib import cm
#import plotly
from tensorflow import keras
import keras
import math
#from scipy.stats import halfnorm
import random
from keras.models import Model,Sequential
from keras.layers import Input,Dense, Dropout, Flatten
from keras import regularizers
#import pandas as pd
import keras.backend as K
import matplotlib as mpl
import random
import matplotlib 
from matplotlib.colors import LinearSegmentedColormap
from numpy.random import seed 
seed(1) 
import os
from random import randint
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from keras.layers import Input, Dense
from numpy import linalg as LA

from keras import backend as K
from keras import regularizers
import numpy as np
from keras.regularizers import Regularizer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.layers import Dropout, Flatten
import scipy.io as sio
from keras.callbacks import EarlyStopping
from tensorflow.compat.v1 import set_random_seed 
import itertools
import sklearn.preprocessing
from sklearn.neighbors import KernelDensity, KDTree, NearestNeighbors
# from kernelll2 import add
from sklearn.linear_model import Ridge, lars_path
import prep as pp
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command    #
      
set_random_seed(2) 
""" =============MAIN================="""
#random.seed(1)
plotuncert=1## 1 predict with uncertainty estimates, else0 
fullinput=0## 1 all features, dont plot trend, else0
backward=0 ## backward 1 starting backward elimination
plotdetail=1
revised=1
ca=2
space=5 ##3, 1

inputmodel=0
dataall=sio.loadmat('trendall'+str(ca)+'.mat')  

dataall=dataall['dataall']

train=sio.loadmat('trendtrain'+str(ca)+'.mat')  
test=sio.loadmat('trendtest'+str(ca)+'.mat')  

traindata=train['traindata']
testdata=test['testdata']
selectrainout=[traindata.shape[1]-1]


if ca==1:
    inputt_descrp=[r'$X_1$','Y']
else:
    inputt_descrp=[r'$X_1$',r'$X_2$','Y']
    
    
traindata=traindata[np.arange(0,traindata.shape[0],space),:]
testdata=testdata[np.arange(0,testdata.shape[0],space),:]
minn=np.min([np.min(traindata[:,selectrainout]),np.min(testdata[:,selectrainout])])*.9
traindata[:,selectrainout]=traindata[:,selectrainout]-minn
testdata[:,selectrainout]=testdata[:,selectrainout]-minn
splitt=.8

inputt=np.vstack((traindata,testdata))
traindataorig=traindata+0
testdataorig=testdata+0

numall=500
fr=np.random.choice(traindata.shape[0],int(numall*splitt))
traindata=traindataorig[fr,:]
fr=np.random.choice(testdata.shape[0],int(numall*(1-splitt)))
testdata=testdataorig[fr,:]

traindataind=traindata+0
testdataind=testdata+0

persizetr=np.max([traindata.shape[0],int(traindata.shape[0]*4)])
persizete=np.max([testdata.shape[0],int(testdata.shape[0]*4)])

findrow=np.random.choice(traindata.shape[0], persizetr)
traindataind2=traindata[findrow,:]
findrow=np.random.choice(testdata.shape[0], persizete)
testdataind2=testdata[findrow,:]
""" Define input hyperparameters"""

thres=1001
if ca==1:
    paramstotune=[0, 0.0000005, 8000,.01,0,30,100]#
else:
    paramstotune=[0.001,0.00001, 16000,.006,0,30,100]#233] ##include strakes, good first attempt, full=0 is also good## for new gl1
pruning=0
l1r=0
noutput=1
ttsize=testdata.shape[0]
trsize=traindata.shape[0]
EPOCHS=paramstotune[2]
n_dimensions=1
countiter=1
wweeall=[0,.1,.2,.3,.4,.45]  #this is theta

if ca==1:
    traindata=np.vstack((traindata,traindataind2))
    testdata=np.vstack((testdata,testdataind2))
    selectt=[0]
    mapee=3
    nnnumall=[13]
    nnnum=nnnumall[0]    
elif ca>=2:
    if wweeall[0]>=0:
        traindata=np.vstack((traindata,traindataind2))
        testdata=np.vstack((testdata,testdataind2))
        mapee=4
    else:
        mapee=1        
    selectt=[0,1]
    nnnumall=[14]
    nnnum=nnnumall[0]

for jjj in range(0,2):
    if jjj==0:
        temp=traindata+0
        temp0=traindataind+0
        temp2=traindataind2+0
        templ=np.zeros((traindata.shape[0],1))
    else:
        temp=testdata+0
        temp0=testdataind+0
        temp2=testdataind2+0
        templ=np.zeros((testdata.shape[0],1))
    if ca==1:        
        for iii in range(0,temp.shape[0]):
            if iii>=temp0.shape[0] and iii<temp0.shape[0]+temp2.shape[0]:
                dt=.07+np.random.uniform(0,.03,1)
                temp[iii,selectt]=temp[iii,selectt]+dt
                templ[iii]=-thres*np.sign(temp[iii,selectt]-dt-0.25)
                temp[iii,selectrainout]=temp[iii,selectrainout]
                
                
    elif ca>=2:
        for iii in range(0,temp.shape[0]):
            if iii>=temp0.shape[0] and iii<temp0.shape[0]+temp2.shape[0]:
                templ[iii]=thres
                dt=(0.02+np.random.uniform(0,.02,1))*space
                temp[iii,0]=temp[iii,0]+dt
                temp[iii,selectrainout]=temp[iii,selectrainout]
                
    if jjj==0:
        traindata=temp+0
        trainlabel=templ+0
    else:
        testdata=temp+0
        testlabel=templ+0
        
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
plt.xlabel('label', fontsize = 20)
plt.ylabel('X', fontsize = 20)        
plt.xticks(fontsize=16)  
plt.yticks(fontsize=16)
plt.rc('font',family='Times New Roman')          
plt.plot(testlabel,testdata[:,0],'o', linewidth=2,markersize=10)
plt.grid()

if mapee<2:
    trainlabel=0*traindata[:,0]
    testlabel=0*testdata[:,0]
for wwee in wweeall:

        train_data=traindata[:,selectt]
        trainout_data=traindata[:,selectrainout] 
        
        test_data=testdata[:,selectt]
        testout_data=testdata[:,selectrainout]
        train_data_normin=train_data+0
        test_data_normin=test_data+0
        print("Training set: {}".format(train_data.shape))  #eg 404 examples, 13 features
        print("Testing set:  {}".format(test_data.shape))   #eg 102 examples, 13 features

        meanstats=np.zeros((train_data.shape[1],2))
        for ii in range(0,train_data.shape[1]):
            meanstats[ii,0]=np.mean(train_data[:,ii])
            if abs(np.std(train_data[:,ii])) <0.001:# and abs(np.mean(inputt[:,ii]))>0.01:
                meanstats[ii,1]=1
            else:
                meanstats[ii,1]=np.std(train_data[:,ii])            
        meanstats[:,0]=0
        meanstats[:,1]=1
        
        for ii in range(0,len(selectt)):
            test_data_normin[:,ii]=(test_data[:,ii]-meanstats[ii,0])/meanstats[ii,1]
            train_data_normin[:,ii]=(train_data[:,ii]-meanstats[ii,0])/meanstats[ii,1]
        """ TRAINING """
        [model,droprat,regurat,layernums,l1r] = pp.build_modelmapee(noutput,nnnum,paramstotune,l1r,pruning,train_data,mapee,thres,wwee,0)
        model.summary()   
    
        # Store training stats
        splitt=0.1
        batchsize=2**5
        trainout_data_normin=trainout_data
        
        if inputmodel==0:
            if mapee<2:
                histories = model.fit([train_data_normin,trainlabel], trainout_data,  batch_size=batchsize, epochs=EPOCHS,
                                    validation_split=splitt, shuffle=True, verbose=0)
            
                [mapetrainout]=pp.plot_history(histories,splitt,mapee)
            else:
                xb=train_data_normin[0:trsize,:]
                xb0=trainlabel[0:trsize,:]
                yb=trainout_data[0:trsize,:]
                histories = model.fit([xb,xb0], yb,  batch_size=batchsize, epochs=int(2*EPOCHS/3),
                                validation_split=splitt, shuffle=True, verbose=0)
                [mapetrainout]=pp.plot_history(histories,splitt,mapee)
                splitt=0
                batchsize=np.min([2**7,int(traindata.shape[0])])
                xb=train_data_normin
                xb0=trainlabel
                yb=trainout_data
                histories = model.fit([xb,xb0], yb,  batch_size=batchsize, epochs=int(EPOCHS/3),
                                validation_split=splitt, shuffle=True, verbose=0)
                [mapetrainout]=pp.plot_history(histories,splitt,mapee)
                        
            testout_data_normin=testout_data
        else:
            stringsave='rigidcfviv'+str(selectt)
            json_file = open(stringsave+'.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            
            loaded_model = tf.keras.models.model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights(stringsave+'.h5')
            print("Loaded model from disk")
             
            # evaluate loaded model on test data
            loaded_model.compile(loss='MAPE', optimizer='rmsprop', metrics=['MAPE'])
    #        score=loaded_model.evaluate(train_data_normin, trainout_data, verbose=0)
            testout_data_normin=testout_data
            score = loaded_model.evaluate(test_data, testout_data_normin, verbose=0)
            print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]))
            model=loaded_model
        """ TESTING"""
    
        
        xlinear1=np.linspace(min(testout_data),max(testout_data),30)
        ylinear1=xlinear1
        test_predictions = model.predict([test_data_normin,testlabel]).flatten()
        train_predictions = model.predict([train_data_normin,trainlabel]).flatten()
        fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        plt.xlabel('Measured output', fontsize = 20)
        plt.ylabel('Predicted output', fontsize = 20)        
        plt.xticks(fontsize=16)  
        plt.yticks(fontsize=16)
        plt.rc('font',family='Times New Roman')          
        plt.plot(testout_data[0:ttsize],test_predictions[0:ttsize],'b*',xlinear1,ylinear1, 'r', linewidth=2,markersize=10)
        plt.grid()
        if mapee>1:
            mape = model.evaluate([test_data,testlabel], testout_data_normin, verbose=0)
            mapetrain=model.evaluate([train_data,trainlabel], trainout_data_normin, verbose=0)
        else:
            mape = model.evaluate([test_data,testlabel], testout_data_normin, verbose=0)[1]
            mapetrain=model.evaluate([train_data,trainlabel], trainout_data_normin, verbose=0)[1]
        print(mape)
        print(mapetrain)
        if mapee>1 and testdata.shape[0]>ttsize:
            losstrend= model.evaluate([test_data[ttsize:test_data.shape[0],:],testlabel[ttsize:test_data.shape[0]]], testdata[ttsize:test_data.shape[0],selectrainout[0]])
        else:
            losstrend=0
        errormape=np.mean(abs(test_predictions[0:ttsize]-(testout_data[0:ttsize].flatten()))/(testout_data[0:ttsize].flatten()))*100
        out=[minn,losstrend,errormape,mape,mapetrain]
    
    
    ## EXTRAPOLATION constant x2
        if len(selectt)>1:
            ainput=np.linspace(0,0.4,30)
            fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)
            plt.xlabel(r'$X_1$', fontsize = 20)
            plt.ylabel('Y', fontsize = 20)        
            plt.xticks(fontsize=16)  
            plt.yticks(fontsize=16)
            plt.rc('font',family='Times New Roman')
            ainput2all=np.linspace(0.2,0.5,5)
            for ainput2 in ainput2all:
                ainputall=np.zeros((ainput.shape[0],2))
                ainputall[:,0]=ainput
                ainputall[:,1]=ainput2
                if ca==1 or len(selectt)==1:
                    aoutput = model.predict([ainput,0*ainput]).flatten()
                else:
                    aoutput = model.predict([ainputall,0*ainput]).flatten()
                plt.plot(ainput,aoutput+minn, linewidth=2,markersize=10)
            plt.grid()
            if ca==1 or len(selectt)==1:
                ax.legend(['Prediction'], fontsize = 20,loc='upper left')    #    plt.ylabel('Predicted '+inputt_descrp[0], fontsize = 16)        
            else:
                ax.legend(['Prediction'+r'$X_2=$'+str(xx) for xx in ainput2all], fontsize = 20,loc='upper left')    #    plt.ylabel('Predicted '+inputt_descrp[0], fontsize = 16)        


            num2plot=selectt
    
        
        ainput=np.linspace(0,0.4,30)
        ainput2=ainput+0#.25*np.sin(2*np.pi*ainput*10)*np.sign(ainput-0.25);
        for ii in range(0,len(ainput)):
            if ainput[ii]<0.25:
                ainput2[ii]=0
        ainputall=np.zeros((ainput.shape[0],2))
        ainputall[:,0]=ainput
        ainputall[:,1]=ainput2
        if ca==1 or len(selectt)==1:
            aoutput = model.predict([ainput,0*ainput]).flatten()
        else:
            aoutput = model.predict([ainputall,0*ainput]).flatten()
            
        fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111)
        plt.xlabel(r'$X_1$', fontsize = 20)
        plt.ylabel('Y', fontsize = 20)        
        plt.xticks(fontsize=16)  
        plt.yticks(fontsize=16)
        plt.rc('font',family='Times New Roman')          
        plt.plot(inputt[:,0],inputt[:,selectrainout]+minn,'g*',dataall[:,0],dataall[:,selectrainout],'r',ainput,aoutput+minn, 'b', linewidth=2,markersize=10)
        plt.grid()
        ax.legend(['Measurement','Desired','Prediction'], fontsize = 20,loc='upper left')    #    plt.ylabel('Predicted '+inputt_descrp[0], fontsize = 16)        
        plt.savefig('./pred'+'.png', dpi=300) 
        stringsavepenal='trendexpt'+str(ca)+str(space)+'wwee'+str(wwee)
        df=pd.concat([pd.DataFrame({'x0':traindataind[:,0]}),pd.DataFrame({'x1':traindataind[:,selectrainout[0]]}),pd.DataFrame({'preds':aoutput[:]}),pd.DataFrame({'xx':ainput[:]}),pd.DataFrame({'outs':out})], axis=1)
        if len(selectt)<2:
            df.to_csv(stringsavepenal+'.csv',index=False,sep=',')        
        else:
            df.to_csv(stringsavepenal+'2.csv',index=False,sep=',')        
    
