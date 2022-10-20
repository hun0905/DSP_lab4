#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:40:09 2020

@author: wschien
"""
#%%
from glob import glob
import os
from re import X
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from MFCC import MFCC
from sklearn.svm import SVC
from model_CNNclassifier import CNN_classifier
import torch
from sklearn.preprocessing import StandardScaler
from plot_cm import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn.metrics import f1_score
import copy
#%%
DataPath = '/home/huan/Desktop/DSP_lab4/2022dsplab-detecting-baby-sounds/Baby_Data'
class_name = {'Canonical':0,'Crying':1,'Junk':2,'Laughing':3,'Non-canonical':4}
class_name_key = {0:'Canonical',1:'Crying',2:'Junk',3:'Laughing',4:'Non-canonical'}
#%% Functions
def MFCC_feat(file):
    '''
    Please put your MFCC() function learned from lab9 here.
    '''
    # mfcc = MFCC(file,frame_length=512,frame_step=256,emphasis_coeff=0.95,num_FFT=512,num_bands=24)
    signal,sr = librosa.load(file,sr=22050)
    LMFB = librosa.feature.melspectrogram(y=signal,sr=sr,n_fft=512,n_mels=40,hop_length=256,fmax = sr / 2.0)
    stft = np.abs(librosa.stft(signal,n_fft=256,hop_length=128,win_length=256))
    # contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    # chromagram=np.mean(librosa.feature.chroma_stft(S=stft_spectrogram, sr=sr),axis=1)
    return stft

def cross_val(cv=5,train_data=None, train_target=None,batch_size=32,epoch=20,lr=1e-4):
    '''
    You can do cross validation here to find the best 'c' for training.
    '''
    Kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    y_dev_cv = []
    y_predict_cv = []
    for cvIdx, (trainIdx, devIdx) in enumerate(Kf.split(range(len(train_data)))):
        ######
        #CODE HERE
        model = CNN_classifier(1,128,128,5).cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr = lr,weight_decay=0.0)
        model.fit(trainData=train_data[trainIdx],trainlabel=train_target[trainIdx],batch_size=256,use_cuda=True,device='cuda',epoch=epoch,optimizer=optimizer)
        #選擇使用multilayer perceptron作為分類器模型
        y_predict=model.predict(train_data[devIdx])
        y_dev_cv.extend(train_target[devIdx])
        
        y_predict_cv.extend(y_predict)
       
        f1_macro = f1_score(y_pred=y_predict,y_true=train_target[devIdx],average='macro')    
        f1_weighted = f1_score(y_pred=y_predict_cv,y_true=y_dev_cv,average='weighted')    
        print('f1 score(macro) = ', f1_macro)
        print('f1 score(weighted) = ', f1_weighted)
        '''
        clf 為所使用的classifier,這裡最初選用svc classifier,但為了達成更高accuracy,選擇了使用multilayer perceptron.
        將資料分成training set 和validation,trainIdx表示training set的index
        而devIdx表示validation set 的index,利用training set進行訓練後在對
        validation set作預測,把預測的結果和真正的結果分別存在y_predict_cv和y_dev_cv
        '''
        ######
        pass
    y_predict_cv = np.array(y_predict_cv).flatten()
    y_dev_cv = np.array(y_dev_cv).flatten()
    f1_macro = f1_score(y_pred=y_predict_cv,y_true=y_dev_cv,average='macro')  
    f1_weighted = f1_score(y_pred=y_predict_cv,y_true=y_dev_cv,average='weighted')     
    print('ALL fscore(macro) = ',  f1_macro)
    print('ALL fscore(weighted) = ',  f1_weighted)
    cm = confusion_matrix(y_true=y_dev_cv,y_pred=y_predict_cv)
    plot_confusion_matrix(cm , ['Canonical','Crying','Junk','Laughing','Non-canonical'])

#%% Loading training and test data
data_all_path = sorted(glob(os.path.join(DataPath, 'wav_train', 'train*.wav')))
train_path = data_all_path
# test_path = data_all_path[3500:]
data_all = [MFCC_feat(path) for path in data_all_path]

#%%pad data
# longest1 = np.max([np.shape(i)[0] for i in data_all])
# longest2 = np.max([np.shape(i)[1] for i in data_all])
# for i in range(len(data_all)):
#     data_all[i] = [np.pad(data_all[i],((0,longest1-np.shape(data_all[i])[0]),(0,longest2-np.shape(data_all[i])[1])),'constant',constant_values=(0,0))]
# train_data = np.array(data_all)
longest = np.max([len(i) for i in data_all])
print(longest)

for i in range(len(data_all)):
    data_all[i] = np.pad(data_all[i],(0,longest-np.shape(data_all[i])[0]),'constant',constant_values=(0,0))
train_data = data_all
labels = pd.read_csv(os.path.join(DataPath, 'label_raw_train.csv'))
name2label = dict((row['file_name'], row['label']) for idx, row in labels.iterrows())
train_label =  [ class_name[ name2label[os.path.basename(path)] ] for path in train_path] 



#%% Training SVM model
# X_train = np.vstack(train_data)
y_train = np.array(train_label)
#%%

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(train_data.reshape(-1, train_data.shape[-1])).reshape(train_data.shape)


# optimizer = torch.optim.Adam(clf.parameters(),lr = 1e-4,weight_decay=0.0)

cross_val(cv=5,train_data=X_train_norm,train_target=y_train,batch_size=32,epoch=20,lr=1e-4)
#%%
clf = CNN_classifier(1,128,128,5).cuda()
optimizer = torch.optim.Adam(clf.parameters(),lr = 1e-4,weight_decay=0.0)
clf.fit(X_train_norm,y_train,batch_size=256,epoch=20,optimizer=optimizer)

data_dev_path = sorted(glob(os.path.join(DataPath, 'wav_dev', 'devel*.wav')))
data_dev = [MFCC_feat(path) for path in data_dev_path]
for i in range(len(data_dev)):
    data_dev[i] = [np.pad(data_all[i],((0,longest1-np.shape(data_dev[i])[0]),(0,longest2-np.shape(data_dev[i])[1])),'constant',constant_values=(0,0))]
X_dev = np.vstack(data_dev)
X_dev_norm = scaler.transform(X_dev.reshape(-1, X_dev.shape[-1])).reshape(X_dev.shape)
file_name=[p.split('/')[-1] for p in data_dev_path]
result = clf.predict(X_dev_norm)
predict_class=[class_name_key[r] for r in result]
dataframe = pd.DataFrame({'file_name':file_name,'Predicted':predict_class})
dataframe.to_csv("result.csv",index=False,)

