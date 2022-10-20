#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:40:09 2020

@author: wschien
"""
#%%
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from turtle import clone
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from glob import glob
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold
from MFCC import MFCC
from sklearn.svm import SVC,SVR
from model_LSTM import LSTM
from sklearn.neural_network import MLPClassifier
import torch
from sklearn.preprocessing import StandardScaler
from plot_cm import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
#%%
DataPath = '/home/huan/Desktop/DSP_lab4/2022dsplab-detecting-baby-sounds/Baby_Data'
class_name5 = {'Canonical':0,'Crying':1,'Junk':2,'Laughing':3,'Non-canonical':4}
class_name_key = {0:'Canonical',1:'Crying',2:'Junk',3:'Laughing',4:'Non-canonical'}
class_name2 = {'Canonical':0,'Crying':0,'Junk':1,'Laughing':0,'Non-canonical':0}

#%% Functions
def MFCC_feat(file):
    '''
    Please put your MFCC() function learned from lab9 here.
    '''
    # mfcc = MFCC(file,frame_length=512,frame_step=256,emphasis_coeff=0.95,num_FFT=512,num_bands=24)
    signal,sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=40,n_fft=512,hop_length=256,fmax = sr / 2.0)
    stft = np.abs(librosa.stft(signal,n_fft=512,hop_length=256,win_length=512))
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr),axis=1)
    chromagram=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr),axis=1)

    # melspectrogram=librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=40,n_fft=256,hop_length=256,fmax=sr/2)

    mean_mfcc, std_mfcc = np.mean(mfcc,axis=1),np.std(mfcc,axis=1)
    # mean_mel, std_mel = np.mean(melspectrogram,axis=1),np.std(melspectrogram,axis=1)
    LPC = librosa.lpc(y=signal, order=3)
    # mean_contrast = np.mean(contrast,axis=1)
    return np.concatenate((mean_mfcc,std_mfcc))
    #return np.concatenate((mean_mfcc,std_mfcc,LPC,contrast,chromagram))

def cross_val(cv=5,train_data=None, train_target2=None,train_target5=None,clf1=None,clf2=None):
    '''
    You can do cross validation here to find the best 'c' for training.
    '''
    Kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    y_dev_cv = []
    y_predict_cv = []
    for cvIdx, (trainIdx, devIdx) in enumerate(Kf.split(range(len(train_data)))):
        ######
        #CODE HERE
        train = train_data[trainIdx] 
        dev = train_data[devIdx]
        target2 = train_target2[trainIdx]
        target5 = train_target5[trainIdx]
        dev_target2 = train_target2[devIdx]
        dev_target5 = train_target5[devIdx]
        model1 = clone(clf1)
        model2 = clone(clf2)
        #選擇使用multilayer perceptron作為分類器模型
        model1.fit(train,target2)
        nojunk=np.where(model1.predict(train)==0)
        model2.fit(train[nojunk],target5[nojunk])

        y_predict = np.full(len(dev),2)
        nojunk = np.where(model1.predict(dev)==0)
        no_junk_predict=model2.predict(dev[nojunk])
        count = 0
        for i in range(len(y_predict)):
            if i in nojunk[0]:
                y_predict[i]=no_junk_predict[count]
                count+=1
        y_dev_cv.extend(train_target5[devIdx])
        y_predict_cv.extend(y_predict)
        f1_macro = f1_score(y_pred=y_predict,y_true=train_target5[devIdx],average='macro')    
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

longest = np.max([len(i) for i in data_all])
print(longest)

for i in range(len(data_all)):
    data_all[i] = np.pad(data_all[i],(0,longest-np.shape(data_all[i])[0]),'constant',constant_values=(0,0))
train_data = data_all


labels = pd.read_csv(os.path.join(DataPath, 'label_raw_train.csv'))
name2label = dict((row['file_name'], row['label']) for idx, row in labels.iterrows())
train_label_2 =  [ class_name2[ name2label[os.path.basename(path)] ] for path in train_path] 
train_label_5 =  [ class_name5[ name2label[os.path.basename(path)] ] for path in train_path] 

#%% Training SVM model
X_train = np.vstack(train_data)
y_train2 = np.array(train_label_2)
y_train5 = np.array(train_label_5)

#%%
print(np.shape(X_train))
sc =StandardScaler().fit(X_train)
X_train_norm=sc.transform(X_train)
pca =  PCA(n_components=50).fit(X_train_norm)
X_train_norm_reduce = pca.transform(X_train_norm)

clf1 = SVC(C=3,
    gamma='auto',
    class_weight='balanced',
    kernel='rbf',
    random_state=69)
clf2 = KNeighborsClassifier(12,weights='uniform',algorithm='auto',p=2)
# clf2 = SVC(C=5,
#     gamma='auto',
#     kernel='rbf',
#     random_state=69)
cross_val(cv=5,train_data=X_train_norm,train_target2=y_train2,train_target5=y_train5,clf1=clf1,clf2=clf2)
# clf = SVC(C=3,
#     gamma='auto',
#     class_weight='balanced',
#     kernel='rbf',
#     random_state=69)
# clf=RandomForestClassifier(
#     n_estimators = 500, 
#     criterion ='entropy',
#     warm_start = True,
#     max_features = 'sqrt',
#     oob_score = 'True', # more on this below
#     random_state=69  
# ) 
# clf = MLPClassifier(hidden_layer_sizes=(300,200),activation='relu',random_state=1,learning_rate='invscaling',learning_rate_init=1e-4 ,max_iter=1000)



#%%
clf = SVC(C=3,
    gamma='auto',
    class_weight='balanced',
    kernel='rbf',
    random_state=69)

clf.fit(X_train_norm_reduce,y_train)
#%%
data_dev_path = sorted(glob(os.path.join(DataPath, 'wav_dev', 'devel*.wav')))
data_dev = [MFCC_feat(path) for path in data_dev_path]
for i in range(len(data_dev)):
    data_dev[i] = np.pad(data_dev[i],(0,longest-np.shape(data_dev[i])[0]),'constant',constant_values=(0,0))
X_dev = np.vstack(data_dev)
X_dev_norm=sc.transform(X_dev)

X_dev_norm_reduce = pca.transform(X_dev_norm)
#%%
file_name=[p.split('/')[-1] for p in data_dev_path]
result = clf.predict(X_dev_norm_reduce)
predict_class=[class_name_key[r] for r in result]
dataframe = pd.DataFrame({'file_name':file_name,'Predicted':predict_class})
dataframe.to_csv("result.csv",index=False,)
#%%
# cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
# plot_confusion_matrix(cm , ['Non-canonical','Junk','Canonical','Crying','Laughing'])

#%%
# dataframe = pd.DataFrame({'True':y_test,'Predict':y_pred})
# dataframe.to_csv("result.csv",index=False,)