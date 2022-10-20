#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 22:40:09 2020

@author: wschien
"""
# %%
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# signal,sr = librosa.load('/home/huan/Desktop/DSP_lab4/2022dsplab-detecting-baby-sounds/Baby_Data/wav_train/train_0028.wav')
# plt.figure(figsize=(30, 5))
# librosa.display.waveshow(signal,x_axis='ms', sr=sr)
# zr = np.mean(librosa.feature.zero_crossing_rate(signal,frame_length=512, hop_length=256, center=True),axis=1)
# print(zr)
#%%
from fishervector import FisherVectorGMM
from Lab4_108061217_functions import fisher_vector
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from glob import glob
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from plot_cm import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import random
import opensmile
from sklearn.neural_network import MLPClassifier
import pandas as pd
smile = opensmile.Smile(
    # feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_set=opensmile.FeatureSet.emobase,
    feature_level=opensmile.FeatureLevel.Functionals,
)
smile2 = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    # feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
smile3 = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
#%%
DataPath = '/home/huan/Desktop/DSP_lab4/2022dsplab-detecting-baby-sounds/Baby_Data'
class_name = {'Canonical':0,'Crying':1,'Junk':2,'Laughing':3,'Non-canonical':4}
class_name_key = {0:'Canonical',1:'Crying',2:'Junk',3:'Laughing',4:'Non-canonical'}
#%% Functions
def SmileExtract(file):
    y = smile.process_file(file).to_numpy()
    y2 = smile2.process_file(file).to_numpy()
    y3 = smile3.process_file(file).to_numpy()
    y = np.concatenate((y,y2,y3),axis=1)
    return y[0]
def MFCC_feat(file,augement=0):
    '''
    Please put your MFCC() function learned from lab9 here.
    '''
    # mfcc = MFCC(file,frame_length=512,frame_step=256,emphasis_coeff=0.95,num_FFT=512,num_bands=24)
    signal,sr = librosa.load(file)
 
    mfcc = librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=39,n_fft=512,hop_length=256,fmax = sr / 2.0)
    stft = np.abs(librosa.stft(signal,n_fft=512,hop_length=256))
    
    mean_mfcc, std_mfcc = np.mean(mfcc,axis=1),np.std(mfcc,axis=1)

   
    return np.concatenate((mfcc,stft),axis=0)#np.concatenate((mean_mfcc,std_mfcc,np.mean(stft,axis=1)))
    #return np.concatenate((mean_mfcc,std_mfcc,LPC,contrast,chromagram))

def cross_val(cv=5,train_data=None, train_target=None,clf=None):
    '''
    You can do cross validation here to find the best 'c' for training.
    '''
    Kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    y_dev_cv = []
    y_predict_cv = []
    for cvIdx, (trainIdx, devIdx) in enumerate(Kf.split(range(len(train_data)))):
        ######
        #CODE HERE
        model = clone(clf)
        #選擇使用multilayer perceptron作為分類器模型
        model.fit(train_data[trainIdx],train_target[trainIdx])
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
# data_all = [MFCC_feat(path) for path in data_all_path]
data_all = [SmileExtract(path) for path in data_all_path]
longest = np.max([len(i) for i in data_all])
print(longest)

for i in range(len(data_all)):
    data_all[i] = np.pad(data_all[i],(0,longest-np.shape(data_all[i])[0]),'constant',constant_values=(0,0))
train_data = data_all


# longest1 = np.max([np.shape(i)[0] for i in data_all])
# longest2 = np.max([np.shape(i)[1] for i in data_all])
# for i in range(len(data_all)):
#     data_all[i] = np.pad(data_all[i],((0,longest1-np.shape(data_all[i])[0]),(0,longest2-np.shape(data_all[i])[1])),'constant',constant_values=(1e-6,1e-6))
train_data = data_all
labels = pd.read_csv(os.path.join(DataPath, 'label_raw_train.csv'))
name2label = dict((row['file_name'], row['label']) for idx, row in labels.iterrows())
train_label =  [ class_name[ name2label[os.path.basename(path)] ] for path in train_path]  


#%% Training SVM model
X_train = np.vstack(train_data)
y_train = np.array(train_label)


# X_train = np.array(train_data)
# fv_gmm = FisherVectorGMM(n_kernels=128).fit(X_train)
# fv = fv_gmm.predict(X_train)
# X_train_all = np.concatenate((np.mean(X_train,axis=2),np.std(X_train,axis=2),np.mean(fv,axis=2)),axis=1)

#%%
sc=StandardScaler().fit(X_train)
X_train_norm=sc.transform(X_train)
print(np.shape(X_train))
# pca = PCA(n_components=3900).fit(X_train_norm)
# X_train_norm_reduce = pca.transform(X_train_norm)
clf = BaggingClassifier(base_estimator=SVC(C=3,gamma='auto',class_weight='balanced',kernel='rbf',random_state=69),n_estimators=5, random_state=0)

# clf = KNeighborsClassifier(8,weights='uniform',algorithm='auto',p=2)
cross_val(cv=5,train_data=X_train_norm,train_target=y_train,clf=clf)
# clf = KNeighborsClassifier(8,weights='uniform',algorithm='auto',p=2)
# clf = SVC(C=3,
#     gamma='auto',
#     class_weight='balanced',
#     kernel='rbf',
#     random_state=69)
# clf = MLPClassifier(hidden_layer_sizes=(300,200),activation='relu',random_state=1,learning_rate='invscaling',learning_rate_init=1e-4 ,max_iter=1000)


#%%
clf = BaggingClassifier(base_estimator=SVC(C=3,gamma='auto',class_weight='balanced',kernel='rbf',random_state=69),n_estimators=5, random_state=0)

clf.fit(X_train_norm,y_train)
del X_train_norm
del X_train
del data_all
#%%
data_dev_path = sorted(glob(os.path.join(DataPath, 'wav_dev', 'devel*.wav')))
data_dev = [SmileExtract(path) for path in data_dev_path]
for i in range(len(data_dev)):
    data_dev[i] = np.pad(data_dev[i],(0,longest-np.shape(data_dev[i])[0]),'constant',constant_values=(0,0))
X_dev = np.vstack(data_dev)
X_dev_norm=sc.transform(X_dev)

# X_dev_norm_reduce = pca.transform(X_dev_norm)
#%%
file_name=[p.split('/')[-1] for p in data_dev_path]
result = clf.predict(X_dev_norm)
predict_class=[class_name_key[r] for r in result]
dataframe = pd.DataFrame({'file_name':file_name,'Predicted':predict_class})
dataframe.to_csv("result.csv",index=False,)
#%%
# cm = confusion_matrix(y_true=y_test,y_pred=y_pred)
# plot_confusion_matrix(cm , ['Non-canonical','Junk','Canonical','Crying','Laughing'])

#%%
# dataframe = pd.DataFrame({'True':y_test,'Predict':y_pred})
# dataframe.to_csv("result.csv",index=False,)