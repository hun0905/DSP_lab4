# signal,sr = librosa.load('/home/huan/Desktop/DSP_lab4/2022dsplab-detecting-baby-sounds/Baby_Data/wav_train/train_0014.wav')
# plt.figure(figsize=(30, 5))
# librosa.display.waveshow(signal,x_axis='ms', sr=sr)
# f0, voiced_flag, voiced_probs = librosa.pyin(signal,frame_length=512,hop_length=256,
#     fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'),fill_na=0)
# print(np.mean(f0,axis=0))
# %%
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
import pandas as pd

class_name = {'Canonical':0,'Crying':1,'Junk':2,'Laughing':3,'Non-canonical':4}
def MFCC_feat(file):
    '''
    Please put your MFCC() function learned from lab9 here.
    '''
    # mfcc = MFCC(file,frame_length=512,frame_step=256,emphasis_coeff=0.95,num_FFT=512,num_bands=24)
    signal,sr = librosa.load(file)
    mfcc = librosa.feature.mfcc(y=signal,sr=sr,n_mfcc=40,n_fft=512,hop_length=256,fmax = sr / 2.0)
    mean_mfcc, std_mfcc = np.mean(mfcc,axis=1),np.std(mfcc,axis=1)
    return [mean_mfcc,std_mfcc]
# %%
DataPath = '/home/huan/Desktop/DSP_lab4/2022dsplab-detecting-baby-sounds/Baby_Data'
data_all_path = sorted(glob(os.path.join(DataPath, 'wav_train', 'train*.wav')))
# data_all = [MFCC_feat(path) for path in data_all_path]
names = [os.path.basename(p) for p in data_all_path]

labels = pd.read_csv(os.path.join(DataPath, 'label_raw_train.csv'))

name2label = dict((row['file_name'], row['label']) for idx, row in labels.iterrows())

train_label =  np.array([  class_name[name2label[os.path.basename(path)]]  for path in data_all_path])
Canonical=np.where(train_label==0)[0]
Crying=np.where(train_label==1)[0]
Junk=np.where(train_label==2)[0]
Laughing=np.where(train_label==3)[0]
Non_canonical=np.where(train_label==4)[0]
# %%
data_all = [MFCC_feat(path) for path in data_all_path]

# %%
# plt.scatter(data_all[,0],data_all[,1])
# plt.scatter(data_all[,0],data_all[,1])
# plt.scatter(data_all[,0],data_all[,1])
# plt.scatter(data_all[,0],data_all[,1])
# plt.scatter(data_all[,0],data_all[,1])
# plt.show()
# %%
