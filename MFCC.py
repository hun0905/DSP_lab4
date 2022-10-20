from random import sample
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from librosa.filters import mel as librosa_mel_fn
from scipy.fftpack import dct
def pre_emphasis(signal, coefficient = 0.95):
    return np.append(signal[0], signal[1:] - coefficient*signal[:-1])

def STFT(time_signal, num_frames, num_FFT, frame_step, frame_length, signal_length, verbose=False):
    padding_length = int((num_frames - 1) * frame_step + frame_length)
    padding_zeros = np.zeros((padding_length - signal_length,))
    padded_signal = np.concatenate((time_signal, padding_zeros))#填充要確保能分成整數個frame，且每個frame等長
    # split into frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_length, 1)).T
    indices = np.array(indices,dtype=np.int32) ###找出對應到每個frame的index
    # slice signal into frames
    frames = padded_signal[indices]
    # apply window to the signal
    frames *= np.hamming(frame_length)
    # FFT
    complex_spectrum = np.fft.rfft(frames, num_FFT).T
    absolute_spectrum = np.abs(complex_spectrum)
    
    if verbose:
        print('Signal length :{} samples.'.format(signal_length))
        print('Frame length: {} samples.'.format(frame_length))
        print('Frame step  : {} samples.'.format(frame_step))
        print('Number of frames: {}.'.format(len(frames)))
        print('Shape after FFT: {}.'.format(absolute_spectrum.shape))

    return absolute_spectrum

def mel2hz(mel):
    '''
    Transfer Mel scale to Hz scale
    '''
    ###################
    # YOUR CODE HERE
    hz=(10**(mel/2595)-1)*700
    '''
    hz=(10**(mel/2595)-1)*700 就是 f = (10^(Mel/2595)-1)*700 將單位由Mel對應至hz
    '''
    ###################
    
    return hz

def hz2mel(hz):
    '''
    Transfer Hz scale to Mel scale
    '''
    ###################
    # YOUR CODE HERE
    mel=2595*np.log10(1+hz/700)
    '''
    mel=2595*np.log10(1+hz/700) 就是Mel =  2595*log_10(1+f/7000)的function 用來將單位由hz對應至Mel
    '''
    ###################
    return mel

def get_filter_banks(num_filters, num_FFT, sample_rate, freq_min = 0, freq_max = None):
    ''' Mel Bank
    num_filters: filter numbers
    num_FFT: number of FFT quantization values
    sample_rate: as the name suggests
    freq_min: the lowest frequency that mel frequency include
    freq_max: the Highest frequency that mel frequency include
    '''
    # convert from hz scale to mel scale
    low_mel = hz2mel(freq_min)
    high_mel = hz2mel(freq_max)
    # define freq-axis
    mel_freq_axis = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_freq_axis = mel2hz(mel_freq_axis)
    # Mel triangle bank design (Triangular band-pass filter banks)
    ##bin是每個頻點的頻率數，也就是sample_rate/num_FFT。代表每個FFT的點的頻率數。
    bins = np.floor((num_FFT + 1) * hz_freq_axis / sample_rate).astype(int) #=hz_freq_axis*num_FFT/sample_rate;hz_freq_axis所對到的FFT的點
    fbanks = np.zeros((num_filters, int(num_FFT / 2 + 1)))
    ###################
    # YOUR CODE HERE
    for i in range(num_filters):
        fbanks[i][bins[i]:bins[i+2]] = np.concatenate((np.linspace(0,1,bins[i+1]-bins[i]),np.linspace(1,0,bins[i+2]-bins[i+1])))
    '''
    總共要產生i個filter,在fbanks[i]存放第i個filter的頻域的數值
    要在第i個filter要在 bins[i]到bins[i+2]之間產生以bins[i+1]為peak的三角波
    np.linspace(0,1,bins[i+1]-bins[i])在bins[i]到bins[i+1]-1間產生由0-1的等量遞增的數列
    np.linspace(1,0,bins[i+2]-bins[i+1])在bins[i+1]到bins[i+2]-間產生由0-1的等量遞減的數列
    把np.linspace(0,1,bins[i+1]-bins[i])和np.linspace(1,0,bins[i+2]-bins[i+1])合併成一個三角波
    '''
    ###################
    return fbanks
def MFCC(path,frame_length=512,frame_step=256,emphasis_coeff=0.95,num_FFT=512,num_bands=12):
    filename = path
    source_signal, sr = sf.read(filename) #sr:sampling rate
    ### hyper parameters
    freq_min = 0
    freq_max = int(0.5 * sr)                #Nyquist
    signal_length = len(source_signal)    # Signal length
    # # number of frames it takes to cover the entirety of the signal
    num_frames = 1 + int(np.ceil((1.0 * signal_length - frame_length) / frame_step)) #frame_length-frame_step是每次重疊的，在分的過程，通常相鄰兩個會設一定量的重疊。前面的1是要確保無論如何都有1個frame
    # ##########################
    # '''
    # Part I:
    # (1) Perform STFT on the source signal to obtain one spectrogram (with the provided STFT() function)
    # (2) Pre-emphasize the source signal with pre_emphasis()
    # (3) Perform STFT on the pre-emphasized signal to obtain the second spectrogram
    # (4) Plot the two spectrograms together to observe the effect of pre-emphasis

    # hint for plotting:
    # you can use "plt.subplots()" to plot multiple figures in one.
    # you can use "axis.pcolor" of matplotlib in visualizing a spectrogram. 
    # '''
    # #YOUR CODE STARTS HERE:

    # source_spectrum=STFT(source_signal,num_frames,num_FFT,frame_step,frame_length,signal_length)
    pre_emphasis_signal= pre_emphasis(source_signal,emphasis_coeff)
    pre_emphasis_spectrum= STFT(pre_emphasis_signal,num_frames,num_FFT,frame_step,frame_length,signal_length)
    '''
    source_spectrum:是原始的signal 經過frame blocking和windowing之後再由Fast Fourier Transform轉換至頻域的表示
    pre_emphasis_signal:原始signal 經過pre-emphasis後改變頻率的分亮後所產生的signal
    pre_emphasis_spectrum:是pre_emphasis_signal經過frame blocking和windowing之後再由Fast Fourier Transform轉換至頻域的表示
    '''
    # #YOUR CODE ENDS HERE;
    # ##########################

    # '''
    # Head to the import source 'Lab1_functions_student.py' to complete these functions:
    # mel2hz(), hz2mel(), get_filter_banks()
    # '''
    # # get Mel-scaled filter
    # fbanks = get_filter_banks(num_bands, num_FFT , sr, freq_min, freq_max)
    # ##########################
    # '''
    # Part II:
    # (1) Convolve the pre-emphasized signal with the filter
    # (2) Convert magnitude to logarithmic scale
    # (3) Perform Discrete Cosine Transform (dct) as a process of information compression to obtain MFCC
    #     (already implemented for you, just notice this step is here and skip to the next step)
    # (4) Plot the filter banks alongside the MFCC
    # '''
    # #YOUR CODE STARTS HERE:
    fbanks = get_filter_banks(num_bands, num_FFT , sr, freq_min, freq_max)
    features=np.dot(fbanks,pre_emphasis_spectrum).T
    features_log = np.log(features)
    MFCC = dct(features_log, norm = 'ortho')[:,:num_bands].T
    # # step(3): Discrete Cosine Transform 
    # MFCC = dct(features, norm = 'ortho')[:,:num_bands]
    # # equivalent to Matlab dct(x)
    # # The numpy array [:,:] stands for everything from the beginning to end.

    '''
    features=np.dot(fbanks,pre_emphasis_spectrum).T features是存放pre_emphasis_spectrum經過了filter banks中各個filter之後所取得的頻譜
    features_log 是將feature的所有數值直接取log得到的結果
    MFCC = dct(features_log, norm = 'ortho')[:,:num_bands].T 是將取log的數值經過dct後取得MFCC features
    '''
    # #YOUR CODE ENDS HERE;
    # ##########################
    return MFCC
