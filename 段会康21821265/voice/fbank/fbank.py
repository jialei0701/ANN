# -*- coding:utf-8 -*-
"""
   该脚本用于提取语音特征，包括MFCC、FBANK以及语谱图特征；
"""

from python_speech_features import mfcc, delta, logfbank
import wave
import numpy as np
from scipy.fftpack import fft
import os


def read_wav_data(filename):
    '''
    获取文件数据以及采样频率；
    输入为文件位置，输出为wav文件数学表示和采样频率；
    '''
    wav = wave.open(filename, 'rb')
    num_frames = wav.getnframes()
    num_channels = wav.getnchannels()
    framerate = wav.getframerate()
    str_data = wav.readframes(num_frames)
    wav.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    # wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data.shape = -1, num_channels
    wave_data = wave_data.T
    return wave_data, framerate


def get_mfcc_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率，输出为语音的MFCC特征+一阶差分+二阶差分；
    '''
    feat_mfcc = mfcc(wavsignal, fs)
    print(feat_mfcc)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature


def get_fbank_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率，输出为语音的FBANK特征+一阶差分+二阶差分；
    '''
    feat_fbank = logfbank(wavsignal, fs, winlen=0.032, winstep=0.016, nfilt=40)
    feat_fbank_d = delta(feat_fbank, 2)
    feat_fbank_dd = delta(feat_fbank_d, 2)
    wav_feature = np.column_stack((feat_fbank, feat_fbank_d, feat_fbank_dd))
    return wav_feature


def get_frequency_feature(wavsignal, fs):
    '''
    输入为wav文件数学表示和采样频率,输出为语谱图特征，特征维度是200；
    '''
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))
    time_window = 25
    wav_array = np.array(wavsignal)
    wav_length = wav_array.shape[1]
    first2end = int(len(wavsignal[0]) / fs * 1000 - time_window) // 10
    data_input = np.zeros(shape=[first2end, 200], dtype=np.float)
    for i in range(0, first2end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_array[0, p_start:p_end]
        data_line = data_line * w
        data_line = np.abs(fft(data_line)) / wav_length
        data_input[i] = data_line[0:200]
    data_input = np.log(data_input)
    return data_input


def cut_and_fb(file_path, folder_path):

    print(file_path)
    wavsignal, fs = read_wav_data(file_path)
    # print(fs)
    # print(wavsignal.shape)

    num_seg = (wavsignal.shape[1] - 16064) // 16064
    # print(num_seg)
    for i in range(num_seg):
        st_point = i * 16064
        ed_point = (i + 2) * 16064
        # print(st_point, ed_point)
        segment_001 = wavsignal[0][st_point:ed_point].reshape((1, -1))
        # print(segment_001.shape)

        b = get_fbank_feature(segment_001, fs)
        # print(b.shape)

        str1 = "_" + '{:04d}'.format(i)
        save_path = folder_path + "/" + file_path[-17:-10] + str1 + ".txt"
        # print(save_path)
        np.savetxt(save_path, b)


if __name__ == '__main__':

    input_path = "D:/sre16/aftervad"
    output_path = "D:/vscode/voice/fbank/data"
    enrollment_path = "sre16_eval_enrollment.txt"
    namelist = []
    with open('sre16_eval_enrollment.txt', 'r') as f:
        name_list = f.readlines()
        print(len(name_list))

    pre_id = ""
    for str in name_list:
        # print(str[0:4], str[11:24])
        now_id = str[0:4]
        if pre_id != now_id:
            # create folder
            pre_id = now_id
            folder_path = output_path + "/" + now_id
            os.makedirs(folder_path)

        wav_input_path = input_path + "/" + str[11:24] + ".wav"
        folder_path = output_path + "/" + now_id
        cut_and_fb(wav_input_path, folder_path)
