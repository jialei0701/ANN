import librosa
import librosa.core
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import json
import math

from sklearn.preprocessing import StandardScaler


"""
hyper-parameters for music data
"""
hop_length = 512
window_length = hop_length * 2
fps = 25
spf = 0.04  # 40 ms
sample_rate = 44100  #
resample_rate = hop_length * fps

data_dir = '../data/'

class Music:

    '''
    @:param
        path:   music file path
        sr:     music sample rate
        start:  start offset(s) of the music
        end:    end offset(s) of the music
    '''
    def __init__(self, path, sr, start, duration):
        self.path = path
        self.start = start
        self.music_data, self.sr = librosa.core.load(path=path, sr=sr, offset=start,duration=duration) # 210秒
        librosa.output.write_wav(path=path.replace("mp3","wav"),y=self.music_data, sr=self.sr)

    def print(self):
        print(self.path)
        print(self.music_data)

    def draw_wav(self):
        plt.figure()
        plt.subplot(3, 1, 1)
        librosa.display.waveplot(self.music_data, sr=self.sr)
        plt.title('{0} wave plot'.format(self.path))


    def draw_onset(self):
        plt.subplot(3, 1, 2)
        librosa.display.waveplot(self.music_data, sr=self.sr)
        plt.title('{0} wave plot'.format(self.path))
        plt.show()

    def extract_features(self):
        # 40ms / frame
        mel_spectrum = librosa.feature.melspectrogram(y=self.music_data, sr=self.sr,n_fft=window_length, hop_length=hop_length)

        mfcc = librosa.feature.mfcc(S=mel_spectrum,n_mfcc=20) # mfcc[3]
        mfcc_delta = librosa.feature.delta(mfcc) # mfcc_delta[3]

        cqt_chroma = librosa.feature.chroma_cqt(y=self.music_data, sr=self.sr, hop_length=hop_length,tuning=0,n_chroma=4) # cqt_chroma[4]
        onset_envelope = librosa.onset.onset_strength(S=mel_spectrum) # onset_envelope[1]

        tempogram = librosa.feature.tempogram(win_length=5, onset_envelope=onset_envelope)  # tempogram[5]

        tempo,beats = librosa.beat.beat_track(y=self.music_data, sr = self.sr,hop_length=hop_length)
        temporal_indexes_1 = np.array([i for i in range(mel_spectrum.shape[1])])
        temporal_indexes_2 = np.array([1 if i in set(beats) else 0  for i in range(mel_spectrum.shape[1])])
        temporal_indexes_3 = np.array(temporal_indexes_2.copy())
        in_frame_count=0
        for i in range(len(temporal_indexes_3)):

            if temporal_indexes_3[i] == 1:
                temporal_indexes_3[i] = 0
                in_frame_count = 1
            else:
                temporal_indexes_3[i] = in_frame_count
                in_frame_count += 1

        temporal_indexes = np.vstack((temporal_indexes_1[:-beats[0]], temporal_indexes_2[beats[0]:], temporal_indexes_3[beats[0]:]))
        acoustic_features = np.vstack((mfcc, mfcc_delta, cqt_chroma, tempogram, onset_envelope))
        acoustic_features = acoustic_features[:, beats[0]:]

        return acoustic_features.transpose(), temporal_indexes.transpose() # feature[16]

        pass



def load_start_end_frame_num(config_fp):
    with open(config_fp, 'r') as f:
        data = json.load(f)
        start = data["start_position"]
        end = data["end_position"]
        return start,end
    pass




'''
Get frame len, center array, skeletons array
'''
def load_skeleton(skeleton_json):
    with open(skeleton_json, 'r') as f:
        data = json.load(f)
        return data['length'],data['center'],data['skeletons']
    pass

def audio_feature_extract(data_dir):

    config_path = os.path.join(data_dir, "config.json")
    skeleton_path = os.path.join(data_dir, "skeletons.json")
    music_path = os.path.join(data_dir,"audio.mp3")

    acoustic_features_path = os.path.join(data_dir, "acoustic_features.npy")
    temporal_indexes_path = os.path.join(data_dir, "temporal_features.npy")

    if  os.path.exists(acoustic_features_path) and os.path.exists(temporal_indexes_path):
        print("load from %s and %s" % (acoustic_features_path, temporal_indexes_path))
        return np.load(acoustic_features_path), np.load(temporal_indexes_path)


    start_frame, end_frame = load_start_end_frame_num(config_fp=config_path) # frame num
    duration,_,_ = load_skeleton(skeleton_json=skeleton_path)

    print("%s %d" % (data_dir,duration))
    music = Music(music_path, sr=resample_rate, start=start_frame / fps, duration=(duration-1) / fps) # 25fps


    acoustic_features, temporal_indexes = music.extract_features()  # 16 dim

    np.save(acoustic_features_path, acoustic_features)
    np.save(temporal_indexes_path, temporal_indexes)

    return acoustic_features, temporal_indexes

def rotate_one_skeleton_by_axis(skeleton, axis, angle):
    delta_x = skeleton[0] - axis[0]
    delta_z = skeleton[2] - axis[2]
    skeleton_new = skeleton
    skeleton_new[0] = delta_x * math.cos(angle) + delta_z * math.sin(angle)
    skeleton_new[2] = -delta_x * math.sin(angle) + delta_z * math.cos(angle)


    return skeleton_new

def rotate_skeleton(frames):

    frames = np.asarray(frames)

    for i in range(len(frames)):
        this_frame = frames[i]
        waist_lf = this_frame[16]
        waist_rt = this_frame[7]

        axis = this_frame[2]

        lf = waist_lf - axis
        rt = waist_rt - axis
        mid = lf+rt

        theta = math.atan2(mid[2], mid[0]) # from x+ axis

        for j in range(len(this_frame)):
            frames[i][j] =  rotate_one_skeleton_by_axis(this_frame[j], axis, theta)
            frames[i][j] =  rotate_one_skeleton_by_axis(this_frame[j], axis, -math.pi/2) # turn to y- axis

    return frames


def motion_feature_extract(data_dir, with_rotate, with_centering):
    skeleton_path = os.path.join(data_dir, "skeletons.json")
    duration, center, frames = load_skeleton(skeleton_json=skeleton_path)

    # length * 3
    # length * skeletons * 3

    center = np.asarray(center)
    frames = np.asarray(frames)

    if with_centering:
        for i in range(len(frames)):
            for j in range(len(frames[i])):
                frames[i][j] -= center[i]

    if with_rotate:
        frames = rotate_skeleton(frames)



    frames = frames.reshape(len(frames), -1)
    return frames

    pass

if __name__ == '__main__':
    All_dirs = os.listdir(data_dir)
    C_dirs = []
    R_dirs = []
    T_dirs = []
    W_dirs = []
    for one in All_dirs:
        if one.split('_')[1] == 'C':
            C_dirs.append(one)
            pass
        elif one.split('_')[1] == 'R':
            R_dirs.append(one)
            pass
        elif one.split('_')[1] == 'T':
            T_dirs.append(one)
            pass
        elif one.split('_')[1] == 'W':
            W_dirs.append(one)
            pass

    for one in C_dirs:
        one_dir = os.path.join(data_dir, one)
        motion_features = motion_feature_extract(one_dir)
        acoustic_features = audio_feature_extract(one_dir).transpose()  # dim 16 [n_features, n_samples]

        scaler = StandardScaler()
        scaler.fit(X=acoustic_features)
        trans_data = scaler.transform(acoustic_features)
        print(acoustic_features.shape)


    pass

