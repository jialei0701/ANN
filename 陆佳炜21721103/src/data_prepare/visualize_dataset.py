import numpy as np
import cv2

import json
import subprocess
import math
from moviepy.editor import *
from moviepy.video import  VideoClip
from data_prepare.feature_extract import rotate_skeleton

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

with_rotate = True
to_video = True
show_ground_truth = True
show_pred = True
show_bones = 21
skeleton_target_color = (0, 0, 0) # Line color
skeleton_pred_color = (0, 0, 0) # Line color
bone_pred_color = (255, 0, 0)
bone_target_color = (0, 0, 255)
fps = 25
scale = 3
CANVAS_SIZE = (scale * 300,scale * 200,3)


dance_type = 'C'
index = 9
root_dir = '../data/DANCE_%c_%d/' %(dance_type,index)

# model_name = 'LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_TempoNor_Threshold_0.060_Masking_Reduced_10'
model_name = 'LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_TempoNor_Threshold_0.100_Masking_Reduced_10'
# model_name = 'LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_TempoNor_Threshold_0.200_Masking_Reduced_10'
# model_name = 'LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_TempoNor_Threshold_0.400_Masking_Reduced_10'



# model_name = 'LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_Threshold_0.015_Masking_Reduced_10'
# model_name = 'LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_Threshold_0.030_Masking_Reduced_10'
# model_name = 'LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_Threshold_0.045_Masking_Reduced_10'
# model_name = 'LSTM-AE_rotate_Ortho_Leaky_Temporal_InputSize_50_Seq_120_Threshold_0.030_Reduced_10'

pred_path = root_dir + '%s.json' % model_name
audio_path = root_dir +'audio.mp3'
target_path = root_dir +'skeletons.json'
pred_video = root_dir +'output.mp4'
pred_video2 = root_dir +'%s.mp4' % model_name
config_path = root_dir +'config.json'
tempo_path = root_dir+'temporal_features.npy'


def draw_hints(cvs):
    cv2.putText(cvs, 'Prediction' , (CANVAS_SIZE[0]//4 - scale * 25,  scale * 15), cv2.FONT_ITALIC, scale * 0.3, bone_pred_color, 1,
                False)
    cv2.putText(cvs, 'Ground Truth' , (CANVAS_SIZE[0]*3//4 - scale * 25, scale * 15), cv2.FONT_ITALIC, scale * 0.3, bone_target_color, 1,
                False)

def draw_beat(cvs, this_beat):
    cv2.putText(cvs, 'frame:' + str(int(this_beat[0])), (scale * 3, CANVAS_SIZE[1] - scale * 3), cv2.FONT_ITALIC, scale * 0.3, (0, 0, 255), 1,
                False)
    cv2.putText(cvs, 'beat:' + str(int(this_beat[1])), (scale * 3, CANVAS_SIZE[1] - scale * 9), cv2.FONT_ITALIC, scale * 0.3, (0, 0, 255), 1,
                False)
    cv2.putText(cvs, 'in beat frame:' + str(int(this_beat[2])), (scale * 3, CANVAS_SIZE[1] - scale * 15), cv2.FONT_ITALIC, scale * 0.3,
                (0, 0, 255), 1, False)

def draw_skeleton_number(cvs, frame):
    for j in range(show_bones):
        cv2.putText(cvs, str(j), (int(frame[j][0]), (CANVAS_SIZE[1] - int(frame[j][1]))), cv2.FONT_ITALIC, scale * 0.3,
                    (0, 0, 255), 1, False)

def draw_skeleton(cvs, frame, bone_color, skeleton_color,):
    for j in range(show_bones):
        cv2.circle(cvs, (int(frame[j][0]), int(frame[j][1])), radius=scale * 3, thickness=-1, color=bone_color)
    cv2.line(cvs, (int(frame[0][0]), int(frame[0][1])), (int(frame[1][0]), int(frame[1][1])), skeleton_color, 2)
    cv2.line(cvs, (int((frame[0][0] + frame[1][0]) / 2), int((frame[0][1] + frame[1][1]) / 2)),
             (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])),
             (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[4][0]), int(frame[4][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[4][0]), int(frame[4][1])), (int(frame[5][0]), int(frame[5][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[5][0]), int(frame[5][1])), (int(frame[6][0]), int(frame[6][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])),
             (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[13][0]), int(frame[13][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[13][0]), int(frame[13][1])), (int(frame[14][0]), int(frame[14][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[14][0]), int(frame[14][1])), (int(frame[15][0]), int(frame[15][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])),
             (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[7][0]), int(frame[7][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[8][0]), int(frame[8][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[8][0]), int(frame[8][1])), (int(frame[9][0]), int(frame[9][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[9][0]), int(frame[9][1])),
             (int((frame[10][0] + frame[11][0]) / 2), int((frame[10][1] + frame[11][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[10][0]), int(frame[10][1])), (int(frame[11][0]), int(frame[11][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[16][0]), int(frame[16][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[16][0]), int(frame[16][1])), (int(frame[17][0]), int(frame[17][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[17][0]), int(frame[17][1])), (int(frame[18][0]), int(frame[18][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[18][0]), int(frame[18][1])),
             (int((frame[19][0] + frame[20][0]) / 2), int((frame[19][1] + frame[20][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[19][0]), int(frame[19][1])), (int(frame[20][0]), int(frame[20][1])), skeleton_color, 2)


def draw(frames,center, video_path, tempo_path, target_frames=None):

    if with_rotate:
        rotate_skeleton(target_frames)
    frames[:, :, 0] *= scale
    frames[:, :, 1] *= scale
    frames[:,:,0] += CANVAS_SIZE[0]//4
    frames[:,:,1] += CANVAS_SIZE[1]//2
    # center[:,0] += CANVAS_SIZE[1]//2
    # center[:,1] += CANVAS_SIZE[0]//2
    # rotate_skeleton(target_frames)
    if target_frames is not None:
        target_frames[:, :, 0] *= scale
        target_frames[:, :, 1] *= scale
        target_frames[:, :, 0] += CANVAS_SIZE[0] * 3 // 4
        target_frames[:, :, 1] += CANVAS_SIZE[1] // 2


        if len(target_frames) > len(frames):
            target_frames = target_frames[:len(frames)]
        assert(len(target_frames)==len(frames))
    
    
    tempo = np.load(tempo_path)
    video = cv2.VideoWriter(video_path, fourcc, fps, (CANVAS_SIZE[0],CANVAS_SIZE[1]), 1)
    image_seq = []
    for i in range(len(frames)):
        cvs = np.ones((CANVAS_SIZE[1], CANVAS_SIZE[0], CANVAS_SIZE[2]))
        cvs[:,:,:] = 255


        this_beat = tempo[i]
        frame = frames[i]

        if target_frames is not None:
            target_frame = target_frames[i]
            # for j in range(len(target_frame)):
            #     target_frame[j] += center[i]
            draw_skeleton(cvs, target_frame, bone_target_color, skeleton_target_color)

        if show_pred:
            draw_skeleton(cvs,frame,bone_pred_color, skeleton_pred_color)

        ncvs = np.flip(cvs, 0).copy()

        if show_pred:
            draw_skeleton_number(ncvs,frame)
        if target_frames is not None:
            draw_skeleton_number(ncvs,target_frame)
        draw_beat(ncvs,this_beat)
        draw_hints(ncvs)
        if not to_video:
            cv2.imshow('canvas',ncvs)
            cv2.waitKey(0)


        # image_seq = ncvs if image_seq is None else np.append(image_seq,ncvs)

        print(i)
        video.write(np.uint8(ncvs))
    # clip = ImageSequenceClip(image_seq,fps=fps)
    # clip.write_videofile(filename=video_path)
    video.release()
    pass

from data_prepare.feature_extract import load_start_end_frame_num, load_skeleton



if __name__ == '__main__':

    with open(pred_path,'r') as predf, open(target_path, 'r') as tarf:
        print(pred_path)
        print(target_path)
        pred_data = json.load(predf)
        tar_data = json.load(tarf)
        if show_ground_truth:
            draw(frames=np.array(pred_data['skeletons']),center=np.array(pred_data['center']),video_path=pred_video, tempo_path=tempo_path, target_frames=np.array(tar_data['skeletons']))
        else:
            draw(frames=np.array(pred_data['skeletons']),center=np.array(pred_data['center']),video_path=pred_video, tempo_path=tempo_path)

    if to_video:
        start, end = load_start_end_frame_num(config_path)
        gr_duration,_,_ = load_skeleton(target_path)
        pred_duration,_,_ = load_skeleton(pred_path)
        audio = AudioFileClip(audio_path)
        start = start + gr_duration -pred_duration
        end = start + pred_duration
        sub = audio.subclip(start/fps, end/fps)
        print('Analyzed the audio, found a period of %.02f seconds' % sub.duration)
        video = VideoFileClip(pred_video, audio=False)
        video = video.set_audio(sub)
        video.write_videofile(pred_video2)
        pass
    # with open(input_path2,'r') as fin:
    #     data2 = json.load(fin)
    #
    #     draw(np.array(data2['skeletons']), center=np.array(data2['center']), avi_path=file_path2, tempo_path=tempo_path)