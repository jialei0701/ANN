import collections
import contextlib
import sys
import wave
import os
import webrtcvad


def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
        wf.writeframes(audio)


def write_wave2(path, audios, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for audio in audios:
            wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad,
                  frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    for frame in frames:
        # sys.stdout.write('1' if vad.is_speech(frame.
        #                                       bytes, sample_rate) else '0')
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([
                f for f in ring_buffer if vad.is_speech(f.bytes, sample_rate)
            ])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('+(%s)' % (ring_buffer[0].timestamp, ))
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([
                f for f in ring_buffer
                if not vad.is_speech(f.bytes, sample_rate)
            ])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    # if triggered:
    #     sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    # sys.stdout.write('\n')
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def SearchFile(path, str_suffix):
    filelist = []
    try:
        files = os.listdir(path)

        for f in files:
            fl = os.path.join(path, f)
            if os.path.isdir(fl):
                SearchFile(fl, str_suffix)
            elif os.path.isfile(fl) and f.endswith(str_suffix):
                # print(type(fl))
                fl = fl.replace('\\', '/')
                print(fl)
                filelist.append(fl)

    except Exception:
        print(u'+++++++++++')
    print(len(filelist))
    return filelist


# def main(args):
#     if len(args) != 2:
#         sys.stderr.write(
#             'Usage: example.py <aggressiveness> <path to wav file>\n')
#         sys.exit(1)
#     audio, sample_rate = read_wave(args[1])
#     vad = webrtcvad.Vad(int(args[0]))
#     frames = frame_generator(30, audio, sample_rate)
#     frames = list(frames)
#     segments = vad_collector(sample_rate, 30, 300, vad, frames)
#     for i, segment in enumerate(segments):
#         # path = 'chunk-%002d.wav' % (i,)
#         print('--end')
#         # write_wave(path, segment, sample_rate)


def main():
    path = "D:/sre16/wav_file"
    tarpath = "C:/Users/dhk/Desktop/aftervad"

    filelist = SearchFile(path, ".wav")

    print(len(filelist))
    # for path in filelist:
    #     # print(path[-17:])
    #     audio, sample_rate = read_wave(path)
    #     vad = webrtcvad.Vad(1)
    #     frames = frame_generator(30, audio, sample_rate)
    #     frames = list(frames)
    #     segments = vad_collector(sample_rate, 30, 300, vad, frames)
    #     audiolist = []
    #     for i, segment in enumerate(segments):
    #         # print('--end')
    #         audiolist.append(segment)
    #     temp = tarpath + "/" + path[-17:]
    #     print(temp)
    #     write_wave2(temp, audiolist, sample_rate)


if __name__ == '__main__':
    # main(sys.argv[1:])
    main()
