import wave
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sig
import soundfile
from pydub import AudioSegment
import os, wave, math, collections


def convert_24_to_16_bits(file1, file2):
    # Transform from 24-bit to 16-bit
    data, samplerate = soundfile.read(file1)
    soundfile.write('CMA-002 Lav-K-16.wav', data, samplerate, subtype='PCM_16')

    data, samplerate = soundfile.read(file2)
    soundfile.write('CMA-002 Lav-Khaki-16.wav', data, samplerate, subtype='PCM_16')



# Split the wav file from t1 to t2 seconds
# t1 = 0 * 1000  # Works in milliseconds
# t2 = 580 * 1000

# newAudio = AudioSegment.from_wav("CMA-002 Lav-K-16-mic1.wav")
# # newAudio = newAudio[t1:t2] från t1 till t2
# newAudio.export('K.wav', format="wav")  # Exports to a wav file in the current path.
#
# newAudio = AudioSegment.from_wav("CMA-002 Lav-Khaki-16-mic2.wav")
# # newAudio = newAudio[t1:t2]
# newAudio.export('Khaki.wav', format="wav")  # Exports to a wav file in the current path.
#
# fs, x1 = wav.read("K.wav")
# fs, x2 = wav.read("Khaki.wav")
#
# x1 = x1.astype('float')
# x2 = x2.astype('float')
#
# x1 = x1[:1200 * fs]
# x2 = x2[:1200 * fs]
#
# print(x1.shape, x2.shape)
#
# # calculate rms energy in dB at a rate of 100 Hz (hop length 0.01 s)
# e1 = librosa.core.amplitude_to_db(
#     librosa.feature.rms(x1, frame_length=int(fs * 0.02), hop_length=int(fs * 0.01))).flatten()
# e2 = librosa.core.amplitude_to_db(
#     librosa.feature.rms(x2, frame_length=int(fs * 0.02), hop_length=int(fs * 0.01))).flatten()
#
# # dB thresholds.
# # tha: absolute dB level for when to consider there to be speech activity in a channel
# tha = 20
# # thb: minimum difference between channels to consider it to be one speaker only
# thb = 3
#
# # boolean vectors at 100 Hz, s1: only speaker 1. s2: only speaker 2.
# s1 = np.logical_and(np.greater(e1, tha), np.greater(e1, e2 + thb))
# s2 = np.logical_and(np.greater(e2, tha), np.greater(e2, e1 + thb))
#
# # up-sample s1 and s2 to audio rate
# s1x = sig.resample(s1, x1.shape[0], window='hamming')
# s2x = sig.resample(s2, x2.shape[0], window='hamming')
#
# # generate two wave signals, with each speaker in isolation
# y1_A = (s1x * x1).astype('int16')
# y1_B = (s1x * x2).astype('int16')
# y2_A = (s2x * x1).astype('int16')
# y2_B = (s2x * x2).astype('int16')
#
# wav.write('output_files/y1_A.wav', fs, y1_A)
# wav.write('output_files/y1_B.wav', fs, y1_B)
# wav.write('output_files/y2_A.wav', fs, y2_A)
# wav.write('output_files/y2_B.wav', fs, y2_B)
#
#

# Mixes 2 .wav files
def mixture2(wav1, wav2):
    d, samplerate = soundfile.read(wav1)
    soundfile.write(wav1, d, samplerate, subtype='PCM_16')
    d, samplerate = soundfile.read(wav2)
    soundfile.write(wav2, d, samplerate, subtype='PCM_16')

    sound1 = AudioSegment.from_file(wav1)
    sound2 = AudioSegment.from_file(wav2)

    combined = sound1.overlay(sound2)

    combined.export("mixture.wav", format='wav')

# Make the 2 input .wav files the same length
def make_same_length(wav1, wav2):
    wav1_len = librosa.get_duration(filename=wav1)
    wav2_len = librosa.get_duration(filename=wav2)
    if wav1_len < wav2_len:
        y, sr = librosa.load(wav2, duration=wav1_len)
        librosa.output.write_wav(wav2, y, sr)
    else:
        y, sr = librosa.load(wav1, duration=wav2_len)
        librosa.output.write_wav(wav1, y, sr)
    return [wav1, wav2]

# Splits the .wav file in split_length in several chunks. Every chunk is split_length seconds
def split_wav(wav_file, split_length):
    t1 = 0 * 1000  # Works in milliseconds
    t2 = split_length * 1000
    newAudio = AudioSegment.from_wav(wav_file)
    outfile = "sample"
    i = 0
    for i in range(int(newAudio.duration_seconds / split_length)):
        tmp = newAudio[t1:t2]
        tmp.export(f"Mixtures/{outfile}{i}.wav", format="wav")  # Exports to a wav file in the current path.
        t1 += split_length * 1000
        t2 += split_length * 1000


# directory = "mics"
# split_wav(directory, 0.5)
infiles = make_same_length("mics/clean_mic1/out1.wav", "mics/clean_mic2/out2.wav")
mixture2("mics/clean_mic1/out1.wav", "mics/clean_mic2/out2.wav")
split_wav("mixture.wav", 0.5)
# Splitting wav file on silence

#Split on silence (run in terminal)
# sox -V3 CMA-002\ Lav-K-16.wav out.wav silence 1 0.5 0.1% 1 0.5 0.1% : newfile : restart

# Remove all silence (run in terminal)
# sox y2-Khaki.wav out4.wav silence 1 0.1 1% -1 0.1 1%

# # '''
# plt.plot(e1)
# plt.plot(e2)
# plt.plot(e1 - e2)
# plt.plot(s1 * 10)
# plt.plot(s2 * 10)
# plt.figure()
# plt.plot(s1x)
# plt.show()
# # '''
