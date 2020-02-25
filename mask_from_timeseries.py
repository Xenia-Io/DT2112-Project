# Written by Jonas Eriksson, 25/2-20
import os
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa


# Creates a ndarray filter mask from a time series.
# sr, x is what is returned from wav.read(filename.wav)
# sr: sampling rate (int), x: time series (nd.array)
def mask_from_timeseries(sr, x):
    x = x.astype('float')
    S, ph = librosa.magphase(librosa.stft(x))
    # i'm not sure what the value of "time" should be. 0.1 works well for segment lengths of 0.5 seconds.
    time = 0.1
    S_filter = librosa.decompose.nn_filter(S,
                                        aggregate=np.median,
                                        metric='cosine',
                                        width=int(librosa.time_to_frames(time, sr=sr)))
    S_filter = np.minimum(S, S_filter)
    margin = 5
    power = 2
    mask =  librosa.util.softmask(S - S_filter,
                                margin * S_filter,
                                power=power)
    return mask

directory = os.fsencode("/home/jonas/Documents/Kurser/Master/Talteknologi (DT2112)/Project_local")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".wav"): 
        #print(os.path.join(directory, filename))
        sr, x = wav.read(filename) 
        mask = mask_from_timeseries(sr, x)
    else:
        continue
