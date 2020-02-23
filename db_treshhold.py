import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa

import librosa.display

# load files
sr, x1 = wav.read("/content/overlap_mic1.wav")
sr, x2 = wav.read("/content/overlap_mic2.wav")
x1 = x1.astype('float')
x2 = x2.astype('float')

# RMS energy in dB at a rate of 100 Hz. WE CAN FINE TUNE THE SAMPLE RATE MAYBE
e1 = librosa.core.amplitude_to_db(librosa.feature.rms(x1,frame_length=int(sr*0.02),hop_length=int(sr*0.01))).flatten()
e2 = librosa.core.amplitude_to_db(librosa.feature.rms(x2,frame_length=int(sr*0.02),hop_length=int(sr*0.01))).flatten()

# dB treshholds. COULD ALSO BE FINE TUNED
# tha: minimum dB level for a channel to be considered activated by speech
tha = 40
# thb: minimum difference between channels to consider it to be a single speaker
thd = 5

# boolean vectors at 100 Hz. s1: only speaker 1 and vice versa.
# could mawe also demand that the other mic is not too loud?
s1 = np.logical_and(np.greater(e1,tha),np.greater(e1,e2+thd))
s2 = np.logical_and(np.greater(e2,tha),np.greater(e2,e1+thd))

# up-sample s1 and s2 to audio rate
s1x = sig.resample(s1,x1.shape[0],window='hamming')
s2x = sig.resample(s2,x2.shape[0],window='hamming')

# generate two wave signals, with each speaker in isolation
y1 = (x1*s1x).astype('int16')
y2 = (x2*s2x).astype('int16')

wav.write('y1.wav',sr,y1)
wav.write('y2.wav',sr,y2)

fig=plt.figure(figsize=(10, 5))
x = np.linspace(0, len(e1)/100, len(e1))
plt.xticks(np.arange(min(x), max(x), 0.5))
plt.plot(x,e1)
plt.plot(x,e2)
plt.plot(x,e1-e2)
plt.plot(x,s1*10)
plt.plot(x,s2*10)

fig=plt.figure(figsize=(10, 5))
x = np.linspace(0, len(s1x)/sr, len(s1x))
plt.xticks(np.arange(min(x), max(x), 0.5))
plt.plot(x,s1x)
plt.plot(x,s2x)
plt.show()

# Hi, the basic idea is quite simple really, to threshold on difference
# in energy between the two mics. Whenever difference in db is greater
# than a given threshold, this means only that speaker is talking.
# No i dont have a paper describing it but I wrote a simple python program
# that shows the idea, you will have to refine it, e.g. discard all segments 
# shorter than some minimum length etc. And you might have to tweak the 
# thresholds. But data is plentiful and you can afford to set thresholds 
# conservatively, even if it means discarding a lot of data.