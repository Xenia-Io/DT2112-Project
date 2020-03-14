import os
import scipy.io.wavfile as wav
import librosa
import numpy as np
import soundfile
#from pydub import AudioSegment, silence
#from pydub.silence import detect_silence
#from train_tuples import Trainer
import pathlib
import tensorflow as tf
from keras.models import Sequential
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.optimizers import rmsprop
from keras.layers import LeakyReLU, Dense, Flatten
import cv2 


def mask_from_timeseries(sr, x):
    x = x.astype('float16')
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
    mask = librosa.util.softmask(S - S_filter,
                                 margin * S_filter,
                                 power=power)
    return mask


# diretory is the directory where all the samples is located
def getMasks(directory):
    masks = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            # print(filename)
            sr, x = wav.read(directory + "/" + filename)
            masks.append(mask_from_timeseries(sr, x))
        else:
            continue
    return masks


# diretory is the directory where all the samples is located
def getSpectogram(directory):
    spect = []
    phases = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            sr, x = wav.read(directory + "/" + filename)
            x = x.astype('float16')
            S, ph = librosa.magphase(librosa.stft(x))
            spect.append(S)
            phases.append(ph)
        else:
            continue
    return spect, phases
# Change path here
print ("Loading data....")
y1ADir = "../Originals/y1_A_clean"
y2BDir = "../Originals/y2_B_clean"
y1Ay2ADir = "../Originals/y1_A_y2_A_MIX"
y1By2BDir = "../Originals/y1_B_y2_B_MIX"
masks1 = getMasks(y1ADir)
masks2 = getMasks(y2BDir)
spectoA, phaseA = getSpectogram(y1Ay2ADir)
spectoB, phaseB = getSpectogram(y1By2BDir)


####################################################
print ("Transform data....")

img_rows = 64
img_cols = 1024 
sample_size = len(spectoA)

# Input 1 and Output 1
spectoA = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in spectoA]).astype('float16')
masks1 = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in masks1]).astype('float16')
spectoA = np.reshape(spectoA, (sample_size, img_cols, img_rows, 1))
#phaseA = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in phaseA])
# Input 2 and Output 2
spectoB = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in spectoB]).astype('float16')
masks2 = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in masks2]).astype('float16')
spectoB = np.reshape(spectoB, (sample_size, img_cols, img_rows, 1))
#phaseB = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in phaseB])

print (spectoA.shape)

input_shape = spectoA.shape[1:]
#input_shape = (1024, 64, 1)

####################################################
print ("Creating model....")

def conv_block(input_tensor, num_filters):
  encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
  encoder = layers.BatchNormalization()(encoder)
  encoder = layers.LeakyReLU(alpha=0.3)(encoder)
  return encoder

def encoder_block(input_tensor, num_filters):
  encoder = conv_block(input_tensor, num_filters)
  encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)

  return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU(alpha=0.3)(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU(alpha=0.3)(decoder)
  return decoder

def output_block(input_tensor, concat_tensor, num_filters):
  decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU(alpha=0.3)(decoder)
  decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
  decoder = layers.BatchNormalization()(decoder)
  decoder = layers.LeakyReLU(alpha=0.3)(decoder)
  
  return decoder


# config normal config: 32, 64, 128, 256, 512 center: 256
in1 = 1
in2 = 2
in3 = 4 
img_cols = 1024
img_rows = 64

input_shape = (img_cols, img_rows, 1)
print (input_shape)
input1 = layers.Input(shape=input_shape, name='input1')
input2 = layers.Input(shape=input_shape, name='input2')
inputs = [input1, input2]
#print (inputs)

center = 256
##### Network 1 
encoder0_pool, encoder0 = encoder_block(input1, in1) # 128
encoder1_pool, encoder1 = encoder_block(encoder0_pool, in2) # 64
encoder2_pool, encoder2 = encoder_block(encoder1_pool, in3) # 32
#encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256) # 16
#encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512) # 8
center = conv_block(encoder2_pool, center) # center
#decoder4 = decoder_block(center, encoder4, 512) # 16
#decoder3 = decoder_block(decoder4, encoder3, 256) # 32
decoder2 = decoder_block(center, encoder2, in3) # 64
decoder1 = decoder_block(decoder2, encoder1, in2) # 128
decoder0 = decoder_block(decoder1, encoder0, in1) # 256
output1 = layers.Dense(1, activation='sigmoid')(decoder0)
output1 = layers.Reshape((img_cols, img_rows), name='output1')(output1)

center2 = 256 
##### Network 2
encoder_pool, encoder00 = encoder_block(input2, in1) # Input 
encoder_pool, encoder01 = encoder_block(encoder_pool, in2) #CNN
encoder_pool, encoder02 = encoder_block(encoder_pool, in3) #CNN
center2 = conv_block(encoder_pool, center2) # CENTER
decoder02 = decoder_block(center2, encoder02, in3)
decoder01 = decoder_block(decoder02, encoder01, in2)
decoder00 = decoder_block(decoder01, encoder00, in1)
output2 = layers.Dense(1, activation='sigmoid')(decoder00)
output2 = layers.Reshape((img_cols, img_rows), name='output2')(output2)
 
##### Outputs 
outputs = [output1, output2]

model = models.Model(inputs=[inputs], outputs=[outputs])


print ("Compiling model....")
model.compile(
    optimizer='rmsprop',
    loss={
        'output1': 'mean_squared_error',
        'output2': 'mean_squared_error'},
    metrics={'output1': 'mse',
             'output2': 'mse'}
)

print ("Training...")
model.fit({'input1': spectoA, 'input2': spectoB}, {'output1': masks1, 'output2': masks2}, verbose=1, epochs=10, validation_split=0.2)

####################################################
print ("Loading data....")
y1ADir = "../Originals/y1_A_clean/rest_data"
y2BDir = "../Originals/y2_B_clean/rest_data"
y1Ay2ADir = "../Originals/y1_A_y2_A_MIX/rest_data"
y1By2BDir = "../Originals/y1_B_y2_B_MIX/rest_data"
testMasks1 = getMasks(y1ADir)
testMasks2 = getMasks(y2BDir)
testSpectoA, phaseA = getSpectogram(y1Ay2ADir)
testSpectoB, phaseB = getSpectogram(y1By2BDir)


####################################################
# Evaluate 
print ("Transform data....")
img_rows = 64
img_cols = 1024 
sample_size = len(testSpectoA)

# Input 1 and Output 1
testSpectoA = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in testSpectoA]).astype('float16')
testMasks1 = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in testMasks1]).astype('float16')
testSpectoA = np.reshape(testSpectoA, (sample_size, img_cols, img_rows, 1))
#phaseA = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in phaseA])
# Input 2 and Output 2
testSpectoB = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in testSpectoB]).astype('float16')
testMasks2 = np.array([cv2.resize(x, (img_rows,img_cols), interpolation=cv2.INTER_CUBIC) for x in testMasks2]).astype('float16')
testSpectoB = np.reshape(testSpectoB, (sample_size, img_cols, img_rows, 1))

print ("Evaluating....")
#score1, score2, mse1, mse2 = model.evaluate({'input1': testSpectoA, 'input2': testSpectoB}, {'output1': testMasks1, 'output2': testMasks2})
print (model.evaluate({'input1': testSpectoA, 'input2': testSpectoB}, {'output1': testMasks1, 'output2': testMasks2}))
#print ("Score 1: " + score1)
#print ("Score 2: " + score1)
#print ("MSE 1: " + mse1)
#print ("MSE 2: " + mse1)