import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from numpy import argmax
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, BatchNormalization



# Multiple Outputs
from keras.utils import plot_model


DATA_PATH = "C:/Users/Kate/Desktop/Py_proj/Python_Neural_Network_Progress/train_data/"

def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)
    
def wav2mfcc(file_path, n_mfcc=20, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = np.asfortranarray(wave[::3])
    mfcc = librosa.feature.mfcc(wave, sr=44100, n_mfcc=n_mfcc)
    
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    else:
        mfcc = mfcc[:, :max_len]
        
    return mfcc

def save_data_to_array(path=DATA_PATH, max_len=11, n_mfcc=20):
    labels, _, _ = get_labels(path)

    for label in labels:
        mfcc_vectors = []
 
        wavfiles = [path +'/'+ label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len, n_mfcc=n_mfcc)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)
        
def get_train_test(split_ratio=0.6, random_state=42):
    labels, indices, _ = get_labels(DATA_PATH)
    
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])
    
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))
        
    assert X.shape[0] == len(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True)

def prepare_dataset(path=DATA_PATH):
    lables, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        
        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data
    
def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))
            
    return dataset[:100]

#################### Actual code with ML ####################

from preprocessing import *
import keras
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM, Conv3D
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import sklearn.linear_model as lr

import librosa
import librosa.display
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "input.wav"  # поменять имя файла


p = pyaudio.PyAudio()

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)
print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


wandb.init()
config = wandb.config
config.max_len = 11
config.buckets = 20
save_data_to_array(max_len=config.max_len, n_mfcc=config.buckets)

labels = ["включи","вперед", "выключи","дополнительный", "загрузи","музыку",
"назад","основной","отсканируй", "паузу","печать","свет","Эл"]

lables, _, _ = get_labels(DATA_PATH)
X_train, X_test, y_train, y_test = get_train_test()

channels = 1
config.epochs = 100 #100
config.batch_size = 440 #100 30

num_classes = 13

X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len, channels)
X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len, channels)

# plt.imshow(X_train[100, :, :, 0])
# print(y_train[100])

# plt.imshow(X_test[100, :, :, 0])
# print(y_train[100])

y_train_hot = to_categorical(y_train, num_classes)
y_test_hot = to_categorical(y_test, num_classes)

#X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len)
#X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len)

#X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len)
#X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len)

#X_train = X_train.reshape(X_train.shape[0], config.buckets, config.max_len)
#X_test = X_test.reshape(X_test.shape[0], config.buckets, config.max_len)

model = Sequential()
#model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(7, 7), input_shape=(config.buckets, config.max_len,1),padding = 'same',activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2),padding ='same'))


model.add(Conv2D(64, kernel_size=(5, 5),activation='relu',padding='same'))

model.add(Conv2D(128, kernel_size=(3, 3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))



model.add(Conv2D(256, kernel_size=(2, 2), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))


model.add(Conv2D(128, kernel_size=(2, 2),activation='relu',padding='same'))
model.add(Flatten())

model.add(Dropout(0.16))
model.add(Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(32, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Dense(num_classes, activation='softmax',kernel_regularizer=keras.regularizers.l2(0.001)))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])


wandb.init()
model.fit(X_train, y_train_hot, epochs=config.epochs, validation_data=(X_test, y_test_hot), callbacks=[WandbCallback(data_type="image", labels=labels)])
#model.save('voice_rec.h5')

x_final_vector = []
x_final_vector_vector = []
#x_final_vector = wav2mfcc("input.wav")
x_final_vector = wav2mfcc("C:/Users/Kate/Desktop/Py_proj/Python_Neural_Network_Progress/train_data/паузу/паузу_2_1.wav")
x_final_vector_vector.append(x_final_vector)
x_final = np.array(x_final_vector_vector)
x_final = x_final.reshape(x_final.shape[0], 20,11,1)
print(x_final.shape)
y_final_oneHotEncoded = model.predict(x_final, batch_size=20, verbose=0)

"""for i in range(len(y_final_oneHotEncoded)):
    print(y_final_oneHotEncoded[i],' - ',labels[i])
    print()
"""

y_final_num = argmax(y_final_oneHotEncoded)
print(y_final_num)
#print('Class labels:', np.unique(labels))
print(labels[y_final_num])
#y_train_pred = lda_model.predict(X_train_lda)
#y_test_pred  = lda_model.predict(X_test_lda)
#print '      %s        %7.4f    %7.4f' % (n_comp, np.mean(y_train != y_train_pred), np.mean(y_test != y_test_pred))