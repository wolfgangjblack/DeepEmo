## This script contains functions and classes used in model training/development. Some functions may be duplicated in src/utils or ../eda/utils. 

import os

import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Input
from tensorflow.keras.applications.inception_v3 import InceptionV3

import librosa
import librosa.display

import seaborn as sns
import pandas as pd
import numpy as np
import pandas as pd

def check_artifacts_dir(artifacts_dir:str ):
    '''
    This function checks if there is an artifacts dir in our root dir, 
    if not it'll create the aritifacts dir to prevent an error
    Args:
        artifacts_dir: a str representing a path -ex './artifacts/'
    '''
    try:
         os.listdir(artifacts_dir)
    except FileNotFoundError:
        os.mkdir(artifacts_dir)

    return 

def generate_tf_dataset(files_to_keep:list, class_labels:list, sample_rate:int, frames:int, batch_size: int, spec_type = 'spec'):
    '''
    Prepares a tf dataset object for use in a model. This function provides a data pipeline
    generating a dataset object of file paths, then using the Dataset.map() function to convert
    file paths to spectrograms (represented in tensors) and labels. 
    '''

    for i in range(len(class_labels)):
        temp_files = [j for j in files_to_keep if class_labels[i] in j]
        temp_file_list = tf.data.Dataset.list_files(temp_files)
        temp_labels =  tf.data.Dataset.from_tensor_slices(tf.ones(len(temp_file_list))*i)
        temp = tf.data.Dataset.zip((temp_file_list, temp_labels))

        if i == 0:
            data = temp
        else:
            data = data.concatenate(temp)

    data = data.shuffle(int(len(data)/4))
    frames = tf.constant(frames, dtype=tf.int64)
    sample_rate = tf.constant(sample_rate, dtype=tf.int64)

    if spec_type == 'spec':
        data = data.map(lambda filepath, label: wav_to_spectrogram(filepath, sample_rate ,label, frames))
    else:
        data = data.map(lambda filepath, label: wav_to_mels_spectrogram(filepath, label, frames))

    data = data.batch(batch_size)
    data = data.cache()
    data = data.prefetch(1)

    return data



def get_label_to_int_mapping(class_labels:list)-> dict:
    """
    This function exists to repeatedly and easily generate a label to integer mapping.
    Args: 
        class_labels: a list containing the class labels - ex: ['happy', 'sad']
    Output
        labels_to_int: a dictionary containing the class labels as keys and an assigned int as
         values
    """
    labels_to_int = {}
    labels = sorted(class_labels)
    lab_int = 0
    for label in labels:
        labels_to_int[label] = lab_int
        lab_int += 1
    return labels_to_int

def load_wav_output_mono_channel_file(filename:str, sample_rate_out:int)-> tf.Tensor:
    """
    This function takes a filename, which is the full path of a specific .wav file, then decodes
     that file to find the tensor associated with the sound - this is later used to get the spectrograms
    Args:
        filename: a full relative path represented by a str to a .wav file -
         ex "../../data/Happy/happymale.wav"
        sample_rate_out: an int, the intended sample rate post conversion -ex 16000
    Outputs:
        wav: a tf.Tensor containing an array representing the audio file 
    """
    #load encoded wav file
    file_contents = tf.io.read_file(filename)
    
    #Decode wav (tensors by channel)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)

    #Remove trailing axis
    wav = tf.squeeze(wav, axis = -1)
    sample_rate = tf.cast(sample_rate, dtype = tf.int64)

    #Goes from 44100Hz to 16000Hz - amplitude of the audio signal 
    wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out = sample_rate_out)

    return wav


def add_noise_to_wav(wav: tf.Tensor)-> tf.Tensor:
    """
    This function was adapted from https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
    and adds noise to the processed wav. This will eventually be used in data augmentation for modeling
    Args:
        wav: a tf.Tensor representing the audio file
    Output:
        wav: a tf.Tensor + some randomly generated noise representing the changed wav
    """
    wav = wav.numpy()
    noise_amp = 0.05*np.random.uniform()*np.amax(wav)
    wav = wav+noise_amp*np.random.normal(size = wav.shape[0])
    wav = tf.convert_to_tensor(wav)
    return wav  

def stretch_wav(wav:tf.Tensor, rate:float) -> np.ndarray:
    """
    This function was adapted from https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
    and stretches the original wav. This will eventually be used in data augmentation for modeling.
    Args:
        wav: a tf.Tensor representing the audio file
    Output:
        wav: a tf.Tensor with augmented frequency space of the original wav

    Note: the purpose of this function is to either stretch or shrink the features found in the amplitude
    space across the frequency space
    """
    length = wav.shape[0]
    temp_wav = tf.convert_to_tensor(librosa.effects.time_stretch(wav.numpy(), rate =rate))
     ##Select as much wav as fills frames, if len(wav) < frames, this will be less than frames and will need padding
    temp_wav = temp_wav[:length]

    ##Calculate the number of zeros for padding, note if the wav >= frames, this will be empty
    
    zero_padding = tf.zeros([length] - tf.shape(temp_wav), dtype = tf.float32)

    ##Add zeros at the start IF the wav length < frames
    temp_wav = tf.concat([zero_padding, temp_wav], 0)

    return temp_wav

def change_pitch(wav:tf.Tensor, sampling_rate: int, n_steps: int) -> np.ndarray:
    """
    This function was adapted from https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition
    and changes the pitch of the original wav. This will eventually be used in data augmentation for
    modeling.
    Args:
        wav: a tf.Tensor representing the audio file
    Output:
        wav: a tf.Tensor at a different pitch from the original audio file

    Note: the n_steps should be selected careful, see librosa documentation for guidance
    """
    return tf.convert_to_tensor(librosa.effects.pitch_shift(wav.numpy(), sr= sampling_rate, n_steps = n_steps))

def wav_to_spectrogram(filename:str, sample_rate_out:int, label:int, frames: int) -> tuple[tf.Tensor, int]:
    '''
    This function reads in a single file path, a label (as an int), and the desired output max frame count 
    to produce a spectrogram. This will be used in tf.data.Dataset.map() to convert filepaths into 
    spectrograms after the data has been groupped together
    Args:
        filename: a full relative path represented by a str to a .wav file - ex "../../data/Happy/happymale.wav"
        sample_rate_out: an int, the intended sample rate post conversion -ex 16000
        label: an int meant to map from string label to int - ex: 'Happy' -> 0, for modeling maps will be made explicit
        frames: an int, the max frames to be comsidered. Both frames and sample_rate_out are from quantile analysis
    Output
        spectrogram: a tf.Tensor that has been comverted from a wav to a zero_padded tensor representing a spectrogram
        label: an int meant to map from string label to int - ex: 'Happy' -> 0, for modeling maps will be made explicit
    
    Note: label is repeated for the output so it can be used in the .map() method

    '''
    wav = load_wav_output_mono_channel_file(filename, sample_rate_out)
    
    ##Select as much wav as fills frames, if len(wav) < frames, this will be less than frames and will need padding
    wav = wav[:frames]

    ##Calculate the number of zeros for padding, note if the wav >= frames, this will be empty
    
    zero_padding = tf.zeros([frames] - tf.shape(wav), dtype = tf.float32)

    ##Add zeros at the start IF the wav length < frames
    wav = tf.concat([zero_padding, wav], 0)

    #use short time fourier transform
    spectrogram = tf.signal.stft(wav, frame_length = 320, frame_step = 32)

    #Get the magnitude of the signal (los direction)
    spectrogram = tf.abs(spectrogram)

    #Adds a second dimension 
    spectrogram = tf.expand_dims(spectrogram, axis = 2)

    return spectrogram, label

def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator
    
    # Scale magnitude relative to maximum value in S. Zeros in the output 
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec

def wav_to_mels_spectrogram(filepath:str, sample_rate: int, label:int, frames:int) -> tuple[tf.Tensor, int]:
    '''
    This function reads in a signle file path, a label, and the desired output max frame count to produce a spectrogram. 
    This will be used in tf.data.Dataset.map() to convert filepaths into mels spectrograms after the data has been 
    groupped together A Mels Spectrogram is a variant of the spectrogram that is obtained by applying a mel scale to the 
    frequency axis.  The mel scale is a perceptual scale of pitches that is based on how humans perceive sound. 
    Mel spectrograms are useful because they allow the representation of audio signals in a way that is more aligned with 
    human perception of sound.
    Args:
        filepath: a full relative path represented by a str to a .wav file - ex "../../data/Happy/happymale.wav"
        sample_rate_out: an int, the intended sample rate post conversion -ex 16000
        label: an int meant to map from string label to int - ex: 'Happy' -> 0, for modeling maps will be made explicit
        frames: an int, the max frames to be comsidered. Both frames and sample_rate_out are from quantile analysis
    Output
        log_magnitude_mel_spectrograms: a tf.Tensor that has been comverted from a wav to a zero_padded tensor representing a
         spectrogram
        label: an int meant to map from string label to int - ex: 'Happy' -> 0, for modeling maps will be made explicit
    
     Note: label is repeated for the output so it can be used in the .map() method'''

    wav = load_wav_output_mono_channel_file(filepath, sample_rate)
    
    ##Select as much wav as fills frames, if len(wav) < frames, this will be less than frames and will need padding
 
    wav = wav[:frames]

    ##Calculate the number of zeros for padding, note if the wav >= frames, this will be empty
    
    zero_padding = tf.zeros([frames] - tf.shape(wav), dtype = tf.float32)

    ##Add zeros at the start IF the wav length < frames
    wav = tf.concat([zero_padding, wav], 0)

    #use short time fourier transform
    spectrogram = tf.signal.stft(wav,
     frame_length = 320, ##This is fft_size
     frame_step = 32 ## this is hop_size
        ) #

    #Get the magnitude of the signal (los direction)
    spectrogram = tf.abs(spectrogram)
    
    mel_filter = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=100,
        num_spectrogram_bins = 257,
        sample_rate=frames,
        lower_edge_hertz=frames/100,
        upper_edge_hertz=frames/2,
        dtype=tf.dtypes.float32)

    mel_power_spectrogram = tf.matmul(tf.square(spectrogram), mel_filter)

    log_magnitude_mel_spectrograms = power_to_db(mel_power_spectrogram)

    log_magnitude_mel_spectrograms = tf.expand_dims(log_magnitude_mel_spectrograms, axis = 2)

    return log_magnitude_mel_spectrograms, label

def get_input_shape(training_data):
    '''
    after the training data has been generated,  this function gets the input shape. This will be used 
    when developing a model
    '''
    samples, _ = training_data.as_numpy_iterator().next()
        
    input_shape = samples.shape[1:]

    return input_shape

def build_shallow_cnn(input_shape: tuple, class_labels: list):
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation = 'relu', input_shape = input_shape)) ##This is 16 kernels of shape 3x3, input shape 
    model.add(Conv2D(16, (3,3), activation = 'relu')) ## as this is the Second layer, we no longer need an input shape - its connected directly to the prior layer
    model.add(MaxPooling2D()),
    model.add(Dropout(0.5)),
    model.add(Flatten()) #This combines all the nodes from the previous conv2d layer into a single dimension
    model.add(Dense(128, activation = 'relu', kernel_regularizer = 'l2'))

    model.add(Dense(len(class_labels), activation = 'softmax'))

    return model

def build_transfer_inception_model(input_shape, class_labels):
    # 1.create the base model but omit the classification layer
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # 2.freeze the convolutional base (i.e. retain the weights)
    base_model.trainable = False

    # 3.add classification head and prediction layer
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(len(class_labels), activation='softmax')

    # 4.build the model
    inputs = tf.keras.Input(shape=(input_shape[0],input_shape[1],1))
    x = tf.keras.layers.Conv2D(3,(3,3), padding = 'same')(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    outputs = prediction_layer(x)
    transfer_model = tf.keras.Model(inputs, outputs)

    return transfer_model

def save_model_performance(history, model_type:str, savepath = './artifacts/'):
    metrics = history.history
    fig = plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel('Epoch')
    plt.ylabel('Loss [CrossEntropy]')

    plt.subplot(1,2,2)
    plt.plot(history.epoch, 100*np.array(metrics['sparse_categorical_accuracy']), 100*np.array(metrics['val_sparse_categorical_accuracy']))
    plt.legend(['accuracy', 'val_accuracy'])
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    fig.savefig(savepath+model_type+'_training_metrics.png')

    return