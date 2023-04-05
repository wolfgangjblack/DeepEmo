## This script contains functions and classes used in EDA. Some functions may be duplicated in src/utils or model/utils. 

import os

import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np

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

def get_files_to_ignore(data_dir:str, class_labels:list) -> dict:
    """
    This function checks all of the data in each class labels data-subdirectory to verify that the files can be provessed by tf.audio.decode_wav.
    This is a minimum requirement to be processed and used in the model. It will return a dictionary of files that WILL BE EXCLUDED from analysis and modeling efforts.
    Args:
        data_dir: the master data directory -ex: '../data'
        class_labels: a list containing the class labels - ex: ['happy', 'sad']
    Output:
        files_to_ignore: a dictionary containing key, values pairs where keys are the class label and the values are lists of full filepaths to ignore within that key
    """
    files_to_ignore = {}
    for label in class_labels:
        bad = []
        for file in os.listdir(data_dir+label+'/'):
            if 'wav' not in file:
                continue
            else:
                file_contents = tf.io.read_file(data_dir+label+'/'+file)
                try:
                    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)
                except:
                    bad.append(data_dir+label+'/'+file)
        
        files_to_ignore[label] = bad
    
    return files_to_ignore

def get_files_to_keep(data_dir:str, class_labels:list, files_to_ignore: dict) -> dict:
    """
    This function utilizes the files_to_ignore, which have filtered out all potential wav files that are not able to be processed. All files in this dictionary
    will be included in the modeling and analysis.
    Args:
        data_dir: the master data directory -ex: '../data'
        class_labels: a list containing the class labels - ex: ['happy', 'sad']
        files_to_ignore: a dictionary containing key, values pairs where keys are the class label and the values are lists of full filepaths to ignore within that key
    Output:
        files_to_keep: a dictionary containing key, values pairs where keys are the class label and the values are lists of full filepaths
    """
    files_to_keep = {}
    for label in class_labels:
        temp_file_list = []
        path = data_dir+label+'/'
        for file in os.listdir(path):
            if 'wav' not in file:
                continue
            elif path+file in files_to_ignore[label]:
                continue
            else:
                temp_file_list.append(path+file)
        files_to_keep[label] = temp_file_list
    
    return files_to_keep

def save_files_to_keep(files_to_keep:dict, artifacts_dir: str):
    flat_list = []
    for key in files_to_keep:
        for file in files_to_keep[key]:
            flat_list.append(file)

    f = open(artifacts_dir+'files_to_keep.txt', 'w')
    for path in flat_list:
        f.write(path+'\n')
    f.close()
    return


def get_native_sample_rates(data_dir:str, class_labels:list, files_to_ignore:dict) -> dict:
    """
    This function finds the native sample rate of each viable file and saves that value to a list per label in class label. 
    These labels and lists are then saved in a dictionary as a key, value pair. 

    The native sample rate is important as we move to standardize the file lengths/output rates. If we choose to small a 
    sample rate we'll end up majorly downsampling our data and potentially missing values, if we choose too large a sample 
    rate we'll end up potentially over sampling features as we scale up the sample rate. 
    
    Args:
        data_dir: the master data directory -ex: '../data'
        class_labels: a list containing the class labels - ex: ['happy', 'sad']
        files_to_ignore: a dictionary containing key, values pairs where keys are the class labels and the values are lists 
                of full filepaths to ignore within that key -ex: {'happy': ['../data/happy/a_sentence.wav', ...], ...}
    Output:
        native_sample_rates: a dictionary containing key, value pairs where keys are the class labels and the values are lists
                of native_sample_rates as found by tf.audio.decode_wav


    Notes: 
        1. Christopher D’Ambrose puts the hearing ability of the normal middle-aged adult at 12-14 kHz).
        2. High quality audio is currently considered 82kHz and above
        3. Nyquist theorum: you must sample at a rate of at least 2x the highest expected frequency
        4. Telephones sample around 8kHz
    """
    native_sample_rates = {}
    for label in class_labels:
        native_sample_rate = []
        path = data_dir+label+'/'
        for file in os.listdir(path):
            if 'wav' not in file:
                continue
            elif path+file in files_to_ignore[label]:
                continue
            else:
                file_contents = tf.io.read_file(data_dir+label+'/'+file)
                wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)
  
                wav = tf.squeeze(wav, axis = -1)
                sample_rate = tf.cast(sample_rate, dtype = tf.int64).numpy()
                
                native_sample_rate.append(sample_rate)

        native_sample_rates[label] = native_sample_rate

    return native_sample_rates

def get_avg_sample_rate_by_quantile(native_sample_rates: dict, quantile: float)-> int:
    """
    This function returns the suggested avg sample rather which will cover 1-quantile % of the data across all labels (keys).
    Args:
        native_sample_rates: a dictionary containing key, value pairs where keys are the class labels and the values are lists
                of native_sample_rates as found by tf.audio.decode_wav
        quantile: a float between 0 and 1, restricted to quantiles (0.1, .2, .3, ...) which denotes the range of data covered by the calculation
    Output:
        avg_sample_rate: an int meant to be used in wav to spectrogram conversions. 

    Note: This is a potential nob for model tuning, we should strongly considered trying to model with various quantiles
    """
    quant_values = [] #Placeholder LARGE value

    for key in native_sample_rates.keys():
        df = pd.DataFrame(native_sample_rates[key])
        quant_values.append(df.quantile(quantile).to_numpy())
        
    return int(np.average(quant_values))

def get_min_sample_rate_by_quantile(native_sample_rates: dict, quantile: float)-> int:
    """
    This function returns the suggested min sample rather which will cover 1-quantile % of the data across all labels (keys).
    Args:
        native_sample_rates: a dictionary containing key, value pairs where keys are the class labels and the values are lists
                of native_sample_rates as found by tf.audio.decode_wav
        quantile: a float between 0 and 1, restricted to quantiles (0.1, .2, .3, ...) which denotes the range of data covered by the calculation
    Output:
        min_sample_rate: an int meant to be used in wav to spectrogram conversions. 

    Note: This is a potential nob, we should strongly considered trying to model with various quantiles
    """
    min_val = 1000000 #Placeholder LARGE value

    for key in native_sample_rates.keys():
        df = pd.DataFrame(native_sample_rates[key])
        val = df.quantile(quantile).to_numpy()
        min_val = min(min_val, val)

    return int(min_val)


def get_wav_lengths(data_dir:str, class_labels:list, files_to_ignore:dict, sample_rate_out: int) -> dict:
    """
    This function finds the native sample rate of each viable file and saves that value to a list per label in class label. 
    These labels and lists are then saved in a dictionary as a key, value pair. 

    The native sample rate is important as we move to standardize the file lengths/output rates. If we choose to small a 
    sample rate we'll end up majorly downsampling our data and potentially missing values, if we choose too large a sample 
    rate we'll end up potentially over sampling features as we scale up the sample rate. 
    
    Args:
        data_dir: the master data directory -ex: '../data'
        class_labels: a list containing the class labels - ex: ['happy', 'sad']
        files_to_ignore: a dictionary containing key, values pairs where keys are the class labels and the values are lists 
                of full filepaths to ignore within that key -ex: {'happy', ['../data/happy/a_sentence.wav', ...], ...}
    Output:
        lengths: a dictionary containing key, value pairs where keys are the class labels and the values are lists
                of lengths for the resampled tensors produced by tfio.audio.resample
    """
    lengths = {}
    for label in class_labels:
        length = []
        path = data_dir+label+'/'
        for file in os.listdir(path):
            if 'wav' not in file:
                continue
            elif path+file in files_to_ignore[label]:
                continue
            else:
                file_contents = tf.io.read_file(data_dir+label+'/'+file)
                wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels = 1)
  
                wav = tf.squeeze(wav, axis = -1)
                sample_rate = tf.cast(sample_rate, dtype = tf.int64).numpy()

                #Goes from 44100Hz to 16000Hz - amplitude of the audio signal 
                wav = tfio.audio.resample(wav, rate_in = sample_rate, rate_out = sample_rate_out)
                length.append(len(wav))

        lengths[label] = length
        
    return lengths

def get_metric_subplots(class_labels:list, metric_dict: dict, metric_type: str, save_path: str):
    """
    This function plots m x 2 subplots for a specified metric across all class labels. 
    Here, m  = C + 3, where C is the number of class labels. The first C plots are the
    individual metric plot per C, the next 3 then are the mean, min, and max value 
    barplots across all labels. 
    Arg: 
        class_labels: a list containing the class labels - ex: ['happy', 'sad']
        metric_dict: a dictionary containing key,value pairs for all class labels and a specified metric - ex: lengths = {'happy': ['1224224', ...],...}
        metric_type: a human readable string used in the title of subplots and in the saved artifact name
        save_path: a path str where the subplot artifact will be saved -ex "./artifacts/"
    Output:

    """
    iterator = 1

    nrows = round(len(class_labels)/2)+2
    ncols = 2

    fig = plt.figure(figsize = (10,35))

    plt.tight_layout()

    for key, value in metric_dict.items():
        ax = plt.subplot(nrows, ncols, iterator)
        ax.plot([i for i in range(len(value))], value)
        ax.set_title(key+" "+metric_type)
        iterator += 1    

    ##Plot average sample rates
    ax = plt.subplot(nrows, ncols, iterator)
    sns.barplot(x= [ i for i in range(len(class_labels))], y = [np.mean(value) for key, value in metric_dict.items()])
    plt.xticks([i for i in range(len(class_labels))], [key for key, values in metric_dict.items()], rotation = 45)
    ax.set_title('average '+metric_type)

    ##Plot average sample rates
    ax = plt.subplot(nrows, ncols, iterator+1)
    sns.barplot(x= [ i for i in range(len(class_labels))], y = [np.min(value) for key, value in metric_dict.items()])
    plt.xticks([i for i in range(len(class_labels))], [key for key, values in metric_dict.items()], rotation = 45)
    ax.set_title("min "+ metric_type)

    ##Plot average sample rates
    ax = plt.subplot(nrows, ncols, iterator+2)
    sns.barplot(x= [ i for i in range(len(class_labels))], y = [np.max(value) for key, value in metric_dict.items()])
    plt.xticks([i for i in range(len(class_labels))], [key for key, values in metric_dict.items()], rotation = 45)
    ax.set_title("max "+ metric_type)

    filename = save_path+metric_type.replace(' ', '_')+'.png'
    plt.savefig(filename)
    plt.close()

    return

def get_sample_frame_df(quantiles: list, native_sample_rates: dict, lengths: dict) -> pd.DataFrame():
    """
    This function takes in the native sample rates and a list of quantiles to generate a pd dataframe
    containing results for the minimum sample rate and averaged sample rate across the classes at a 
    specified quantile. This data is then used to calculate and populate columns for the suggested max
    frames for spectrogram conversion. 
    Args: 
        quantiles: a list of quantiles - ex: [i/10 for i in range(1, 10)]
        native_sample_rates: a dictionary containing key, value pairs where keys are the class labels and the values are lists
                of native_sample_rates as found by tf.audio.decode_wav
        lengths: a dictionary containing key, value pairs where keys are the class labels and the values are lists
                of lengths for the resampled tensors produced by tfio.audio.resample
    Outputs:
        data: a pd.DataFrame() containing columns for min and average sample rates and frames to be used at each quantile
    """
    min_sample_rates = []
    avg_sample_rates = []
    min_frames = []
    avg_frames = []

    for quant in quantiles:
        min_sample = get_min_sample_rate_by_quantile(native_sample_rates, quant)
        avg_sample = get_avg_sample_rate_by_quantile(native_sample_rates, quant)
        min_sample_rates.append(min_sample)
        avg_sample_rates.append(avg_sample)

        frames_list = []
        for key in lengths.keys():
            df = pd.DataFrame(lengths[key])
            frames_list.append(df.quantile(quant).to_numpy())
        min_frames.append(int(round(np.average(frames_list)/min_sample, 2)*min_sample))
        avg_frames.append(int(round(np.average(frames_list)/avg_sample, 2)*avg_sample))

    data = pd.DataFrame()
    data['quantile'] = quantiles
    data['avg_sample_rates'] = avg_sample_rates
    data['min_sample_rates'] = min_sample_rates
    data['frames_at_min'] = min_frames
    data['frames_at_avg'] = avg_frames
    
    return data

def save_sample_frame_subplots(sample_rate_df: pd.DataFrame(), artifact_dir: str):
    """
    This function saves off the sample rates and frame rates found at quantiles as graphs for later review
    Args:
        sample_rate_df: pd.DataFrame() containing columns for min and average sample rates and frames to be used at each quantile
        artifact_dir: a string path pointing to a directory meant for artifacts
    Output:
        a matplotlib subplot
    """
    fig = plt.figure(figsize = (10, 10))
    ax =plt.subplot(2, 2, 1)
    ax.set_title('Sample Rates')
    plt.plot(sample_rate_df['quantile'].to_list(), sample_rate_df['min_sample_rates'].to_list(), 'r--')
    plt.plot(sample_rate_df['quantile'].to_list(), sample_rate_df['avg_sample_rates'].to_list(), 'b--')
    plt.legend(['min', 'avg'])

    ax = plt.subplot(2, 2, 2)
    ax.set_title("Suggested max Frames")
    plt.plot(sample_rate_df['quantile'].to_list(), sample_rate_df['frames_at_min'].to_list(), 'r--')
    plt.plot(sample_rate_df['quantile'].to_list(), sample_rate_df['frames_at_avg'].to_list(), 'b--')
    plt.legend(['min', 'avg'])

    plt.savefig(artifact_dir+'sample_frame_subplots.png')
    plt.close()
    return

def get_label_to_int_mapping(class_labels:list)-> dict:
    """
    This function exists to repeatedly and easily generate a label to integer mapping.
    Args: 
        class_labels: a list containing the class labels - ex: ['happy', 'sad']
    Output
        labels_to_int: a dictionary containing the class labels as keys and an assigned int as values
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
    This function takes a filename, which is the full path of a specific .wav file, then decodes that file 
    to find the tensor associated with the sound - this is later used to get the spectrograms
    Args:
        filename: a full relative path represented by a str to a .wav file - ex "../../data/Happy/happymale.wav"
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

def get_sample_wav_subplot(labels_to_int:dict, files_to_keep:dict, index:int, sample_rate:int, artifacts_dir:str):
    """
    This function takes an index common to all labels within the files_to_keep dictionary and generates a wav 
    subplot, saving the entire thing in artifacts with the sample rate and index recorded.

    Args:
        labels_to_int: a dictionary mapping string labels to integers, used in wav_to_spectrogram -ex {'Angry': 0, 'Happy': 1, ...}
        files_to_keep: 
        index: an int meant to denote which value in the files_to_keep key, value pairs will be plotted as a spectrogram - ex: 0
        sample_rate: an int, the intended sample rate post conversion -ex 16000
        artifacts_dir: a string path pointing to a directory meant for artifacts -ex './artifacts/'
    Output:
        A  subplot of wavs at some index per class label
    """
    nrows = len(labels_to_int.keys())
    iterator = 1
    plt.figure(figsize = (20, 10*nrows))
    plt.tight_layout
    for key in labels_to_int.keys():
        wav = load_wav_output_mono_channel_file(files_to_keep[key][index], sample_rate)
        ax = plt.subplot(nrows, 1, iterator)
        ax.set_title(files_to_keep[key][index])
        plt.plot(wav)
        iterator += 1
    filename = 'sample_wav_subplot_with_index_{0}_and_{1}_sample_rate.png'.format(index, sample_rate)

    plt.savefig(artifacts_dir+filename)
    plt.close()
    return

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
        log_magnitude_mel_spectrograms: a tf.Tensor that has been comverted from a wav to a zero_padded tensor representing a spectrogram
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

def get_sample_spectrogram_subplot(labels_to_int:dict, files_to_keep:dict, index:int, sample_rate:int, frames:int, spec_type:str, artifacts_dir:str):
    """
    This function takes an index common to all labels within the files_to_keep dictionary and generates a spectrogram on the spectrogram
    subplot, saving the entire thing in artifacts with the sample rate, frames, and index recorded. This can produce either base spectrogram
    or mel spectrogram subplots depending on the specification in spec
    Args:
        labels_to_int: a dictionary mapping string labels to integers, used in wav_to_spectrogram -ex {'Angry': 0, 'Happy': 1, ...}
        files_to_keep: 
        index: an int meant to denote which value in the files_to_keep key, value pairs will be plotted as a spectrogram - ex: 0
        sample_rate: an int, the intended sample rate post conversion -ex 16000
        frames: an int, the max frames to be comsidered. Both frames and sample_rate_out are from quantile analysis -ex 43360
        spec_type: a str meant to be either "spec" or "mels", if this is undefined the program will automatically assign spec
        artifacts_dir: a string path pointing to a directory meant for artifacts -ex './artifacts/'
    Output:
        A spectrogram subplot
    """
    nrows = len(labels_to_int.keys())
    iterator = 1
    plt.figure(figsize = (20, 30))
    plt.tight_layout

    ##Verify spec_type is correctly assigned
    if spec_type not in ['spec', 'mels']:
        print("spec_type not recognized, assigning spec")
        spec_type = 'spec'

    for key in labels_to_int.keys():
        if spec_type == 'spec':
            spec, _ = wav_to_spectrogram(files_to_keep[key][index], sample_rate,labels_to_int[key],frames)
            filename = 'sample_spectrogram_subplot_with_index_{0}_{1}_sample_rate_and_{2}_frames.png'.format(index, sample_rate, frames)

        else:
            spec, _ = wav_to_mels_spectrogram(files_to_keep[key][index], sample_rate,labels_to_int[key],frames)
            filename = 'sample_mels_spectr_subplot_with_index_{0}_{1}_sample_rate_and_{2}_frames.png'.format(index, sample_rate, frames)

        ax = plt.subplot(nrows, 1, iterator)
        ax.set_title(files_to_keep[key][index])
        plt.imshow(tf.reshape(spec, [spec.shape[1], spec.shape[0]]))
        iterator += 1
    plt.savefig(artifacts_dir+filename)
    plt.close()

    return

