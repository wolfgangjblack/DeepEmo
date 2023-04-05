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

def check_artifacts_dir(save_path = './artifacts/'):
    '''
    This function checks if there is an artifacts dir in our root dir, 
    if not it'll create the aritifacts dir to prevent an error
    '''
    try:
         os.listdir(save_path)
    except FileNotFoundError:
        os.mkdir(save_path)

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
ß
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
    ax =plt.subplot(2, 2, 1))
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