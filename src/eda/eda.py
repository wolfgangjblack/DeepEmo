## This script is meant to generate artifacts and files relevant to the exploratory data analysis
# for the DeepEmo Project. At the end of this script running, the user should have artifacts for
# 1. suggested resample rates at specified quantiles
# 2. native sample rate subplots covering each class as well as average, min, and max native sample rates across classes
# 3. converted wav length subplots covering lengths of wavs at suggested resampled rates covering each class
# 4. suggested max frames to consider for modeling
# 5. example spectrograms across class labels
# 6. example wav plots across class labels
# 7. example mel-spectrograms across class labels
#
# Run Instructions: 
# users should change the initial data directories and artifact directories as necessary. 
# the quantile list can also be changed to accomodate the users preferences
#---------------------------------
import os
from utils.utils import (check_artifacts_dir, 
                        get_files_to_ignore, 
                        get_files_to_keep,
                        save_files_to_keep,
                        get_native_sample_rates,
                        get_min_sample_rate_by_quantile, 
                        get_wav_lengths,
                        get_metric_subplots,
                        get_sample_frame_df,
                        save_sample_frame_subplots,
                        get_label_to_int_mapping,
                        get_sample_spectrogram_subplot)


##Below are things users can change
#Data dir Here
data_dir = '../../data/'
#artifacts dir here
artifacts_dir = './artifacts/'
quantiles = [i/10 for i in range(1, 10)] ##use this to save a graph later
quantile = 0.2
#---------------------------------

check_artifacts_dir(artifacts_dir)

##read in class labels
class_labels = [i for i in os.listdir(data_dir) if '_' not in i]

#Find the bad files - these are files that can't be read into tf.audio.wav_decode. We'll save a list of these off as 
# well as a list of the files we CAN use. This will be kept in artifacts, but later read into the modeling scripts
files_to_ignore =  get_files_to_ignore(data_dir, class_labels)
files_to_keep = get_files_to_keep(data_dir, class_labels, files_to_ignore)
save_files_to_keep(files_to_keep, artifacts_dir)

#All audio files have some native sampling rate. We want to understand how much we'll change the sampling rate
native_sample_rates = get_native_sample_rates(data_dir, class_labels, files_to_ignore)

##For this, we're assuming a quantile of 0.2 is sufficient, however please note some changes in file lengths occur if we adjust this
sample_rate_out = get_min_sample_rate_by_quantile(native_sample_rates, quantile)

lengths = get_wav_lengths(data_dir, class_labels, files_to_ignore,sample_rate_out)

##save off metric subplots
get_metric_subplots(class_labels, lengths, "file lengths at "+ str(quantile)+" quantile", artifacts_dir)
get_metric_subplots(class_labels, native_sample_rates, "native sample rates", artifacts_dir)

##Get Pandas DF for later review and subplot calculation
sampleFrameDf = get_sample_frame_df(quantiles, native_sample_rates, lengths)
sampleFrameDf.to_csv(artifacts_dir+'sample_rate_data.csv')
save_sample_frame_subplots(sampleFrameDf, artifacts_dir)

##Get labels_to_int - this is an alphabetical numeric mapping that will be used throughtout the project
labels_to_int = get_label_to_int_mapping(class_labels)

##to understand how sample rate and frames affect the data for the model, we'll fix the index and explore changing the
#  quantile (driving up sample rate and max frames)
for quant in [0.1, 0.5, 0.7,  0.8, 0.9]:
    sample_rate = int(sampleFrameDf[sampleFrameDf['quantile'] == quant]['min_sample_rates'].to_numpy())
    frames = int(sampleFrameDf[sampleFrameDf['quantile'] == quant]['frames_at_min'].to_numpy())
    get_sample_spectrogram_subplot(labels_to_int, files_to_keep, 0, sample_rate, frames, 'spec', artifacts_dir)
    get_sample_spectrogram_subplot(labels_to_int, files_to_keep, 0, sample_rate, frames, 'mels', artifacts_dir)

