## This script is meant to generate a single model including training rtifacts and relevant files
# for the DeepEmo Project. At the end of this script running, the user should have the final h5 for
# a trained mode as well as artifacts for
# 1. training/validation loss
# 2. validation confusion matrix

# Run Instructions:  
# This script does not require the use of a config, instead users are expected to change this file directly
# this is intention as the users should be aware of what they're training and the base script used to train the models
#---------------------------------


##imports
import os
import matplotlib.pyplot as plt

from utils.utils import check_artifacts_dir, generate_tf_dataset, get_input_shape, build_shallow_cnn, build_transfer_inception_model, save_model_performance

import tensorflow as tf

#Note: This is in development and will be abstracted in the future.
## *future config params
##paths
files_to_keep_artifact = '../eda/artifacts/files_to_keep.txt'
data_dir = '../data/emotions/'
artifacts_dir = './artifacts/'

#dataset stuffs
frames = 50000
sample_rate = 16000
batch_size = 32
spec_type = 'spec'

##model stuffs
model_type = 'transfer'
model_dir = './saved_models/'+model_type+'/'

optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
model_metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseCategoricalCrossentropy()]
split = 0.75
epochs = 25
callback = tf.keras.callbacks.EarlyStopping(verbose=1, patience=4, restore_best_weights = True)

###script start

##check artifacts dir and model dir
#artifacts_dir will contain plots and any confusion matrix data
#model_dir will contain model files
check_artifacts_dir(artifacts_dir)
check_artifacts_dir(model_dir)

with open(files_to_keep_artifact) as f:
    files_to_keep = [line.rstrip() for line in f]

class_labels = [i for i in os.listdir(data_dir) if '.' not in i]

#Generaet TF dataset
data = generate_tf_dataset(files_to_keep, class_labels, sample_rate, frames, batch_size, spec_type)

input_shape = get_input_shape(data)

if model_type == 'shallow':
    model = build_shallow_cnn(input_shape, class_labels)
elif model_type == 'transfer':
    model = build_transfer_inception_model(input_shape, class_labels)

model.compile(
        optimizer=optimizer,
        loss =tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics = model_metrics)

split_count = int(len(data)*split)+1

train = data.take(split_count)
val = data.skip(split_count).take(len(data)-split_count)

history = model.fit(train,
                 epochs = epochs,
                 validation_data = val,
    callbacks=callback)

save_model_performance(history, model_type, artifacts_dir)

##Save Model
model.save(model_dir +model_type+'_'+spec_type+ '_model.h5')
