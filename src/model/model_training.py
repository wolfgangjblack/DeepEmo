## This script is meant to generate a single model including training rtifacts and relevant files
# for the DeepEmo Project. At the end of this script running, the user should have the final h5 for
# a trained mode as well as artifacts for
# 1. training/validation loss
# 2. validation confusion matrix

# Run Instructions:  
# This script requires the modifcation of a config, found in relative path ./configs/
# the change the config, see ./configs/training_config.py - a script with detailed explanations for each
#  parameter after any changes in the config script, run the main.py script which will call first the config
#  script then run this model_training.py script
#---------------------------------


##imports
import os
import matplotlib.pyplot as plt

from utils.utils import generate_tf_dataset, get_input_shape, build_shallow_cnn

import tensorflow as tf


#Note: This is in development and will be abstracted in the future.
## *future config params
##paths
files_to_keep_artifact = '../eda/artifacts/files_to_keep.txt'
data_dir = '../../data'
artifacts_dir = './artifacts/'

#dataset stuffs
frames = 50000
sample_rate = 16000
batch_size = 32
spec_type = 'spec'

##model stuffs
model_type = 'shallow'
model_dir = './saved_models/'+model_type+'/'

optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
model_metrics = [tf.keras.metrics.SparseCategoricalAccuracy(), tf.keras.metrics.SparseCategoricalCrossentropy()]
split = 0.75
epochs = 10
callback = tf.keras.callbacks.EarlyStopping(verbose=1, patience=4, restore_best_weights = True)

###script start
with open(files_to_keep_artifact) as f:
    files_to_keep = [line.rstrip() for line in f]

class_labels = [i for i in os.listdir(data_dir) if '.' not in i]

#Generaet TF dataset
data = generate_tf_dataset(files_to_keep, class_labels, sample_rate, frames, batch_size, spec_type)

input_shape = get_input_shape(data)

if model_type == 'shallow':
    model = build_shallow_cnn(input_shape, class_labels)

##Currently unsupported
# elif model_type == 'transfer':
#     model = transfer_inceptionV3(input_shape, class_labels)
# elif model_type == 'visTransformer':
#     model = build_visTransformer(input_shape, class_labels)

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