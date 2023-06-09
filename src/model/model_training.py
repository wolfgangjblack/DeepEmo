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
import numpy as np
import tensorflow as tf

from utils.utils import check_artifacts_dir, generate_tf_dataset, get_input_shape, build_shallow_cnn, build_transfer_inception_model, save_model_performance, get_preds_array, save_confusion_matrix

##paths
files_to_keep_artifact = '../eda/artifacts/files_to_keep.txt'
data_dir = '../../data/'
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
split = 0.9
epochs = 20
callback = tf.keras.callbacks.EarlyStopping(verbose=1, patience=7, restore_best_weights = True)

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
##Currently unsupported
# elif model_type == 'visTransformer':
#     model = build_visTransformer_model(input_shape, class_labels)

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

#Get validation y and model predictions
y_true = np.concatenate([y for x, y in val], axis = 0)
preds = model.predict(val)
y_pred = get_preds_array(preds)

## Save Validation Confusion Matrix
save_confusion_matrix(y_true, y_pred, class_labels, model_type, artifacts_dir)