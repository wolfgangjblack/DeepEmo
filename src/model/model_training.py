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
