## This script contains functions and classes used in model training/development. Some functions may be duplicated in src/utils or ../eda/utils. 

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
