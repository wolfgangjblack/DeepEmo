# DeepEmo
Using Deep Neural Networks to build a robust classifier for Human Emotions. 

Date Modified: 6/16/23 <br>
Author: Wolfgang Black <br>

This repository contains code for training a shallow Convolutional Neural Net (CNN) and using ImageNet for transfer learning for SER (Speech Emotion Recognition) to classify audio data into 7 classes of human emotion. The data consists of 11803 audio recordings of human speech. This data was sourced from [kaggle](https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition) and pulls from the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) emotional speech, Toronto Emotional speech set (TESS), Crowd-Sourced Emotional Multimodal Actors Dataset (CREMA-D), and Surrey Audio-Visual Expressed Emotion (Savee) datasets. It should be noted that with the way we processed the data, some 100 files were not used. So expect some small discrepanies in the data counts if you download the data from the above link. 

## Dependencies
  - Python 3.9
  - Tensorflow 2.4
  - Tensorflow_io
  - NumPy
  - Matplotlib
  - Seaborn
  - Sklearn
  - Pandas
 
## Motivation
Speech Emotion Recognition (SER) aims to automatically detect and classify emotions conveyed through human speech. Building an SER model is motivated by several factors:

Enhanced Human-Computer Interaction: Emotions play a crucial role in human communication. Integrating SER into human-computer interaction systems, such as voice assistants, call centers, or virtual agents, can enhance the overall user experience. By understanding and responding appropriately to users' emotional states, these systems can provide more personalized and empathetic interactions.

Mental Health Applications: Emotions are deeply connected to mental health and well-being. An SER model can assist in diagnosing and monitoring mental health conditions by analyzing speech patterns. It can aid in early detection of disorders like depression, anxiety, or bipolar disorder, and contribute to the development of automated screening tools or virtual therapists.

Social Robotics and Assistive Technologies: SER models find applications in social robotics and assistive technologies. Robots or devices equipped with SER capabilities can better understand and respond to human emotions. This can be particularly useful in contexts such as elderly care, special education, or therapy, where empathetic interactions are crucial.

I myself am most interested in the mental health applications - as I feel there is a lot of good AI can provide in this area. 

## Usage
After the data is downloaded and processed to see which files can be easily converted into numpy arrays - the audio files are explored in the /eda/ directory. The Exploratory Data script is provided in /src/eda/ as eda.py. This script generates wav form, audio file length, audio sampling rates, and spectrogram plots. These should be used to explore who the user wants to process the data. For the example here, we decide to use a standard spectrogram finding the minimum native sampling rate range that allows us to capture 80% of the data and from this we can calculate the max number of frames - though users should feel free to explore these choices in the script and data. 

The figure below shows the native sampling rates per class, and should highlight why we need both minimums and some maximum cutoff re-sampling rate:
![alt text](https://github.com/wolfgangjblack/DeepEmo/blob/main/src/eda/artifacts/native_sample_rates.png)

Since varying the resampling rate changes how long the files are, we can use this graph to understand the max number of frames:

![alt_text](https://github.com/wolfgangjblack/DeepEmo/blob/main/src/eda/artifacts/sample_frame_subplots.png)

Finally once a resampling rate and frame length have been decided on, we can generate spectrograms which will be used in the model for training/scoring. The figure below is an example of the types of spectrogram subplots generated in eda.py:

![alt text](https://github.com/wolfgangjblack/DeepEmo/blob/main/src/eda/artifacts/sample_spectrogram_subplot_with_index_0_24414_sample_rate_and_58837_frames.png)
For a visual representation of how varying the resampling rate, see the image below:

![alt text](https://github.com/wolfgangjblack/DeepEmo/blob/main/src/eda/artifacts/Angry_1022_ITS_ANG_XX.wav%20data%20augmentation%20spectrograms.png)

To generate the model, use /src/model/model_training.py. Users should open the script and pay attention to the parameters used in training, lines: 23-36. This script generates either a shallow CNN or pulls down the inceptionV3 trained on imagenet. The transfer learning model only retrains the output layer. This script uses the spectrogram/mel spectrogram transformation on the audio .wav files to generate images for the CNN. 

## Results
During training, the shallow cnn trains too slowly, hitting the early stopping criteria after around 8 epochs. This limits is accuracy to around 60%. However, the transfer learning model trains for 50 epochs and gets up to 70% accuracy. Its important to note that we could change the early stopping criteria and increase the shallow net, or similarly we could increase the number of epochs to see how quickly the transfer learning model could learn. **Note: I was limited by colab compute hours. If this model is to be used, the generate_dataset function should be optimized and the transfer learning model trained above 100 epochs.**

![alt text]([https://github.com/wolfgangjblack/synthetic_accent_module/blob/main/src/model/artifacts/shallow_cnn_training_metrics.png](https://github.com/wolfgangjblack/DeepEmo/blob/main/src/model/artifacts/transfer_training_metrics.png))

Above we can see that the transfer net ends its training with an accuracy and validation accuracy of around 0.7, however we can see in the confusion matrix on a holdout dataset that the model has trouble predicting on the disguisted and angry classes. 

![alt text](https://github.com/wolfgangjblack/DeepEmo/blob/main/src/model/artifacts/transfer_confusion_matrix.jpg)

## Next Step
For the mean time I'm interested in building an LLM that can help me read through machine learning papers and suggest improved projects, so for now I'll take a pause on this. However on the off chance I come back, the steps below are what I would recommend for exploration

1. optimize the dataset function
  - I get an error that it can properly load in memory
2. retrain with more epochs
  - challenging, I have to walk a fine line here for Colab
3. build an API where users can upload their own audio clips of human voices.
  - For this one, I want to also have an unknown class. So I can build a second model that makes sure its an audio clip of a human voice, and nothing else. 

## Conclusion
In this project, we trained and evaluate two CNNs for emotion classification. Our results suggest that transfer learning with the ImageNet has better performance in both training and validation over a shallow CNN for this task. This code can be used as a starting point for other projects that want to process audio data - as well as the start of a transcripting software which works to recognize individuals emotions.
