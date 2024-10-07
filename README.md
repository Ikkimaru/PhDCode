# Speech Emotion Recognizer

This project implements a speech emotion recognition system using machine learning models. The system is capable of classifying emotions from audio recordings based on features extracted from the speech signals.

## Features

- **Extracted Speech Features**:
  - MFCC (Mel-frequency cepstral coefficients)
  - Chroma
  - MEL Spectrogram Frequency
  - Spectral Contrast
  - Tonnetz

- **Models**:
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
  - Naive Bayes

## Dataset

The project uses the [RAVDESS dataset](https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio) for training and testing. It includes audio files labeled with various emotions.

## How It Works

1. **Extracting Features**: The system extracts the following features from each audio file:
   - MFCC
   - Chroma
   - MEL
   - Spectral Contrast
   - Tonnetz

2. **Emotion Classification**: Based on the extracted features, different models are trained to classify the emotions in the audio. The following emotions are supported:
   - Calm
   - Happy
   - Sad
   - Disgust
   - Angry
   - Fearful
   - Surprised

3. **Results**: The system calculates the accuracy, utility value, and confusion matrix for the selected emotion classifications. The results are saved into an Excel file for analysis.
