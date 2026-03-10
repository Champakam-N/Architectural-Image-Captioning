# Architectural Image Captioning Using Deep Learning with Multilingual Audio Description

## Project Overview

This project focuses on generating descriptive captions for architectural images using deep learning techniques. The system analyzes images of buildings, monuments, temples, and other structures and automatically generates meaningful textual descriptions.

The generated captions are then translated into multiple languages and converted into speech, making the system accessible to visually impaired users and people who prefer regional languages.

## Objectives

* Generate accurate captions for architectural images using deep learning.
* Extract visual features using the InceptionV3 CNN model.
* Generate captions using an LSTM-based sequence model.
* Translate captions into regional languages (Hindi and Kannada).
* Convert captions into audio using Text-to-Speech technology.
* Provide an interactive GUI for user interaction.

## System Architecture

The system follows the following pipeline:

1. Image Upload through GUI
2. Image Preprocessing
3. Feature Extraction using InceptionV3
4. Caption Generation using LSTM
5. Caption Post-processing
6. Translation using Google Translator
7. Text-to-Speech using gTTS
8. Display Caption and Audio Output

## Project Files

* **train.py** – Used to train the CNN-LSTM caption generation model.
* **test.py** – Used to test the trained model and generate captions for new images.
* **captions.py** – Handles caption processing and generation logic.

## Technologies Used

* Python
* TensorFlow / Keras
* InceptionV3 CNN
* LSTM (Recurrent Neural Network)
* OpenCV
* NumPy
* Google Translator API
* gTTS (Google Text-to-Speech)
* Tkinter (GUI)

## Installation

Install the required libraries before running the project:

pip install tensorflow keras numpy pandas opencv-python matplotlib nltk gtts googletrans

## How to Run the Project

### Train the Model

python train.py

### Test the Model

python test.py

### Generate Captions

python captions.py

## Applications

* Architectural image documentation
* Smart tourism applications
* Accessibility for visually impaired users
* AI-based image understanding systems

## Author

Champakam N

## License

This project is developed for educational and research purposes.
