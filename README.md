# SCT_ML_4
CNN-based hand gesture classification using the LEAPGestRecog dataset

# Hand Gesture Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to classify static hand gestures using the LEAPGestRecog dataset. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) and contains grayscale images of hand gestures recorded via the Leap Motion controller.

---

## 👤 Student Details

Name: velaga sasank

---

## 📌 Project Overview

- Objective: To build a deep learning model that can recognize hand gestures from image data.
- Model Used: Convolutional Neural Network (CNN)
- Dataset: LEAPGestRecog – consists of 10 different gesture classes from 10 users.
- Platform Used: Jupyter Notebook (online)

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- scikit-learn

---

## 📁 Folder Structure

leapGestRecog/
├── 00/
│ ├── 01/
│ ├── 02/
│ └── ...
├── 01/
├── ...
├── 09/

gesture_cnn.ipynb
README.md
requirements.txt


Each numbered folder represents a subject. Each gesture is stored in its own subfolder.

---

## 🚀 Features

- Loads and preprocesses grayscale hand gesture images
- Scales and reshapes images to 64x64 for CNN input
- Applies one-hot encoding to labels
- Splits data into training and testing sets
- Trains a CNN to classify 10 hand gesture classes
- Evaluates model accuracy and plots training curves

---

## 📊 Sample Model Output

- Model achieved ~95% accuracy on validation set
- Plots of accuracy and loss over epochs are included

---

## 🔧 Installation

Create a virtual environment and install the dependencies:

bash
pip install -r requirements.txt

### 🧠 Model Architecture
Conv2D → ReLU → MaxPooling

Conv2D → ReLU → MaxPooling

Flatten → Dense(128) → Dropout(0.5)

Output Layer: Softmax (10 classes)

### 📦 Requirements
- tensorflow
- numpy
- opencv-python
- matplotlib
- scikit-learn
