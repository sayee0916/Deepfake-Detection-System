# Deepfake Detection Using Deep Learning

## Overview

Deepfake technology uses artificial intelligence to generate highly realistic manipulated images or videos. These fake media can be misused for spreading misinformation, fraud, identity impersonation, and digital manipulation. Detecting such content manually is difficult because modern deepfakes appear visually convincing.

This project builds a **Deepfake Image Detection System** using **Deep Learning and Transfer Learning**. The model classifies images into two categories:

* **Real Images** – Authentic, unmodified images
* **Fake Images (Deepfake)** – AI‑generated or manipulated images

The model uses **EfficientNetB0**, a pretrained convolutional neural network, to extract deep visual features and perform binary classification.

---

# Problem Statement

Deepfake images are becoming increasingly realistic and difficult to identify by human observation alone. Social media platforms and online communication channels can rapidly spread manipulated content.

Therefore, an **automated AI‑based detection system** is required to analyze images and determine whether they are real or artificially generated.

---

# Objectives

* Build a deep learning model capable of detecting deepfake images
* Use **transfer learning** to improve performance with limited data
* Apply **data preprocessing and augmentation** to improve generalization
* Train and evaluate the model for accurate binary classification

---

# Dataset

The dataset used for this project contains two classes of images.

### Classes

* **Real Images** – Authentic photographs
* **Fake Images** – Deepfake or AI‑generated images

### Dataset Structure

```
Final Dataset
 ├── Real
 └── Fake
```

### Dataset Split

* **80% Training Data** – Used to train the model
* **20% Validation Data** – Used to evaluate model performance

---

# Tech Stack

## Programming Language

* Python

## Deep Learning Framework

* TensorFlow
* Keras

## Supporting Libraries

* NumPy
* Matplotlib
* Scikit‑learn

## Development Platform

* Google Colab (GPU acceleration)

---

# Project Workflow

The project follows a complete deep learning pipeline:

1. Dataset upload and extraction
2. Image preprocessing
3. Data augmentation
4. Train‑validation split
5. Transfer learning using EfficientNetB0
6. Model training
7. Fine tuning
8. Model evaluation
9. Prediction

---

# Data Preprocessing

Before training the model, images were prepared using the following steps:

* **Image Resizing:** All images resized to **224 × 224** pixels
* **Pixel Normalization:** Pixel values scaled for stable training
* **Class Folder Organization:** Images arranged into Real and Fake directories

These preprocessing steps ensure the model receives consistent input data.

---

# Data Augmentation

To increase dataset diversity and improve model robustness, data augmentation techniques were applied using **ImageDataGenerator**.

### Augmentation Techniques

* Horizontal Flip
* Image Rotation
* Zoom Transformation

### Benefits

* Reduces overfitting
* Improves generalization
* Helps the model learn different variations of images

---

# Model Architecture

The project uses **Transfer Learning with EfficientNetB0**.

### Architecture Components

1. **EfficientNetB0**

   * Pretrained on ImageNet
   * Used as a feature extractor

2. **Global Average Pooling Layer**

   * Converts feature maps into a feature vector

3. **Dropout Layer (0.5)**

   * Prevents overfitting

4. **Dense Layer with Sigmoid Activation**

   * Performs binary classification

### Output

* **0 → Fake Image**
* **1 → Real Image**

---

# Training Strategy

The model was trained in two stages.

## Phase 1 – Frozen Base Model

* EfficientNetB0 layers were frozen
* Only classification layers were trained

## Phase 2 – Fine Tuning

* Top layers of EfficientNetB0 were unfrozen
* Model retrained using a lower learning rate

### Techniques Used

* Early Stopping
* Transfer Learning
* Fine Tuning

---

# Model Performance

* **Validation Accuracy:** ~76%
* **Dropout Rate:** 0.5
* **Training Method:** Transfer Learning + Fine Tuning

The model successfully learned visual patterns that help distinguish between real and deepfake images.

---

# How the Model Works

1. User uploads an image
2. Image is resized and preprocessed
3. EfficientNetB0 extracts deep features
4. Classification layers predict the probability
5. Model outputs whether the image is **Real or Fake**

---

# Applications

* Fake news detection
* Social media content verification
* Cybercrime prevention
* Digital media authentication
* Identity protection systems

---

# Future Improvements

* Train the model on larger datasets
* Improve accuracy using advanced architectures
* Add **face detection before classification**
* Extend the system for **video deepfake detection using CNN + LSTM**

---

# Author

**Sayali Sanjay Chidrawar**

Domain: Computer Vision | Deep Learning

