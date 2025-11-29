# Digit Classifier from Scratch (NumPy)

A neural network implemented entirely from scratch using NumPy to classify the MNIST dataset.  
This project demonstrates deep learning concepts, including feedforward networks, ReLU and Softmax activations, the Adam optimizer, and manual data preprocessing—all without using high-level ML libraries such as Sci-Kit.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Results](#results)
7. [Future Improvements](#future-improvements)  

---

## Project Overview

This project builds a simple multi-layer neural network capable of recognizing digits (0–9) from the MNIST dataset.  
It emphasizes understanding the underlying mechanics of neural networks rather than relying on pre-built libraries like TensorFlow or PyTorch.

Key concepts demonstrated:

- Feedforward fully connected layers  
- ReLU and Softmax activations  
- Cross-entropy loss  
- Adam optimizer  
- Manual train/test split and normalization  
- Model saving and loading  
- Evaluation with per-class accuracy  

---

## Features

- **Neural network from scratch** using NumPy  
- **Manual implementation** of layers, activations, forward and backward propagation  
- **Adam optimizer** for parameter updates  
- **Model saving/loading** via `.npz` files  
- **Detailed evaluation** with per-class accuracy and prediction distribution  
- **Optional interactive web app** to draw digits and get predictions in real-time  
- **Simple, modular project structure** ready for extension  
- **Mini-batch  Training** to improve efficiency for large data sets (MNIST)
---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/digit-classifier-from-scratch.git
cd digit-classifier-from-scratch

```
2. Create A Virtual Environment
pip install -r requirements.txt
python interactive_testing.py


## Results
Training accuracy: ~95% 
Test accuracy: ~92%
Provides per-class accuracy and prediction distribution for detailed analysis
