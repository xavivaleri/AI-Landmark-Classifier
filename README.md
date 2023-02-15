# AI Landmark Classifier System

This repository contains code for building an AI Landmark Classifier system using PyTorch and CNNs. The project is divided into three main parts:

1. **Building the System from Scratch**: In this part, we build the neural network architecture from scratch using PyTorch. The architecture is designed using convolutional neural networks to learn features from images and make predictions based on them.

2. **Improving the Model using Transfer Learning**: In the second part, we improve the accuracy of the model by using transfer learning from a pre-trained model, RESNET18. Transfer learning is a technique where we use a pre-trained model's learned features and fine-tune it to perform a new task.

3. **Developing an App for Tagging Landmarks**: In the final part, we develop an application that uses the trained model to classify landmarks. The app allows users to upload an image of a landmark and get the predicted label.

## Content

This repository contains the following files:

- `cnn_from_scratch.ipynb`: Jupyter Notebook for building the AI Landmark Classifier system from scratch using PyTorch.
- `transfer_learning.ipynb`: Jupyter Notebook script for improving the accuracy of the AI Landmark Classifier system using transfer learning from the pre-trained RESNET18 model.
- `app.ipynb`: Jupyter Notebook script for developing an app to use the trained model for tagging landmarks.

All the functions developed are contained in the src/ directory.

## Getting Started

To get started with this project, you will need to have Python and PyTorch installed on your machine. Once you have these dependencies, you can clone the repository and run the scripts in each part to train the model and test it on sample images.

## Conclusion

In summary, this project demonstrates how to build an AI Landmark Classifier system using PyTorch and CNNs. By following the three parts, you can learn how to build a neural network architecture, improve its performance using transfer learning, and develop an application to use the model for classifying landmarks.

