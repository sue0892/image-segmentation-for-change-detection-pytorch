# A deep learning-driven change detection model for martian surface segmentation
This is the code for the paper: Detection of Detached Ice-fragments at Martian Polar Scarps Using a Convolutional Neural Network.

## Introduction
This is a deep learning-driven change detection method for extracting the detached ice-fragments at martian polar scarps. The customized model is using:
- ResU-Net architecture, which combines ResNet and U-Net;
- Siamese network architecture
- An augmented attention module
- A hybrid loss function: dice loss and focal loss

## Datasets
The examples of the dataset are provided, but the entire dataset is currently not available to the public. However, users can easily apply our model to various training datasets available online for testing. The training dataset consists of three files: T1, T2, and Mask. 
