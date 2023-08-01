# Hemorrhage Detection in Neurobiological Brain CT Scans using CNN

## Overview

This repository contains a Convolutional Neural Network (CNN) model designed to detect hemorrhage in neurobiological brain CT scans. Hemorrhage, characterized by abnormal bleeding in the brain, is a critical medical condition requiring early detection and intervention. Our CNN model leverages advanced deep learning techniques to analyze brain CT scans and identify regions indicative of hemorrhage with high accuracy.

## Dataset

The model was trained on a curated dataset of 200 CT brain scan images, equally divided into hemorrhage-positive and hemorrhage-negative cases. Each CT scan was preprocessed and normalized to ensure consistent representation across the dataset. The data was split into training and validation sets to facilitate robust model evaluation.

The dataset can be found here - https://www.kaggle.com/datasets/anmspro/head-ct-hemorrhage-227-x-227 

## Computational Tools and Deep Learning Techniques

- Python: The entire model and its components were developed using Python, a versatile programming language, enabling seamless integration with popular deep learning libraries.

- Keras: The Keras deep learning library, built on top of TensorFlow, was employed for model creation and training. Keras offers a high-level, user-friendly API for neural network design and optimization.

- Convolutional Neural Networks (CNN): CNNs are a class of deep neural networks uniquely suited for image recognition tasks. The model utilizes a series of convolutional and pooling layers to extract meaningful features from the brain CT scans, capturing both local and global patterns.

- Dropout Regularization: To mitigate overfitting and improve generalization, dropout layers were strategically incorporated in the network. Dropout randomly deactivates neurons during training, forcing the model to learn robust representations.

- Adam Optimizer: The Adam optimization algorithm was employed to efficiently update model parameters during the training process. Adam adapts learning rates on a per-parameter basis, leading to faster convergence and superior performance.

- Learning Rate Scheduling: Learning rate scheduling was applied to further fine-tune the model's convergence. A gradual decrease in the learning rate during training helps navigate the loss landscape effectively, resulting in more stable and accurate models.

## Hemorrhage Detection and Biological Context

Hemorrhage detection in neurobiological brain CT scans is a complex task due to the diverse appearance of hemorrhagic regions and potential similarities with other brain abnormalities. Our CNN model employs a data-driven approach, learning intricate patterns and distinctive features from the training dataset. By leveraging neural networks, the model gains an understanding of the complex relationships within the brain CT scans, enabling it to distinguish between healthy and hemorrhage-affected regions.

The model's convolutional layers perform feature extraction, identifying spatial patterns, edges, and textures indicative of hemorrhage. The subsequent pooling layers reduce the spatial dimensions, retaining crucial information while minimizing computational complexity. The fully connected layers then combine the extracted features to make a binary classification decision - hemorrhage or no-hemorrhage.

## Model Performance and Validation

The CNN model underwent rigorous training on the provided CT scan dataset and was evaluated on an independent validation set. Extensive hyperparameter tuning and regularization techniques were employed to optimize its performance and prevent overfitting. The model achieved an impressive accuracy of 95% on the validation set, underscoring its efficacy in hemorrhage detection.

## Usage

To utilize the model for hemorrhage detection on new neurobiological brain CT scans, simply provide the image path to the `preprocess_single_image` function and run the prediction code. The model will output a confidence score, indicating the likelihood of hemorrhage in the given brain CT scan.

## Conclusion

The CNN model presented here represents a powerful tool for early hemorrhage detection in neurobiological brain CT scans. By integrating advanced deep learning techniques with a curated CT scan dataset, our model showcases impressive performance and holds great promise for assisting medical professionals in timely hemorrhage diagnosis and treatment.

For more information and technical details, refer to the model's code and the provided Jupyter Notebook.
