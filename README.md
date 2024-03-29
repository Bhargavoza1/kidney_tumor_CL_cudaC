# end to end kidney tumor classification with cuda C++
 
Welcome to the repository for my cutting-edge kidney tumor classification system! This project represents the culmination of my efforts, leveraging CUDA C++ and OpenCV to develop a sophisticated neural network for accurately classifying kidney tumors. Through extensive work, I have created a system that seamlessly integrates with a Golang server and is complemented by a React-powered frontend. Deployment on Azure AKS with GitHub Actions ensures streamlined deployment and management.

## Features

- **CUDA C++ Neural Network:** At the core of this system is a robust neural network implemented entirely in CUDA C++, enabling efficient parallel processing and leveraging the power of GPUs for accelerated computations.

- **Golang Server Integration:** The neural network seamlessly integrates with a Golang server, facilitating smooth communication between the backend and frontend components.

- **React Frontend:** The frontend interface, powered by React, offers users an intuitive and interactive experience, allowing them to interact with the classification system effortlessly.

- **Deployment on Azure AKS:** Leveraging Azure AKS (Azure Kubernetes Service) for deployment ensures scalability, reliability, and ease of management, providing a robust infrastructure for hosting the system.

- **GitHub Actions Deployment:** The deployment process is streamlined using GitHub Actions, enabling continuous integration and deployment with minimal manual intervention.


# Neural network architecture

The image shows a neural network architecture for image classification. The network takes an input image of size 225x225 and outputs a probability distribution over the possible classes. The network consists of several layers, including convolutional layers, ReLU activation layers, max-pooling layers, batch normalization layers, a flatten layer, linear layers, and a sigmoid activation layer.

The convolutional layers extract features from the input image. The ReLU activation layers introduce non-linearity into the network. The max pooling layers downsample the feature maps. The batch normalization layers normalize the activations. The flatten layer converts the feature maps into a single vector. The linear layers learn to classify the input image based on this vector. The sigmoid activation layer outputs a probability distribution over the possible classes.

This network architecture is a common choice for image classification tasks. It is effective because it can learn to extract features from the input image that are relevant to the classification task. The network is also relatively efficient to train and can be used to classify images with high accuracy.

<p align="center">
  <img src="./gitresource/image_classification.png" />
</p>