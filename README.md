# Overcoming Overfitting and Underfitting in Neural Network

There are two major problems when training neural networks: overfitting and underfitting.

Overfitting is a problem that can occur when the model is too sensitive to the training data. The model will then fail to generalize and perform well on new data. This can happen when there are too many parameters in the model.

This is noticeable in the learning curve by a big gap between the training and validation loss/accuracy.

Underfitting is the opposite of overfitting. It occurs when the model is not sensitive enough to the training data and as a result, the model fails to learn the most important patterns in the training data.

Underfitting is often not really a problem because you can prevent it by simply making your model more deep (add more layers/neurons to the model) or training for a few more epochs.

There are two main ways to avoid overfitting: gather more training data and apply regularization techniques to penalize the model.

In this template we have trained a convolutional neural network on the CIFAR-10 dataset available in tensorflow keras library.

Two main methods we have used to reduce overfitting and underfitting are:
#### 1-Reduce Overfitting with Dropout
#### 2-Reduce Overfitting with Data Augmentation

Hopefully this template will find you useful for understanding overfitting and underfitting and overcome it.
