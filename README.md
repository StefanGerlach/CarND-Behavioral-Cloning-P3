# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

![Simulator Screenshot](/documentation/splash.png)

Overview
---

This repository contains my solution of the Udacity Self Driving Car Nanodegree Behavioral Cloning Project. The target of this project is to create a Deep Learning Model that autonomously can drive the tracks in the simulator application. 

The simulator can record images and meta data like speed, throttle and steering angle while a human is driving the course. This material will be used to train a neural network to clone the behaviour of the human driver.


#### The steps of this project are the following: ####
* Use the simulator to collect data of good driving behavior
* Exploration of the collected data, preprocessing and augmentation
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Monitor the training process using Tensorboard
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./documentation/recover_center.png "Get to center image"
[image4]: ./documentation/angles_histograms.png "Steering Angle Histograms"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./documentation/imgaug_0.png "Image Augment 0"
[image9]: ./documentation/imgaug_1.png "Image Augment 1"
[image10]: ./documentation/imgaug_2.png "Image Augment 2"
[image11]: ./documentation/imgaug_3.png "Image Augment 3"
[image12]: ./documentation/imgaug_4.png "Image Augment 4"
[image13]: ./documentation/imgaug_5.png "Image Augment 5"
[image14]: ./documentation/monitoring.png "Tensorboard"
[image15]: ./documentation/youtube.png "YT Link"

Files Submitted
---

#### 1. Submission includes the following files:

* packages/training.py containing the complete pipeline to train a deep learning model
* packages/kerasmodels.py containing the definition of the keras models
* packages/dataset.py for in-memory representation of a training/validation dataset
* packages/datapoint.py for a single 'snapshot' of the 3 camera images plus meta data
* packages/batchgenerator.py for training batchwise with image preprocessing and augmentation
* packages/imageaugmentation.py for extended image augmentation

* video.py create a video from frames
* drive.py for driving in autonomous mode in the simulator

* model.h5 containing a trained convolution neural network


### Dependencies
This lab requires:

* [Image Augmentation by aleju/imgaug](https://github.com/aleju/imgaug)


Collection of training data from the simulator
---

I used the simulator to create my training dataset and a separate validation dataset. For this purpose I did drive one lap of the track 1 and track 2 in both directions. Driving the course I tried to keep the center of the track, smooth steering with the mouse and avoid zero-angles. In my point of view, controlling the car with keyboard would have let to mostly 0.0 angles and sharp and high steering angles when hitting the A or D keys. I tried to avoid that. 


#### Train to avoid the track border

To let the model learn how to avoid the border of the track I created a small training set that I added to the initial one. In this new training set I did only steer the car to get back on a driving direction that will lead to the track center (wavy line driving). In all other situations I let the steering angle be 0.0. When loading the training set I excluded all samples with a steering angle of 0.0. So all samples in the set are situations, where I used the control to get a driving direction back to the center of the track.

![RecoverCenter][image3]


Exploration of the collected data, preprocessing and augmentation
---

For loading the simulator output into the python program I wrote the class **SimulatorDatasetImporter**. This class is capabile of reading in the csv file and create a list of instances of the class **Datapoint**. Using the reading method of the SimulatorDatasetImporter it is possible to exclude some specific steering angles or samples with steering angles that occur less than n times in the csv file. 


![AnglesHistograms][image4]

### Preprocessing

The image data was converted from RGB to grayscale to reduce input-depth complexicity and remove color-information. Afterwards, a pixelwise normalization with the help of a lambda layer is done: (x - 128) / 128. 

### Augmentation

For extending the dataset and as a method to reduce overfitting, I used a very sophisticated [library for image augmentation](https://github.com/aleju/imgaug). *To get my code working, it is necessary to clone this repository next to mine on the filesystem.* Additionally I used the keras imagedatagenerator to perform image augmentation. I included the following augmentation methods:

Image Rotation, Flip and Translation

![Imgaug0][image8]

Local intensity shifts (simplex noise)

![Imgaug1][image9]
![Imgaug2][image10]

Coarse dropout in the image data and gaussian noise

![Imgaug5][image13]
![Imgaug3][image11]


Model Architecture and Training Strategy
---

In this context I decided to use different architectures that are computational efficient. I was inspired by [MobileNet](https://arxiv.org/abs/1704.04861), [SqueezeNet](https://arxiv.org/abs/1602.07360) and the architecture in this [NVIDIA Blogpost](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). For my final result I decided for the NVIDIA-inspired model with the following architecture:

| Layer | Description | Output Shape (Height x Width x Channels) |
| :---: | :---------: | :----------: |
| 0 | input | 160 x 320 x 1 |
| 1 | cropping2d | 80 x 320 x 1 | 
| 2 | lambda (x: (x-128.0) / 128.0 | 80 x 320 x 1 |
| 3 | Conv2D 5x5 same-padding | 80 x 320 x 24 |
| 4 | BN + ReLU | 80 x 320 x 24 |
| 5 | maxpool 2x2 | 40 x 160 x 24 |
| 6 | Conv2D 5x5 same-padding | 40 x 160 x 36 |
| 7 | BN + ReLU | 40 x 160 x 36 |
| 8 | maxpool 2x2 | 20 x 80 x 36 |
| 9 | Conv2D 5x5 same-padding | 20 x 80 x 48 |
| 10 | BN + ReLU | 20 x 80 x 48 |
| 11 | maxpool 2x2 | 10 x 40 x 48 |
| 12 | Conv2D 3x3 same-padding | 10 x 40 x 64 |
| 13 | BN + ReLU | 20 x 80 x 64 |
| 14 | maxpool 2x2 | 10 x 40 x 64 |
| 15 | Conv2D 5x5 same-padding | 10 x 40 x 64 |
| 16 | BN + ReLU | 10 x 40 x 64 |
| 17 | maxpool 2x2 | 5 x 20 x 64 |
| 18 | global maxpool | 64 |
| 19 | dropout | 64 |
| 20 | dense + ReLu | 100 |
| 21 | dropout | 100 |
| 22 | dense + ReLu | 50 |
| 23 | dropout | 50 |
| 22 | dense + ReLu | 10 |
| 23 | dropout | 10 |
| 24 | dense + Sigmoid | 1 |


The model contains regularization by dropout and batch-normalization layers. I left out weight decay because the model then did not perform like without weight decay. The activations (non-linearities) are Rectified Linear Units. The last output has a Sigmoid activtion to predict the steering angle. 

Since this model is quite small and simple, it was nicely trainable on my GTX 1070 within about 1 hour. 

This was not the case for my SqueezeNet and MobileNet-Like implementations. Because these implementations design a much deeper network, these models have **overfitted** very fast. The validation-loss increased after a really short time of training (some epochs).

### Training Strategy

To train the deep learning model, I created a **BatchGenerator** that is derived from the keras Iterator class. The function **_get_batches_of_transformed_samples** implements the former **next** method that returns a random batch of preprocessed and augmented images. The BatchGenerator will return a batch of n_b elements where it is random, whether the image from the left, right or center -camera is taken. If a frame from the left or right camera is choosen, the respective steering angle is corrected by a random offset within the intervall of 0.2 and 0.3.

I used a batchsize of 32 - chosen empirically. 
The Adam-Optimizer adjusts the learning rate in an adaptive way. It is used to optimized a mean squared error - loss function. As a monitoring metric I used mean absolute error. Using the fit_generator-method of keras starts the optimizer loop.

#### Montoring the training process

For monitoring the training process I used a keras tensorboard -callback.

![Tensorboard][image14]


Results and autonomous driving in simulator
---

To visualize my result on track 1, [I uploaded a video of the autonomous drive on Youtube](https://youtu.be/r9HVnz7Q2AU). There a picture-in-picture visualizes the images that are fed to the neural network.

[![Thumbnail][image15]](https://youtu.be/r9HVnz7Q2AU "Video Title")

The model is able to drive safely through the course without leaving the track. When the speed is set too high, there seems to be some lag in the prediction and the car begins to oscillate a litte bit (at speed 30), so I set the speed to 20. 


Reflection and Problems
---

* The initial experiments seemed to overfit very very fast on this dataset. Even with aggressive augmentation, the models could not generalize enough to drive on track 2, too. I first tried to learn only on track 1 and test on track 2. For this purpose, I tried a lot of augmentation (median filtering, inverting the image) - to force the model to only learn to detect the edges of the track. But this did not work.

* So, the initial experiments had too much parameters (too large models)

* Too much augmentation broke the training

* Reinforcement Learning may be a better strategy to solve this problem


