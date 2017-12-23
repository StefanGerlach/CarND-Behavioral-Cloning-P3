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


The model contains regularization by dropout and batchnormalization. I left out weight decay because the model then did not perform like without weight decay. The activations (non-linearities) are Rectified Linear Units. The last output has a Sigmoid activtion to predict the steering angle. 

Since this model is quite small and simple, it was nicely trainable on my GTX 1070 within about 1 hour. 

This was not the case for my SqueezeNet and MobileNet-Like implementations. Because these implementations design a much deeper network, these models have **overfitted** very fast. The validation-loss increased after a really short time of training (some epochs).



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.










Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

