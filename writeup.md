# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image3]: ./examples/normal.gif "Normal Image (Udacity)"
[image4]: ./examples/recovering.gif "Recovery Image (Udacity)"

[image6]: ./examples/normal_me.gif "Normal Image (My Data Set)"
[image5]: ./examples/recovery_me.gif "Recovery Image (My Data Set)"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `trainer.ipynb` The notebook containing the training information
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `nvidia_model_2.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results
* `video.mp4` a sample video of the given track with autonomous drive

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing:

```sh
python drive.py nvidia_model_2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. For an example run see the trainer.ipynb file which includes all the logs.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The base model used here is the one described in the [Nvidia Paper](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The images are also normalized at first using a single Lambda layer. The first layers are several convulsion layers that are connected with batch normalized one. This maintains the mean activation close to 0. The fully connected layers then are connected with drop out layers, to avoid overfitting.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The validation and training data set are not related an are different. This makes the chance of overfitting much smaller.

#### 3. Model parameter tuning

For the parameter tunning the model uses the AdamOptimizer.

#### 4. Appropriate training data

To create the training data I used the training data given by Udacity. I also tried to drive the track one with good behavior one time forward and once backward. The same procedure was done on the track 2.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

To design the model, I tried the to implement model used in traffic classifier first, with last part of the model done, different. This didn't work well, so I started to implement the Nvidia paper given model. It had some overfitting, so I added drop outs to the flat fully connected layers. During the start of the training my memory was full really fast so, I had to start to use the generator, I implemented the generators for the fitting part of the model. With this I felt that I dont need to change the image size with cropping and resize as all my batches will be fitted in the memory. The only change on the image was normalization. Between multiple layers, I also added batch normalization to hold the mean activation near 0.

#### 2. Final Model Architecture

Here is a visualization of the architecture:

|  Layer (type)               | Output Shape             | Param #  |
|-----------------------------|--------------------------|----------|
|lambda_3 (Lambda)            |(None, 160, 320, 3)       |  0       |                    
|batch_normalization_11 (Batc |(None, 160, 320, 3)       |12        |
|conv2d_11 (Conv2D)           |(None, 78, 158, 24)       |1824      |   
|conv2d_12 (Conv2D)           |(None, 37, 77, 36)        |21636     |     
|conv2d_13 (Conv2D)           |(None, 17, 37, 48)        |43248     |    
|batch_normalization_12 (Batc |(None, 17, 37, 48)        |192       |     
|conv2d_14 (Conv2D)           |(None, 15, 35, 64)        |27712     |    
|conv2d_15 (Conv2D)           |(None, 13, 33, 64)        |36928     |    
|flatten_3 (Flatten)          |(None, 27456)             |0         |  
|dense_9 (Dense)              |(None, 100)               |2745700   |  
|batch_normalization_13 (Batc |(None, 100)               |400       |     
|dropout_5 (Dropout)          |(None, 100)               |0         |     
|dense_10 (Dense)             |(None, 50)                |5050      |    
|batch_normalization_14 (Batc |(None, 50)                |200       |     
|dropout_6 (Dropout)          |(None, 50)                |0         |     
|dense_11 (Dense)             |(None, 10)                |510       |     
|batch_normalization_15 (Batc |(None, 10)                |40        |     
|dense_12 (Dense)             |(None, 1)                 |11        |     


Total params: 2,883,463

Trainable params: 2,883,041

Non-trainable params: 422

#### 3. Creation of the Training Set & Training Process

To have a good training set I used two sets of data:

1. The data set provided by Udacity for the first track.
2. My own created data set from the first and second track.

The Udacity created data set had both normal driving behavior and recovery. This can be seen in the images:

![alt text][image3]
![alt text][image4]

For my own data set creation I tried to do the same. It includes good driving behavior and also recovery. The data set also includes one time forward/backward run in both of the tracks. With having a backward run I generalized my model much better.

![alt text][image6]
![alt text][image5]

I also used the flipping feature with negative steering to have a larger data set. As the flipping was done on the fly, there is no image to see.
 
Steering value of `0` which is for straight road happens much and weights on the whole data as clutter. I tried to remove it from the data set with 70% chance, to have a more balanced data set. The result was better turning in the sharp curves.

For the left and right images I added the following corrections to the steering value:

- `0.27`
- `0.23`

and included them from both data set in my sample. At the end my data set would have `27101` samples. I finally randomly shuffled the data set and put `20%` of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Then I tried the run the trainer with 10 epochs, From the 7th epoch the model performance got worse so, I tried again with 7 epochs to avoid overfitting.