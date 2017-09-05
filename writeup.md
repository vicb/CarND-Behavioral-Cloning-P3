#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-1.py / model-1.h5 - basic model with a single dense layer used to get familiar with the code, 
* model-2.py / model-2.h5 - more elaborate model based on NVIDIA paper,
* model-3.py / model-3.h5 - adding a few refinements and tuning to the NVIDIA model,
* model-3.mpa - video of model-3 output
* writeup.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model-3.h5
```

####3. Submission code is usable and readable

The model-3.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The architecture implemented in model-3 is based on the NIVIDIA network mentioned in the course:

The network starts with 5 convolutional nertwork:
- filter size 5x5, depth: 24,
- filter size 5x5, depth: 36,
- filter size 5x5, depth: 48,
- filter size 3x3, depth: 64,
- filter size 3x3, depth: 64.

There is then 3 fully connected layers before the output layers:
- 100 neurons,
- 50 neurons,
- 10 neurons.

All layers use a relu activation to introduce nonlinearity excluding the output layer that uses a tanh activation.

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout after each dense layer in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 

Also the model was training for only 5 epochs to reduce the chance of overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I started with the very simple model described in the course: A single dense layer after some normalization layers:
- adjusting the pixel values to be in the range -0.5 to 0.5,
- cropping the image to keep only the bottom portion where the road is.

This first trial was mainly done to test the workflow.

After having validated the workflow (capturing video, learning and simulating), I decided to try the architecture from the NVIDIA paper with 5 convolutional layers followed by 4 dense layers. This architecture is known to have given good results.

While the results (model-2) were promising and better than the first naive model (model-1), there were still times when the car would drive off road and would fail to recover.

I then tried to add dropout, play with the number of epochs, the size of the layers, ... but without much success. The loss would remain around 0.7-0.8. So many of my attempts to improve the model were unsuccessful.

As a software engineer when something is wrong, I would first double and triple check the algorithm which is what I did here without success. It took me a lot of time to figure out that the error was actually in the correction factor for the left & right camera. For some reason I had it set to 1.6 which was way too much.
By tuning this parameter I was finally able to dramatically reduce the loss and have the car stay on the track for several laps.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Creation of the Training Set & Training Process

Because the simulator is not easy to drive with the keyboard I have only used the provided training data as a starting point.

To augment the data set:
- I flipped images and angles so that the car can do as well for left and right turns,
- I used the images from the side cameras with a correction factor.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 

The model was trained for only 5 epochs (more would have increased the risk of over fitting eventhough the dropout should limit it). I stopped after 5 epochs because the loss was at a plateau both on the training set and on the validation set. The loss on the validation set is similar to the loss on the training set (even a little bit better, consistently) which indicates that there is no over fitting.

####3. Takeaway

As described in this document I started with simple models to validate the worflow before making the model more complex and augmenting the training data.

However when I foung the output not to be good enough I first (and for long) thought that the model was wrong and tried to debug it and tweak it.

It took me a long time to realize that my mistake was in the data augmentation and that the correction factor for side images was way too high.

One of the main takeaway for me is that every single step should be validated and the data is as important as the model (using TensorFlow or Kera make authoring a model quite simple).