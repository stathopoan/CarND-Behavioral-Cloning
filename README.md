# Behavioral Cloning

---

The goals / steps of this project are the following:

- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Prerequisites

#### Please download the simulator [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip).
#### Find more info in simulator repository [here](https://github.com/udacity/self-driving-car-sim)
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using Udacity's simulator and my drive.py file, the car can be driven in autonomous mode around the first track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py contains all the code for building up the model, training and saving it at a later stage.

### Model Architecture and Training Strategy
#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 or 5X5 filter sizes and depths between 24 and 64 (model.py lines 78-112). Next a series of fully connected layers have been added to enchance the results.

The model includes RELU layers between convolutional and fully connected layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 80). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 100,106). 

Using the training and validation loss diagram at the end of each run along with dropout layers i ensured the network was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a number of approaches to ensure the car stays on the road. 

For details about how I created the training data, see the next section. 
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a very simple network that works at least for the first meters and then building it up untill the results are satisfying. As a reference model i used the the Lenet model.

My first step was to use a convolution neural network model similar to the network on the lectures. To proceed with validation I split my image and steering angle data into a training and validation set with a ratio of 80% and 20% respectively.
After my initial network was working i started building it up addding more convolutional layers and fully connected layers.
The first attempts had a low mean squared error on the training set but a high mean squared error on the validation set. The model was overfitting. 

To overcome overfitting, I modified the model and added dropout layers with probability of 0.1 and finally 0.2. One mistake i corrected was that i had not added activation layers between the fully connected layers so i missed not-linearity relations. Furthermore i selected the number of epochs so the validation error was at minimum.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track especially in turns. To improve the driving behavior in these cases, I added more convolutional layers and generally i tried many layer sizes and depth combinations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 78-112) consisted of a convolution neural network along with a fully connected neural network with the following layers and layer sizes:

- Layer for normalizing images using Lambda function
- Layer for cropping images
- Convolutional layer with 5X5 filter size and depth 24
- Activation layer of type 'RELU'
- Convolutional layer with 5X5 filter size and depth 36
- Activation layer of type 'RELU'
- Convolutional layer with 5X5 filter size and depth 48
- Activation layer of type 'RELU'
- Convolutional layer with 3X3 filter size and depth 64
- Activation layer of type 'RELU'
- Convolutional layer with 3X3 filter size and depth 64
- Activation layer of type 'RELU'
- Flatten Layer
- Fully connected layer with size of 100 nodes
- Activation layer of type 'RELU'
- Dropout layer with probability of 0.2
- Fully connected layer with size of 50 nodes
- Activation layer of type 'RELU'
- Dropout layer with probability of 0.2
- Fully connected layer with size of 10 nodes
- Activation layer of type 'RELU'
- Fully connected layer with size of 1 node (Output)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane image](http://i.imgur.com/C1vZYFv.jpg)

Then i recorded another three laps driving on the opposite direction.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct the steering angle in case it approaches the sides of the road. These images show what a recovery looks like:

![](http://i.imgur.com/zRzXzLW.jpg)
![](http://i.imgur.com/CIy5nUK.jpg)
![](http://i.imgur.com/dW6QlPE.jpg)
![](http://i.imgur.com/sRddCBR.jpg)

While i was testing my model on the simulator i noticed that the car had a major problem in corners. I started to collect more data but only on the corners as smoothly as i could. I did this for 3 to 5 times on all corners of the lap.

To augment the data set, I also used multiple cameras (left and right cameras) and i adjusted the steering angle accordingly. Some images from the left and right cameras can be seen below:

![](http://i.imgur.com/x1kuWBJ.jpg)
![](http://i.imgur.com/pozHmQA.jpg)



After the collection process, I had around 70000 number of data points. The preprocessing is being conducted in the model network with two layers:

- Normalizing image with mean value 0 and boundaries: -0.5 to 0.5
- Cropping image (70 pixels from top and 25 from bottom)


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by validation and training loss diagram. The final solution diagram can be seen below:

![](http://i.imgur.com/lhaRuTj.jpg)

 I used an adam optimizer so that manually training the learning rate wasn't necessary.

The optimal batch size is 32 since any larger value results in more epochs and different model architecture.

### Final video output
Here's a link to my [video result](./result_video.mp4)