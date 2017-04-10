import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib
matplotlib.use('agg')
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D, Lambda
from sklearn.utils import shuffle
from keras.models import load_model


# Generator function for storing a specific number of images in the memory
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []# Initialize list of images
            angles = [] # Initialize list of angles
            
            for batch_sample in batch_samples:
                #name = '.\\mydata\\IMG\\'+batch_sample[0].split('\\')[-1] # For use in Windows
                # Change hardcoded image path
                name = './mydata/IMG/'+batch_sample[0].split('\\')[-1] # For use in Linux
                # Read image
                img = cv2.imread(name)
                # Fetch stored angle
                angle = float(batch_sample[1])
                # Append info
                images.append(img)
                angles.append(angle)

            # Convert lists to numpy arrays
            X_train = np.array(images)
            y_train = np.array(angles)
            # Shuffle and return
            yield sklearn.utils.shuffle(X_train, y_train)


def main():
    samples = []# Initialize the master list containing all info
    with open('./mydata/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            steering_center = float(row[3])
            # create adjusted steering measurements for the side camera images
            correction = 0.3 # this is a parameter to tune
            steering_left = steering_center + correction # Left image steering correction
            steering_right = steering_center - correction # Right image steering correction
            # read paths from center, left and right cameras
            path_img_center = row[0]
            path_img_left = row[1]
            path_img_right = row[2]            
            # add images and angles to data set as tuples
            samples.append( (path_img_center,steering_center) )
            samples.append( (path_img_left,steering_left) )
            samples.append( (path_img_right,steering_right) )

    
    # Split to train and test set. 20% -> test set and 80% -> train set
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    
    # Compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    
    ch, row, col = 3, 160, 320  # The default image format
    
    # Create Keras sequential model
    model = Sequential()
    # Preprocess incoming data, centered around zero and between -0.5 and 0.5 . Output: 160,320,3
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
    # Preprocess incoming data, cropping image. New dimensions : 3x65X320
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch )))
    # Add 2D convolutional layer with 5x5 filter size and depth 24. Add activation of type: "RELU"
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", activation = "relu"))
    # Add 2D convolutional layer with 5x5 filter size and depth 36. Add activation of type: "RELU"
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", activation = "relu"))
    # Add 2D convolutional layer with 5x5 filter size and depth 48. Add activation of type: "RELU"
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", activation = "relu"))
    # Add 2D convolutional layer with 3x3 filter size and depth 64. Add activation of type: "RELU"
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation = "relu"))
    # Add 2D convolutional layer with 3x3 filter size and depth 64. Add activation of type: "RELU"
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation = "relu"))
    # Flatten the structure so the input now is a vector
    model.add(Flatten())
    # Add fully connected layer of size 100
    model.add(Dense(100))
    # Add activation of type: "RELU"
    model.add(Activation('relu'))
    # Add dropout layer with probability 0.2
    model.add(Dropout(0.2))
    # Add fully connected layer of size 50
    model.add(Dense(50))
    # Add activation of type: "RELU"
    model.add(Activation('relu'))
     # Add dropout layer with probability 0.2
    model.add(Dropout(0.2))
    # Add fully connected layer of size 10
    model.add(Dense(10))
    # Add activation of type: "RELU"
    model.add(Activation('relu'))
    # Add FC layer of one node to display the output (regression)
    model.add(Dense(1))
    # Select optimizer, metric and loss
    model.compile(loss='mse', optimizer='adam')
    # Begin training and evaluation
    history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples),  validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)
    
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    # Save fig
    #plt.savefig('diagram.jpg')
    
    # Save the model
    model.save('model.h5')

if __name__ == '__main__':
    main()
