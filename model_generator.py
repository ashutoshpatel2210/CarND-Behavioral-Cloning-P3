import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import keras
import sklearn
#import imgaug
from keras.models import Sequential, Model
from keras.layers import Lambda
from keras.optimizers import Adam
from keras.layers import Convolution2D, Cropping2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random

# Read csv from directory
# Return data frame 
def read_csv(path):
    columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = columns)
    pd.set_option('display.max_colwidth', -1)
    data.head()
    return data;

def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail

data = read_csv("data/")
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)

print("total data::", len(data))

#print(data.iloc[1])

#Check data 
def load_image_steering_data(datadir, dataframe):
    image_path=[]
    steering=[]
    correction = 0.2
    for i in range(1,len(data)):
        index = data.iloc[i]
        center, left, right = index[0], index[1], index[2]
        image_path.append(os.path.join(datadir, center.strip()))
        image_path.append(os.path.join(datadir, left.strip()))
        image_path.append(os.path.join(datadir, right.strip()))
        steering.append(float(index[3]))
        steering.append(float(index[3]) + 0.2)
        steering.append(float(index[3]) - 0.2)
    image_paths = np.asarray(image_path)
    steering = np.asarray(steering)
    #print(image_paths)
    #print(steering)
    return image_paths, steering

def img_preprocess(img):
    img = img[60:135,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

def validation_images(image, steering):
    random_number = np.random.rand()
    if random_number < 0.20:
        brightness = iaa.Multiply((0.2, 1.2))
        image = brightness.augment_image(image)
    elif random_number > 0.20 and random_number < 0.40:
        zoom = iaa.Affine(scale=(1, 1.3))
        image = zoom.augment_image(image)
    elif random_number > 0.40 and random_number < 0.60:
        pan = iaa.Affine(translate_percent= {"x" : (-0.1, 0.1), "y": (-0.1, 0.1)})
        image = pan.augment_image(image)
    else:
        image = cv2.flip(image,1)
        steering = -steering 
    return image, steering


def behavior_model():
    model=Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    #model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation = 'elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(50, activation = 'elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'elu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1))
    optimizer = Adam(lr=0.0001)
    model.compile(loss='mse', optimizer=optimizer)
    return model

def batch_generator (image_paths, steering, batch_size, in_training):
    
    while 1: # Loop forever so the generator never terminates
        batch_images = []   
        batch_steerings = []
        #X_train, y_train = shuffle(X_train, Y_train)
        for i in range(batch_size):
            random_number = random.randint(0, len(image_paths) - 1)
            if in_training == True:
                img = mpimg.imread(image_paths[random_number])
                angle = steering[random_number]
                randomness = np.random.rand()
                if randomness < 0.5:
                    img, angle = validation_images(img, angle)                
            else:
                img = mpimg.imread(image_paths[random_number])
                angle = steering[random_number]
                
            img = img_preprocess(img)
            batch_images.append(img)
            batch_steerings.append(angle)
        yield (np.asarray(batch_images), np.asarray(batch_steerings))

# Set our batch size
batch_size=32

# compile and train the model using the generator function

image_paths,steering = load_image_steering_data("data/IMG/", data)


X_train, X_valid, Y_train, Y_valid = train_test_split(image_paths, steering, test_size=0.2, random_state=6)
print(X_train.shape)
print(Y_train.shape)
model = behavior_model()
print(model.summary())

history = model.fit_generator(batch_generator(X_train, Y_train, 100, True),
                                  steps_per_epoch=300, 
                                  epochs=10,
                                  validation_data=batch_generator(X_valid, Y_valid, 100, False),
                                  validation_steps=200,
                                  verbose=1,
                                  shuffle=1)
model.save('model.h5')