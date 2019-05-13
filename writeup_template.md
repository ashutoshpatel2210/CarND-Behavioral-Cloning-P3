# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualizing_loss.png "Visualizing loss"
[image2]: ./examples/Model.svg "Model Visualization"
[image3]: ./examples/center_2016_12_01_13_41_15_584.jpg "Center Camera Image"
[image4]: ./examples/left_2016_12_01_13_41_15_584.jpg "Left Camera Image"
[image5]: ./examples/right_2016_12_01_13_41_15_584.jpg "Right Camera Image"
[image6]: ./examples/original2043.jpg "Original Image"
[image7]: ./examples/2043.jpg "Flipped Image"
[image8]: ./examples/original9907.jpg "Original Image"
[image9]: ./examples/9907.jpg "Flipped Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or summarizing the results
* video.mp4 A video recording of vehicle driving autonomously 2 laps around the track.


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5 x 5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 89-120) 
```python
    model=Sequential()
    #Convolution layer for Input size = 66x200x3 , Filters=24,  5x5 kernel filter size, activation ='elu'
    model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2), input_shape=(66, 200, 3)))
    #Convolution layer for Filters=36,  5x5 kernel filter size, activation ='elu'
    model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
    #Convolution layer for Filters=48,  5x5 kernel filter size, activation ='elu'
    model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
    #Convolution layer for Filters=68,  3x3 kernel filter size, activation ='elu'
    model.add(Conv2D(64, (3, 3), activation="elu"))
    #Convolution layer for Filters=68,  3x3 kernel filter size, activation ='elu'
    model.add(Conv2D(64, (3, 3), activation="elu"))
    
    #Flatten 
    model.add(Flatten())
    #Dense layer size 100 , activation ='elu'
    model.add(Dense(100, activation = 'elu'))
    #Dense layer size 50 , activation ='elu'
    
    model.add(Dense(50, activation = 'elu'))
    
    #Dense layer size 10 , activation ='elu'
    model.add(Dense(10, activation = 'elu'))
    
    #Dense layer size 1 , activation ='elu'
    model.add(Dense(1))
    #Adam optimizer learning rate = 0.0001
    optimizer = Adam(lr=0.0001)
    #Loss function mse
    model.compile(loss='mse', optimizer=optimizer)
```
The model includes ELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using image processing function (code line 69-75). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 134-136). For 50% images , The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
```python
'''
Apply data augmentation technique to training data
'''
def training_images(image, steering):
    #Flip remaining 50% of images
    image = cv2.flip(image, 1)
    steering = -steering 
    return image, steering
```
#### 3. Model parameter tuning

The model used an adam optimizer with learning rate 0.0001, so the learning rate was reduce from 0.001 to 0.0001 to reduce loss. (model.py line 116).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and flip images 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use transfer learning from Nvidia Behavioural model. 

My first step was to use a convolution neural network model similar to the Nvidia model. I thought this model might be appropriate because Nvidia model is already proven model for similar project. So I tried with Nvidia model directly on this project. 

##### Data selection and Augmentation
In order to gauge how well the model was working, I split my image and steering angle data into a training(80%) and validation set(20%). I found that my first model had a high mean squared error on the training set but a low mean squared error on the validation set.  I already added dropout after I trained model with only center images first. To avoid high traning loss(~0.500), i added images from left and right camera and logic to flip 50% of images randomly by cv2.flip() function. Number of images in the data/ folder are in the rangle ~20000. 

##### Activation function changed from relu to elu 

Steering angle(output) consists of positive and negative numbers. So i decided to use ELU function instead of RELU. RELU which works well for > 0 numbers while ELU provides smooth output for negative numbers. It combines advantages of RELU and Leaky RELU. 

After above changes, 
```
Epoch 1/10
300/300 [==============================] - 173s 577ms/step - loss: 0.0322 - val_loss: 0.0250
Epoch 2/10
300/300 [==============================] - 165s 549ms/step - loss: 0.0224 - val_loss: 0.0209
Epoch 3/10
300/300 [==============================] - 164s 547ms/step - loss: 0.0210 - val_loss: 0.0210
Epoch 4/10
300/300 [==============================] - 164s 548ms/step - loss: 0.0204 - val_loss: 0.0202
Epoch 5/10
300/300 [==============================] - 165s 550ms/step - loss: 0.0196 - val_loss: 0.0194
Epoch 6/10
300/300 [==============================] - 165s 550ms/step - loss: 0.0195 - val_loss: 0.0191
Epoch 7/10
300/300 [==============================] - 164s 547ms/step - loss: 0.0185 - val_loss: 0.0197
Epoch 8/10
300/300 [==============================] - 165s 548ms/step - loss: 0.0183 - val_loss: 0.0183
Epoch 9/10
300/300 [==============================] - 165s 548ms/step - loss: 0.0173 - val_loss: 0.0183
Epoch 10/10
300/300 [==============================] - 164s 548ms/step - loss: 0.0169 - val_loss: 0.0176
```
![alt text][image1]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I updated drive.py to use similar image processing pipeline (line 52-58, drive.py)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines lines 89-120) consisted of a convolution neural network with the following layers and layer sizes

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300    
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________

```


Here is a visualization of the architecture 

![alt text][image2]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:



I added data of left and right camera angle with tweaking sterring measurement

![alt text][image3]
![alt text][image4]
![alt text][image5]


After the collection process, I had ~20000 number of data points. 
To augment the data sat, I also flipped images and angles thinking that this would help to recude training loss For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]


![alt text][image8]
![alt text][image9]

I then preprocessed this data by 
```python
def img_preprocess(img):
        img = img[60:135,:,:]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,  (5, 5), 0)
        img = cv2.resize(img, (200, 66))
        img = img/255
        return img
```

I finally randomly shuffled the data set and put randomly chosed 50% of the data while generating batches into a training set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

```
Epoch 1/10
300/300 [==============================] - 173s 577ms/step - loss: 0.0322 - val_loss: 0.0250
Epoch 2/10
300/300 [==============================] - 165s 549ms/step - loss: 0.0224 - val_loss: 0.0209
Epoch 3/10
300/300 [==============================] - 164s 547ms/step - loss: 0.0210 - val_loss: 0.0210
Epoch 4/10
300/300 [==============================] - 164s 548ms/step - loss: 0.0204 - val_loss: 0.0202
Epoch 5/10
300/300 [==============================] - 165s 550ms/step - loss: 0.0196 - val_loss: 0.0194
Epoch 6/10
300/300 [==============================] - 165s 550ms/step - loss: 0.0195 - val_loss: 0.0191
Epoch 7/10
300/300 [==============================] - 164s 547ms/step - loss: 0.0185 - val_loss: 0.0197
Epoch 8/10
300/300 [==============================] - 165s 548ms/step - loss: 0.0183 - val_loss: 0.0183
Epoch 9/10
300/300 [==============================] - 165s 548ms/step - loss: 0.0173 - val_loss: 0.0183
Epoch 10/10
300/300 [==============================] - 164s 548ms/step - loss: 0.0169 - val_loss: 0.0176
```
