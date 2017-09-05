import csv
import cv2
import numpy as np

def read_image(filename):
    f = filename.split('/')[-1]
    path = './data/IMG/' + f
    img = cv2.imread(path)
    return img

lines = []
with open('./data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
            lines.append(line)

images = []
angles = []

for line in lines[1:]:
    steering = float(line[3])

    # create adjusted steering measurements for the side camera images
    correction = 0.15 # this is a parameter to tune
    steering_left = steering + correction
    steering_right = steering - correction

    # read in images from center, left and right cameras
    img = read_image(line[0])
    img_left = read_image(line[1])
    img_right = read_image(line[2])

    # add images and angles to data set
    images.extend([img, img_left, img_right])
    angles.extend([steering, steering_left, steering_right])

    if (steering != 0):
        img_flip = np.fliplr(img)
        images.append(img_flip)
        angles.append(-steering)

X_train = np.array(images)
y_train = np.array(angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Convolution2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model-3.h5')
