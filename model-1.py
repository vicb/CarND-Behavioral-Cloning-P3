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
    correction = 0.2 # this is a parameter to tune
    steering_left = steering + correction
    steering_right = steering - correction

    # read in images from center, left and right cameras
    img = read_image(line[0])
    img_flip = np.fliplr(img)
    img_left = read_image(line[1])
    img_right = read_image(line[2])

    # add images and angles to data set
    images.extend([img, img_flip, img_left, img_right])
    angles.extend([steering, -steering, steering_left, steering_right])

X_train = np.array(images)
y_train = np.array(angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model-1.h5')



