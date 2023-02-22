# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:06:25 2020

@author: salar
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 01:31:57 2020

@author: salar
"""

import datetime
import glob
import os

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Conv2D, MaxPool2D, Input, Dropout, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

os.environ['CUDA_VISIBLE_DEVICES'] = ''

IMG_PATH = "./images/"
LBL_PATH = "./labels/"

# Train dataset and labels
train_set = pd.read_csv(LBL_PATH + "train_subset1.csv")
train_set = train_set.to_numpy()
# ======================================
'''
remov_train=(train_set[:,1]!=8)
train_set=train_set[remov_train,:]
'''
# ======================================
train_imgs = train_set[:, 0]
train_lbls = train_set[:, 1]

x_train = []
for i in train_imgs:
    img = cv2.imread(IMG_PATH + i)
    img = cv2.resize(img, dsize=(224, 224))
    img = img.astype('float16')
    img /= np.max(img)
    x_train.append(img)
# ======================================
'''
image=x_train[5]
plt.imshow(image)
'''
# ======================================
x_train = np.asarray(x_train)

y_train_nc = train_lbls.astype('int')
y_train = np_utils.to_categorical(y_train_nc)

# Test dataset and labels
test_set = pd.read_csv(LBL_PATH + "test_subset1.csv")
test_set = test_set.to_numpy()
# ======================================
'''
remov_test=(test_set[:,1]!=8)
test_set=test_set[remov_test,:]
'''
# ======================================
test_imgs = test_set[:, 0]
test_lbls = test_set[:, 1]

x_test = []
for i in test_imgs:
    img = cv2.imread(IMG_PATH + i)
    img = cv2.resize(img, dsize=(224, 224))
    img = img.astype('float16')
    img /= np.max(img)
    x_test.append(img)

x_test = np.asarray(x_test)

y_test_nc = test_lbls.astype('int')
y_test = np_utils.to_categorical(y_test_nc)

# Creating the model
myinput = Input(shape=(224, 224, 3))

x = Conv2D(32, 3, activation='relu', padding='same', strides=2)(myinput)
x = Dropout(0.2)(x)
x = Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)
x = Dropout(0.2)(x)
x = Conv2D(64, 3, activation='relu', padding='same', strides=2)(x)
x = Dropout(0.2)(x)
x = Conv2D(128, 3, activation='relu', padding='same', strides=2)(x)
x = Dropout(0.2)(x)
x = Conv2D(128, 2, activation='relu', padding='same', strides=2)(x)
x = Dropout(0.2)(x)
x = Conv2D(256, 2, activation='relu', padding='same', strides=2)(x)
x = Dropout(0.2)(x)
x = Conv2D(256, 2, activation='relu', padding='same', strides=2)(x)
x = Dropout(0.2)(x)

x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
out_layer = Dense(9, activation='softmax')(x)

mymodel = Model(myinput, out_layer)

mymodel.summary()
mymodel.compile(optimizer=Adam(0.0001),
                loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


# Defining the plotting method
def plot_history(net_history):
    history = net_history.history
    losses = history['loss']
    val_losses = history['val_loss']
    accuracies = history['accuracy']
    val_accuracies = history['val_accuracy']

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['loss', 'val_loss'])

    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    plt.legend(['accuracy', 'val_accuracy'])


# Training the model
es = EarlyStopping(monitor='val_accuracy', patience=10)
start = datetime.datetime.now()
network_history = mymodel.fit(x_train, y_train, batch_size=32, shuffle=True,
                              epochs=500, validation_split=0.2,
                              callbacks=[es])

plot_history(network_history)
end = datetime.datetime.now()

# Training time
elapsed = end - start
print('Total training time =', str(elapsed))

# Evaluation
test_loss, test_acc = mymodel.evaluate(x_test, y_test)
test_labels_p = mymodel.predict(x_test)

test_labels_p = np.argmax(test_labels_p, axis=1)

# Saving the model
mymodel.save('project.h5')

# Saving the model architecture
plot_model(mymodel, to_file='project.pdf', show_shapes=True)

# Confusion Matrix
import plt_confusion_matrix as plt_cm

plt_cm.plot_confusion_matrix(y_test_nc, test_labels_p,
                             title='Output of CNN (Test Data)',
                             labels=['1', '2', '3', '4', '5', '6', '7', '8', '9'])
plt.show()
