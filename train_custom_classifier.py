import os
import cv2
import numpy as np
import tensorflow as tf

import keras.utils
from keras import backend as kbe
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv3D, MaxPooling2D, Activation
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

if __name__ == "__main__":

    print("============== train_custom_classifier.py ==============")

    #  Class for digits, class for no digits
    num_classes = 10

    if 1:  # for Floydhub cloud computing
        training_dir = '/data/classifier_train'
        testing_dir = '/data/classifier_test'
        model_path = "/output/svhn-classifier-model.hdf5"
        log_path = "/output/custom_classifier_log.csv"
    if 0:  # For local machine
        training_dir = 'processed_data/classifier_train'
        testing_dir = 'processed_data/classifier_test'
        model_path = "models/svhn-classifier-model-new.hdf5"
        log_path = "logs/custom_classifier_log_new.csv"

    #  hyperparameters
    batch_size = 128
    num_epoch = 8

    print("=====Start Training=====")

    #  Model works on 64 x 64 images cropped from SVHN dataset 1
    input_shape = (64, 64, 1)
    target_size = (64, 64)


    #  Source:
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=5, strides=5, border_mode='same',
                            input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))

    model.add(Conv2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))

    model.add(Conv2D(64, 1, 1, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 1, 1, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    #  Stochastic Gradient Descent
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    # destination for the saved trained model
    filepath = model_path
    checkpoint = ModelCheckpoint(filepath,
                                monitor='val_acc',
                                verbose=1,
                                save_best_only=True,
                                mode='max')

    #  Callback for early stopping
    stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, \
                                      patience=1, min_lr=0.001)

    logger = CSVLogger(log_path, separator=',', append=False)

    #  Generators needed to "avoid out of memory" problem
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=False)

    test_datagen = ImageDataGenerator(
            rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            training_dir,
            color_mode='grayscale',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
            testing_dir,
            color_mode='grayscale',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical')

    #  Fit the model
    model.fit_generator(
            train_generator,
            steps_per_epoch=1000,
            epochs=num_epoch,
            validation_data=test_generator,
            validation_steps=100,
            callbacks=[checkpoint, stopping, logger])
