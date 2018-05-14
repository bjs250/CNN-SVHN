import os

from keras import backend as K
import keras.utils
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

if __name__ == "__main__":

    print("============== train_VGG16.py ==============")

    if 1:  # for Floydhub cloud computing
        training_dir = '/data/classifier_train'
        testing_dir = '/data/classifier_test'
        model_path = "/output/svhn-classifier-vgg-scratch-model.hdf5"
        log_path = "/output/vgg_scratch_log.csv"
    if 0:  # For local machine
        training_dir = 'processed_data/classifier_train'
        testing_dir = 'processed_data/classifier_test'
        model_path = "models/svhn-classifier-vgg-scratch-model_new.hdf5"
        log_path = "logs/vgg_scratch_log_new.csv

    #  Class for digits, class for no digits
    num_classes = 10
    training_dir = 'processed_data/classifier_train'
    testing_dir = 'processed_data/classifier_test'
    input_shape = (64, 64, 1)
    target_size = (64, 64)
    batch_size = 128
    num_epoch = 20

    model = Sequential()

    model.add(Conv2D(64, (3, 3), border_mode='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3),  border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dense(4096, activation='relu'))
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

    #  Callbacks
    stopping = EarlyStopping(monitor='val_loss', patience=3, mode='auto')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, \
                                      patience=1, min_lr=0.001)

    logger = CSVLogger(log_path, separator=',', append=False)

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
