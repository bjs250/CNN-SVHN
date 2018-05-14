from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import numpy as np

"""
    This file will train top layers of VGG16 using imagenet weights for 10-digit CNN classifier
"""

if __name__ == "__main__":

    print("============== pretrained_VGG16.py ==============")

    if 1:  # for Floydhub cloud computing
        training_dir = '/data/classifier_train'
        testing_dir = '/data/classifier_test'
        model_path = "/output/svhn-classifier-vgg-pretrained-model.hdf5"
        log_path = "/output/vgg_pretrained_log.csv"
    if 0:  # For local machine
        training_dir = 'processed_data/classifier_train'
        testing_dir = 'processed_data/classifier_test'
        model_path = "models/svhn-classifier-vgg-pretrained-model-new.hdf5"
        log_path = "logs/vgg_pretrained_log_new.csv"

    batch_size = 128
    num_epoch = 20
    target_size = (64, 64)

    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    input = Input(shape=(64, 64, 3), name = 'image_input')
    output_vgg16_conv = model_vgg16_conv(input)
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(10, activation='softmax', name='predictions')(x)
    model = Model(input=input, output=x)

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
            color_mode='rgb',
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
            testing_dir,
            color_mode='rgb',
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
