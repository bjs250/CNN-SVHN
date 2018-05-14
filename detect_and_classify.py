import os
import cv2
import numpy as np
import tensorflow as tf
import scipy.io as sio

from keras import backend as kbe
import keras.utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv3D, MaxPooling2D, Activation
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
    This file will run MSER, digit/non-digit detector, and 10-digit CNN classifier
"""

def predict(img, model):
    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0)
    img = img.reshape((img.shape[0],img.shape[1],img.shape[2],1))
    p = model.predict_proba(img)
    y_classes = p.argmax(axis=-1)
    m = np.max(p[0])
    v = p.argmax(axis=-1)
    return (m,v[0])

def add_noise(image, channel, sigma):
    temp_image = np.copy(image)
    noise = np.random.normal(0, sigma, (temp_image.shape[0], temp_image.shape[1]))
    temp_image[:, :, channel] = temp_image[:, :, channel] + noise
    return temp_image

if __name__ == "__main__":

    print("============== detect_and_classify.py ==============")

    #  Load the CNN models for detection and classification
    detect_model = load_model("models/svhn-detector-custom-model.hdf5")
    class_model = load_model("models/svhn-classifier-custom-model.hdf5")

    #  Load in the image of interest and convert to grayscale
    INPUT_DIR = "input_images"
    OUTPUT_DIR = "graded"
    #INPUT_DIR = "input_images/shortcomings"
    #OUTPUT_DIR = "shortcomings"

    file_list = ["dark","noisy","orientation","scale","location"]

    for f in file_list:
        #  configure read and write names
        read_filename = INPUT_DIR + "/" + f + ".png"
        write_filename = OUTPUT_DIR + "/" + f + ".png"
        vis = cv2.imread(read_filename)

        if f == "noisy":
            vis = add_noise(vis, 0, 50)

        img = vis.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if f == "dark":
            img = cv2.medianBlur(img, 9)
        else:
            img = cv2.medianBlur(img, 7)

        #  Use MSER to detect regions of interest
        mser = cv2.MSER()
        regions = mser.detect(img, None)

        #  Get the bounding boxes for the regions of interest
        bbs = list()
        hulls = list()
        for i, region in enumerate(regions):
            (x, y, w, h) = cv2.boundingRect(region.reshape(-1,1,2))

            #  Rule: no wide rectangles
            if w > 1.25*h:
                continue

            #  Rule: no rectangles that are too long
            if h > 3*w:
                h = 3 * w

            bb = ((y, y+h, x, x+w))
            bbs.append(bb)
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            hulls.append(hull)

        #  Generate some stats on the hulls/bbs
        hull_length = list()
        for h in hulls:
            hull_length.append(len(h))
        mean_hull = np.mean(hull_length)
        std_hull = np.std(hull_length)

        #  Remove some bounding boxes based on the detector CNN
        s = list()
        for i, bb in enumerate(bbs):
            if len(hulls[i]) > mean_hull - .2*std_hull:
                crop = vis[bb[0]:bb[1],bb[2]:bb[3]]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                m, v = predict(crop, detect_model)
                if v == 1:
                    y = bb[0]
                    x = bb[2]
                    h = bb[1]-bb[0]
                    w = bb[3]-bb[2]
                    A = h * w
                    m, v = predict(crop, class_model)
                    s.append((bb, A, v))

        #  Poor man's NMS
        sorted(s, key=lambda tup: tup[1])
        s = s[::-1]

        r = list()
        for i in range(len(s)):
            r.append(0)

        for i in range(len(s)):
            for j in range(len(s)):
                if i != j and \
                (s[i][0][2] < s[j][0][2]) and \
                (s[i][0][3] > s[j][0][3]) and \
                (s[i][0][0] < s[j][0][0]) and \
                (s[i][0][1]> s[j][0][1]):
                    # i is big, j is small
                    if s[i][2] != 0 and s[j][2] == 0:
                        r[j] = 1
                    elif s[j][2] == s[i][2]:
                        r[j] = 1

        s2 = list()
        r2 = list()
        for i in range(len(r)):
            if r[i] == 0:
                s2.append(s[i])
                r2.append(0)

        for i in range(len(s2)):
            for j in range(len(s2)):
                if i != j and \
                (s2[i][0][2] < s2[j][0][2]) and \
                (s2[i][0][3] > s2[j][0][3]) and \
                (s2[i][0][0] < s2[j][0][0]) and \
                (s2[i][0][1]> s2[j][0][1]):
                    # i is big, j is small
                    if s2[i][2] != s2[j][2]:
                        r2[i] = 1

        cand = list()
        for i in range(len(r2)):
            if r2[i] == 0:
                cand.append(s2[i][0])

        #  Apply the CNN's the bounding boxes
        for i,bb in enumerate(cand):
            y = bb[0]
            x = bb[2]
            h = bb[1]-bb[0]
            w = bb[3]-bb[2]
            crop = vis[bb[0]:bb[1],bb[2]:bb[3]]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            m, v = predict(crop, class_model)
            if m > .90:
                cv2.putText(vis, str(v), (x, y+int(0.2*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
                cv2.rectangle(vis, (x, y), (x+w,y+h), (0, 255, 0), 1)
                if 0:
                    cv2.imshow('img', crop)
                    cv2.imshow("resize", cv2.resize(crop, (64, 64)))
                    cv2.imshow('full', vis)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

        cv2.imwrite(filename=write_filename, img=vis)

        if 0:
            cv2.imshow('full', vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

print("=== DONE ===")
