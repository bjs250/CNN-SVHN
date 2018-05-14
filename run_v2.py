import os
import sys
import cv2
import numpy
import scipy
import tensorflow

"""
    This file is only used to run detect_and_classify.py, which will generate 5 images into the graded directory
"""

if __name__ == "__main__":

    print("============== run_v2.py ==============")

    if 0:  # version check
        print("Python version:" + sys.version)
        print("OpenCV version :  {0}".format(cv2.__version__))
        print("Numpy version: " + numpy.version.version)
        print("Scipy version: " + scipy.__version__)
        print("Tensorflow version: " + tensorflow.__version__)

    os.system('python detect_and_classify.py')
