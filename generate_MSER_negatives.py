import os
import cv2
import numpy as np
import scipy.io as sio
import json

import re

_nsre = re.compile('([0-9]+)')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

"""
    This file will generate MSER regions outside the cropped area of the SVHN dataset
"""

if __name__ == "__main__":

    print("============== generate_MSER_negatives.py ==============")


    if 1:  #toggle true to add to the train directory
        #data = json.load(open('labels/digitStruct_train.json'))
        #data_dir = "raw_data/train"
        #data = json.load(open('labels/digitStruct_extra.json'))
        #data_dir = "raw_data/extra"
        data = json.load(open('labels/digitStruct_test.json'))
        data_dir = "raw_data/test"
        out_dir = "processed_data/detect_test/0/"
        #out_dir = "detect_train/0/"

    #  Load the images of interest in normalized grayscale
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(".png")]
    imagesFiles.sort(key=natural_sort_key)
    c = -1
    #c = 64190

    for i in range(len(imagesFiles)):
        print(i)
        vis = cv2.imread(os.path.join(data_dir, imagesFiles[i]))
        img = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)

        # get the bounding box of the cropped image
        filename = data['digitStruct'][i]['name']
        width = list()
        height = list()
        top = list()
        left = list()
        try:
            nums = 1
            width.append(data['digitStruct'][i]['bbox']['width'])
            height.append(data['digitStruct'][i]['bbox']['height'])
            top.append(data['digitStruct'][i]['bbox']['top'])
            left.append(data['digitStruct'][i]['bbox']['left'])
        except:
            nums = len(data['digitStruct'][i]['bbox'])
            for n in range(nums):
                width.append(data['digitStruct'][i]['bbox'][n]['width'])
                height.append(data['digitStruct'][i]['bbox'][n]['height'])
                top.append(data['digitStruct'][i]['bbox'][n]['top'])
                left.append(data['digitStruct'][i]['bbox'][n]['left'])

        for n in range(nums):
            T = top[n]
            L = left[n]
            if T < 0:
                T = 0
            if L < 0:
                L = 0
            B = (top[n] + height[n])
            R = (left[n] + width[n])
            #print(L,T,R,B)
            cv2.rectangle(vis, (L, T), (R, B), (0, 255, 0), 1)

        #  Construct combined bounding box for all digits found
        mintop = int(0.9*min(top))
        minleft = int(0.9*min(left))
        maxtop = int(1.1*(max(top) + max(height)))
        maxleft = int(1.1*(max(left) + max(width)))
        centx = int((minleft +maxleft)/2.0)
        centy = int((mintop +maxtop)/2.0)

        #cv2.rectangle(vis, (minleft, mintop), (maxleft, maxtop), (0, 0, 255), 1)

        #  generate candidates
        mser = cv2.MSER()

        regions = mser.detect(img, None)

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

        hull_length = list()
        for h in hulls:
            hull_length.append(len(h))
        mean_hull = np.mean(hull_length)
        std_hull = np.std(hull_length)

        for i,bb in enumerate(bbs):
            y = bb[0]
            x = bb[2]
            h = bb[1]-bb[0]
            w = bb[3]-bb[2]
            bb_centx = int(x + w/2.0)
            bb_centy = int(y + h/2.0)
            if (bb_centx > minleft and bb_centx < maxleft) or (centx > x and centx < (x+w)) or (w < 16 or h < 16):
                pass
            else:
                c +=1
                crop = vis[y:(y+h),x:(x+w)]
                crop = cv2.resize(crop, (64, 64))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                print(out_dir+str(c)+'.png')
                cv2.imwrite(filename=out_dir+str(c)+'.png', img=crop)

        if 0:
            cv2.imshow('full', vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
