import os
import cv2
import json
import numpy as np
import scipy.io as sio

if __name__ == "__main__":

    print("=============== generate_positives.py =================")

    # Set if generating examples for class or detect
    flag = "class"
    cflag = "color"

    #  Load data from the prepared digitStruct
    if 0:  #  toggle true to add to the train directory
        data = json.load(open('labels/digitStruct_train.json'))
        data_dir = "raw_data/train"
        if flag == "class":
            out_dir = "processed_data/classifier_train"
        else:
            out_dir = "processed_data/detector_train"

    if 1:   #toggle true to add to the test directory
        data = json.load(open('labels/digitStruct_test.json'))
        data_dir = "raw_data/test"
        if flag == "class":
            out_dir = "processed_data/classifier_test"
        else:
            out_dir = "processed_data/detector_test"

    #  Number of images
    N = len(data['digitStruct'])

    #  for separating into 10 classes
    f_dict = {}
    for i in range(1, 11):
        f_dict[i] = -1

    c = -1

    #  For each file in digitStruct...
    for i in range(N):
        #  Extract the following properties
        filename = data['digitStruct'][i]['name']
        width = list()
        height = list()
        top = list()
        left = list()
        label = list()

        #  This is needed because of the way the json was constructed (handle mult digits in 1 image)
        try:
            nums = 1
            width.append(data['digitStruct'][i]['bbox']['width'])
            height.append(data['digitStruct'][i]['bbox']['height'])
            top.append(data['digitStruct'][i]['bbox']['top'])
            left.append(data['digitStruct'][i]['bbox']['left'])
            label.append(data['digitStruct'][i]['bbox']['label'])
        except:
            nums = len(data['digitStruct'][i]['bbox'])
            for n in range(nums):
                width.append(data['digitStruct'][i]['bbox'][n]['width'])
                height.append(data['digitStruct'][i]['bbox'][n]['height'])
                top.append(data['digitStruct'][i]['bbox'][n]['top'])
                left.append(data['digitStruct'][i]['bbox'][n]['left'])
                label.append(data['digitStruct'][i]['bbox'][n]['label'])


        for n in range(nums):
            #print(c)
            img = cv2.imread(os.path.join(data_dir, filename))
            LF = 1
            HF = 1
            T = int(LF*top[n])
            L = int(LF*left[n])
            if T < 0:
                T = 0
            if L < 0:
                L = 0
            B = int(HF * (top[n] + height[n]) )
            R = int(HF * (left[n] + width[n]) )
            crop = img[T:B, L:R]

            if flag == "class":
                f_dict[label[n]] += 1
                if label[n] != 10:
                    print(out_dir+"/"+str(label[n])+str("/")+str(f_dict[label[n]])+'.png')
                    cv2.imwrite(filename=out_dir+"/"+str(label[n])+str("/")+str(f_dict[label[n]])+'.png', img=crop)
                else:
                    print(out_dir+"/"+str(0)+str("/")+str(f_dict[label[n]])+'.png')
                    cv2.imwrite(filename=out_dir+"/"+str(0)+str("/")+str(f_dict[label[n]])+'.png', img=crop)
            else:
                c += 1
                print(out_dir+"/"+str(1)+str("/")+str(c)+'.png')
                cv2.imwrite(filename=out_dir+"/"+str(1)+str("/")+str(c)+'.png', img=crop)

            if 0:
                cv2.imshow("Image", img)
                cv2.imshow("Crop", crop)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

print("Done")
print("Stats:", f_dict)
