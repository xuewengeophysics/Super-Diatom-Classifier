import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import isfile, join
import errno
import pickle

def showImg(img, scale=1):
    plt.figure(figsize=(5,5))
    plt.imshow(img,cmap='gray')
    plt.show()
    
def check_dirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    
def saveImg(img, path):
    check_dirs(path)
    cv2.imwrite(path,img)
    
def savePickle(obj, path):
    check_dirs(path)
    pickle_out = open(path,"wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()

def round_rectangle(radius, w, h, value=255):
    thickness = -1
    color = value
    rr = np.zeros((w, h))
    rr = cv2.circle(rr, (radius, radius), radius, color, thickness) 
    rr = cv2.circle(rr, (h-radius, radius), radius, color, thickness) 
    rr = cv2.circle(rr, (radius, w-radius), radius, color, thickness) 
    rr = cv2.circle(rr, (h-radius, w-radius), radius, color, thickness) 
    rr = cv2.rectangle(rr, (0, radius), (h, w-radius), color, thickness) 
    rr = cv2.rectangle(rr, (radius, 0), (h-radius, w), color, thickness) 
    return rr

def resize_img(img, scale_percent):
    if scale_percent==1:
        return img
    else:
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height) 
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
        return resized
    
def benchmark(n, fastp):
    total_time = 0
    n = 0
    random.seed()
    for i in range(n):
        start = time.time()
        final_image, annotations = main_generator(simple_angles = False, size_px = 1000, fast=fastp, verbose=False, overlapping=True)
        total_time += time.time()-start
        n+=1
        print(n)
    print(total_time/n)
