# %%
import cv2
import numpy
import random

from os import listdir
from os.path import isfile, join
DATASET_PATH = "./data/ra"

# %%
images = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]

# %% LOADING n RANDOM images
n = random.randint(7,10)
tmp_images = []
for i in range(n):
    tmp = random.choice(images)
    while(len(a.split('.'))<2 or tmp.split('.')[1]!='png'):
        tmp = random.choice(images)
    tmp_images.append(tmp)

# %%
img_path = join(DATASET_PATH, f)