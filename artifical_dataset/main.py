# %%
import cv2
import numpy as np
import random

from os import listdir
from os.path import isfile, join
DATASET_PATH = "./data/ra"

# %%
images = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]

# %% LOADING n RANDOM images
n = random.randint(7,10)
tmp_images = []
mean_brightness = 0
for i in range(n):
    # CHOOSING RANDOM IMAGE
    img_path = random.choice(images)
    while(len(img_path.split('.'))<2 or img_path.split('.')[1]!='png'):
        img_path = random.choice(images)
    # LOADING THE IMAGE
    img_path = join(DATASET_PATH, img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mean_brightness += np.mean(img)
    tmp_images.append(img)
mean_brightness /= len(tmp_images)

# %% NORMALIZING BRIGHTNESS
for img in tmp_images:
    img += (mean_brightness-np.mean(img)).astype(np.uint8)
    img[img>255] = 255

# %%
