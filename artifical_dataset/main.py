# %%
import cv2
import imutils
import numpy as np
import random

from os import listdir
from os.path import isfile, join
DATASET_PATH = "./data/ra"

%run util.py

# %% LOADING n RANDOM images
images = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]
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
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness += np.mean(img)
    tmp_images.append(img)
mean_brightness /= len(tmp_images)

# %% NORMALIZING BRIGHTNESS
for img in tmp_images:
    # showImg(img)
    img += (mean_brightness-np.mean(img)).astype(np.uint8)
    img[img>254] = 254
    # showImg(img)

# %%
random.seed(19)
size_px = 1000
art_img = (np.ones((size_px, size_px))*mean_brightness).astype(np.uint8)
for img in tmp_images:
    angle = random.randint(0,360)
    rotated = imutils.rotate_bound(img, angle)
    px, py = int(rotated.shape[0]/2), int(rotated.shape[1]/2)
    x, y = random.randint(0,size_px-1), random.randint(0,size_px-1) 
    # xmin, xmax = (x-px, 0)[x-px<0], (x+px, size_px-1)[x+px>size_px-1]
    # ymin, ymax = (y-py, 0)[y-py<0], (y+py, size_px-1)[y+py>size_px-1]
    xmin, xmax, ymin, ymax = x-px, x+px, y-py, y+py
    dxmin, dxmax = (0, -xmin)[xmin<0], (0, size_px-1-xmax)[xmax>size_px-1]
    dymin, dymax = (0, -ymin)[ymin<0], (0, size_px-1-ymax)[ymax>size_px-1]

    print("Center point: ", x, y)
    print("Patch size: ", px, py)
    print("Initial: ", xmin, xmax, ymin, ymax)
    print("Deltas: ", dxmin, dxmax, dymin, dymax)
    art_img[xmin+dxmin:xmax+dxmax, ymin+dymin:ymax+dymax] = rotated[dxmin:2*px+dxmax, ]
showImg(art_img)


# %%

# %%
""