# %%
import cv2
import imutils
import numpy as np
import random

from os import listdir
from os.path import isfile, join
DATASET_PATH = "./data/ra"

from sklearn.neighbors import KDTree

random.seed(19)
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
size_px = 1000
art_img = (np.ones((size_px, size_px))*mean_brightness).astype(np.uint8)
global_patch = np.zeros_like(art_img)
global_patch_mask = np.zeros_like(art_img)
for img in tmp_images:
    mask = np.ones_like(img)*255
    # ROTATING
    angle = random.randint(0,360)
    rotated = imutils.rotate_bound(img, angle)
    rotated_mask = imutils.rotate_bound(mask, angle)
    #PLACING THE IMAGE WITHOUT OVERLAPPING
    overlap_test = 1
    while overlap_test != 0:
        # TRANSLATING
        px, py = int(rotated.shape[0]/2), int(rotated.shape[1]/2)
        x, y = random.randint(0,size_px-1), random.randint(0,size_px-1) 
        xmin, xmax, ymin, ymax = x-px, x+px, y-py, y+py
        dxmin, dxmax = (0, -xmin)[xmin<0], (0, size_px-1-xmax)[xmax>size_px-1]
        dymin, dymax = (0, -ymin)[ymin<0], (0, size_px-1-ymax)[ymax>size_px-1]
        # PLACING ON TEMPORARY PATCH/MASL
        patch = np.zeros_like(art_img)
        patch_mask = np.zeros_like(art_img)
        patch[xmin+dxmin:xmax+dxmax, ymin+dymin:ymax+dymax] = rotated[dxmin:2*px+dxmax, dymin:2*py+dymax]
        patch_mask[xmin+dxmin:xmax+dxmax, ymin+dymin:ymax+dymax] = rotated_mask[dxmin:2*px+dxmax, dymin:2*py+dymax]
        # Testing if there is overlapping by comparing to global mask
        print(np.nonzero(np.logical_and(patch_mask, global_patch_mask)))
        overlap_test = len(np.nonzero(np.logical_and(patch_mask, global_patch_mask))[0]) 
    # (erosion to get rid of black edges)
    kernel_size = 3
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    patch_mask = cv2.erode(patch_mask,kernel,iterations = 1)
    # filling global patches
    cv2.copyTo(patch, patch_mask, global_patch)
    cv2.copyTo(patch_mask, patch_mask, global_patch_mask)
#CREATING FINAL IMAGE
cv2.copyTo(global_patch, global_patch_mask, art_img)
showImg(art_img)
showImg(global_patch_mask)

# %%
final_img = global_patch.copy()
showImg(final_img)
# Constructing kd tree with known values
known = np.argwhere(global_patch_mask!=0)
kdt = KDTree(known, leaf_size=30, metric='euclidean')
# Finding neirest neighbors of unknownn values
unknown = np.argwhere(global_patch_mask==0)
nn = kdt.query(unknown, k=5, return_distance=False)
# Filling
for i in range(len(unknown)):
    final_img[unknown[i][0], unknown[i][1]] = np.mean(global_patch[unknown[nn[i]][0], unknown[nn[i]][1]]).astype(np.uint8)
showImg(final_img)

# %%
