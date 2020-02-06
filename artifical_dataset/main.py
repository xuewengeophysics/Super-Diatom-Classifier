# %%
import cv2
import imutils #pip install imutils
import numpy as np
# import cupy as np
import random

from os import listdir
from os.path import isfile, join
DATASET_PATH = "./data/ra"

# from sklearn.neighbors import KDTree

random.seed(19)
def showImg(img, scale=1):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', scale*img.shape[1], scale*img.shape[0])
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    print("DEBUG: waitKey returned:", chr(k))
    cv2.destroyAllWindows()

# %% LOADING n RANDOM images
images = [f for f in listdir(DATASET_PATH) if isfile(join(DATASET_PATH, f))]
n = random.randint(9,12)
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
# for img in tmp_images:
#     # showImg(img)
#     img += (mean_brightness-np.mean(img)).astype(np.uint8)
#     img[img>254] = 254
#     # showImg(img)

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
showImg(np.hstack([global_patch, global_patch_mask]))

# %% TEST 03 - INFLUENCE MAP
sigma = 10e3
final_img = global_patch.copy()
showImg(final_img)
acc, accw = np.zeros_like(final_img).astype(np.float64), np.zeros_like(final_img).astype(np.float64)
# Finding contours
kernel_size = 5
kernel = np.ones((kernel_size,kernel_size),np.uint8)
mask_tmp = cv2.erode(global_patch_mask,kernel,iterations = 0)
conts, h = cv2.findContours(mask_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Getting indices
indices = np.indices(final_img.shape)
xMap = indices[0]
yMap = indices[1]
# Looping
i = 0
known = np.concatenate(conts)
for kp in known:
    # Counter
    if i%100==0:
        print(i, "/", len(known))
    i += 1
    # Init
    xkp, ykp = kp[0][1], kp[0][0]
    val = final_img[xkp, ykp]
    # FILLING
    d2 = np.square(xMap - xkp) + np.square(yMap - ykp)
    w = np.exp(-d2/sigma)
    w[w<1e-10] = 1e-10
    acc += w*val
    accw += w
    # print(w*val, w)
    # if i==1000:
    #     break
acc = np.divide(acc, accw)

# %%
acc_img = acc.astype(np.uint8)
final_img[global_patch_mask==0]=acc_img[global_patch_mask==0]
showImg(acc_img)
showImg(final_img)
# %%
cv2.imwrite( "./yey.png", final_img)

# %%
kernel_size = 5
kernel = np.ones((kernel_size,kernel_size),np.uint8)
showImg(global_patch_mask)
mask_tmp = cv2.erode(global_patch_mask,kernel,iterations = 0)
showImg(mask_tmp)
conts, h = cv2.findContours(mask_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
known = np.concatenate(conts)
test=np.ones_like(global_patch)
for kp in known:
    xkp, ykp = kp[0][1], kp[0][0]
    test[xkp, ykp] = global_patch[xkp, ykp]
    # test[xkp, ykp] = 255
showImg(test)

# %%
