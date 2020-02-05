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
showImg(global_patch)
showImg(global_patch_mask)

# %% TEST 01 - SIMPLE KNN
# Constructing kd tree with known values
known = np.argwhere(global_patch_mask!=0)
kdt = KDTree(known, leaf_size=30, metric='euclidean')
# Finding neirest neighbors of unknownn values
unknown = np.argwhere(global_patch_mask==0)
nn_query = kdt.query(unknown, k=10, return_distance=True)
nn = nn_query[1]
nn_weights = nn_query[0]
# Filling

# %%
final_img = global_patch.copy()
showImg(final_img)
for i in range(len(unknown)):
    x_u, y_u = unknown[i][0], unknown[i][1] 
    avg_values = global_patch[known[nn[i]][:,0], known[nn[i]][:,1]]
    # final_img[x_u, y_u] = np.mean(global_patch[known[nn[i]][:,0], known[nn[i]][:,1]]).astype(np.uint8)
    final_img[x_u, y_u] = np.average(avg_values, weights=nn_weights[i]).astype(np.uint8)
showImg(final_img)
cv2.imwrite( "./yey.png", final_img)

# %% TEST 02 - COMPLEX KNN
final_img = global_patch.copy()
retval, labels = cv2.connectedComponents(global_patch_mask)
kdts = []
for i in range(1,retval):
    cc_indexes = np.argwhere(labels==i)
    tmp = KDTree(cc_indexes, leaf_size=30, metric='euclidean')
    kdts.append(tmp)

# %% TEST 03 - INFLUENCE MAP
sigma = 30^2
final_img = global_patch.copy()
showImg(final_img)
acc, accw = np.zeros_like(final_img).astype(np.float64), np.zeros_like(final_img).astype(np.float64)
known = np.argwhere(global_patch_mask!=0)
unknown = np.argwhere(global_patch_mask==0)
xMap=np.ones(size_px,1)*[1:w]
yMap=[1:h]'*ones(1,w)
i = 0
for kp in known:
    xkp, ykp = kp[0], kp[1]
    print(i, len(known))
    i += 1
    for up in unknown:
        xup, yup = up[0], up[1]
        d2 = np.linalg.norm(up-kp)
        w = max(np.exp(-0.5*d2/sigma),1e-10)
        acc[xup, yup] += float(final_img[xkp, ykp])*w
        accw[xup, yup] += w
acc = np.divide(acc, accw)

# %%
sigma = 30^2
final_img = global_patch.copy()
showImg(final_img)
acc, accw = np.zeros_like(final_img).astype(np.float64), np.zeros_like(final_img).astype(np.float64)
known = np.argwhere(global_patch_mask!=0)
unknown = np.argwhere(global_patch_mask==0)
# Getting indices
indices = np.indices((size_px,size_px))
xMap = indices[1]
yMap = indices[0]
# Looping
i = 0
for kp in known:
    xkp, ykp = kp[0], kp[1]
    val = final_img[xkp, ykp]
    # print(xkp, ykp, val)
    print(i, len(known))
    i += 1
    # FILLING
    d2 = (xMap - xkp)*(xMap - xkp) + (yMap - ykp)*(yMap - ykp)
    # xup, yup = up[0], up[1]
    # d2 = np.linalg.norm(up-kp)
    w = np.exp(-0.5*d2/sigma)
    w[w<1e-10] = 1e-10
    acc += w*val
    accw += w
    # print(w*val, w)
    if i==100:
        break
acc = np.divide(acc, accw)

# %%
acc_img = acc.astype(np.uint8)
final_img[global_patch_mask==0]=acc_img[global_patch_mask==0]
showImg(final_img)
showImg(acc_img)

# %%
