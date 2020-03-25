# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import figure


# %%
def showImg(img, scale=1):
    cv.namedWindow('image',cv.WINDOW_NORMAL)
    cv.resizeWindow('image', scale*img.shape[1], scale*img.shape[0])
    cv.imshow('image',img)
    k = cv.waitKey(0)
    print("DEBUG: waitKey returned:", chr(k))
    cv.destroyAllWindows()


# %%
img = cv.imread('sample01.tiff',0)
print(img.shape)
showImg(img)

# %%
blur_size = 10
img_blur = cv.blur(img, (blur_size,blur_size))
showImg(img_blur)


# %%
kernel_size = 30
kernel = np.ones((kernel_size,kernel_size),np.uint8)
# kernel=cv.getstructuringelement
img_opening = cv.morphologyEx(img_blur, cv.MORPH_OPEN, kernel)
showImg(img_opening)


# %%
sigma = 1
v = np.median(img_opening)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))

img_edges = cv.Canny(img_opening, 20, 30)
showImg(img_edges)


# %%
ret, img_thresh = cv.threshold(img_opening,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU) #voir le | à la place du +
showImg(img_thresh)


# %%
kernel_size = 100
kernel = np.ones((kernel_size,kernel_size),np.uint8)
img_closed = cv.morphologyEx(img_thresh, cv.MORPH_CLOSE, kernel)
showImg(img_closed)


# %%
# open cv -> find contours
# open cv -> find moments (of connex parts)

