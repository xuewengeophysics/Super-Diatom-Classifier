# %%
import cv2
import numpy
import os

# %%
IMAGES_PATH = "."
subfolders = [x[0] for x in os.walk(IMAGES_PATH)]
images = []
for folder in subfolders:
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if filename.split(".")[-1].lower() == "jpg":
               images.append(os.path.join(folder, filename)) 
        # f.extend(filenames)
        # break

# %%
drawing = False
def draw_rectangle(event,x,y,flags,param):
    global drawing
    print(event)

#%%
img = cv2.imread(images[0])
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

# %%
