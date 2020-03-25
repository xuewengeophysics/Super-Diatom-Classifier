# %%
import cv2
import numpy
import os

# %%
IMAGES_PATH = "."
OUTUPUT_PATH = "samples"
images = []
for (dirpath, dirnames, filenames) in os.walk(IMAGES_PATH):
    for filename in filenames:
        ext = filename.split(".")[-1].lower()
        if ext in ["jpg", "tiff", "png"]:
            images.append(os.path.join(dirpath, filename))

# %%
drawing = False
ix, iy = None, None
def draw_rectangle(event,x,y,flags,param):
    global img, img_tmp, img_original, drawing, ix, iy, n_id
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img = img_tmp.copy()
            cv2.rectangle(img,(ix,iy),(x,y),-1,5)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img = img_tmp.copy()
        cv2.rectangle(img,(ix,iy),(x,y),-1,5)
        img_tmp = img.copy()
        # Saving sample
        string_id = '{:04d}'.format(n_id)
        patch = img_original[iy:y, ix:x]
        cv2.imwrite(os.path.join(OUTUPUT_PATH, string_id+".png"), patch) 
        n_id+=1
    

#%%
n_id = 0
for image_path in images:
    img = cv2.imread(image_path)
    img_tmp = img.copy()
    img_original = img.copy()
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', draw_rectangle)
    while(1):
        cv2.imshow('image',img)
        cv2.resizeWindow('image', 600, 600)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
cv2.destroyAllWindows()

# %%
