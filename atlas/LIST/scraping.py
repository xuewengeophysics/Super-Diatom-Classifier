# %%
# PDF handling
import fitz # pip install PyMuPDF
# Regex
import re
# Plotting
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im

def showImg(img, scale=1):
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', scale*img.shape[1], scale*img.shape[0])
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    print("DEBUG: waitKey returned:", chr(k))
    cv2.destroyAllWindows()

def saveImg(img, root, folder, id):
    if folder==None:
        path_to_folder = "./"+root
    else:
        path_to_folder = "./"+root+"/"+name
        if not os.path.exists(path_to_folder):
            os.makedirs(path_to_folder)
    full_path = path_to_folder+"/"+str(id)+".png"
    print(path_to_folder,full_path)
    cv2.imwrite(full_path,img)

# %%
pdf_path = "./data/atlas_LIST.pdf"
input1 = fitz.open(pdf_path)
taxon_regex = "[A-Z]{4}"
taxons = {}
for section in input1.getToC():
    # Extracting informations
    title = section[1]
    page_num = section[2]
    # Finding info
    p = re.compile(taxon_regex)
    match = p.search(title)
    if match and page_num != (-1):
        taxons[match.group()] = page_num
taxons = {k: v for k, v in sorted(taxons.items(), key=lambda item: item[1])}

# %%
key_list = list(taxons.keys())
for i, name in enumerate(key_list):
    n_start = taxons[key_list[i]]
    n_end = taxons[key_list[i+1]]-1
    for page_num in range(n_start, n_end+1):
        print(name, page_num)
        images = {}
        for img in input1.getPageImageList(page_num):
            xref = img[0]
            pix = fitz.Pixmap(input1, xref)
            if pix.n >=5:        # CMYK: convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)
            if pix.width not in images:
                images[pix.width] = []
            images[pix.width].append(pix2np(pix))
            # print(type(pix2np(pix)))
            # print(pix2np(pix))
            pix = None
        i = 0
        for size in images:
            image_name = "./tmp0/"+ name + "/" + str(i) + "_"
            tmp = np.vstack(images[size])
            handleImage(tmp, "tmp1", name, i, 0)
            i += 1
        
        #cv morpho ->  conected components -> bounding box -> 

# %%
def handleImage(img, root, folder, id, mode):
    # img = cv2.imread('./tmp/AAMB_0.png',cv2.IMREAD_UNCHANGED)
    height, width = img.shape[0], img.shape[1]
    img_area = height*width
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clipping equals
    ret,img_clip = cv2.threshold(255-img_gs,230,255,cv2.THRESH_TOZERO_INV)
    img_clip = img_clip
    # showImg(img_clip)
    # erosion
    kernel_size = 2
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    img_opened = cv2.erode(img_clip,kernel,iterations = 2)
    img_opened = 255-img_opened
    # showImg(img_opened)
    # thresolding
    ret,img_thr = cv2.threshold(img_opened,240,255,cv2.THRESH_BINARY_INV)
    # showImg(img_thr)
    # showImg(img_thr)
    #CONNECTED COMPONENTS
    contours,h = cv2.findContours(img_thr,1,2)
    j = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if area > img_area*0.005 and area < img_area*0.9:
            # showImg(ROI)
            # saveImg(ROI, root, folder, str(id)+str(j))
            if mode == 0:
                ROI = img[y:y+h,x:x+w]
                saveImg(ROI, root, folder, str(id)+str(j))
            else:
                # showImg(img[y:y+h,x:x+w])
                img[y:y+h,x:x+w] = -img[y:y+h,x:x+w] + 255
            j+=1
    if mode == 2:
        saveImg(img, root, None, folder+str(id))
    if mode == 1:
        showImg(img)
    print(j, "component(s) extracted!")

# test=["FAPP_0", "DSEP_0", "DVUL_0"]
# img = cv2.imread('./tmp/CAEX_0.png',cv2.IMREAD_UNCHANGED)
# handleImage(img, "", "", "", 1)

# %%
def handleImage1(img, root, folder, id, mode):
    # test=["FAPP_0", "DSEP_0", "DVUL_0"]
    # img = cv2.imread('./tmp/'+test[]+'.png',cv2.IMREAD_UNCHANGED)
    height, width = img.shape[0], img.shape[1]
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,img_thr = cv2.threshold(img_gs,254,255,cv2.THRESH_BINARY_INV)
    # showImg(img_thr)
    kernel_size = 20
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_OPEN, kernel)
    # showImg(img_thr)
    contours,h = cv2.findContours(img_thr,1,2)
    # showImg(img_thr)
    j=0
    for cnt in contours:
        for eps in np.arange(0.05, 0.20, 0.01):
            approx = cv2.approxPolyDP(cnt,eps*cv2.arcLength(cnt,True),True)
            if len(approx)==4:
                x, y, w, h = cv2.boundingRect(cnt)
                if w*h > height*width*0.005:
                    ROI = img[y:y+h,x:x+w]
                    
                    if mode == 0:
                        saveImg(ROI, root, folder, str(id)+str(j))
                    else:
                        img[y:y+h,x:x+w] = -img[y:y+h,x:x+w] + 255
                    j+=1
            # print("square")
            # cv2.drawContours(img,[cnt],0,(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)),-1)
    if mode == 2:
        saveImg(img, root, None, folder+str(id))
    if mode == 1:
        showImg(img)
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
# test=["FAPP_0", "DSEP_0", "DVUL_0"]
# img = cv2.imread('./tmp/CAEX_0.png',cv2.IMREAD_UNCHANGED)
# handleImage1(img, "", "", "", 1)

# %%
img = cv2.imread('./tmp/AOVA_0.png',cv2.IMREAD_UNCHANGED)
# img = cv2.imread('./tmp/AAMB_0.png',cv2.IMREAD_UNCHANGED)
height, width = img.shape[0], img.shape[1]
img_area = height*width
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# clipping equals
ret,img_clip = cv2.threshold(255-img_gs,230,255,cv2.THRESH_TOZERO_INV)
img_clip = img_clip
showImg(img_clip)
# erosion
kernel_size = 2
kernel = np.ones((kernel_size,kernel_size),np.uint8)
img_opened = cv2.erode(img_clip,kernel,iterations = 2)
img_opened = 255-img_opened
showImg(img_opened)
# thresolding
ret,img_thr = cv2.threshold(img_opened,235,255,cv2.THRESH_BINARY_INV)
showImg(img_thr)

# %%


# %%
