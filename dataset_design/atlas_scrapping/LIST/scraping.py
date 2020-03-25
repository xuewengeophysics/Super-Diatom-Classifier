# %%
# PDF handling
import fitz # pip install PyMuPDF
# Regex
import re
# Plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import cv2
import os
import math

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
        path_to_folder = "./"+root+"/"+folder
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

    # FIRST THRESOLD:
    ret,img_bin_thre = cv2.threshold(img_gs,253,255,cv2.THRESH_BINARY_INV)
    # showImg(img_bin_thre)

    # OPENING
    kernel_size = 3
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    opening = cv2.morphologyEx(img_bin_thre, cv2.MORPH_OPEN, kernel)
    # showImg(opening)

    contours,h = cv2.findContours(opening,1,2)
    j = 0
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        area = w*h
        if area > img_area*0.005 and area < img_area*0.9:
            # showImg(ROI)
            # saveImg(ROI, root, folder, str(id)+str(j))
            # showImg(img2)
            if mode == 0:
                cimg = np.zeros_like(img_gs)
                img2 = cv2.drawContours(cimg, contours, i, color=255, thickness=-1)
                imgb = np.copy(img_gs)
                # showImg(img2)
                imgb[img2 != 255] = 255
                ROI = imgb[y:y+h,x:x+w]
                print("HEYY")
                handleROI(ROI, folder+str(id)+str(j))
                # saveImg(ROI, root, folder, str(id)+str(j))
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
def handleROI(roi, id):
    # roi = cv2.imread('./test.png',cv2.IMREAD_UNCHANGED)
    # showImg(roi)

    roi_height, roi_width = roi.shape[0], roi.shape[1]
    ret,img_thr = cv2.threshold(roi,250,255,cv2.THRESH_BINARY_INV)
    # showImg(img_thr)

    edges = cv2.Canny(img_thr,30,50)
    # showImg(edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,5))
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # showImg(closing)
    # closing=edges
    nlines = 0
    lines = cv2.HoughLinesP(closing, 1, math.pi/2, int(roi_height/4), minLineLength=roi_height*0.7, maxLineGap=roi_height/2)
    if lines is not None:
        for line in lines[:,0,:]:
            x = line[0]
            pad = 10
            if line[0]==line[2] and x<(roi_width/2+pad) and x>(roi_width/2-pad):
                nlines +=1
                # pt1 = (line[0],line[1])
                # pt2 = (line[2],line[3])
                # cv2.line(roi, pt1, pt2, (0,0,255), 3)
    if nlines>0 or roi_width>1.2*roi_height:
        saveImg(roi, "tmp1", "bad", id)
    else:
        saveImg(roi, "tmp1", "good", id)

    # cv2.imshow('image',roi)
    # k = cv2.waitKey(0)
    # print("DEBUG: waitKey returned:", chr(k))
    # cv2.destroyAllWindows()
# %%
# ret,img_max = cv2.threshold(img_gs,254,255,cv2.THRESH_BINARY_INV)

# img = cv2.imread('./tmp/AOVA_0.png',cv2.IMREAD_UNCHANGED)
img = cv2.imread('./tmp/AUBR_0.png',cv2.IMREAD_UNCHANGED)
height, width = img.shape[0], img.shape[1]
img_area = height*width
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# FIRST THRESOLD:
ret,img_bin_thre = cv2.threshold(img_gs,252,255,cv2.THRESH_BINARY_INV)
showImg(img_bin_thre)

# OPENING
kernel_size = 2
kernel = np.ones((kernel_size,kernel_size),np.uint8)
opening = cv2.morphologyEx(img_bin_thre, cv2.MORPH_OPEN, kernel)
showImg(opening)

# Contours
contours,h = cv2.findContours(opening,1,2)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w*h > height*width*0.005:
        ROI = img[y:y+h,x:x+w]
        # handleROI(ROI)
        showImg(ROI)

# %%
