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
from os import listdir
from os.path import isfile, join

def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if im.shape[2] == 3:
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
def handlePDF(doc):
    taxon_regex = "(?:^|\W)[A-Z]{4}(?:^|\W)"
    p = re.compile(taxon_regex)
    for i in range(20, len(doc)):
        page = doc.loadPage(i)
        text = page.getText("text")
        match = p.search(text)
        if match and match.group().find("TOME")==-1:
            print("---------------------------------")
            print(page)
            print(match.group(), page.number)
            taxon = ''.join(e for e in match.group() if e.isalnum())
            i = 0
            for img in doc.getPageImageList(page.number):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n >=5:        # CMYK: convert to RGB first
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img = pix2np(pix)
                if img.shape[0]>10 and img.shape[1]>10:
                    area = img.shape[0]*img.shape[1]
                    if img.shape[2]!=1:
                        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    ret,img_thr = cv2.threshold(img_gs,254,255,cv2.THRESH_BINARY)
                    n = len(np.where(img_thr==255)[0])
                    if n<area*0.2:
                        saveImg(img_gs, "tmp", taxon, i)
                        i+=1


# %%
root_path = "./data/"
pdfs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
for atlas_path in pdfs:
    split = atlas_path.split(".")
    if split[1]=='pdf':
        doc = fitz.open(root_path+atlas_path)
        taxon = split[0]
        for img in doc.getPageImageList(1):
            if img[7] == "I5":
                xref = img[0]
                print(img)
                pix = fitz.Pixmap(doc, xref)
                if pix.n >=5:        # CMYK: convert to RGB first
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                img = pix2np(pix)
                # showImg(img)
                saveImg(img, "tmp", None, taxon)


# %%
# img = cv2.imread('./tmp/AAMB.png',cv2.IMREAD_UNCHANGED)
root_path = "./tmp/"
imgs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
for img_name in imgs:
    split = img_name.split(".")
    if split[1]=='png':
        img = cv2.imread(root_path+img_name,cv2.IMREAD_UNCHANGED)
        taxon = split[0]
        print(taxon)
        height, width = img.shape[0], img.shape[1]
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_comp = cv2.cvtColor(img_gs, cv2.COLOR_GRAY2BGR)
        # Finding common parts
        sub = img-img_comp
        sub_thr = cv2.inRange(sub, (0,0,0),(1,1,1))
        kernel_size = 3
        kernel = np.ones((kernel_size,kernel_size),np.uint8)
        mask = cv2.morphologyEx(sub_thr, cv2.MORPH_CLOSE, kernel)
        # showImg(img_thr)

        contours,h = cv2.findContours(mask,1,2)
        # showImg(img_thr)
        j=0
        padding = 1
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w*h > height*width*0.005 and w*h<height*width*0.95:
                ROI = img_gs[y-padding:y+h+padding,x-padding:x+w+padding]
                if ROI.shape[0]>10 and ROI.shape[1]>10:
                    saveImg(ROI, "tmp0", taxon, j)
                    j+=1

# %%
