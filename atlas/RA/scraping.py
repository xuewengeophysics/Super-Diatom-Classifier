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
    # print(path_to_folder,full_path)
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
                ROI = pix2np(pix)
                ROI_w, ROI_h = ROI.shape[0], ROI.shape[1]
                if ROI_w>10 and ROI_h>10:
                    #CHECKING IF IMAGE CONTAINS TO MUCH WHITE
                    area = ROI_w*ROI_h
                    if ROI.shape[2]!=1:
                        ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                    else:
                        ROI_gray = ROI
                    ret,ROI_thr = cv2.threshold(ROI_gray,254,255,cv2.THRESH_BINARY)
                    n = len(np.where(ROI_thr==255)[0])
                    # IF NOT, SAVE IMAGE
                    if n<area*0.2:
                        saveImg(ROI_gray, "tmp", taxon, i)
                        i+=1


# %%
root_path = "./data/"
# pdf_path = "./data/atlas_dreal_rhone_alpes_2013_volume1.pdf"
pdfs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
for atlas_path in pdfs:
    doc = fitz.open(root_path+atlas_path)
    handlePDF(doc)
print("OVER")


# %%
