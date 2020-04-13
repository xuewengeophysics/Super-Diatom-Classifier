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
root_path = "/media/pierrefg/01D341A80E60C350/Users/pierr/Google Drive/School/Georgia Tech/Super Diatomee Classifier/Atlas/PDFs/BRG (LIST)/"
pdfs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
pdf_path = os.path.join(root_path, pdfs[3])
print("Opening: ", pdf_path)
doc = fitz.open(pdf_path)

# %%
taxon_pages = {}
taxon_regex = "(?:^|\W)[A-Z]{4}(?:^|\W)"
p = re.compile(taxon_regex)
for i in range(39, len(doc)):
    page = doc.loadPage(i)
    text = page.getText("text")
    match = p.search(text)
    if match:
        taxon = ''.join(x for x in match.group().strip() if x.isalpha())
        taxon_pages.setdefault(taxon, []).append(i)
        print(taxon, i+1)
        
    # if match and match.group().find("TOME")==-1:
    #     print("---------------------------------")
    #     print(page)
    #     print(match.group(), page.number)
    #     taxon = ''.join(e for e in match.group() if e.isalnum())
    #     i = 0
    #     for img in doc.getPageImageList(page.number):
    #         xref = img[0]
    #         pix = fitz.Pixmap(doc, xref)
    #         if pix.n >=5:        # CMYK: convert to RGB first
    #             pix = fitz.Pixmap(fitz.csRGB, pix)
    #         img = pix2np(pix)
    #         if img.shape[0]>10 and img.shape[1]>10:
    #             area = img.shape[0]*img.shape[1]
    #             if img.shape[2]!=1:
    #                 img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #             ret,img_thr = cv2.threshold(img_gs,254,255,cv2.THRESH_BINARY)
    #             n = len(np.where(img_thr==255)[0])
    #             if n<area*0.2:
    #                 saveImg(img_gs, "tmp", taxon, i)
    #                 i+=1

# %%
