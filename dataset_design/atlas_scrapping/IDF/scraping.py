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

import sys
sys.path.insert(1, '../..')
from utils import *

selected_taxons = get_selected_taxons("../../../selected_taxons.txt")

# %% 01 - EXTRACTING BIG PICTURES
root_path = "/mnt/01D341A80E60C350/Users/pierr/Google Drive/School/Georgia Tech/Super Diatomee Classifier/Atlas/PDFs/IDF (DREAL)/"
pdfs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
n_id = 0
for atlas_path in pdfs:
    split = atlas_path.split(".")
    if split[1]=='pdf':
        doc = fitz.open(root_path+atlas_path)
        taxon = split[0]
        if taxon in selected_taxons.keys():
            for img in doc.getPageImageList(1):
                if img[7] == "I5":
                    xref = img[0]
                    print(img)
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n >=5:        # CMYK: convert to RGB first
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img = pix2np(pix)
                    # saving image
                    string_id = '{:04d}'.format(n_id)
                    path = "./tmp/"+taxon+"_"+string_id+".png"
                    n_id += 1
                    check_dirs(path)
                    # print(path)
                    cv2.imwrite(path, img)

# %%
def handle_image(img, taxon, id, verbose=False):
    n_id = 0
    height, width = img.shape[0], img.shape[1]
    # img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gs = img[:,:,0]
    # img_comp = cv2.cvtColor(img_gs, cv2.COLOR_GRAY2BGR)
    img_comp = np.repeat(img_gs[:,:,np.newaxis], 3, axis=2)
    # Finding common parts
    sub = img-img_comp
    sub_thr = cv2.inRange(sub, (0,0,0),(20,20,20))
    kernel_size = 3
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    mask = cv2.morphologyEx(sub_thr, cv2.MORPH_CLOSE, kernel, iterations = 1)

    if verbose: 
        showImg(img)
        showImg(sub)
        showImg(sub_thr)
        showImg(mask)

    contours,h = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    padding = 1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w*h > 900 and w*h<height*width*0.95):
            ROI = img_gs[y-padding:y+h+padding,x-padding:x+w+padding]
            if ROI.shape[0]>10 and ROI.shape[1]>10:
                # saving image
                string_id = '{:03d}'.format(id)+'{:03d}'.format(n_id)
                filename = get_file_name("IDF", taxon, string_id)
                path = os.path.join("./tmp0/", filename)
                n_id += 1
                check_dirs(path)
                # print(path)
                cv2.imwrite(path, ROI)

# root_path = "./tmp/"
# imgs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
# img_name = imgs[2]
# img = cv2.imread(root_path+img_name,cv2.IMREAD_UNCHANGED)
# handle_image(img, "TEST", 0, verbose=True)

# %% EXTRACTING IMAGES
# img = cv2.imread('./tmp/AAMB.png',cv2.IMREAD_UNCHANGED)
root_path = "./tmp/"
imgs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
for i, img_name in enumerate(imgs):
    split = img_name.split(".")
    if split[1]=='png':
        img = cv2.imread(root_path+img_name,cv2.IMREAD_UNCHANGED)
        taxon = split[0].split("_")[0]
        print(taxon)
        handle_image(img, taxon, i)
