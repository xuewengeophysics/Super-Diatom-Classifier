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
sys.path.insert(1, '..')
from utils import *

selected_taxons = get_selected_taxons("../../../selected_taxons.txt")

# %%
root_path = "/media/pierrefg/01D341A80E60C350/Users/pierr/Google Drive/School/Georgia Tech/Super Diatomee Classifier/Atlas/PDFs/BRG (LIST)/"
pdfs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
pdf_path = os.path.join(root_path, pdfs[1])
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
        if taxon in selected_taxons:
            taxon_pages.setdefault(taxon, []).append(i)
            print(taxon, i+1)

# %%
n_id = 0
for taxon in taxon_pages:
    pages_n = taxon_pages[taxon]
    # first page is never usefull
    for page_n in pages_n[1:]:
        print("Page ", page_n)
        for img in doc.getPageImageList(page_n):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n >=5:        # CMYK: convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)
            fimg = pix2np(pix)
            if fimg.shape[0]>10 and fimg.shape[1]>10:
                if fimg.shape[2]!=1:
                    fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2GRAY)
                saveImg(fimg, "./tmp", taxon, n_id)
                cv2.imwrite("./tmp/BRG_"+taxon+"_",fimg)
                n_id+=1
        
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
