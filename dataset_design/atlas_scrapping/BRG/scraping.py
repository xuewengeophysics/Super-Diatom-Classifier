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
def handle_pdf(doc, id_prefix):
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
                # print(taxon, i+1)
    n_id = 0
    for taxon in taxon_pages:
        pages_n = taxon_pages[taxon]
        # first page is never usefull
        for page_n in pages_n[1:]:
            # print("Page ", page_n)
            for img in doc.getPageImageList(page_n):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n >=5:        # CMYK: convert to RGB first
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                fimg = pix2np(pix)
                if fimg.shape[0]>10 and fimg.shape[1]>10:
                    if fimg.shape[2]!=1:
                        fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2GRAY)
                    string_id = str(id_prefix)+'{:04d}'.format(n_id)
                    filename = get_file_name("BRG", taxon, string_id)
                    path = os.path.join("./tmp/", filename)
                    # saveImg(fimg, "./tmp", taxon, n_id)
                    # print(path)
                    cv2.imwrite(path, fimg)
                    n_id+=1

    print(n_id, "images extracted!")


# %%
root_path = "/mnt/01D341A80E60C350/Users/pierr/Google Drive/School/Georgia Tech/Super Diatomee Classifier/Atlas/PDFs/BRG (LIST)"
pdfs = [f for f in listdir(root_path) if isfile(join(root_path, f))]
for i, file_name in enumerate(pdfs):
    split = file_name.split(".")
    if len(split)>1 and split[1]=='pdf':
        pdf_path = os.path.join(root_path, file_name)
        print("Opening: ", pdf_path)
        doc = fitz.open(pdf_path)
        handle_pdf(doc, i)

# %%
