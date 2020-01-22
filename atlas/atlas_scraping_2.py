# %%
# PDF handling
import fitz
# Regex
import re
# Plotting
import matplotlib.pyplot as plt
import numpy as np
import cv2

def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im

# %%
input1 = fitz.open("./data/atlas_LIST_uncompressed.pdf")
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
        i = 0
        for img in input1.getPageImageList(page_num):
            image_name = "./tmp/"+ name + "_" + str(i) + ".png"
            xref = img[0]
            pix = fitz.Pixmap(input1, xref)
            if pix.n < 5:       # this is GRAY or RGB
                pix.writePNG(image_name)
            else:               # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.writePNG(image_name)
                pix1 = None
            pix = None
            i += 1

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
            image_name = "./tmp/"+ name + "_" + str(i) + ".png"
            tmp = np.vstack(images[size])
            cv2.imwrite(image_name,tmp)
            i += 1
        
        #cv morpho ->  conected components -> bounding box -> 