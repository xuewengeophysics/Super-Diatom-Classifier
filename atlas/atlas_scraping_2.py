# %%
# PDF handling
import fitz
# Regex
import re
# Plotting
import matplotlib.pyplot as plt
import cv2

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