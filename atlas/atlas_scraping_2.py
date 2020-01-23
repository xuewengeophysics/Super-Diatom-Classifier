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
            handleImage(tmp, "tmp0", name)
            i += 1
        
        #cv morpho ->  conected components -> bounding box -> 

# %%
def handleImage(img, root, folder):
# img = cv2.imread('./tmp/AAMB_0.png',cv2.IMREAD_UNCHANGED)
    height, width = img.shape[0], img.shape[0]
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret,img_thr = cv2.threshold(img_gs,230,255,cv2.THRESH_BINARY_INV)
    #OPENING
    kernel_size = 20
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    img_thr = cv2.morphologyEx(img_thr, cv2.MORPH_OPEN, kernel)
    #CONNECTED COMPONENTS
    output = cv2.connectedComponentsWithStats(img_thr)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]
    j = 0
    for i in range(num_labels):
        area = stats[i][cv2.CC_STAT_AREA]
        x, y = stats[i][cv2.CC_STAT_LEFT], stats[i][cv2.CC_STAT_TOP]
        w, h = stats[i][cv2.CC_STAT_WIDTH], stats[i][cv2.CC_STAT_HEIGHT]
        if area > height*width*0.01:
            ROI = img[y:y+h,x:x+w]
            # showImg(ROI)
            j+=1
            path_to_folder = "./"+root+"/"+name
            full_path = path_to_folder+"/"+str(j)+".png"
            print(path_to_folder,full_path)
            if not os.path.exists(path_to_folder):
                os.makedirs(path_to_folder)
            cv2.imwrite(full_path,ROI)
    print(j, "component(s) extracted!")

# %%
ret,img_thr = cv2.threshold(img_gs,20,255,cv2.THRESH_BINARY_INV)
print(img_thr.shape)
showImg(img_thr)
contours,h = cv2.findContours(img_thr,1,2)
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print(len(approx))
    if len(approx)==4:
        cv2.drawContours(img,[cnt],0,(0,0,255),-1)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
