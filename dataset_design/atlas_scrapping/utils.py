import cv2 
import numpy as np
import os

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


def get_selected_taxons(file_path):
    selected_taxons = {}
    f = open(file_path, 'r')
    lines = f.readlines()
    del lines[0]
    for line in lines:
        line = line.strip()
        taxon, taxon_id = line.split(',')[0], int(line.split(',')[1])
        selected_taxons[taxon] = taxon_id
    return selected_taxons

def get_file_name(atlas, taxon, id):
    return atlas+"_"+taxon+"_"+id+".png"