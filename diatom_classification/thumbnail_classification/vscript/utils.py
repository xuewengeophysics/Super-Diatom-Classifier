import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir, walk
from os.path import isfile, join
import errno

from variables import *
    
def check_dirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
def get_selected_taxons():
    selected_taxons = {}
    f = open(SELECTED_TAXONS, 'r') 
    lines = f.readlines() 
    del lines[0]
    for line in lines:
        line = line.strip()
        taxon, taxon_id = line.split(',')[0], int(line.split(',')[1])
        selected_taxons[taxon] = taxon_id
    return selected_taxons

def convert_to_square(image):
    square_size = np.max(image.shape)
    h, w = image.shape
    delta_w = square_size - w
    delta_h = square_size - h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    square_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    return square_image

def expand(image):
    if len(image.shape)==2:
        image = np.expand_dims(image, -1)
        image = np.repeat(image, 3, 2)
    return image

def get_dataset():
    id_map = get_selected_taxons()
    x_set = []
    y_set = []
    taxons_dirs = next(walk(DATASET_PATH))[1]
    n_taxons = len(taxons_dirs)
    for i, taxon in enumerate(taxons_dirs):
        if not taxon in id_map:
            print("WARNING: Taxon",taxon,"not found in id_map !")
            continue
        taxon_id = id_map[taxon]
        path = join(DATASET_PATH, taxon)
        files = [f for f in listdir(path) if isfile(join(path, f))]
        for file in files:
            x_set.append(join(path, file))
            y_set.append(taxon_id)
    return x_set, y_set