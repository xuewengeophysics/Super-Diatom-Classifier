# general imports
from os import listdir
from os.path import isfile, join
import pickle
import PIL.Image
import multiprocessing

# imports
from variables import *
from util import *
from generator import *

def save_maps():
    images = []
    for path in DATASET_PATH:
        images.extend([f for f in listdir(path) if isfile(join(path, f))])

    tmp_code = {}
    i = 1
    for file in images:
        taxon = file.split('_')[1]
        if not (taxon in tmp_code):
            tmp_code[taxon] = i
            i+=1

    savePickle(tmp_code, SAVE_PATH+"/maps/multiclass_label_map.pickle")
    binary_tmp_code = {}
    binary_tmp_code["diatom"] = 1
    savePickle(binary_tmp_code, SAVE_PATH+"/maps/binary_label_map.pickle")
    print("Binary and multiclass label maps saved successfully !")

def worker(lock, n_id, wnumber, verbose):
    print("Worker ", wnumber, " ok!")
    random.seed()
    lock.acquire()
    wid = n_id.value
    if verbose:
        print("Worker ", wnumber, ": ", (wid+1),"/", N_IMAGES)
    n_id.value += 1
    lock.release()
    while(wid<=N_IMAGES):
        string_id = '{:05d}'.format(wid)
        final_img, annotations = main_generator(simple_angles=False, size_px=1000, fast=True, verbose=False, overlapping=True)
        path_img = "images/"+string_id+".png"
        saveImg(final_img, join(SAVE_PATH, path_img));

        ## Saving individual masks
        taxon_n = {}
        paths = []
        for annotation in annotations:
            taxon = annotation["taxon"]
            if taxon in taxon_n:
                taxon_n[taxon] += 1
            else:
                taxon_n[taxon] = 0
            path_mask = "masks/"+string_id+"_"+taxon+"_"+'{:03d}'.format(taxon_n[taxon])+".png"
            # Saving mask
            img = PIL.Image.fromarray(annotation["patch_mask"])
            annotation.pop("patch_mask")
            #output = io.BytesIO()
            check_dirs(join(SAVE_PATH, path_mask))
            img.save(join(SAVE_PATH, path_mask), format='PNG')
            annotation["mask_path"] = path_mask
        
        # Building and saving final_annotation
        full_annotations = {}
        full_annotations["img_path"] = path_img
        full_annotations["labels"] = annotations
        savePickle(full_annotations, SAVE_PATH+"/annotations/bb_"+string_id+".pickle")
        
        # Incrementing id
        lock.acquire()
        wid = n_id.value
        if verbose:
            print("Worker ", wnumber, ": ", (wid+1), "/", N_IMAGES)
        n_id.value += 1
        lock.release()
    return 0

if __name__ == '__main__':
    jobs = []
    n_process = 7
    n_id = multiprocessing.Value('i', 0)
    lock = multiprocessing.Lock()
    verbose = False
    save_maps()
    print("Generating", N_IMAGES, "images with ", n_process, " workers !")
    for i in range(n_process):
        p = multiprocessing.Process(target=worker, args=(lock, n_id, i, verbose,))
        jobs.append(p)
        p.start()