# %%
import cv2
import pandas as pd
import os
import shutil as sh
import math

import sys
sys.path.insert(1, '..')
from utils import *

root = "/mnt/1184aa73-c854-40a5-9a6e-30ae55a1cbf8/MT/datasets/ADIAC/"

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum[0]

# %% DIMITRI DELTA=24
name = "dimitri/"
df_adiac = pd.read_csv(os.path.join(root, "index.csv"))
df_adiac = df_adiac[df_adiac["processing"]!='scale bar added']
df_adiac = df_adiac[df_adiac["focus"]=='standard focus']
df_adiac = df_adiac[~df_adiac["image_type"].str.contains("problem")]
#df_adiac = df_adiac[~df_adiac["authority"].str.contains("Lange-Bertalot", na=False)]
#df_adiac = df_adiac[df_adiac["image_type"]=='single valve, testset2']
df_adiac.head()

label_dict = {}
input_folder = os.path.join(root, "images/")
output_folder = os.path.join(root, name)
for index, row in df_adiac.iterrows():
    fileroot = row["image"].split(".")[0].upper()
    fileext = row["image"].split(".")[1].upper()
    filename = fileroot+"."+fileext
    genra = row["genus"]
    specie = row["species"]
    # print([genra, specie])
    if type(genra)==str:
        if type(specie)!=str: #chec for nan
            specie="sp"
        label = "_".join([genra, specie])
        label_dict.setdefault(label, []).append({
            "filename": filename,
            "variety": row["variety"]
        })

unwanteds = ["002192AA.TIF", "002193AA.TIF", "000236.TIF"]
for label in label_dict:
    varieties = [img["variety"] for img in label_dict[label]]
    majority_variety = find_majority(varieties)
    #print(label, len(label_dict[label]))
    for img in label_dict[label]:
        if (type(majority_variety)!=str and type(img["variety"])!=str) or img["variety"]==majority_variety :
            filename = img["filename"]
            source_file = os.path.join(input_folder, filename)
            target_file = os.path.join(output_folder, label, filename)
            check_dirs(target_file)
            if (filename not in unwanteds) and (os.path.exists(source_file)):
                sh.copy(source_file, target_file)
            else:
                #print(label, ":", filename, " not found!")
                pass

# %%
source_file = os.path.join(input_folder, filename)
target_file = os.path.join(output_folder, label, filename)
check_dirs(target_file)
if (os.path.exists(source_file)):
    sh.copy(source_file, target_file)
else:
    print(label, ":", filename, " not found!")

# %% TEST_SET2
name = "testset2/"
df_adiac = pd.read_csv(os.path.join(root, "index.csv"))
df_adiac = df_adiac[df_adiac["image_type"].str.contains("testset2", na=False)]
#df_adiac = df_adiac[df_adiac["image_type"]=='single valve, testset2']
df_adiac.head()

# %%
input_folder = os.path.join(root, "images/")
output_folder = os.path.join(root, name)
for index, row in df_adiac.iterrows():
    fileroot = row["image"].split(".")[0].upper()
    fileext = row["image"].split(".")[1].upper()
    filename = fileroot+"."+fileext
    genra = row["genus"]
    specie = row["species"]
    # print([genra, specie])
    if type(genra)==str:
        if type(specie)==str: #chec for nan
            label = "_".join([genra, specie])
        else:
            label = genra
        source_file = os.path.join(input_folder, filename)
        target_file = os.path.join(output_folder, label, filename)
        check_dirs(target_file)
        if (os.path.exists(source_file)):
            sh.copy(source_file, target_file)
        else:
            print(label, ":", filename, " not found!")
