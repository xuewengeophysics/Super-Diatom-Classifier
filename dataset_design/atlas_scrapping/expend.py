# %%
import shutil
from os import listdir
from os.path import isfile, join
from utils import *

# %%
path = "/home/pierrefg/Desktop/tmp"
pngs = [f for f in listdir(path) 
    if (isfile(join(path, f)) and len(f.split("."))==2 and f.split(".")[1]=="png")]

# %%
output = "/home/pierrefg/Desktop/RA"
selected_taxons = get_selected_taxons("../../selected_taxons.txt")
for png in pngs:
    taxon = png.split("_")[1]
    if taxon in list(selected_taxons.keys()):
        folder_path = join(output, taxon+"/")
        check_dirs(folder_path)
        shutil.copyfile(join(path, png), join(folder_path, png))
    # print(folder_path)

# %%
