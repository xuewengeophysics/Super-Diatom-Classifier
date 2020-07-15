Soon the greatest diatom classifier on planet earth.

# This git's manual

I have been using almost excluively notebooks for all this project.
Most of the notebooks in this git use config files generally named variables.ipynb
Those configs allow me to quickly switch between datasets and configurations.

## dataset_design/

Where we handled all dataset operation.
* atlas_scrapping/: Atlas scrapping to extract diatoms with their labels from the pdfs (availble on the Drive).
* ADIAC/: Where we extracted the sub datasets from the general public ADIAC dataset (https://rbg-web2.rbge.org.uk/ADIAC/pubdat/downloads/public_images.htm)
* dust_extraction/: exract.py was a manual thing based on opencv to crop debris from microscope images but finally we used LABELIMG and extract the labeled debris.

## diatom_identification/utils/

* Various utilities used in almost all diatom_identification/ notebooks

## diatom_identification/full_image_detection/

Where we handle the detection operations.
* artificial_dataset/: Generation of the artificial dataset. "generator.ipynb" is the brain of the generation and "dataset_generator.ipynb" uses "generator.ipynb" and a multithread approach to generate the dataset in the VOC format
* trainin_setup/: mainly conversion to tf_record but also the scripts I used for training and evaluation (the whole structure is in balrog->nvme->pfaurgi->training) As it has been used for many purposed, the script is highly customizable which also brings a bit of confusion in the code. Do no hesite if you have any question.

## diatom_identification/thumbnail_classification/

This folder is dedicated to diatom classification (Atlas, ADIAC and Aqualitas datasets). The datasets are stored fully and the subsets used for classification are gathered using the "filters" files (in the filter folder from the dataset root). Switching from one dataset to another (including filters) is handled from diatom_identification/thumbnail_classification/variables.ipynb
The crossval is hand coded and outputs in a log file when ran (allows to check and keep the progress as it runs in a notebook).
There is the begining of a hierarchical approach which, for now, creates a dendogram.
