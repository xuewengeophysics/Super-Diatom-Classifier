from os import listdir, walk
from os.path import isfile, join
import numpy as np
import time
from sys import getsizeof
import random
import math
import datetime
import sys

import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 32
N_EPOCHS = 20
TRAIN_P = 0.70
TEST_P = 0.15
VAL_P = 0.15

from utils import *
from variables import *

files, labels = get_dataset()
NB_SAMPLES = len(files)

def cb_load_image(image_path, label):
    img_file = tf.io.read_file(image_path)
    img = tf.image.decode_png(img_file, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32) #0-1 range
    return img, label

full_dataset = tf.data.Dataset.from_tensor_slices((files, labels))
full_dataset = full_dataset.map(cb_load_image, num_parallel_calls=AUTOTUNE)

DATASET_SIZE = len(files)
train_size = int(TRAIN_P * DATASET_SIZE)
val_size = int(VAL_P * DATASET_SIZE)
test_size = int(TEST_P * DATASET_SIZE)

full_dataset = full_dataset.shuffle(len(files))
train_dataset = full_dataset.take(train_size).repeat(3).batch(BATCH_SIZE)
tmp_dataset = full_dataset.skip(train_size)
val_dataset = tmp_dataset.skip(val_size).repeat(N_EPOCHS+1).batch(BATCH_SIZE)
test_dataset = tmp_dataset.take(test_size).repeat(N_EPOCHS+1).batch(BATCH_SIZE)

if __name__ == "__main__":
    base_model = InceptionV3(weights='imagenet', input_shape=(256, 256, 3), include_top=False)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(185, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, 
              epochs=2, 
              steps_per_epoch=int(train_size/BATCH_SIZE)-1,
              use_multiprocessing=True, 
              validation_data=val_dataset,
              validation_steps=int(val_size/BATCH_SIZE)-1)