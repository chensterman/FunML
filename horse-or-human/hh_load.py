import tensorflow as tf
import numpy as np
import os
import cv2
import random
import pickle
#import matplotlib.pyplot as plt

DIRECTORY = "C:/Users/Leon Chen/Documents/Projects/ML-basics/horse-or-human"
SETS = ["train", "validation"]
LABELS = ["horses", "humans"]
IMAGE_SIZE = 50

train_set = []
val_set = []

for set in SETS:
    for label in LABELS:
        path = os.path.join(DIRECTORY, set)
        path = os.path.join(path, label)
        label = LABELS.index(label)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_arr = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE))
            if set == "train":
                train_set.append((img_arr, label))
            else:
                val_set.append((img_arr, label))

random.shuffle(train_set)
random.shuffle(val_set)

train_X = []
train_Y = []
val_X = []
val_Y = []

for x, y in train_set:
    train_X.append(x)
    train_Y.append(y)

for x, y in val_set:
    val_X.append(x)
    val_Y.append(y)

train_X = np.array(train_X)
train_X = tf.keras.utils.normalize(train_X, axis=1)
train_Y = np.array(train_Y)
val_X = np.array(val_X)
val_X = tf.keras.utils.normalize(val_X, axis=1)
val_Y = np.array(val_Y)

pickle_out = open("hh_train_X.pickle", "wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

pickle_out = open("hh_train_Y.pickle", "wb")
pickle.dump(train_Y, pickle_out)
pickle_out.close()

pickle_out = open("hh_val_X.pickle", "wb")
pickle.dump(val_X, pickle_out)
pickle_out.close()

pickle_out = open("hh_val_Y.pickle", "wb")
pickle.dump(val_Y, pickle_out)
pickle_out.close()