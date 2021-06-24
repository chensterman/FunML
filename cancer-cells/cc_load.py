import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

DIRECTORY = "C:/Users/Leon Chen/Documents/Projects/ML-basics/cancer-cells/data.csv"
SETS = ["train", "validation"]
LABELS = {"B": 0, "M": 1}

df = pd.read_csv(DIRECTORY)
df = df.sample(frac=1).reset_index(drop=True)

df_size = len(df)
train_size = int(df_size * 0.8)
df_train = df.iloc[:train_size]
df_val = df.iloc[train_size:]

train_X = df_train.iloc[:, 2:].replace([0], 0.0).to_numpy(dtype='float32', copy=True)[:, :30]
train_Y = df_train["diagnosis"].map(LABELS).to_numpy(dtype='float32', copy=True)
val_X = df_val.iloc[:, 2:].replace([0], 0.0).to_numpy(dtype='float32', copy=True)[:, :30]
val_Y = df_val["diagnosis"].map(LABELS).to_numpy(dtype='float32', copy=True)

pickle_out = open("cc_train_X.pickle", "wb")
pickle.dump(train_X, pickle_out)
pickle_out.close()

pickle_out = open("cc_train_Y.pickle", "wb")
pickle.dump(train_Y, pickle_out)
pickle_out.close()

pickle_out = open("cc_val_X.pickle", "wb")
pickle.dump(val_X, pickle_out)
pickle_out.close()

pickle_out = open("cc_val_Y.pickle", "wb")
pickle.dump(val_Y, pickle_out)
pickle_out.close()