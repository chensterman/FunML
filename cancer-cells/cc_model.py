import tensorflow as tf
import numpy as np
import pickle

pickle_in = open("cc_train_X.pickle", "rb")
train_X = pickle.load(pickle_in)
pickle_in = open("cc_train_Y.pickle", "rb")
train_Y = pickle.load(pickle_in)
pickle_in = open("cc_val_X.pickle", "rb")
val_X = pickle.load(pickle_in)
pickle_in = open("cc_val_Y.pickle", "rb")
val_Y = pickle.load(pickle_in)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(30,)))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=5000)

val_loss, val_acc = model.evaluate(val_X, val_Y)
print(val_loss, val_acc)
model.save('cc_dense.model')