import tensorflow as tf
import pickle

pickle_in = open("hh_train_X.pickle", "rb")
train_X = pickle.load(pickle_in)
pickle_in = open("hh_train_Y.pickle", "rb")
train_Y = pickle.load(pickle_in)
pickle_in = open("hh_val_X.pickle", "rb")
val_X = pickle.load(pickle_in)
pickle_in = open("hh_val_Y.pickle", "rb")
val_Y = pickle.load(pickle_in)

NUM_CHANNELS = 1
IMAGE_SIZE = 50
train_X = train_X.reshape(train_X.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
val_X = val_X.reshape(val_X.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=train_X.shape[1:], activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=train_X.shape[1:], activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_Y, epochs=25)

val_loss, val_acc = model.evaluate(val_X, val_Y)
print(val_loss, val_acc)
model.save('hh_conv2d.model')