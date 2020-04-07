%tensorflow_version 2.x

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from progressbar import progressbar
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

def build_fc_model():
  fc_model = tf.keras.Sequential([
      tf.keras.layers.Flatten(),
      Dense(64, activation='relu', input_shape=(784,)),
      Dense(64, activation='relu'),
      Dense(10, activation='softmax'),
  ])
  return fc_model

num_classes = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
def build_cnn_model():
    cnn_model = tf.keras.Sequential()
    cnn_model.add(Conv2D(24, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(36, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    #cnn_model.add(Dropout(0.25))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    #cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(num_classes, activation='softmax'))
    return cnn_model

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = np.expand_dims(train_images, axis=-1)/255.
train_labels = np.int64(train_labels)
test_images = np.expand_dims(test_images, axis=-1)/255.
test_labels = np.int64(test_labels)

plt.figure(figsize=(10,10))
random_inds = np.random.choice(60000,36)
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[image_ind])
    
model = build_fc_model()
BATCH_SIZE = 64
EPOCHS = 5

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  train_images, # training data
  to_categorical(train_labels), # training targets
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
)


print('Test accuracy:', test_acc)

cnn_model = build_cnn_model()
print(cnn_model.summary())

cnn_model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

cnn_model.fit(train_images, to_categorical(train_labels),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(test_images, to_categorical(test_labels)))

loss,test_acc = cnn_model.evaluate(test_images, to_categorical(test_labels), verbose=0)
print('Test accuracy:', test_acc)


print('Test accuracy:', test_acc)
predictions = cnn_model.predict(test_images)

predictions[0]

#TODO: identify the digit with the highest confidence prediction for the first image in the test dataset
print("arg max" , np.argmax(predictions[0])) 

test_labels[0]