%tensorflow_version 2.x

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
import h5py
from IPython.display import SVG
from tensorflow.keras.utils import plot_model
from tensorflow.keras import applications
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from tensorflow.keras.optimizers import SGD
import random

%matplotlib inline

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in tensorflow.keras
    """

    #TODO: Start Code Here, Make CNN by using tf.keras.layers, put last layer into tf.keras.models.Model

    #TODO: End Code Here
    
    return model
  

def load_dataset():
    path_to_train = "assignment3_train.h5"
    path_to_test = "assignment3_test.h5"

    train_dataset = h5py.File(path_to_train, "r")
    train_x = np.array(train_dataset['train_set_x'][:])
    train_y = np.array(train_dataset['train_set_y'][:])

    test_dataset = h5py.File(path_to_test, "r")
    test_x = np.array(test_dataset['test_set_x'][:])
    test_y = np.array(test_dataset['test_set_y'][:])

    # reshape y from (samples, ) to (1, samples)
    train_y = train_y.reshape((1, train_x.shape[0]))
    test_y = test_y.reshape((1, test_x.shape[0]))

    # transpose y
    train_y = train_y.T
    test_y = test_y.T

    return train_x, train_y, test_x, test_y
 

def preprocess_data():
    train_x, train_y, test_x, test_y = load_dataset()

    # Normalize image vectors
    train_x = train_x/255.
    test_x = test_x/255.

    print ("number of training examples = " + str(train_x.shape[0]))
    print ("number of test examples = " + str(test_x.shape[0]))

    print ("X_train shape: " + str(train_x.shape))
    print ("Y_train shape: " + str(train_y.shape))
    print ("X_test shape: " + str(test_x.shape))
    print ("Y_test shape: " + str(test_y.shape))

    return train_x, train_y, test_x, test_y


def plot(training_results, validation_results, results_type, model_name):
    fig = plt.figure(figsize=[8, 6])

    plt.plot(training_results, 'r', linewidth=3.0)
    plt.plot(validation_results, 'b', linewidth=3.0)
    plt.legend(['Training ' + results_type, 'Validation ' + results_type], fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel(results_type, fontsize=16)
    plt.title(results_type + ' of ' + model_name, fontsize=16)


def plot_accuracy(history, model_name):
    plot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy', model_name)
    

def plot_loss(history, model_name):
    plot(history.history['loss'], history.history['val_loss'], 'Loss', model_name)


def test_model(model, model_name, epochs=10, print_summary=True, plot_results=True):
    train_x, train_y, test_x, test_y = preprocess_data()

    x_tr, x_val, y_tr, y_val = train_test_split(train_x, train_y, random_state = 3)

    # TODO Try different optimizers, optimizers hyperparameters, losses
    
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    if (print_summary):
        model.summary()

    hist = model.fit(x_tr, y_tr, validation_data = (x_val, y_val), verbose=1, epochs=epochs, batch_size=32)
    preds = model.evaluate(test_x, test_y, batch_size=32, verbose=1, sample_weight=None)

    print ("Testing Loss = " + str(preds[0]))
    print ("Testing Accuracy = " + str(preds[1]))

    if(plot_results):
        plot_loss(hist, model_name)
        plot_accuracy(hist, model_name)


# TODO redefine this faulty freeze layers method to perform as you studied in the lecture
def freeze(model, number_of_freezed_layers):
    layers = model.layers

    # Randomly choose layer to freeze
    for layer in layers:
        # toss a coin
        if(random.choice([True, False])):
            layer.trainable = False
        else:
            layer.trainable = True

    return model


def test_CNN(epochs=10, print_summary=True, plot_results=True, model_name="CNN"):
    # TODO Call test_model from here with your defined CNN Model


def test_Resnet(pretrained=True, freeze_layers=False, number_of_freezed_layers=0, epochs=10, print_summary=True, plot_results=True, model_name="ResNet"):

    if(pretrained):
        model_name = "Pretrained " + model_name
        # TODO define ResNet pretrained model
    else:
        model_name = "Untrained " + model_name
        # TODO define Untrained Resnet model

    if freeze_layers:
        base_model = freeze(base_model, number_of_freezed_layers)

    # TODO define the top of your model (the output layers)

    test_model(resnet_model, model_name, epochs, print_summary, plot_results)


def test_VGG(pretrained=True, freeze_layers=False, number_of_freezed_layers=0, epochs=10, print_summary=True, plot_results=True, model_name="VGG"):


    if(pretrained):
        model_name = "Pretrained " + model_name
        # TODO define VGG pretrained model
    else:
        model_name = "Untrained " + model_name
        # TODO define Untrained VGG model

    if freeze_layers:
        base_model = freeze(base_model, number_of_freezed_layers)

    # TODO define the top of your model (the output layers)

    test_model(vgg_model, model_name, epochs, print_summary, plot_results)