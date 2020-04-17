%tensorflow_version 2.x
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM ,GRU, SimpleRNN
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split

import random
%matplotlib inline
# fix random seed for reproducibility
np.random.seed(7)

top_words = 5000
max_review_length = 500
embedding_vecor_length = 32

def load_dataset():
  (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
  return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
  X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
  X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
  return X_train,X_test

def RNN_model():
  model = Sequential()
  model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
  model.add(SimpleRNN(128))
  model.add(Dense(1, activation='sigmoid'))
  return model

def LSTM_model():
  model = Sequential()
  model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
  model.add(LSTM(100))
  model.add(Dense(1, activation='sigmoid'))
  return model

def GRU_model():
  model = Sequential()
  model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
  model.add(GRU(256, return_sequences=True))
  model.add(Dense(1, activation='sigmoid'))
  return model

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

def test_model(train_x, train_y, test_x, test_y ,model, model_name, epochs=10, print_summary=True, plot_results=True):

    x_tr, x_val, y_tr, y_val = train_test_split(train_x, train_y, random_state = 3)
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    if (print_summary):
        model.summary()

    hist = model.fit(x_tr, y_tr, validation_data = (x_val, y_val), verbose=1, epochs=epochs, batch_size=64)
 
    preds = model.evaluate(test_x, test_y, batch_size=64, verbose=1, sample_weight=None)
    print ("Testing Loss = " + str(preds[0]))
    print("Accuracy: %.2f%%" % (preds[1]*100))
    if(plot_results):
        plot_loss(hist, model_name)
        plot_accuracy(hist, model_name)

X_train, y_train, X_test, y_test = load_dataset()
X_train ,X_test =  preprocess_data(X_train, X_test)
model = LSTM_model()
test_model(X_train, y_train, X_test, y_test,model, "LTSM", epochs=20)
model = RNN_model()
test_model(X_train, y_train, X_test, y_test,model, "RNN Model", epochs=20)
model = GRU_model()
test_model(X_train, y_train, X_test, y_test,model, "GRU Model", epochs=20)
