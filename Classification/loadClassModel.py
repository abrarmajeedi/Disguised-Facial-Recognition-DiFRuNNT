import pandas as pd
import cv2
import os
import numpy as np
import timeit
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D,MaxPooling2D,AveragePooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten, Concatenate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from keras.utils import to_categorical


json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("classifier.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

df = pd.read_csv('anglesratiostrain.csv', header = None)
X_train = df.iloc[:,0:94]
y_train = df.iloc[:,-1:]
enc = LabelEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train)
y_train = y_train.reshape(1,-1)

scaler = StandardScaler(with_mean = False).fit(X_train)
X_train = scaler.transform(X_train)

y_train = np.array(y_train).T

df = pd.read_csv('anglesratiospredictions.csv', header = None)

X_pred = df.iloc[:,0:94]

y_pred = df.iloc[:,-1:]

enc.fit(y_pred)

y_pred = enc.transform(y_pred)

y_pred = to_categorical(y_pred)

y_pred = pd.DataFrame(y_pred)

X_pred = scaler.transform(X_pred)


loss,score = loaded_model.evaluate(X_pred, y_pred, batch_size=200, verbose=1)

print("pred acc: "+ str(score*100))

print("pred loss: "+ str(loss))


