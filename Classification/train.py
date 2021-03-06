import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dropout, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras import callbacks
from sklearn.preprocessing import StandardScaler


tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#train
df = pd.read_csv('anglesratiostrain.csv', header = None)
X_train = df.iloc[:,0:94]
y_train = df.iloc[:,-1:]
enc = LabelEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train)
y_train = to_categorical(y_train)
y_train = pd.DataFrame(y_train)
scaler = StandardScaler(with_mean = False).fit(X_train)
X_train = scaler.transform(X_train)
scalery = StandardScaler(with_mean = False).fit(y_train)
y_train = scalery.transform(y_train)

#test
df = pd.read_csv('anglesratiostest.csv', header = None)
X_test = df.iloc[:,0:94]
y_test = df.iloc[:,-1:]
enc.fit(y_test)
y_test = enc.transform(y_test)
y_test = to_categorical(y_test)
y_test = pd.DataFrame(y_test)
X_test = scaler.transform(X_test)
y_test = scalery.transform(y_test)



#predictions
df = pd.read_csv('anglesratiospredictions.csv', header = None)

X_pred = df.iloc[:,0:94]

y_pred = df.iloc[:,-1:]

enc.fit(y_pred)

y_pred = enc.transform(y_pred)

y_pred = to_categorical(y_pred)

y_pred = pd.DataFrame(y_pred)

X_pred = scaler.transform(X_pred)

y_pred = scalery.transform(y_pred)







#model
model = Sequential()

model.add(Dense(2000, input_dim = 94, activation="relu"))

model.add(Dropout(0.1))

model.add(Dense(1500, activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(1500, activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(1000, activation = 'relu'))

model.add(Dropout(0.3))

model.add(Dense(500, activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(200, activation = 'relu'))

model.add(Dropout(0.1))

model.add(Dense(20, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=200, epochs=300, verbose=1, shuffle=True, callbacks=[tbCallBack])

loss,score = model.evaluate(X_test, y_test, batch_size=200, verbose=1)

print("test acc: "+ str(score*100))

print("test loss: "+ str(loss))


loss,score = model.evaluate(X_pred, y_pred, batch_size=200, verbose=1)

print("pred acc: "+ str(score*100))

print("pred loss: "+ str(loss))

#saving model

model_json = model.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("classifier.h5")
print("Saved model to disk")


