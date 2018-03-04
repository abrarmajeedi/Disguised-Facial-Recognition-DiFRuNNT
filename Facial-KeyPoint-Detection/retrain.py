import pandas as pd
import cv2
import os
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D,MaxPooling2D,AveragePooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers, callbacks
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten, Concatenate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


dim = 227
tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#train
df1 = pd.read_csv('train.csv',header = None)
print "Program started"
X = np.stack([cv2.imread("trainimages/"+str(img)) for img in df1.iloc[:,-1]]).astype(np.float)[:, :, :, np.newaxis]
print "Resizing done"
y = np.vstack(df1.iloc[:,:-1].values)
X_train = X / 255
y_train = y


#test
df2 = pd.read_csv('test.csv',header = None)
print "Program started"
X = np.stack([cv2.imread("testimages/"+str(img)) for img in df2.iloc[:,-1]]).astype(np.float)[:, :, :, np.newaxis]
print "Resizing done"
y = np.vstack(df2.iloc[:,:-1].values)
X_test = X / 255
y_test = y


print "Model Started"

X_train = X_train.reshape(3500,dim,dim,3)
X_test = X_test.reshape(500,dim,dim,3)

print "X_train.shape" + str(X_train.shape)
print "y_train.shape" + str(y_train.shape)
print "X_test.shape" + str(X_test.shape)
print "y_test.shape" + str(y_test.shape)


print "Model Started"

X_train = X_train.reshape(3500,dim,dim,3)
X_test = X_test.reshape(500,dim,dim,3)

print "X_train.shape" + str(X_train.shape)
print "y_train.shape" + str(y_train.shape)
print "X_test.shape" + str(X_test.shape)
print "y_test.shape" + str(y_test.shape)


json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

history = model.fit(X_train, y_train, batch_size=50, epochs=400, verbose=1, shuffle=True,callbacks = [tbCallBack])

#saving model

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

#evaluate model
scores = model.evaluate(X_test, y_test, verbose=1, batch_size = 50)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
