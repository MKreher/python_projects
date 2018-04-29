# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.optimizers import SGD
from sklearn import preprocessing

print("TensorFlow version: {}".format(tf.VERSION))

# Location of data files
datafilePath = "./data/BedarfByDayOfWeek.csv"

# Ist-Bestandsverlauf plotten
X, Y = np.loadtxt(datafilePath, delimiter=',', skiprows=1, unpack=True)
#dataframeX = pd.read_csv(datafilePath, delimiter=',', skiprows=1, usecols=[0])
#dataframeY = pd.read_csv(datafilePath, delimiter=',', skiprows=1, usecols=[1])

# pandas DataFrame to numpy Array and reshape nD-Array to a 1D-Array for matplot
#X = dataframeX.values.reshape([dataframeX.size])
#Y = dataframeY.values.reshape([dataframeY.size])

fig, ax = plt.subplots(figsize=(15, 5))
plt.scatter(X, Y)
plt.xlabel('Tag')
plt.ylabel('Bestand')
plt.title('Ist-Bestand (unscaled)')
plt.show()

scaler = preprocessing.MinMaxScaler()
#X = scaler.fit_transform(dataframeX)
Y = scaler.fit_transform(Y.reshape(-1,1))

fig, ax = plt.subplots(figsize=(15, 5))
plt.plot(X, Y, '')
plt.plot(X, Y, 'ro')
plt.xlabel('Tag')
plt.ylabel('Bestand')
plt.title('Ist-Bestand (scaled)')
plt.show()

#Needed to use the TensorBoard tool to visualize the model training.
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./tmp/model_graph', write_graph=True)

# Define the NN as with fully connected layers with 12 nodes each that are using the sigmoid function.
model = Sequential()
model.add(Dense(40, activation='sigmoid', kernel_initializer='uniform', input_shape=(1,)))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
sgd = SGD(0.001);
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])

H = model.fit(X, Y, epochs=1000, batch_size=50, verbose=1, validation_split=0.3, callbacks=[tbCallBack])

plt.plot(H.history['mean_squared_error'])
plt.show()

# Predict
Y_ = model.predict(X)

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(X, Y, '')
ax.plot(X, Y, 'ro')
ax.plot(X, Y_, 'y')
ax.plot(X, Y_, 'r+')
ax.set_xlabel('Tag')
ax.set_ylabel('Bestand')
plt.title('Ist-Bestand (scaled)')
plt.show()
