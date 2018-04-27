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
datafilePath = "./data/Bedarfsvorhersage.csv"

# Ist-Bestandsverlauf plotten
# tag, bestand = np.loadtxt(datafilePath, delimiter=',', skiprows=1, unpack=True)
dataframeX = pd.read_csv(datafilePath, delimiter=',', skiprows=1, usecols=[0])
dataframeY = pd.read_csv(datafilePath, delimiter=',', skiprows=1, usecols=[1])

# pandas DataFrame to numpy Array and reshape nD-Array to a 1D-Array for matplot
#x = dataframeX.values.reshape([dataframeX.size])
#y = dataframeY.values.reshape([dataframeY.size])

scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(dataframeX)
Y = scaler.fit_transform(dataframeY)


#x = np.arange(200).reshape(-1,1) / 50
#y = np.sin(x)

#Needed to use the TensorBoard tool to visualize the model training.
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./tmp/model_graph', write_graph=True)

# Define the NN as with fully connected layers with 12 nodes each that are using the sigmoid function.
model = Sequential()
model.add(Dense(40, activation='sigmoid', kernel_initializer='uniform', input_shape=(1,)))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(1, activation='linear'))
sgd = SGD(0.001);
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mean_squared_error'])

H = model.fit(X, Y, epochs=10000, batch_size=50, verbose=2, validation_split=0.3, callbacks=[tbCallBack])

#plt.plot(H.history['mean_squared_error'])
#plt.show()

# Predict
X_= []
for x_ in range(0, 70):
    X_.append(x_)

Y_ = model.predict(X_)

plt.plot(X, Y, 'b')
plt.plot(X, Y_, 'r')
plt.show()

#print(X_)
#print(Y_)

#plt.title('Bestandsprognose')
#plt.subplot(211)
#plt.plot(x, y)
#plt.xlabel('Tag')
#plt.ylabel('Bestand')
#plt.subplot(212)
#plt.plot(X_, Y_)
#plt.xlabel('Tag')
#plt.ylabel('Prognose')
#plt.show()

