import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

print("TensorFlow version: {}".format(tf.VERSION))

# Location of data files
datafilePath = "./data/Bedarfsvorhersage.csv"

# Ist-Bestandsverlauf plotten
# tag, bestand = np.loadtxt(datafilePath, delimiter=',', skiprows=1, unpack=True)
dataframeX = pd.read_csv(datafilePath, delimiter=',', skiprows=1, usecols=[0])
dataframeY = pd.read_csv(datafilePath, delimiter=',', skiprows=1, usecols=[1])

# pandas DataFrame to numpy Array and reshape nD-Array to a 1D-Array for matplot
x = dataframeX.values.reshape([dataframeX.size])
y = dataframeY.values.reshape([dataframeY.size])
#plt.bar(x, y)
#plt.xlabel('Tag')
#plt.ylabel('Bestand')
#plt.title('Bestandsverlauf')
#plt.show()

# We use a constant seed for our random number generator to create the same pseudo-random numbers each time.
# This comes handy when we want to try different models and to compare their performances.
seed = 7
np.random.seed(seed)

#Needed to use the TensorBoard tool to visualize the model training.
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./tmp/model_graph', write_graph=True)

# Define the NN as with fully connected layers with 12 nodes each that are using the sigmoid function.
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Activation('linear'))
opt = Adam(0.01)
model.compile(loss='mse', optimizer=opt, metrics=['mse'])
#Train our model using the defined optimizer and loss function.
# Epoch is the number of times (iterations) the whole data set will go through the network,
# validation_split is how much data from the dataset to hold back just to validate the performance of the model.
# validation_split is how much data from the dataset to hold back just to validate the performance of the model.

H = model.fit(dataframeX.values, dataframeY.values, epochs=100, batch_size=10, verbose=1, validation_split=0.3, callbacks=[tbCallBack])

plt.plot(H.history['mean_squared_error'])
plt.show()

# Predict
X_= []
Y_= []
#for x_ in range(71, 140):
for x_ in range(0, 70):
    X_.append(x_)
    #print("Prognose:")
    #print("Tag {x_:5d} -> Bestand {y_:8.2f}").format(x_, y_)

Y_ = model.predict(X_)

plt.plot(x, y, 'b',
         X_, Y_, 'r')
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

