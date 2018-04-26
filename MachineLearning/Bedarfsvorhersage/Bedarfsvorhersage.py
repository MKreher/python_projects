import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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

plt.bar(x, y)
plt.xlabel('Tag')
plt.ylabel('Bestand')
plt.title('Bestandsverlauf')
#plt.show()

# We use a constant seed for our random number generator to create the same pseudo-random numbers each time.
# This comes handy when we want to try different models and to compare their performances.
seed = 7
np.random.seed(seed)

#https://github.com/kmclaugh/fastai_courses/blob/master/ai-playground/Keras_Linear_Regression_Example.ipynb

#Needed to use the TensorBoard tool to visualize the model training.
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./tmp/model_graph', write_graph=True)

# Define the NN as with fully connected layers with 12 nodes each that are using the sigmoid function.
model = Sequential()
model.add(Dense(60, input_shape=(1,), activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#Train our model using the defined optimizer and loss function.
# Epoch is the number of times (iterations) the whole data set will go through the network,
# validation_split is how much data from the dataset to hold back just to validate the performance of the model.
# validation_split is how much data from the dataset to hold back just to validate the performance of the model.
model.fit(dataframeX.values, dataframeY.values, epochs=100, batch_size=10, verbose=1, validation_split=0.3, callbacks=[tbCallBack])

# Predict
prediction = model.predict(np.array([30]))
print(prediction)
