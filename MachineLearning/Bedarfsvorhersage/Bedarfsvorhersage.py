import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense

print("TensorFlow version: {}".format(tf.VERSION))

# Location of data files
datafilePath = "./data/Bedarfsvorhersage.csv"

# Ist-Bestandsverlauf plotten
tag, bestand = np.loadtxt(datafilePath, delimiter=',', skiprows=1, unpack=True)
plt.bar(tag, bestand)
plt.xlabel('Tag')
plt.ylabel('Bestand')
plt.title('Bestandsverlauf')
#plt.show()

model = Sequential()
model.add(Dense(32, input_shape=[2,]))
model.add(Activation('relu'))