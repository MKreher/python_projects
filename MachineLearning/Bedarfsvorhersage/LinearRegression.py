import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation
from tensorflow.python.keras.optimizers import SGD

from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

print("TensorFlow version: {}".format(tf.VERSION))

# Generate Testdata
# 100 data points.
# TrainX has values between –1 and 1 and TrainY has 3 times the TrainX and some randomness.
X = np.linspace(0, 10, 100)
Y = X + np.random.randn(*X.shape) * 0.5

#Needed to use the TensorBoard tool to visualize the model training.
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./tmp/model_graph', write_graph=True)

# We shall create a sequential model. All we need is a single connection so we use a Dense layer with linear activation.
model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.add(Activation("linear"))

# This will take input x and apply weight, w, and bias, b followed by a linear activation to produce output.
# Let’s look at values that the weights are initialized with:
weights = model.layers[0].get_weights()
w_init = weights[0][0][0]
b_init = weights[1][0]
print('Linear regression model is initialized with weights w: %.2f, b: %.2f' % (w_init, b_init))
## Linear regression model is initialized with weight w: -0.03, b: 0.00

sgd = SGD(0.01)
model.compile(loss='mse', optimizer=sgd, metrics=['mse'])

H = model.fit(X, Y, epochs=200, batch_size=10, verbose=2, validation_split=0.3, callbacks=[tbCallBack])

plt.plot(H.history['loss'])
plt.show()

weights = model.layers[0].get_weights()
w_final = weights[0][0][0]
b_final = weights[1][0]
print('Linear regression model is trained to have weight w: %.2f, b: %.2f' % (w_final, b_final))
##Linear regression model is trained to have weight w: 2.94, b: 0.08

# Predict
Y_ = model.predict(X)

plt.plot(X, Y, 'ob')
plt.plot(X, Y_, '+r')
plt.show()

#print(X_)
#print(Y_)
