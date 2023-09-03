import pandas as pd

dataset = pd.read_csv('cancer.csv')

# INPUT LAYERS

x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])

y = dataset['diagnosis(1=m, 0=b)']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=(455, 30, 1), activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# COMPILE THE MODEL

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(x_train, y_train, epochs=1000)