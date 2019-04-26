import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


input_layer = tf.keras.Input(shape=(7056,))
hidden_layer_1 = tf.keras.layers.Dense(500, activation=tf.nn.sigmoid)(input_layer)
hidden_layer_2 = tf.keras.layers.Dense(500, activation=tf.nn.sigmoid)(hidden_layer_1)
output_layer = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(hidden_layer_2)

network_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

network_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='sparse_categorical_accuracy')

