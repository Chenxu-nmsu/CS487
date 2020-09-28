import tensorflow as tf
import numpy as np

model = tf.keras.Sequential()
print('Input:    ', model.compute_output_shape(input_shape=(100, 28, 28, 1)))

model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(5, 5), strides=(1, 1), padding='valid', data_format='channels_last', name='conv_1', activation='relu'))
print('Conv_1:   ', model.compute_output_shape(input_shape=(100, 28, 28, 1)))

model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool_1'))
print('Pooling_1:', model.compute_output_shape(input_shape=(100, 28, 28, 1)))

model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), strides=(3, 3), padding='valid', name='conv_2', activation='relu'))
print('Conv_2:   ', model.compute_output_shape(input_shape=(100, 28, 28, 1)))

model.add(tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), name='pool_2'))
print('Pooling_2:', model.compute_output_shape(input_shape=(100, 28, 28, 1)))

model.add(tf.keras.layers.Flatten())


model.add(tf.keras.layers.Dense(units=1024, name='fc_1', activation='relu'))
print('FC_1:     ', model.compute_output_shape(input_shape=(100, 28, 28, 1)))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(units=10, name='fc_2', activation='softmax'))
print('FC_2:     ', model.compute_output_shape(input_shape=(100, 28, 28, 1)))

tf.random.set_seed(1)
model.build(input_shape=(None, 28, 28, 1))
