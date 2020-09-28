################################################
import tensorflow as tf


def cnn(filters1, filters2, kernel_size1, kernel_size2, poolingfun1, poolingfun2):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=filters1, kernel_size=kernel_size1, strides=(1, 1), padding='valid', data_format='channels_last', name='conv_1', activation='relu'))

    if poolingfun1 == 'MaxPool':
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool_1'))
    elif poolingfun1 == 'AveragePool':
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='pool_1'))

    model.add(tf.keras.layers.Conv2D(filters=filters2, kernel_size=kernel_size2, strides=(3, 3), padding='valid', name='conv_2', activation='relu'))

    if poolingfun2 == 'MaxPool':
        model.add(tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), name='pool_2'))
    elif poolingfun2 == 'AveragePool':
        model.add(tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), name='pool_2'))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, name='fc_1', activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.5))
    model.add(tf.keras.layers.Dense(units=10, name='fc_2', activation='softmax'))
    tf.random.set_seed(1)
    model.build(input_shape=(None, 28, 28, 1))

    print(model.summary())
    return model
##########################