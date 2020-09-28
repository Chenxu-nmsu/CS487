from time import process_time
import tensorflow as tf
import mycnn
import sys
import matplotlib.pyplot as plt
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

######################
# load_mnist func
import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels
##########################

# load_mnist
X_train, y_train = load_mnist('./mnist/', kind='train')
X_test, y_test = load_mnist('./mnist/', kind='t10k')

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train_centered = (X_train - mean_vals) / std_val
X_test_centered = (X_test - mean_vals) / std_val

### ---> Preprocess dataset
# convert to tensors [train]
x_train = tf.convert_to_tensor(X_train_centered)
x_train = tf.reshape(x_train, [-1, 28, 28, 1])
y_train = tf.convert_to_tensor(y_train)

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.int32)

train_x = tf.data.Dataset.from_tensor_slices(x_train)
train_y = tf.data.Dataset.from_tensor_slices(y_train)
train_joint = tf.data.Dataset.zip((train_x, train_y))

# convert to tensors [test]
x_test = tf.convert_to_tensor(X_test_centered)
x_test = tf.reshape(x_test, [-1, 28, 28, 1])
y_test = tf.convert_to_tensor(y_test)

x_test = tf.cast(x_test, tf.float32)
y_test = tf.cast(y_test, tf.int32)

test_x = tf.data.Dataset.from_tensor_slices(x_test)
test_y = tf.data.Dataset.from_tensor_slices(y_test)
mnist_test = tf.data.Dataset.zip((test_x, test_y))

BUFFER_SIZE = 10000
BATCH_SIZE = 100
NUM_EPOCHS = 10
tf.random.set_seed(1)

mnist_train = train_joint.shuffle(buffer_size=BUFFER_SIZE, reshuffle_each_iteration=False)

# mnist_train + valid dataset
mnist_valid = mnist_train.take(10000).batch(BATCH_SIZE)
mnist_train = mnist_train.skip(10000).batch(BATCH_SIZE)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # parameters in kmeans
    parser.add_argument('--filters1', help='filters in [conv_1]', type=int)
    parser.add_argument('--filters2', help='filters in [conv_2]', type=int)
    parser.add_argument('--kernel_size1', nargs='+', help='kernel_Size in [conv_1]', type=int)
    parser.add_argument('--kernel_size2', nargs='+', help='kernel_Size in [conv_2]', type=int)
    parser.add_argument('--poolingfun1', help='poolingfun in [pool_1]', type=str)
    parser.add_argument('--poolingfun2', help='poolingfun in [pool_2]', type=str)

    args = parser.parse_args()
    args = vars(parser.parse_args())

    model = mycnn.cnn(filters1=args['filters1'], filters2=args['filters2'],
                      kernel_size1=tuple(args['kernel_size1']), kernel_size2=tuple(args['kernel_size2']),
                      poolingfun1=args['poolingfun1'], poolingfun2=args['poolingfun2'])

    # compile and fit model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        metrics=['accuracy'])
    # same as `tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')`

    # record initial time
    start = process_time()

    history = model.fit(mnist_train, epochs=NUM_EPOCHS,
                              validation_data=mnist_valid,
                              shuffle=True)

    # calculate the time elapse
    elapse = process_time() - start

    print('Running time: {0:.5f} s'.format(elapse))

    # plot training/testing loss and accuracy curves
    hist = history.history
    x_arr = np.arange(len(hist['loss'])) + 1

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, hist['loss'], '-o', label='Train loss')
    ax.plot(x_arr, hist['val_loss'], '--<', label='Validation loss')
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Loss', size=15)
    ax.legend(fontsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, hist['accuracy'], '-o', label='Train acc.')
    ax.plot(x_arr, hist['val_accuracy'], '--<', label='Validation acc.')
    ax.legend(fontsize=15)
    ax.set_xlabel('Epoch', size=15)
    ax.set_ylabel('Accuracy', size=15)

    plt.savefig('15_12.png', dpi=300)
    plt.show()

    # test accuracy
    test_results = model.evaluate(mnist_test.batch(20))
    print('\nTest Acc. {:.2f}%'.format(test_results[1] * 100))

    ###
    batch_test = next(iter(mnist_test.batch(12)))

    preds = model(batch_test[0])

    tf.print(preds.shape)
    preds = tf.argmax(preds, axis=1)
    print(preds)

    fig = plt.figure(figsize=(12, 4))
    for i in range(12):
        ax = fig.add_subplot(2, 6, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        img = batch_test[0][i, :, :, 0]
        ax.imshow(img, cmap='gray_r')
        ax.text(0.9, 0.1, '{}'.format(preds[i]),
                size=15, color='blue',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

    plt.savefig('15_13.png', dpi=300)
    plt.show()
