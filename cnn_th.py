import keras
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import os

def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def construct_model(classes=35):
    model = Sequential()
    model.add(Convolution2D(32, 5, 5, activation='tanh', border_mode='same', input_shape=(1, 28, 28)))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(64, 2, 2, border_mode='same', activation='tanh'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model

def predict_from_features(X, y, X_test, y_test, save_progress=None):
    model = construct_model()
    if save_progress != None and os.path.isfile(save_progress):
        model.load_weights(save_progress)
    sgd = SGD(lr=1e-4, decay=0.0, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, batch_size=50, nb_epoch=1, verbose=1, validation_data=(X_test, y_test))
    if save_progress != None:
        model.save_weights(save_progress)
    return model.evaluate(X_test, y_test, verbose=0)

# deal with O/0 and L/1 confusion
def fine_tune(classified_chars):
    num_result = len(classified_chars)
    num_digits = 0
    for d in classified_chars:
        if d >= '0' and d <= '9':
            num_digits += 1
    if (float(num_digits) / num_result) > 0.6:
        return classified_chars
    for i in xrange(0, num_result):
        if classified_chars[i] == '0':
            classified_chars[i] = 'O'
        if classified_chars[i] == '1':
            classified_chars[i] = 'L'
    return classified_chars

def classify_digits(pretraine_weights, filename):
    keras.backend.set_image_dim_ordering("th")
    labels = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
    class_to_char = {i:labels[i] for i in xrange(0, len(labels))}
    images = np.load(filename) # load data
    model = construct_model()
    model.load_weights(pretraine_weights) # load model

    w, h = 28, 28
    images = images.reshape(images.shape[0], 1, w, h)
    images[images < 128.0] = 0
    images[images > 127.0] = 1

    labels = np.argmax(model.predict(images), axis=1)

    classified_chars = [class_to_char[label] for label in labels]
    return "".join(fine_tune(classified_chars))

if __name__ == '__main__':
    save_progress = "pretrained_model.w"
    keras.backend.set_image_dim_ordering("th")

    pixels_filename = "pixels.ubyte.npy"
    labels_filename = "labels.ubyte.npy"

    dev_pixels_filename = "dev-pixels.ubyte.npy"
    dev_labels_filename = "dev-labels.ubyte.npy"

    classes = 35
    w, h = 28, 28
    train_images = np.load(pixels_filename)
    train_labels = dense_to_one_hot(np.load(labels_filename), classes)
    dev_images = np.load(dev_pixels_filename)
    dev_labels = dense_to_one_hot(np.load(dev_labels_filename), classes)

    train_images = train_images.reshape(train_images.shape[0], 1, w, h)
    dev_images = dev_images.reshape(dev_images.shape[0], 1, w, h)

    train_images[train_images < 128.0] = 0
    train_images[train_images > 127.0] = 1
    dev_images[dev_images < 128.0] = 0
    dev_images[dev_images > 127.0] = 1

    loss, accuracy = predict_from_features(train_images, train_labels, dev_images, dev_labels, save_progress)

    print "loss:", loss, "accuracy", accuracy



