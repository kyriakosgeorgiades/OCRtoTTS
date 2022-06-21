import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from data import train_data, y_train, y_test, num_classes, test_data, map_txt
import numpy as np
from keras_visualizer import visualizer
from keract import get_activations, display_activations
from keras.regularizers import l2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.models import Model

import logging

os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# For training samples

# How many training data there is
original_test_data = test_data
y_train_print = y_train
y_test_print = y_test
print(f"Original shape test data {original_test_data.shape}")
train_data_number = train_data.shape[0]
print("How many train data:")
print(train_data_number, "\n")
# Setting the height & width (28x28)
train_data_height = 28
train_data_width = 28
train_data_size = train_data_width * train_data_height

# Different pre process of data for LSTM algorithm
train_data_LSTM = train_data
train_data_LSTM = train_data_LSTM.reshape(train_data_number, train_data_width, train_data_height)
print("LOOK AT LSTM SHAPE TRAIN")
print(train_data_LSTM.shape)

# Reshaping data to 4D dimension to work in CNN model. (batch size, height, width, channel)
train_data = train_data.reshape(train_data_number, train_data_height, train_data_width, 1)

print("The shape of CNN input train data:")
print(train_data.shape, '\n')

# For testing samples

# How many testing data there is
test_data_number = test_data.shape[0]
# Setting the height & width (28x28)
test_data_height = 28
test_data_width = 28
test_data_size = test_data_width * test_data_height

test_data = test_data.reshape(test_data_number, test_data_height, test_data_width, 1)

print("The shape of CNN test data:")
print(test_data.shape, "\n")

print("The train data normalized based of unique values range of 0-1")
print(np.unique(train_data), "\n")

print("The test data normalized based of unique values range of 0-1")
print(np.unique(test_data), "\n")

# Creating categorical labels for the target in train and test dataset
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"Label size of y_train: {y_train.shape}")
print(f"Label size of y_test: {y_test.shape} \n")

# Splitting the data for CNN
train_X, test_X, train_y, test_y = train_test_split(train_data, y_train, test_size=0.2, random_state=30)
# Splitting the data for LSMT
train_X_LSMT, test_X_LSMT, train_y_LSMT, test_y_LSMT = train_test_split(train_data_LSTM, y_train, test_size=0.2,

                                                                        random_state=30)


def cnn():
    # CNN creation

    model_CNN = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(units=num_classes, activation='softmax')
    ])

    model_CNN.summary()
    model_CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Storing the best weights and change Learning Rate

    model_checkpoint = ModelCheckpoint('Saves.h5', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, restore_best_weights=True, patience=3, mode='max')
    reduce_lr_p = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)

    history = model_CNN.fit(train_X, train_y, epochs=10, validation_data=(test_X, test_y),
                            callbacks=[model_checkpoint, early_stop, reduce_lr_p])

    # img = np.transpose(test_X[10].reshape(1, 28, 28, 1))
    # fig = plt.figure(figsize=(5, 5))
    # plt.imshow(img[0, :, :, 0], cmap="gray")
    # plt.axis('off')
    # plt.show()
    #
    # activations_keract = get_activations(model_CNN, img)
    # display_activations(activations_keract, cmap="gray", save=False)
    #
    # visualizer(model_CNN, format='png', view=True)
    # activations = activation_model.predict(img)

    # model_CNN.save('CNN')
    cnn_accuracy_best = history.history['accuracy']
    print("CNN ACCURACY: ", max(cnn_accuracy_best))
    cnn_val_acc_best = history.history['val_accuracy']
    print("CNN VAL_ACCURACY: ", max(cnn_val_acc_best))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy of CNN')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss of CNN')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    preds = model_CNN.predict(test_X)
    import random
    for i in range(1, 10):
        nums = random.randint(0, 22560)
        print(f"Pred: {np.argmax(preds[nums])} vs Label {np.argmax(test_y[nums])}")

    fig, axes = plt.subplots(3, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        img_flip = np.transpose(test_X[i + 70].reshape([28, 28]))
        index = np.squeeze(np.argmax(model_CNN.predict(test_X[i + 70].reshape(1, 28, 28, 1)), axis=1), axis=0)
        label = np.argmax(test_y[i + 70])
        ax.set_title(f" Predicted {chr(map_txt[index])} vs Label {chr(map_txt[label])}")
        ax.imshow(img_flip, plt.cm.binary)
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    return acc, val_acc


def lstm():
    img = np.transpose(test_X_LSMT[13].reshape(1, 28, 28, 1))
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img[0, :, :, 0], cmap="gray")
    plt.axis('off')
    plt.show()

    model_LSTM = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(train_X_LSMT.shape[1:]), activation='tanh', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model_LSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_LSTM.summary()

    # Storing the best weights and change Learning Rate

    model_checkpoint = ModelCheckpoint('Saves_LSTM.h5', verbose=1, save_best_only=True, monitor='val_accuracy',
                                       mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, restore_best_weights=True, patience=3, mode='max')
    reduce_lr_p = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)

    history = model_LSTM.fit(train_X_LSMT, train_y_LSMT, epochs=10, validation_data=(test_X_LSMT, test_y_LSMT),
                             callbacks=[model_checkpoint, early_stop, reduce_lr_p])

    # plot_model(model_LSTM, to_file='model_LSTM.png')

    # activations_keract = get_activations(model_LSTM, img)
    # display_activations(activations_keract, cmap="gray", save=True)

    # visualizer(model_LSTM, filename='lstm_graph', format='png', view=True)

    lstm_accuracy_best = history.history['accuracy']
    print("LSTM ACCURACY: ", max(lstm_accuracy_best))
    lstm_val_acc_best = history.history['val_accuracy']
    print("LSTM VAL_ACCURACY: ", max(lstm_val_acc_best))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy of LSTM')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss of LSTM')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    preds = model_LSTM.predict(test_X_LSMT)
    import random
    for i in range(1, 10):
        nums = random.randint(0, 22560)
        print(f"Pred: {np.argmax(preds[nums])} vs Label {np.argmax(test_y_LSMT[nums])}")

    fig, axes = plt.subplots(3, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        img_flip = np.transpose(test_X[i + 70].reshape([28, 28]))
        index = np.squeeze(np.argmax(model_LSTM.predict(test_X_LSMT[i + 70].reshape(1, 28, 28)), axis=1), axis=0)
        label = np.argmax(test_y_LSMT[i + 70])
        ax.set_title(f" Predicted {chr(map_txt[index])} vs Label {chr(map_txt[label])}")
        ax.imshow(img_flip, plt.cm.binary)
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    model_LSTM.save('LSTM')

    return acc, val_acc


def cnn_lstm():
    # THREE PARAMETERS/ RESEARCH THE OPTIMAL NUMBER OF BATCHES, etc... //TODO!!
    cnn_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu',
                                       input_shape=(1, 28, 28,))
    model_LSTM = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D),
        tf.keras.layers.LSTM(32, input_shape=(train_X_LSMT.shape[1:]), activation='tanh', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model_LSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_LSTM.summary()

    # Storing the best weights and change Learning Rate

    model_checkpoint = ModelCheckpoint('Saves_LSTM.h5', verbose=1, save_best_only=True, monitor='val_accuracy',
                                       mode='max')
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=0, restore_best_weights=True, patience=3, mode='max')
    reduce_lr_p = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)

    history = model_LSTM.fit(train_X_LSMT, train_y_LSMT, epochs=10, validation_data=(test_X_LSMT, test_y_LSMT),
                             callbacks=[model_checkpoint, early_stop, reduce_lr_p])

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy of LSTM')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss of LSTM')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    preds = model_LSTM.predict(test_X_LSMT)
    import random
    for i in range(1, 10):
        nums = random.randint(0, 22560)
        print(f"Pred: {np.argmax(preds[nums])} vs Label {np.argmax(test_y_LSMT[nums])}")

    fig, axes = plt.subplots(3, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        img_flip = np.transpose(test_X[i + 70].reshape([28, 28]))
        index = np.squeeze(np.argmax(model_LSTM.predict(test_X_LSMT[i + 70].reshape(1, 28, 28)), axis=1), axis=0)
        label = np.argmax(test_y_LSMT[i + 70])
        ax.set_title(f" Predicted {chr(map_txt[index])} vs Label {chr(map_txt[label])}")
        ax.imshow(img_flip, plt.cm.binary)
