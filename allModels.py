import os

os.environ["PATH"] += os.pathsep + 'C:/Users/kakos/Anaconda3/Library/bin/graphviz'
from data_analysis import test_X, train_X, train_y, test_y, num_classes, \
    map_txt, X_train_arabic, chinese, train_X_lstm, test_X_lstm, train_y_lstm, test_y_lstm
import matplotlib.pyplot as plt
import time
from keras.layers import Input, Dense, Conv2D, LSTM, MaxPooling2D, UpSampling2D, Flatten, \
    Dropout, Reshape
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
import tensorflow as tf
import numpy as np


###########################################
#   CAE
# Amomaly detection function based on the data of EMNIST. Using the arabic and chinese data evaluate the data anomaly.
def cae():
    # Encoder
    input_img = Input(shape=(28, 28, 1))
    x = Conv2D(16, 3, activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    # x = Dropout(0.3)(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    # Deconder
    x = Dense(16)(encoded)
    x = Reshape((1, 1, 16))(x)
    # x = UpSampling2D((7, 7))(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, 3, activation='relu', padding='valid')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy',
                        metrics=['accuracy'])

    autoencoder.summary()
    plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True, show_layer_names=True)
    print("SHOW SHAPE OF DATA")
    print(train_X.shape)
    model_CNN1 = autoencoder.fit(train_X, train_X,
                                 epochs=10,
                                 batch_size=128,
                                 shuffle=True,
                                 validation_data=(train_X, train_X))

    latin_eval = autoencoder.evaluate(train_X, train_X)
    print("Evaluate on Latin characters")
    print("test loss, test acc: ", latin_eval)
    chinese_eval = autoencoder.evaluate(chinese, chinese)
    print("Evaluate on Chinese characters")
    print("test loss, test acc: ", chinese_eval)
    arabic_eval = autoencoder.evaluate(X_train_arabic, X_train_arabic)
    print("Evaluate on Arabic characters")
    print("test loss, test acc: ", arabic_eval)
    decoded_imgs = autoencoder.predict(X_train_arabic)
    decoded_imgs_china = autoencoder.predict(chinese)
    decoded_imgs_original = autoencoder.predict(test_X)

    mqr1 = tf.keras.losses.binary_crossentropy(test_X, decoded_imgs_original)
    print("This is the loss error")
    print(mqr1.shape)

    mqr2 = tf.keras.losses.binary_crossentropy(X_train_arabic, decoded_imgs)
    print("This is the mean square error of arabic")
    print(mqr2.shape)

    # mqr3 = tf.keras.losses.binary_crossentropy(chinese, autoencoder.predict(chinese))

    plt.scatter(np.arange(len(mqr1)), tf.reduce_mean(mqr1, axis=[1, 2]), s=1)
    plt.scatter(np.arange(len(mqr2)), tf.reduce_mean(mqr2, axis=[1, 2]), s=1)
    plt.axhline(y=tf.reduce_mean(mqr1) + tf.math.reduce_std(tf.reduce_mean(mqr1, axis=[1, 2])) * 3, color='r',
                linestyle='-')
    plt.title('Mean Square Error of Normal Vs Anomaly')
    plt.legend(['Threshold', 'Normal', 'Anomaly'], loc='upper left')
    plt.show()

    n = 10

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(chinese[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs_china[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X_train_arabic[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(test_X[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs_original[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    plt.plot(model_CNN1.history["loss"], label="Training Loss")
    plt.plot(model_CNN1.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()


###########################################


def cnn():
    model_CNN = Sequential()
    model_CNN.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
    model_CNN.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))
    model_CNN.add(MaxPooling2D(pool_size=2))
    model_CNN.add(Dropout(.5))
    model_CNN.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_CNN.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model_CNN.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
    model_CNN.add(Dropout(0.25))
    model_CNN.add(Flatten())
    model_CNN.add(Dense(units=256, activation='relu'))
    model_CNN.add(Dropout(.5))
    model_CNN.add(Dense(units=num_classes, activation='softmax'))

    model_CNN.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      metrics=['accuracy'])
    start = time.time()
    history = model_CNN.fit(train_X, train_y, epochs=30, validation_data=(test_X, test_y), batch_size=256)
    end = time.time()
    mins = end - start
    mins = mins // 60
    print("The training took ", mins, " minutes")

    cnn_accuracy_best = history.history['accuracy']
    print("CNN ACCURACY: {:.2f}".format(max(cnn_accuracy_best)))
    cnn_val_acc_best = history.history['val_accuracy']
    print("CNN VAL_ACCURACY: {:.2f}".format(max(cnn_val_acc_best)))
    cnn_loss_best = history.history['loss']
    print("CNN LOSS: {:.2f}".format(min(cnn_loss_best)))
    cnn_val_loss_best = history.history['val_loss']
    print("CNN VAL_LOSS: {:.2f}".format(min(cnn_val_loss_best)))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model_CNN accuracy of CNN')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model_CNN loss of CNN')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, 30)
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
        index = np.squeeze(np.argmax(model_CNN.predict(test_X[i + 255].reshape(1, 28, 28, 1)), axis=1), axis=0)
        label = np.argmax(test_y[i + 255])
        ax.set_title(f" Predicted {chr(map_txt[index])} vs Label {chr(map_txt[label])}")
        ax.imshow(test_X[i + 255], plt.cm.binary)
    plt.show()

    img = test_X[10].reshape(1, 28, 28, 1)
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img[0, :, :, 0], cmap="gray")
    plt.axis('off')
    plt.show()

    # activations_keract = get_activations(model_CNN, img)
    # display_activations(activations_keract, cmap="gray", save=False)

    # visualizer(model_CNN, format='png', view=True)
    plot_model(model_CNN, to_file='model_plot_CNN.png', show_shapes=True, show_layer_names=True)

    # model_CNN.save('CNN')
    return history, test_X[684]


def lstm():
    model_lstm = Sequential()
    model_lstm.add(LSTM(28, input_shape=(train_X_lstm.shape[1:]), activation='tanh', return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(64, activation='tanh', return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(128, activation='tanh'))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(64, activation='relu'))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(64, activation='relu'))
    model_lstm.add(Dense(num_classes, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    model_lstm.compile(loss='categorical_crossentropy', optimizer=opt,
                       metrics=['accuracy'])
    model_lstm.summary()
    start = time.time()
    history = model_lstm.fit(train_X_lstm, train_y_lstm, epochs=30, validation_data=(test_X_lstm, test_y_lstm))
    end = time.time()
    mins = end - start
    mins = mins // 60
    print("The time it took for the model to train is: ", mins, " minutes.")
    lstm_accuracy_best = history.history['accuracy']
    print("LSTM ACCURACY: {:.2f}".format(max(lstm_accuracy_best)))
    lstm_val_acc_best = history.history['val_accuracy']
    print("LSTM VAL_ACCURACY: {:.2f}".format(max(lstm_val_acc_best)))
    lstm_loss_best = history.history['loss']
    print("LSTM LOSS: {:.2f}".format(min(lstm_loss_best)))
    lstm_val_loss_best = history.history['val_loss']
    print("LSTM VAL_LOSS: {:.2f}".format(min(lstm_val_loss_best)))

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model_LSTM accuracy of LSTM')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model_LSTM loss of LSTM')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(0, 30)
    plt.ylim(0, 1)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    preds = model_lstm.predict(test_X_lstm)
    import random
    for i in range(1, 10):
        nums = random.randint(0, 22560)
        print(f"Pred: {np.argmax(preds[nums])} vs Label {np.argmax(test_y_lstm[nums])}")

    fig, axes = plt.subplots(3, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        index = np.squeeze(np.argmax(model_lstm.predict(test_X_lstm[i + 255].reshape(1, 28, 28)), axis=1), axis=0)
        label = np.argmax(test_y_lstm[i + 255])
        ax.set_title(f" Predicted {chr(map_txt[index])} vs Label {chr(map_txt[label])}")
        ax.imshow(test_X_lstm[i + 255], plt.cm.binary)
    plt.show()

    plot_model(model_lstm, to_file='model_plot_LSTM.png', show_shapes=True, show_layer_names=True)

    return history
