import keras.layers
import numpy
from data_analysis import train_data, test_data, y_train, y_test, test_X, train_X, train_y, test_y, num_classes, \
    map_txt, X_train_arabic, chinese, train_X_lstm, test_X_lstm, train_y_lstm, test_y_lstm
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, LSTM, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, \
    LeakyReLU as LR, \
    Dropout, Activation, Reshape, Normalization, GlobalAvgPool2D
from keras.models import Model, Sequential
from keras.applications import vgg16
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


# input_img = Input(shape=(28, 28, 1))
# x = keras.layers.Flatten()(input_img)
# encoder_output = keras.layers.Dense(64, activation= 'relu')(x)
#
# encoder = keras.Model(input_img, encoder_output, name= "encoder")
#
# decoder_input = keras.layers.Dense(784, activation= 'relu')(encoder_output)
# decoder_output = keras.layers.Reshape((28, 28, 1))(decoder_input)
#
# opt = tf.keras.optimizers.Adam(lr =0.001, decay = 1e-6)
#
# autoencoder = keras.Model(input_img, decoder_output ,name= "autoencoder")
# autoencoder.summary()
# autoencoder.compile(optimizer= opt, loss= 'mse')
# autoencoder.fit(train_X, train_X, epochs = 15, batch_size= 32, validation_split=0.1)
#
# example = encoder.predict(test_X[0].reshape(-1, 28, 28, 1))[0]
# print(example)
# print(example.shape)
# plt.imshow(example.reshape(8,8), cmap="gray")
# plt.imshow(test_X[0], cmap = 'gray')
#
# ae_out = autoencoder.predict(test_X[0].reshape(-1, 28, 28, 1))[0]
# plt.imshow(ae_out, cmap = 'gray')

# def cae():


# plt.scatter(np.arange(len(mqr1)), np.mean(np.mean(mqr1, axis=1), axis=1), s=1)
# plt.scatter(np.arange(len(mqr2)), np.mean(np.mean(mqr2, axis=1), axis=1), s=1)
# plt.title('Mean Square Error of Normal Vs Anomaly')
# plt.legend(['Normal', 'Anomaly'], loc='upper left')
# plt.show()


# noise_factor = 0.4
# x_train_noisy = train_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_X.shape)
# x_test_noisy = test_X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_X.shape)
#
# x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# x_test_noisy = np.clip(x_test_noisy, 0., 1.)
#
# n = 10
# plt.figure(figsize=(20, 2))
# for i in range(n):
#     ax = plt.subplot(1, n, i + 1)
#     plt.imshow(np.transpose(x_test_noisy[i].reshape(28, 28)))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
#
# autoencoder.fit(x_train_noisy, train_X,
#                 epochs=100,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test_noisy, test_X)
#                 )
#
# decoded_imgs = autoencoder.predict(test_X)
#
# n = 10
#
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(np.transpose(x_test_noisy[i].reshape(28, 28)))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(np.transpose(decoded_imgs[i].reshape(28, 28)))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
#
# print(train_X_lstm[1:].shape)


#def cnn():
model_CNN = Sequential()
model_CNN.add(Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=(28, 28, 1)))
model_CNN.add(Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=2))
model_CNN.add(Dropout(.5))
model_CNN.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model_CNN.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
model_CNN.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model_CNN.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model_CNN.add(MaxPooling2D(pool_size=2, strides=(2, 2)))
model_CNN.add(Dropout(0.25))
model_CNN.add(Flatten())
model_CNN.add(Dense(units=32, activation='relu')) # Units was at 256
model_CNN.add(Dropout(.5))
model_CNN.add(Dense(units=num_classes, activation='softmax'))

# Storing the best weights and change Learning Rate

model_CNN.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
history = model_CNN.fit(train_X, train_y, epochs=30, validation_data=(test_X, test_y), batch_size=256)

cnn_accuracy_best = history.history['accuracy']
print("CNN ACCURACY: ", max(cnn_accuracy_best))
cnn_val_acc_best = history.history['val_accuracy']
print("CNN VAL_ACCURACY: ", max(cnn_val_acc_best))
###########################################
#   CAE

input_img = Input(shape=(28, 28, 1))
encoded = model_CNN.layers[-3].output
# x = Conv2D(8, 3, activation='relu', padding='same')(input_img)
# x = MaxPooling2D((2, 2), padding='same')(x)
# # x = Dropout(0.3)(x)
# x = Conv2D(16, 3, activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(32, 3, activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(64, 3, activation='relu', padding='same')(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(128, 3, activation='relu', padding='same')(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Dense(64)(encoded)
x = Reshape((1, 1, 64))(x)
x = UpSampling2D((4, 4))(x)
# x = UpSampling2D((7, 7))(x)
# x = Conv2D(128, 3, activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(64, 3, activation='relu', padding='same')(x)
# x = UpSampling2D((2, 2))(x)
x = Conv2D(32, 3, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, 3, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, 3, activation='relu', padding='valid')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, 3, activation='relu', padding='same')(x)
decoded = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

autoencoder = Model(model_CNN.input, decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy',
                    metrics=['accuracy'])

for layer in autoencoder.layers[:14]:
    layer.trainable = False

autoencoder.summary()




print("SHOW SHAPE OF DATA")
print(train_X.shape)
model_CNN1 = autoencoder.fit(train_X, train_X,
                             epochs=10,
                             batch_size=128,
                             shuffle=True,
                             validation_data=(X_train_arabic,  X_train_arabic))

# features = autoencoder.predict(train_X)
# gmm = GaussianMixture(n_components=1)

print(autoencoder.evaluate(train_X, train_X))
print(autoencoder.evaluate(chinese, chinese))
print(autoencoder.evaluate(X_train_arabic, X_train_arabic))
decoded_imgs = autoencoder.predict(X_train_arabic)
decoded_imgs_china = autoencoder.predict(chinese)
decoded_imgs_original = autoencoder.predict(test_X)


def rme_cal(images, predictions):
    mse_list = []
    batch_length = images.shape[0] // 500
    for i in range(500):
        start = i * batch_length
        end = (i + 1) * batch_length
        img_batch = images[start:end]
        pred_batch = predictions[start:end]
        loss = tf.keras.losses.mse(img_batch, pred_batch)
        mask = tf.cast(tf.not_equal(pred_batch, 0), tf.float32)
        mask = tf.squeeze(mask, axis=-1)
        sum = tf.math.reduce_sum(loss * mask, axis=[-1, -2])
        mse_list.append(sum / tf.math.reduce_sum(mask, axis=[-1, -2]))
    return tf.concat(mse_list, axis=0)


print("rme of non black pixels in test_X")
rme1 = rme_cal(test_X, decoded_imgs_original)
print(rme1.shape)
rme2 = rme_cal(chinese, decoded_imgs_china)
print(rme2.shape)
rm3 = rme_cal(X_train_arabic, decoded_imgs)

mqr1 = tf.keras.losses.binary_crossentropy(test_X, decoded_imgs_original)
print("This is the mean squared error")
print(mqr1.shape)

mqr2 = tf.keras.losses.binary_crossentropy(X_train_arabic, decoded_imgs)
# print("This is the mean square error of arabic")
# print(mqr2.shape)

#mqr3 = tf.keras.losses.binary_crossentropy(chinese, autoencoder.predict(chinese))


plt.scatter(np.arange(len(rme1)), rme1, s=1)
plt.scatter(np.arange(len(rm3)), rm3, s=1)
plt.title('Mean Square Error of Normal Vs Anomaly')
plt.legend(['Normal', 'Anomaly'], loc='upper left')
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


def lstm():
    model_lstm = Sequential()
    model_lstm.add(LSTM(128, input_shape=(train_X_lstm.shape[1:]), activation='relu', return_sequences=True))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(LSTM(128, activation='relu'))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(32, activation='relu'))
    model_lstm.add(Dropout(0.2))
    model_lstm.add(Dense(num_classes, activation='softmax'))
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    model_lstm.compile(loss='categorical_crossentropy', optimizer=opt,
                       metrics=['accuracy'])
    model_lstm.summary()
    history = model_lstm.fit(train_X_lstm, train_y_lstm, epochs=30, validation_data=(test_X_lstm, test_y_lstm))

    lstm_accuracy_best = history.history['accuracy']
    print("CNN ACCURACY: ", max(lstm_accuracy_best))
    lstm_val_acc_best = history.history['val_accuracy']
    print("CNN VAL_ACCURACY: ", max(lstm_val_acc_best))

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
        print(f"Pred: {np.argmax(preds[nums])} vs Label {np.argmax(test_y[nums])}")

    fig, axes = plt.subplots(3, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flat):
        index = np.squeeze(np.argmax(model_lstm.predict(test_X_lstm[i + 255].reshape(1, 28, 28)), axis=1), axis=0)
        label = np.argmax(test_y[i + 255])
        ax.set_title(f" Predicted {chr(map_txt[index])} vs Label {chr(map_txt[label])}")
        ax.imshow(test_X_lstm[i + 255], plt.cm.binary)
    plt.show()
