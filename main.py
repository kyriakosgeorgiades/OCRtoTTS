import pyttsx3
from data_analysis import map_txt, test_y, test_X
from allModels import cnn, lstm, cae
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

cae()
# cnn_, character = cnn()
# lstm_ = lstm()


# cnn_accuracy = cnn_.history['accuracy']
# print("CNN ACCURACY: ", max(cnn_accuracy))
# cnn_val_acc = cnn_.history['val_accuracy']
# print("CNN VAL_ACCURACY: ", max(cnn_val_acc))
#
# lstm_accuracy = lstm_.history['accuracy']
# print("LSTM ACCURACY: ", max(lstm_accuracy))
# lstm_val_acc = lstm_.history['val_accuracy']
# print("LSTM VAL_ACCURACY: ", max(lstm_val_acc))
#
# plt.plot(cnn_accuracy)
# plt.plot(cnn_val_acc)
# plt.plot(lstm_accuracy)
# plt.plot(lstm_val_acc)
# plt.xlim(0, 30)
# plt.ylim(0, 1)
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['CNN Acc', 'Val_Acc_CNN', 'LSTM ACC', 'Val_Acc_CNN'], loc='lower right')
# plt.show()
#
# model = keras.models.load_model('CNN')
# pred_label = np.argmax(model.predict(test_X[856].reshape(-1, 28, 28, 1)))
# character_to_speech = chr(map_txt[pred_label])
# print(character_to_speech)
# print(chr(map_txt[np.argmax(test_y[856])]))
#
# plt.title(f" Predicted {chr(map_txt[pred_label])} vs Label {chr(map_txt[np.argmax(test_y[856])])}")
# plt.imshow(test_X[856], plt.cm.binary)
# plt.show()
# engine = pyttsx3.init()
# engine.say(character_to_speech)
# engine.runAndWait()

