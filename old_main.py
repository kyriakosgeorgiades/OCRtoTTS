from models import cnn, lstm
import matplotlib.pyplot as plt


cnn_acc, cnn_val_acc = cnn()
lstm_acc, lstm_val_acc = lstm()

plt.plot(cnn_acc)
plt.plot(cnn_val_acc)
plt.plot(lstm_acc)
plt.plot(lstm_val_acc)
plt.xlim(0, 10)
plt.ylim(0, 1)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_CNN', 'test_CNN', 'train_LSMT', 'test_LSMT'], loc='lower right')
plt.show()

