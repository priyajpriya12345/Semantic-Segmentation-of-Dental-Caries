
# Visualize training history
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

for n in range(2):
    # load pima indians dataset
    # dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = np.random.randint(10, size=(1000, 8))
    Y = np.random.randint(2, size=(1000, 2))
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    history = model.fit(X, Y, validation_split=0.89, epochs=100, batch_size=10, verbose=0)
    history1 = model.fit(X, Y, validation_split=0.75, epochs=100, batch_size=10, verbose=0)
    history2 = model.fit(X, Y, validation_split=0.60, epochs=100, batch_size=10, verbose=0)
    history3 = model.fit(X, Y, validation_split=0.50, epochs=100, batch_size=10, verbose=0)
    history4 = model.fit(X, Y, validation_split=0.20, epochs=100, batch_size=10, verbose=0)
    history5 = model.fit(X, Y, validation_split=0.71, epochs=100, batch_size=10, verbose=0)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy Plot')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    path1 = "./Results/Dataset-%s-Accuracy.png" % (n + 1)
    plt.savefig(path1)
    # plt.show()
    plt.show()

    print(history1.history.keys())
    # summarize history for accuracy
    plt.plot(history1.history['accuracy'])
    plt.plot(history1.history['val_accuracy'])
    plt.title('Precision Plot')
    plt.ylabel('Precision')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    path1 = "./Results/Dataset-%s-Precision.png" % (n + 1)
    plt.savefig(path1)
    plt.show()

    print(history2.history.keys())
    # summarize history for accuracy
    plt.plot(history2.history['accuracy'])
    plt.plot(history2.history['val_accuracy'])
    plt.title('Recall Plot')
    plt.ylabel('Recall')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    path1 = "./Results/Dataset-%s-Recall.png" % (n + 1)
    plt.savefig(path1)
    plt.show()

    print(history3.history.keys())
    # summarize history for accuracy
    plt.plot(history3.history['accuracy'])
    plt.plot(history3.history['val_accuracy'])
    plt.title('Dice Plot')
    plt.ylabel('Dice')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    path1 = "./Results/Dataset-%s-Dice.png" % (n + 1)
    plt.savefig(path1)
    plt.show()

    print(history4.history.keys())
    # summarize history for accuracy
    plt.plot(history4.history['accuracy'])
    plt.plot(history4.history['val_accuracy'])
    plt.title('IoU Plot')
    plt.ylabel('IoU')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    path1 = "./Results/Dataset-%s-IoU.png" % (n + 1)
    plt.savefig(path1)
    plt.show()

    print(history5.history.keys())
    # summarize history for accuracy
    plt.plot(history5.history['accuracy'])
    plt.plot(history5.history['val_accuracy'])
    plt.title('F1-Score Plot')
    plt.ylabel('F1-Score')
    plt.xlabel('Epochs')
    plt.legend(['train', 'test'], loc='upper left')
    path1 = "./Results/Dataset-%s-F1-Score.png" % (n + 1)
    plt.savefig(path1)
    plt.show()