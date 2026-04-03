import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
SEQ_LEN = 10

def load_data(path):
    X, y = [], []
    labels = ["A", "B"]
    for label_idx, label in enumerate(labels):
        folder = os.path.join(path, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            sequence = [img]*SEQ_LEN
            X.append(sequence)
            y.append(label_idx)
    return np.array(X), np.array(y)

X, y = load_data("dataset")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(SEQ_LEN, IMG_SIZE, IMG_SIZE, 3)))
model.add(TimeDistributed(MaxPooling2D(2,2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(32))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=8)
model.save("sign_model.h5")