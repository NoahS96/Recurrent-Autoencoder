import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

def build_model(train_data):
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
        #keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.AdamOptimizer(0.001)

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=['mae'])
    return model

print("Loading Boston Dataset")
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
print("Shuffling the Dataset")
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# Normalize Data
print("Normalizing")
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Build neural network model
print("Building Neural Network")
model = build_model(train_data)

# Start training the model
print("Begin Training")
EPOCHS = 500
print('Train data {}    Train Labels {}'.format(len(train_data), len(train_labels)))
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(train_data, train_labels, epochs=EPOCHS, validation_split=0.2,
                    verbose=0, callbacks=[early_stop, PrintDot()])

# Print the testing error
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# Make predictions and display
print("Generating Predictions")
test_predictions = model.predict(test_data).flatten()
error = test_predictions - test_labels
plt.hist(error, bins=50)
plt.xlabel("Prediction Error [1000$]")
_ = plt.ylabel("Count")
plt.show()