from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(10)
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing

# Parameters
N_EPOCHS = 50
BLOCK_SIZE = 25
N_TESTS = 8
avg_norm_err = 0
avg_mal_err = 0

# Import the normal traffic training dataset
print("Importing Training Dataset...")
train_dataset = pd.read_csv("C:\\Users\\Andrew\\Desktop\\Datasets\\Normal\\Skyhook\\split\\AllTraffic_chunk2.csv",
                      header=[0,1,2,3])

train_dataset2 = pd.read_csv("C:\\Users\\Andrew\\Desktop\\Datasets\\Normal\\Skyhook\\split\\AllTraffic_chunk3.csv",
                      header=[0,1,2,3])

# Import the normal traffic testing dataset
print("Importing Testing Dataset...")
test_dataset = pd.read_csv("C:\\Users\\Andrew\\Desktop\\Datasets\\Normal\\Skyhook\\split\\AllTraffic_chunk1.csv",
                      header=[0,1,2,3])

# Normalize the datasets
print("Normalizing Data...")
train_dataset = preprocessing.MinMaxScaler().fit_transform(train_dataset.values)
train_dataset = preprocessing.Normalizer().fit_transform(train_dataset)
train_dataset2 = preprocessing.MinMaxScaler().fit_transform(train_dataset2.values)
train_dataset2 = preprocessing.Normalizer().fit_transform(train_dataset2)
test_dataset = preprocessing.MinMaxScaler().fit_transform(test_dataset.values)
test_dataset = preprocessing.Normalizer().fit_transform(test_dataset)

# Reshape to 3D (Shave off remainders of dataset if not all data can fit fully into one block)
print("Reshaping data to 3D...")
train_dataset = train_dataset[: -(train_dataset.shape[0] % BLOCK_SIZE)]
train_dataset = train_dataset.reshape(int(train_dataset.shape[0]/BLOCK_SIZE), BLOCK_SIZE, 4)

train_dataset2 = train_dataset2[: -(train_dataset2.shape[0] % BLOCK_SIZE)]
train_dataset2 = train_dataset2.reshape(int(train_dataset2.shape[0]/BLOCK_SIZE), BLOCK_SIZE, 4)

test_dataset = test_dataset[: -(test_dataset.shape[0] % BLOCK_SIZE)]
test_dataset = test_dataset.reshape(int(test_dataset.shape[0]/BLOCK_SIZE), BLOCK_SIZE, 4)

# Create the network model
print("Building Network Model...")
input_dimension = train_dataset.shape[1:]
encoding_dimension = (int(input_dimension[0]/2), int(input_dimension[1]/2))
input_layer = keras.layers.Input(shape=input_dimension)
encoder = keras.layers.LSTM(encoding_dimension[1], activation=tf.nn.tanh, return_sequences=True)(input_layer)
decoder = keras.layers.LSTM(input_dimension[1], activation=tf.nn.tanh, return_sequences=True)(encoder)
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# Non-Recurrent model
#input_dimension = train_dataset.shape[1]
#encoding_dimension = int(input_dimension/2)
#input_layer = keras.layers.Input(shape=(input_dimension,))
#encoder = keras.layers.Dense(input_dimension, activation=tf.nn.tanh)(input_layer)
#encoder = keras.layers.Dropout(0.33)(encoder)
#encoder = keras.layers.Dense(encoding_dimension, activation=tf.nn.relu)(encoder)
#decoder = keras.layers.Dense(input_dimension, activation=tf.nn.relu)(encoder)
#decoder = keras.layers.Dense(input_dimension, activation=tf.nn.tanh)(decoder)
#autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# Round 1: Compile and train 
print("Training Round 1")
autoencoder.compile(optimizer='Adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])
history = autoencoder.fit(train_dataset, train_dataset,
                          epochs=N_EPOCHS,
                          shuffle=True,
                          validation_data=(test_dataset,test_dataset),
                          verbose=1).history

# Round 2: Train'
print("\nTraining Round 2")
history = autoencoder.fit(train_dataset2, train_dataset2,
                          epochs=N_EPOCHS,
                          shuffle=True,
                          validation_data=(test_dataset, test_dataset),
                          verbose=1).history
# End Training

# Make predictions and display error
norm_string = "C:\\Users\\Andrew\\Desktop\\Datasets\\Normal\\Skyhook\\split\\AllTraffic_chunk{}.csv"
mal_string = "C:\\Users\\Andrew\\Desktop\\Datasets\\Malicious\\Skyhook\\split\\DoS_Traffic_chunk{}.csv"
# Norm Test
for i in range(1,N_TESTS):
    build_string = norm_string.format(i)
    dataset = pd.read_csv(build_string, header=[0,1,2,3])
    dataset = preprocessing.MinMaxScaler().fit_transform(dataset.values)
    dataset = preprocessing.Normalizer().fit_transform(dataset)
    dataset = dataset[: -(dataset.shape[0] % BLOCK_SIZE)]
    dataset = dataset.reshape(int(dataset.shape[0]/BLOCK_SIZE), BLOCK_SIZE, 4)

    predictions = autoencoder.predict(dataset)
    mae = np.sum(np.absolute(dataset - predictions))
    avg_norm_err += np.power(mae,2)
    print("Norm MAE {} : {}".format(i, mae))
print("Squared Average Normal MAE {}\n".format(np.sqrt(avg_norm_err/N_TESTS)))

# Mal Test
for i in range(1,N_TESTS):
    build_string = mal_string.format(i)
    dataset = pd.read_csv(build_string, header=[0,1,2,3])
    dataset = preprocessing.MinMaxScaler().fit_transform(dataset.values)
    dataset = preprocessing.Normalizer().fit_transform(dataset)
    dataset = dataset[: -(dataset.shape[0] % BLOCK_SIZE)]
    dataset = dataset.reshape(int(dataset.shape[0]/BLOCK_SIZE), BLOCK_SIZE, 4)

    predictions = autoencoder.predict(dataset)
    mae = np.sum(np.absolute(dataset - predictions))
    print("Mal MAE {} : {}".format(i, mae))
    avg_mal_err += np.power(mae,2)
print("Squared Average Malicious MAE {}\n".format(np.sqrt(avg_mal_err/N_TESTS)))