from numpy.random import seed
seed(5)
from tensorflow import set_random_seed
set_random_seed(10)
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing

# build
#   args:
#       - dataset_shape:    A tuple of the dataset dimensional shape in tensorflow format
#   returns:
#       - A keras network model
def build(dataset_shape):
    input_dimension = dataset_shape[1:]
    encoding_dimension = (int(input_dimension[0]/2), int(input_dimension[1]/2))
    input_layer = keras.layers.Input(shape=input_dimension)
    encoder = keras.layers.LSTM(encoding_dimension[1], activation=tf.nn.tanh, return_sequences=True)(input_layer)
    decoder = keras.layers.LSTM(input_dimension[1], activation=tf.nn.tanh, return_sequences=True)(encoder)
    autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

    return autoencoder

# normalize
#   args:
#       - dataset:  A pandas dataframe containing the data to be processed
#   returns:
#       - A numpy array of the normalized data
def normalize(dataset):
    dataset = preprocessing.MinMaxScaler().fit_transform(dataset.values)
    dataset = preprocessing.Normalizer().fit_transform(dataset)

    return dataset

# reshape3D
#   args:
#       - dataset:      A 2D numpy array 
#       - block_size:   The size of a block in one time slice
#   returns:
#       - A 3D numpy array converted from the provided 2D numpy array
#   note:
#       - If the last data block is not completely filled, it will be dropped. This means not all data provided in the
#         dataset will be present in the returned dataset.
def reshape3D(dataset, block_size):
    feature_count = dataset.shape[-1]
    dataset = dataset[: -(dataset.shape[0] % block_size)]
    dataset = dataset.reshape(int(dataset.shape[0]/block_size), block_size, feature_count)

    return dataset

# Begin main
# Parameters
N_EPOCHS = 50
BLOCK_SIZE = 25
N_TESTS = 8
avg_norm_err = 0
avg_anom_err = 0

norm_filepath = "C:\\Users\\Andrew\\Desktop\\Datasets\\Normal\\Skyhook\\split\\AllTraffic_chunk{}.csv"
mal_filepath = "C:\\Users\\Andrew\\Desktop\\Datasets\\Malicious\\Skyhook\\split\\DoS_Traffic_chunk{}.csv"

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
train_dataset = normalize(train_dataset)
train_dataset2 = normalize(train_dataset2)
test_dataset = normalize(test_dataset)

# Reshape to 3D (Shave off remainders of dataset if not all data can fit fully into one block)
print("Reshaping data to 3D...")
train_dataset = reshape3D(train_dataset, BLOCK_SIZE)
train_dataset2 = reshape3D(train_dataset2, BLOCK_SIZE)
test_dataset = reshape3D(test_dataset, BLOCK_SIZE)

# Create the network model
print("Building Network Model...")
autoencoder = build(train_dataset.shape)

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

# Make predictions and display error
# Norm Test
for i in range(1,N_TESTS):
    build_string = norm_filepath.format(i)
    dataset = pd.read_csv(build_string, header=[0,1,2,3])
    dataset = normalize(dataset)
    dataset = reshape3D(dataset, BLOCK_SIZE)

    predictions = autoencoder.predict(dataset)
    mae = np.sum(np.absolute(dataset - predictions))
    avg_norm_err += np.power(mae,2)
    print("Norm MAE {} : {}".format(i, mae))
print("Squared Average Normal MAE {}\n".format(np.sqrt(avg_norm_err/N_TESTS)))

# Mal Test
for i in range(1,N_TESTS):
    build_string = mal_filepath.format(i)
    dataset = pd.read_csv(build_string, header=[0,1,2,3])
    dataset = normalize(dataset)
    dataset = reshape3D(dataset, BLOCK_SIZE)

    predictions = autoencoder.predict(dataset)
    mae = np.sum(np.absolute(dataset - predictions))
    print("Mal MAE {} : {}".format(i, mae))
    avg_anom_err += np.power(mae,2)
print("Squared Average Malicious MAE {}\n".format(np.sqrt(avg_anom_err/N_TESTS)))