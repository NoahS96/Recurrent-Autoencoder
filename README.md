# NeuralNetworkTest

A recurrent autencoder example which uses the dataset found here: https://sites.google.com/site/dspham/downloads/network-traffic-datasets
An autoencoder is a type of neural network which takes in data, compresses it, decompresses it, then validates the output against the 
input. If a properly trained network produces a high amount of error given a certain input, then the input is likely to be anomalous 
somehow. The recurrent aspect allow the model to make use previous data to make decisions on how to build the output. The objective is to
train the model only on the normal traffic datasets then test it on a normal sample and a DoS sample. The error on the DoS sample should 
be significantly greater than that of the normal sample allowing for easy anomaly detection.
