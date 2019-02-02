# Jamison Campbell CPS 596 Final Project Script

import numpy as np
import sys
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import scipy as sc
 
# Declarations
batch_size = 32
epochs = 100

# Load data MSD subset 10000 samples
train_data = np.loadtxt('song_features_and_tags.csv', delimiter=",", comments="//", skiprows=1, dtype=float, usecols=(4,5,6,7,9,10,11,12,13,14,16,17,18,19))
song_names = np.loadtxt('song_features_and_tags.csv', delimiter=",", comments="//", skiprows=1, dtype=str, usecols=(0,1))

# Split data into train and validation 80-20
N = .8
#train_data = raw_data[:int(N * raw_data.shape[0]),0:]
#val_data = raw_data[int(N * raw_data.shape[0]):,0:]

# Normalize features in training data
mean_train = train_data.mean(axis = 0)
train_data -= mean_train
std_train = train_data.std(axis = 0)
train_data /= std_train

# Testing shape of train data
print(train_data.shape)
feature_length = train_data.shape[1]

# Specify model here, four hidden layers 128-64-2-64-128
input_layer = Input(shape = (feature_length, ))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(3, name = 'encoded_layer', 
		   activity_regularizer = regularizers.l2(0.01))(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(feature_length)(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer = 'adadelta', loss = 'mse')

autoencoder.fit(train_data, train_data,
                epochs = epochs,
                batch_size = batch_size,
		validation_split = .2) 

# Create model from encoded layer
encoder = Model(inputs = [input_layer], outputs = [autoencoder.get_layer("encoded_layer").output])

# Plot predictions
encodings = encoder.predict(train_data) # only using val data results in first 2000 songs only
plt.scatter(encodings[:,0], encodings[:,1])
plt.savefig('encodings.png')
#print(encodings)
#print(encodings.ndim)

song_choice = 1074
song_choice = song_choice - 2 #indexes are off by two from excel row number

print("If you like " + song_names[song_choice, 1] + " by " + song_names[song_choice, 0])

def generate_songsNew(idx):
	for i in range(len(encodings)-1):
		distance = sc.spatial.distance.euclidean(encodings[idx], encodings[i])
		if(distance < .01 and distance > 0):
			print("You would like " + song_names[i, 1] + " by " + song_names[i, 0])
		
generate_songsNew(song_choice)
			
