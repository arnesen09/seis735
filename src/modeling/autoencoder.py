# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 19:21:44 2017

@author: Dane
"""

import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Dropout
import os

# Set the working directory for the project
os.chdir('C://Users/Dane/Documents/GitHub/seis735_project/')

####################################################################
## Step 1: Prep the dataset
####################################################################

# Read the training data
df = pd.read_csv('data/interim/train_freq.gz', compression='gzip', encoding='ISO-8859-1')

# Keep only the predictors
df.drop(['ID','Gene','Variation','Class'], inplace=True, axis=1)

# Convert predictors to matrix format
x = df.as_matrix()

# Cleanup
del df

####################################################################
## Step 2: Build the autoencoder architecture
####################################################################

# This is the shape of our input
inputs = Input(shape=(23899,))

# These are the model layers
encoded = Dropout(rate=0.25)(inputs)
encoded = Dense(2500, activation='relu')(encoded)
encoded = Dropout(rate=0.1)(encoded)
encoded = Dense(2000, activation='relu')(encoded)
encoded = Dropout(rate=0.1)(encoded)
encoded = Dense(1500, activation='relu')(encoded)
encoded = Dropout(rate=0.1)(encoded)
encoded = Dense(1000, activation='relu')(encoded)
decoded = Dense(1500, activation='sigmoid')(encoded)
decoded = Dense(2000, activation='sigmoid')(decoded)
decoded = Dense(2500, activation='sigmoid')(decoded)
decoded = Dense(23899, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(inputs, decoded)

# this model maps an input to its encoded representation
encoder = Model(inputs, encoded)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
autoencoder.fit(x, x, epochs=20, batch_size=50)

# Predict the x-hat values
x_hat = encoder.predict(x)

####################################################################
## Step 3: Save the model
####################################################################

from keras.models import load_model

encoder.save('models/encoder_v1.h5')

# returns a compiled model
model = load_model('models/encoder_v1.h5')