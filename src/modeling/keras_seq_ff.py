# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:19:59 2017

@author: Dane
"""

import pandas as pd
import os

# Set the working directory for the project
os.chdir('C://Users/Dane/Documents/GitHub/seis735_project/')

####################################################################
## Step 1: Prepare the training data and test data
####################################################################

# Read the training data
df_train = pd.read_csv('data/interim/train_freq.gz', compression='gzip', encoding='ISO-8859-1')

# Get the Y values
df_y = df_train[['Class']]

# True y
y_true_train = df_y['Class'].as_matrix()

# Turn y values into dummies
df_y = pd.get_dummies(df_y, prefix='y', columns=['Class'])

# Keep only the predictors
df_train.drop(['ID','Gene','Variation','Class'], inplace=True, axis=1)

# Convert predictors to matrix format
x_train = df_train.as_matrix()
y_train = df_y.as_matrix()

# Cleanup
del df_train, df_y

# Read the test data
df_test = pd.read_csv('data/interim/test_freq.gz', compression='gzip', encoding='ISO-8859-1')

df_test.groupby('Class')['Class'].count()

# Get the Y values
df_y = df_test[['Class']]

# True y
y_true_test = df_y['Class'].as_matrix()

# Turn y values into dummies
df_y = pd.get_dummies(df_y, prefix='y', columns=['Class'])

# Keep only the predictors
df_test.drop(['ID','Gene','Variation','Class'], inplace=True, axis=1)

# Convert predictors to matrix format
x_test = df_test.as_matrix()
y_test = df_y.as_matrix()

# Cleanup
del df_y, df_test

####################################################################
## Step 2: Create and train the model
####################################################################

# Get model input and output shape
n_words = x_train.shape[1]
n_out = y_train.shape[1]

# Import the keras libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Create the architecture
model = Sequential()
model.add(Dense(50, input_dim=n_words, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_out, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=50, epochs=20)

# Make the predictions
preds_test = model.predict_classes(x_test, 1)
preds_train = model.predict_classes(x_train, 1)

preds_test = [i + 1 for i in preds_test]
preds_train = [i + 1 for i in preds_train]

####################################################################
## Step 3: Evaluate the model
####################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true_test, preds_test, labels=[1,2,3,4,5,6,7,8,9])
#cm = confusion_matrix(y_true_train, preds_train, labels=[1,2,3,4,5,6,7,8,9])

acc = np.sum(np.diag(cm)) / np.sum(cm)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=[1,2,3,4,5,6,7,8,9],
                      title='Confusion matrix, without normalization')
plt.show()

