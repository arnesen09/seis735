# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:19:59 2017

@author: Dane
"""

import pandas as pd
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

# Set the working directory for the project
os.chdir('C://Users/Dane/Documents/GitHub/seis735_project/')

####################################################################
## Step 1: Prepare the training data and test data
####################################################################

# Read the training data
df_train = pd.read_csv('data/interim/train_freq.gz', compression='gzip', encoding='ISO-8859-1')

# Keep only the predictors
df_train.drop(['ID','Gene','Variation'], inplace=True, axis=1)

# Convert predictors to matrix format
x_train = df_train.as_matrix()[:,1:]
y_train = df_train.as_matrix()[:,0]

# Cleanup
del df_train

# Read the test data
df_test = pd.read_csv('data/interim/test_freq.gz', compression='gzip', encoding='ISO-8859-1')

# Keep only the predictors
df_test.drop(['ID','Gene','Variation'], inplace=True, axis=1)

# Convert predictors to matrix format
x_test = df_test.as_matrix()[:,1:]
y_test = df_test.as_matrix()[:,0]

# Cleanup
del df_test

########################################################################
## Step 2: Dimensionality reduction
########################################################################

# Get the compiled autoencoder
encoder = load_model('models/encoder_v1.h5')

# Transform the data (reduce the dimensions)
x_train_red = encoder.predict(x_train)
x_test_red = encoder.predict(x_test)

del x_train, x_test

####################################################################
## Step 3: Create and train the model
####################################################################

from xgboost import XGBClassifier

# Initialize the model parameters
xgb = XGBClassifier(learning_rate=0.1,
                    max_depth=3,
                    n_estimators=2000,
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    num_class=9,
                    n_jobs=-1
                    )

# Train the model
xgb.fit(x_train_red,
        y_train,
        verbose=True,
        early_stopping_rounds=50,
        eval_set=[(x_train_red, y_train), (x_test_red, y_test)]
        )

# Make the predictions
preds = xgb.predict(x_test_red)
probs = xgb.predict_proba(x_test_red)

####################################################################
## Step 3: Evaluate the model
####################################################################

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

# Create the confusion matrix
cm = confusion_matrix(y_test, preds, labels=[1,2,3,4,5,6,7,8,9])

# Calculate accuracy
acc = np.sum(np.diag(cm)) / np.sum(cm)
print('Accuracy: %0.5f' % acc)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cm, classes=[1,2,3,4,5,6,7,8,9],
                      title='Confusion matrix, without normalization')
plt.show()

####################################################################
## Step 4: Save the model
####################################################################

import pickle
 
# Save the xgb model
with open('models/xgboost_v2.pickle', 'wb') as handle:
    pickle.dump(xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)

