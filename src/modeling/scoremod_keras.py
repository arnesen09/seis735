import os
import pandas as pd
import numpy as np
from keras.models import load_model
 
# Change the working directory
os.chdir('/home/ec2-user/Notebooks/seis735_aws/')

# Load the model
model = load_model('models/iteration1.h5')
		
# Read the dataset
df = pd.read_csv('data/val_freq.gz', compression='gzip', encoding='ISO-8859-1')
print(df.shape)

# Keep only the predictors
df.drop(['Gene','Variation'], inplace=True, axis=1)

# Convert predictors to matrix format
x = df.as_matrix()[:,1:]
ids = df.as_matrix()[:,0]

# Make the predictions
probs = model.predict(x)

# Concat the ids and probabilities
preds = np.c_[ids, probs]

# Create a dataframe of our predictions
df = pd.DataFrame(preds, columns=['ID','class1','class2','class3','class4','class5','class6','class7','class8','class9'])
df['ID'] = df['ID'].apply(lambda x: int(x))

# Export the csv
df.to_csv('data/keras_preds_20171116.csv', index=False)