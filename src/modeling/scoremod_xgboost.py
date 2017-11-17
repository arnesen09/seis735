import os
import pandas as pd
import numpy as np
from scipy import sparse
import pickle
 
# Set the working directory for the project
os.chdir('C://Users/Dane/Documents/GitHub/seis735_project/')

# Load our svd object that was already trained 
with open('models/svd_500_v1.pickle', 'rb') as obj:
	svd = pickle.load(obj)
	
# Load our xgb object that was already trained 
with open('models/xgboost_v1.pickle', 'rb') as obj:
	xgb = pickle.load(obj)
	
# Read the dataset
df = pd.read_csv('data/interim/val_freq.gz', compression='gzip', encoding='ISO-8859-1')
print(df.shape)

# Keep only the predictors
df.drop(['Gene','Variation'], inplace=True, axis=1)

# Convert predictors to matrix format
x = sparse.csr_matrix(df.as_matrix()[:,1:])
ids = df.as_matrix()[:,0]

# Reduce the dimensions
x_red = svd.transform(x)

# Make the predictions
probs = xgb.predict_proba(x_red)

# Concat the ids and probabilities
preds = np.c_[ids, probs]

# Create a dataframe of our predictions
df = pd.DataFrame(preds, columns=['ID','class1','class2','class3','class4','class5','class6','class7','class8','class9'])
df['ID'] = df['ID'].apply(lambda x: int(x))

# Export the csv
df.to_csv('data/final/xgb_preds_20171109.csv', index=False)