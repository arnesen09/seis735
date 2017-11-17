# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:35:46 2017

@author: Dane
"""

import os
import pandas as pd
import numpy as np

# Set the working directory for the project
os.chdir('C://Users/Dane/Documents/GitHub/seis735_project/')

### Step 1: Read the files
text = pd.read_csv("data/raw/test_text", sep="\|\|", header=None, skiprows=1, names=["ID","Text"], engine="python")
variants = pd.read_csv("data/raw/test_variants")

### Step 2: Basic file statistics

# Counting number of rows in each class
variants.groupby('Class')['Class'].count()

# Word count
text['Length'] = text['Text'].apply(lambda x: len(x.split(' ')))

print('The maximum number of words: %d' % max(text['Length']))
print('The minimum number of words: %d' % min(text['Length']))
print(text['Length'].describe())

text[text['Length'] == 1]


