import os
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
 
# Set the working directory for the project
os.chdir('C://Users/Dane/Documents/GitHub/seis735_project/')

# Training variants
#variants = pd.read_csv("data/raw/test_variants")
variants = pd.read_csv("data/raw/stage2_test_variants.csv")

# Load the data from file
text = pd.read_csv("data/raw/stage2_test_text.csv", 
                   sep="\|\|", 
                   header=None, 
                   skiprows=1, 
                   names=["ID","Text"],
                   engine="python"
                  )

print(variants.shape)
print(text.shape)

# Merge the Variants and Text
# Use inner join to merge the datasets on ID
merged = pd.merge(left=variants, right=text, how="inner", on="ID")

# Dropping the variants and text datasets as we won't need them anymore
del variants, text

print(merged.shape)
print(merged.dtypes)

# Get a list of stop words
stop_words = set(stopwords.words('english'))

# Create a PorterStemmer object
stemmer = PorterStemmer()

# Function that turns a doc into clean tokens
def clean_doc(doc, stemmer, stop_words):
    # Split into individual tokens by white space
    tokens = doc.split()
    # Remove punctuation and set to lowercase
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table).lower() for w in tokens]
    # Remove words that are not entirely alphabetical
    tokens = [w for w in tokens if w.isalpha()]
    # Removing all known stop words
    tokens = [w for w in tokens if not w in stop_words]
    # Remove tokens that aren't at least two characters in length
    tokens = [w for w in tokens if len(w) > 1]
    # Stem the remaining tokens
    tokens = [stemmer.stem(w) for w in tokens]
    return(tokens)

# Read in our previously defined vocabulary
file = open('data/interim/vocab.txt', 'r')

# Read the vocab from file
vocab = file.read()

# Close the file
file.close()

# Unique list of our vocab
vocab = set(vocab.split())
print(len(vocab))

# Next we need to filter each of our texts down to only the words in our defined vocabulary.
# A container object that will hold the words of each individual document
lines = list()

# Iterate over each of the texts in our training sample
for text in merged['Text']:
    # Create a list of tokens
    tokens = clean_doc(text, stemmer, stop_words)
    # Filter the words in the document by our defined vocabulary
    tokens = [w for w in tokens if w in vocab]
    # Concatentate each word in the document by a single space and append to our lines container
    lines.append(' '.join(tokens))

# Printing the size of our lines object. 
print(len(lines))

# Load our tokenizer object that was already trained 
with open('models/tokenizer.pickle', 'rb') as obj:
	tokenizer = pickle.load(obj)
	
# Function that merges our vector representation back to the original dataframe
def vector_to_df(df_orig, vector, tokenizer):
	# Convert the train vector to a dataframe 
	df_vector = pd.DataFrame(vector[:,1:], columns=[key for key in tokenizer.word_counts])
	# Merge the original train dataset to the vectorized dataset
	df_final = pd.concat([df_orig.reset_index(drop=True), df_vector], axis=1)
	# Drop the text blob from the original train dataset
	df_final.drop(['Text'], inplace=True, axis=1)
	return (df_final)

'''
# Now, let's encode our data using the **Frequency** method.
vector = tokenizer.texts_to_matrix(lines, mode='freq')

# Create and export the dataframe
df = vector_to_df(merged, vector, tokenizer)
print(df.shape)
df.to_csv('data/interim/val_freq.gz', index=False, compression='gzip')
'''

# Encode our data using the **binary** method.
vector = tokenizer.texts_to_matrix(lines, mode='binary')

# Create and export the dataframe
df = vector_to_df(merged, vector, tokenizer)
print(df.shape)
df.to_csv('data/interim/val_binary.gz', index=False, compression='gzip')

# Encode our data using the **count** method.
vector = tokenizer.texts_to_matrix(lines, mode='count')

# Create and export the dataframe
df = vector_to_df(merged, vector, tokenizer)
print(df.shape)
df.to_csv('data/interim/val_count.gz', index=False, compression='gzip')

# Encode our data using the **tfidf** method.
vector = tokenizer.texts_to_matrix(lines, mode='tfidf')

# Create and export the dataframe
df = vector_to_df(merged, vector, tokenizer)
print(df.shape)
df.to_csv('data/interim/val_tfidf.gz', index=False, compression='gzip')

