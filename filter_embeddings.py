import numpy as np
import torch
import json
import torch.nn as nn
from dataloader import SentimentDataLoader

# load data
# get tokenized clean train data
train_clean_loader = SentimentDataLoader('train_clean.tsv')
train_clean_loader.load_data()  
_, train_clean_documents, _, _, _, _= train_clean_loader.get_data()

# get tokenized train data
train_loader = SentimentDataLoader('train.tsv')
train_loader.load_data()  
_, train_documents, _, _, _, _= train_loader.get_data()

# get tokenized clean dev data
dev_clean_loader = SentimentDataLoader('dev_clean.tsv')
dev_clean_loader.load_data()  
_, dev_clean_documents, _, _, _, _= dev_clean_loader.get_data()

# get tokenized dev data
dev_loader = SentimentDataLoader('dev_clean.tsv')
dev_loader.load_data()  
_, dev_documents, _, _, _, _= dev_loader.get_data()

# get tokenized clean test data
test_clean_loader = SentimentDataLoader('test_clean.tsv')
test_clean_loader.load_data()  
_, test_clean_documents, _, _, _, _= test_clean_loader.get_data()

# get tokenized test data
test_loader = SentimentDataLoader('test.tsv')
test_loader.load_data()  
_, test_documents, _, _, _, _= test_loader.get_data()

# append all the data together
all_documents = train_documents + train_clean_documents + dev_clean_documents + dev_documents + test_clean_documents + test_documents

# get vocab of all the datasets
vocab = set(word for sublist in all_documents for word in sublist)
# print(vocab)


#glove_file = 'glove.twitter.27B.100d.txt'
fasttext_file = 'wiki-news-300d-1M-subword.vec'  

# initialize an empty dictionary to store glove or fasttext embeddings
fasttext_embeddings = {}

# open the glove file and populate the dictionary

with open(fasttext_file, 'r', encoding='utf-8') as file:
    for line in file:
        if line.startswith(" "):  # skip the first line containing metadata
            continue
        values = line.split()
        word = values[0]
        embedding = np.array([float(val) for val in values[1:]])
        fasttext_embeddings[word] = embedding

print("Number of words in GloVe embeddings:", len(fasttext_embeddings))

# initialize an empty dictionary to store filtered embeddings
filtered_embeddings = {}

# iterate through the pre-trained embeddings
for word, embedding in fasttext_embeddings.items():
    if word in vocab:
        # if the word is in dataset vocab, keep the embedding
        filtered_embeddings[word] = embedding

# convert the filtered embeddings to a NumPy array
filtered_embeddings_matrix = np.array(list(filtered_embeddings.values()))

# convert the NumPy array to a PyTorch tensor
filtered_embeddings_tensor = torch.FloatTensor(filtered_embeddings_matrix)

# convert the PyTorch tensor elements to Python lists
filtered_embeddings_py_lists = {word: embedding.tolist() for word, embedding in filtered_embeddings.items()}

print("Shape of filtered embeddings tensor:", filtered_embeddings_tensor.shape)

# save the filtered embeddings as a json file
with open('fasttext_embeddings.json', 'w') as json_file:
    json.dump(filtered_embeddings_py_lists, json_file)

print("Fasttext embeddings saved to 'fasttext_embeddings.json'.")

