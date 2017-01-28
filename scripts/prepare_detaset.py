#!/usr/bin/env python

# This file takes as input the raw dataset and converts it to be suitable for
# classifer training and testing


import dataset_handler
import preprocessing
import csv
from sklearn import cross_validation


# Open the raw data
with open('../data/raw_data.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

tag = []
data = []

for cls in your_list:
    for i in range(2, len(cls)):

        # Skip if empty string
        if cls[i] == '':
            continue

        # remove stopwords and punctuation and apply stemming to the sentence
        tmp = preprocessing.preprocess(cls[i])

        # Skip if empty result
        if tmp == []:
            continue

        # Append the tag name
        tag.append(cls[0])

        # Append the result of pre-processing
        data.append(tmp)

# Extract the unique words from the dataset
unique_words = preprocessing.extract_unique_words(data)


dataset = [[0 for j in range(len(unique_words))] for i in range(len(tag))]

# convert the dataset to feature vectors
for i in range(len(tag)):
    for j in range(len(data[i])):
        dataset[i][unique_words.index(data[i][j])] += 1

# Split the dataset to training set and test set
dataset_train, dataset_test, tag_train, tag_test = cross_validation.train_test_split(dataset, tag, test_size=.2)

# Save the data
dataset_handler.save_array(unique_words, '../data/unique_words.csv')
dataset_handler.save_data(tag, dataset, '../data/dataset.csv')
dataset_handler.save_data(tag_train, dataset_train, '../data/dataset_train.csv')
dataset_handler.save_data(tag_test, dataset_test, '../data/dataset_test.csv')
