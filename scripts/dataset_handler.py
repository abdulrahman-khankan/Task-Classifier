# This file handles disk operations related to saving and loading the dataset

import csv

# Loads the dataset from the specified csv file
def read_data(csv_filename):
    tag = []
    dataset = []
    with open(csv_filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            tag.append(row[0])
            dataset.append([int(v) for v in row[1:len(row)]])

    return tag, dataset

# Saves the dataset to the specified csv file
def save_data(tag, data, filename):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(tag)):
            writer.writerow([tag[i]] + data[i])

# Writes an array to a csv file
def save_array(array, filename):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(array)

# Reads an array from a csv file
def load_array(filename):
    array = []
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        array = list(reader)[0]
    return  array