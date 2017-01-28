# This file applies pre-precessing steps to provided sentences
# It also handles the allowed words set and generating feature vectors


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import dataset_handler

nltk_stopwords = stopwords.words('english')
custom_stopwords = ['could', 'would', 'please', 'mate', 'us', 'something' ,'may'];
punctuation = set(string.punctuation)
stemmer = PorterStemmer()


# Remove stop words and apply stemming to the provided sentence
def preprocess(sample):
    # Strip leading and trailing spaces and
    # convert to lower case letters
    try:
        tmp = sample.strip().lower()
    except:
        return -1
    # Remove punctuation
    tmp = ''.join(ch for ch in tmp if ch not in punctuation)

    # Split to words and remove nltk stop words
    tmp = [w for w in tmp.split() if w not in nltk_stopwords]

    # Remove custom stop words
    tmp = [w for w in tmp if w not in custom_stopwords]

    # Apply porter stemmer

    return [stemmer.stem(p) for p in tmp]


# Extract unique words from the dataset
def extract_unique_words(data):
    # Extract unique words
    unique_words = []
    for sample in data:
        for w in sample:
            if w not in unique_words:
                unique_words.append(w)
    return unique_words


# generate the feature vector
def generate_feature_vector(allowed_words, sample):
    vector = [0 for i in range(len(allowed_words))]
    for i in range(len(sample)):
        vector[allowed_words.index(sample[i])] += 1
    return vector[:]