#!/usr/bin/env python
# This file takes an input void command from the user and
# tries to classify the given command
from preprocessing import preprocess, generate_feature_vector
from TaskClassifiers import TextClassifiers
from dataset_handler import load_array
import speech_recognition as sr


# Load the classifiers and the set of allowed words
tc = TextClassifiers.load('../data/TextClassifiers.tc')
allowed_words = load_array('../data/unique_words.csv')

# Instantiate the microphone and speech recognizer
r = sr.Recognizer()
m = sr.Microphone()
user_input = ""

# Adjust the threshold based on the background noise
print("A moment of silence, please...")
with m as source: r.adjust_for_ambient_noise(source)
print("Set minimum energy threshold to {}".format(r.energy_threshold))

print "Say 'exit' to quit"

while 1:
    # Receive command from the user
    with sr.Microphone() as source:
        print "How can I help you?\t", 
        audio = r.listen(source)
        print("Hold on")
    try:
        # Convert to text using Google Speech API
        user_input = r.recognize_google(audio)
    # Handle exceptions
    except sr.UnknownValueError:
        print "Google Speech Recognition could not understand audio"
    except sr.RequestError as e:
        print "Could not request results from Google Speech Recognition service; {0}".format(e)

    print user_input
    
    #user_input = raw_input('enter request >> ')

    # Stop the execution loop if the given command is 'exit'
    if user_input == 'exit':
        break

    # Apply pre-processing to the use input to remove stopwords and punctuation and apply stemming
    user_input = preprocess(user_input)


    if user_input == -1:
        print "invalid input"
        continue

    # Remove disallowed words
    user_input = [w for w in user_input if w in allowed_words]

    print user_input
    if user_input == []:
        print "Could you clarify, please? Say 'exit' if you want to quit"
        continue

    # Extract features and generate feature vector
    user_input = generate_feature_vector(allowed_words, user_input)

    # Print the class predicted by the selected classifier
    print tc.classify('knn', [user_input])