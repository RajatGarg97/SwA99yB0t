
### $W@99YB0T ###

# importing libraries
import nltk
import numpy as np
import random
import string

f = open('corpusfile.txt', 'r', errors = 'ignore')

rawtxt = f.read()

rawtxt = rawtxt.lower() #converts to lower case

# nltk.download('punkt')
# nltk.download('wordnet')

# converts to a list of sentences   
sent_tokens = nltk.sent_tokenize(rawtxt)

# converts to a list of words
word_tokens = nltk.word_tokenize(rawtxt)

lemmer = nltk.stem.WordNetLemmatizer()
# WordNet is a dictionary of English included in NLTK.

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

