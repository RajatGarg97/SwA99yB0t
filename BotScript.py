
### $W@99YB0T ###

# importing libraries
import nltk
import numpy as np
import random
import string

f = open('corpusfile.txt', 'r', errors = 'ignore')

rawtxt = f.read()

rawtxt = rawtxt.lower() #converts to lower case

### RUN THESE TWO LINES ONLY ONCE ###
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

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)

GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    
    bot_response = ''

    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if(req_tfidf == 0):
        bot_response = bot_response + "Sorry! I didn't get it"

        return bot_response

    else:
        bot_response = bot_response + sent_tokens[idx]

        return bot_response

flag = True
print ("$W@99YB0T: My name is $W@99YB0T. I will answer your queries about Chatbots. If you want to exit, type Bye! ")

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response!='bye'):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            print("$W@99YB0T: You are welcome...")
    
        else:
            if(greeting(user_response) != None):
                print("$W@99YB0T: " + greeting(user_response))

            else:
                sent_tokens.append(user_response)

                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print("$W@99YB0T: ", end = "")
                print(response(user_response))
                sent_tokens.remove(user_response)
    
    else:
        flag = False
        print("$W@99YB0T: Bye! take care...")
        


