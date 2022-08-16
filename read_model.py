import random
import json
import pickle
import numpy as np

import nltk

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# intents = json.loads(open('./data/Intent.json').read())

words = pickle.load(open('./backend/words.pkl','rb'))
classes = pickle.load(open('./backend/classes.pkl','rb'))

model = load_model('./backend/chatbot_model')



def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def get_numerical_representation_of_words(sentence):
    sentence_words = clean_sentence(sentence)
    result = [0] * len(words)
    for word in sentence_words:
        for index,w in enumerate(words):
            if word == w:
                result[index] = 1
    return np.array(result)


def predict(sentence):
    rep = get_numerical_representation_of_words(sentence)
    prediction = model.predict(np.array([rep]))[0]
    THRESHOLD  = 0.25
    results = [[i,r] for i,r in enumerate(prediction) if r > THRESHOLD]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({ 'intent': classes[r[0]] , 'probability' : str(r[1]) })
    return return_list 


def get_response(predicted_class):
    with open('./backend/data/Intent.json','rb') as file:
        intents = json.loads(file.read())['intents']
        response = None
        for intent in intents:
            if intent['intent'] ==  predicted_class:
                response = random.choice(intent['responses'])
                break 
        return response


while True:
    sentence = input('Enter your sentence: ')
    print(get_response(predict(sentence)[0]['intent']))
    