from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD



import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('./backend/data/Intent.json').read())

words = []
classes = []
documents =[]
ignore_letters = ['?','!','.',',']

nltk.download('punkt')
for intent in intents['intents']:
  for question in intent['text']:
     word_list = nltk.word_tokenize(question)
     words.extend(word_list)
     documents.append((word_list,intent['intent']))
     if intent['intent'] not in classes:
       classes.append(intent['intent'])

# documents

nltk.download('wordnet')
nltk.download('omw-1.4')
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

# words

words = sorted(set(words))

# words

# classes

classes = sorted(set(classes))

pickle.dump(words,open('./backend/words.pkl','wb'))
pickle.dump(classes,open('./backend/classes.pkl','wb'))

training = []
output_empty = [0]*len(classes)

for document in documents:
  bag = []
  word_patterns = document[0]
  word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
  for word in words:
    bag.append(1) if word in word_patterns else bag.append(0)
  output_row = list(output_empty)
  output_row[classes.index(document[1])] = 1
  training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

sgd = SGD(lr = 0.01,momentum=0.9,nesterov=True)
model.compile(loss ='categorical_crossentropy',optimizer = sgd,metrics = ['accuracy'])
hist = model.fit(np.array(train_x),np.array(train_y),epochs = 200, batch_size = 5,verbose = 1)
model.save('./backend/chatbot_model')
print('done')