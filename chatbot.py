import json
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import random


nltk.download('punkt_tab')  
nltk.download('stopwords')


stop_words = set(stopwords.words('turkish'))


with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)


patterns = []
labels = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^\w\s]', '', text) 
    words = word_tokenize(text) 
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)


cleaned_patterns = [clean_text(pattern) for pattern in patterns]


vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(cleaned_patterns)


model = LogisticRegression()
model.fit(X, labels)


def chatbot_response(user_input):
    
    cleaned_input = clean_text(user_input)
    
    input_vector = vectorizer.transform([cleaned_input])
 
    predicted_tag = model.predict(input_vector)[0]
    
   
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "Üzgünüm, seni anlayamadım. Başka ne sorabilirsin?"


def main():
    print("Chatbot'a hoş geldin! Çıkmak için 'çık' yaz.")
    while True:
        user_input = input("Sen: ")
        if user_input.lower() == 'çık':
            print("Görüşmek üzere!")
            break
        response = chatbot_response(user_input)
        print("Bot:", response)


if __name__ == "__main__":
    main()
