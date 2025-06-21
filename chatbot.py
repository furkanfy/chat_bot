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

# NLTK veri setlerini indir
nltk.download('punkt_tab')  # Tokenizasyon için
nltk.download('stopwords')

# Türkçe stop words'leri yükle
stop_words = set(stopwords.words('turkish'))

# intents.json dosyasını oku
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Veriyi hazırla: patterns ve labels
patterns = []
labels = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        labels.append(intent['tag'])

# Metin temizleme fonksiyonu
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Rakamları kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    words = word_tokenize(text)  # Kelimelere ayır
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Patterns'ı temizle
cleaned_patterns = [clean_text(pattern) for pattern in patterns]

# TF-IDF vektörleştirici ile metinleri vektörleştir
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(cleaned_patterns)

# Lojistik regresyon modelini eğit
model = LogisticRegression()
model.fit(X, labels)

# Chatbot fonksiyonu
def chatbot_response(user_input):
    # Kullanıcı girdisini temizle
    cleaned_input = clean_text(user_input)
    # Vektörleştir
    input_vector = vectorizer.transform([cleaned_input])
    # Tahmin yap
    predicted_tag = model.predict(input_vector)[0]
    
    # Tahmin edilen tag'e göre rastgele bir yanıt seç
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "Üzgünüm, seni anlayamadım. Başka ne sorabilirsin?"

# Etkileşimli sohbet döngüsü
def main():
    print("Chatbot'a hoş geldin! Çıkmak için 'çık' yaz.")
    while True:
        user_input = input("Sen: ")
        if user_input.lower() == 'çık':
            print("Görüşmek üzere!")
            break
        response = chatbot_response(user_input)
        print("Bot:", response)

# Chatbot'u başlat
if __name__ == "__main__":
    main()