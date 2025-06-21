import re
import string
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, render_template
from cryptography.fernet import Fernet
import os

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\d{10}\b', ' <phone> ', text)
    text = re.sub(r'\S+@\S+', ' <email> ', text)
    text = re.sub(r'http\S+', ' <url> ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Training the model (run once)
def train_model():
    sms_df = pd.read_csv(
        'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
        sep='\t', header=None, names=['label', 'message']
    )
    sms_df['message'] = sms_df['message'].apply(preprocess_text)
    sms_df['label'] = sms_df['label'].map({'ham': 0, 'spam': 1})

    X_train, X_test, y_train, y_test = train_test_split(
        sms_df['message'], sms_df['label'], test_size=0.2, random_state=42
    )

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(tfidf, 'vectorizer.pkl')

# Encryption
def generate_key():
    if not os.path.exists("secret.key"):
        key = Fernet.generate_key()
        with open("secret.key", "wb") as key_file:
            key_file.write(key)

def load_key():
    return open("secret.key", "rb").read()

def encrypt_text(text):
    key = load_key()
    fernet = Fernet(key)
    return fernet.encrypt(text.encode()).decode()

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    clean_text = preprocess_text(text)
    tfidf = joblib.load('vectorizer.pkl')
    model = joblib.load('spam_model.pkl')
    vector = tfidf.transform([clean_text])
    prediction = model.predict(vector)[0]
    encrypted = encrypt_text(text)
    return render_template('result.html', prediction='Spam' if prediction else 'Not Spam', encrypted=encrypted)

if __name__ == '__main__':
    if not os.path.exists("spam_model.pkl") or not os.path.exists("vectorizer.pkl"):
        train_model()
    generate_key()
    app.run(debug=True)