import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os

# Create models directory if not exists
if not os.path.exists("models"):
    os.makedirs("models")

# Load SMS spam dataset from URL
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=["label", "message"])

# Encode 'spam' as 1, 'ham' as 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X = df['message']
y = df['label']

# Vectorize text data
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train SVC model
model = SVC()
model.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(model, open("models/svc_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/tfidf_vectorizer.pkl", "wb"))

print("✅ Model and vectorizer saved in 'models/' folder.")