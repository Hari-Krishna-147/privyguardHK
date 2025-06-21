import os
from flask import Flask, render_template, request, redirect, session, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle, os
from cryptography.fernet import Fernet

print("privyguard.py is starting...")

# Initialize Flask
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Load ML model
svc_model = pickle.load(open("models/svc_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# Encryption
key = Fernet.generate_key()
cipher = Fernet(key)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()
        if not user:
            error = "Email is incorrect"
        elif user.password != password:
            error = "Password is incorrect"
        else:
            session['user_id'] = user.id
            return redirect('/')
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already exists.")
        new_user = User(email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/')
def home():
    if 'user_id' in session:
        return render_template("dashboard.html")
    return redirect('/login')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect('/login')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    vect = vectorizer.transform([data['message']])
    prediction = svc_model.predict(vect)[0]
    return jsonify({'result': 'Spam' if prediction else 'Not Spam'})

@app.route('/encrypt', methods=['POST'])
def encrypt():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    msg = request.json['message']
    encrypted = cipher.encrypt(msg.encode()).decode()
    return jsonify({'encrypted': encrypted})

@app.route('/decrypt', methods=['POST'])
def decrypt():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    encrypted = request.json['encrypted']
    try:
        decrypted = cipher.decrypt(encrypted.encode()).decode()
        return jsonify({'decrypted': decrypted})
    except:
        return jsonify({'decrypted': 'Invalid or corrupted encryption'}), 400

# Start app
if __name__ == '__main__':
    print("inside __main__")
    with app.app_context():
        if not os.path.exists("users.db"):
            db.create_all()
            print("users.db created")
        else:
            print("users.db already exists")
    print("🚀 Launching Flask server...")
    app.run(host='0.0.0.0',port=int(os.environ.get("PORT",10000)))