from pathlib import Path
import random
import re
import secrets
import os
from flask import Flask, render_template, flash, redirect, url_for, session, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
import numpy as np
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS
import msgConstant as msgCons  # Ensure this file is present

app = Flask(__name__)
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.secret_key = "m4xpl0it"

db = SQLAlchemy(app)
migrate = Migrate(app, db)

userSession = {}

def make_token():
    return secrets.token_urlsafe(16) 

class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)


# --- GLOBALS & DATA LOADING ---
df = None
symptoms_set = set()
diseases_set = set()


# Bulletproof path 
BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / "dataset.xlsx"

try:
    if os.path.exists(dataset_path):
        df = pd.read_excel(dataset_path)
        for s in df['Symptoms']:
            for symptom in s.split(','):
                symptoms_set.add(symptom.strip())
        diseases_set = set(df['Disease'])
        print(">>> SUCCESS: Dataset loaded perfectly!")
    else:
        print(f">>> ERROR: Path doesn't exist: {dataset_path}")
except Exception as e:
    print(f">>> ERROR: Excel read failure: {e}")


# --- TO PREVENT LOCHA IN 'def' FUNCTIONS ---

def predict_symptom(user_input, symptom_list):
    user_input_tokens = user_input.lower().replace("_", " ").split()
    similarity_scores = []
    for symptom in symptom_list:
        symptom_tokens = symptom.lower().replace("_", " ").split()
        combined_tokens = list(set(user_input_tokens + symptom_tokens))
        
        count_vector = np.zeros((2, len(combined_tokens)))
        for i, token in enumerate(combined_tokens):
            count_vector[0][i] = user_input_tokens.count(token)
            count_vector[1][i] = symptom_tokens.count(token)

        similarity = cosine_similarity(count_vector)[0][1]
        similarity_scores.append(similarity)

    max_score_index = np.argmax(similarity_scores)
    return symptom_list[max_score_index]


def predict_disease_from_symptom(symptom_list):
    global df, symptoms_set  # Ensuring global access inside def

    # FIRST: Try ML Model (Random Forest)
    try:
        if not symptoms_set:
            raise ValueError("Symptoms template is empty.")

        # Vector framework standard setup
        s_dict = {s: 0 for s in symptoms_set}
        for s in symptom_list:
            matched_index = predict_symptom(s, list(s_dict.keys()))
            s_dict[matched_index] = 1
        
        df_test = pd.DataFrame([list(s_dict.values())], columns=list(s_dict.keys()))
        
        # ML model parsing execution
        clf = load(BASE_DIR / "model" / "random_forest.joblib")
        result = clf.predict(df_test)
        predicted_disease = result[0]
        
        disease_details = getDiseaseInfo(predicted_disease)
        return f"<b>{predicted_disease}</b><br>{disease_details}", predicted_disease

    except Exception as e:
        print(f">>> ML Model failed, switching to Cosine Similarity fallback. Error: {e}")
        
        # SECOND: Fallback directly to dataset parsing if model fails
        if df is None:
            return "<b>Error: System database could not be loaded. Please contact administrator.</b>", ""
            
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['Symptoms'])
        user_X = vectorizer.transform([', '.join(symptom_list)])
        similarity_scores = cosine_similarity(X, user_X)

        max_score = similarity_scores.max()
        max_indices = similarity_scores.argmax(axis=0)
        
        matched_diseases = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                matched_diseases.add(df.iloc[i]['Disease'])

        if not matched_diseases:
            return "<b>No matching diseases found</b>", ""
        
        first_disease = list(matched_diseases)[0]
        disease_details = getDiseaseInfo(first_disease)
        return f"<b>{first_disease}</b><br>{disease_details}", first_disease


def get_symtoms(user_disease):
    if df is None:
        return False, "Database not available"
        
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([user_disease])
    similarity_scores = cosine_similarity(X, user_X)

    max_score = similarity_scores.max()
    if max_score < 0.7:
        return False, "No matching diseases found"
    
    max_indices = similarity_scores.argmax(axis=0)
    symptoms = set()
    for i in max_indices:
        if similarity_scores[i] == max_score:
            symptoms.update(set(df.iloc[i]['Symptoms'].split(',')))
    return True, symptoms


def getDiseaseInfo(keywords):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords + " disease symptoms treatment", region='wt-wt', max_results=1))
            if results:
                return results[0]['body']
            return ""
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return ""


# --- CONTROLLERS & ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/user")
def index_auth():
    my_id = make_token()
    userSession[my_id] = {
        'state': -1,
        'name': '',
        'age': 0,
        'symptoms': []
    }
    return render_template("index_auth.html", sessionId=my_id)

@app.route("/instruct")
def instruct():
    return render_template("instructions.html")

@app.route("/upload")
def bmi():
    return render_template("bmi.html")

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")

@app.route('/pred_page')
def pred_page():
    pred = session.get('pred_label', None)
    f_name = session.get('filename', None)
    return render_template('pred.html', pred=pred, f_name=f_name)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        login_user = user.query.filter_by(username=uname, password=passw).first()
        if login_user is not None:
            return redirect(url_for("index_auth"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']

        new_user = user(username=uname, email=mail, password=passw)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route('/ask', methods=['GET', 'POST'])
def chat_msg():
    user_message = request.args.get("message", "").lower()
    sessionId = request.args.get("sessionId")

    if not sessionId or sessionId not in userSession:
        sessionId = make_token() if not sessionId else sessionId
        userSession[sessionId] = {'state': -1, 'name': '', 'age': 0, 'symptoms': []}

    rand_num = random.randint(0, 4)
    response = []

    if request.args.get("message") == "undefined":
        userSession[sessionId] = {'state': -1, 'name': '', 'age': 0, 'symptoms': []}
        response.append(msgCons.WELCOME_GREET[rand_num])
        response.append("What is your good name?")
        return jsonify({'status': 'OK', 'answer': response})

    current_userdata = userSession[sessionId]
    currentState = current_userdata['state']

    if currentState == -1:
        response.append(f"Hi {user_message}, To predict your disease based on symptoms, we need some information. Please provide it accordingly.")
        current_userdata['name'] = user_message
        current_userdata['state'] = 0

    elif currentState == 0:
        username = current_userdata['name']
        response.append(f"{username}, what is your age?")
        current_userdata['state'] = 1

    elif currentState == 1:
        result = re.findall(r'\d+', user_message)
        if len(result) == 0 or float(result[0]) <= 0 or float(result[0]) >= 130:
            response.append("Invalid input, please provide a valid age.")
        else:
            current_userdata['age'] = float(result[0])
            username = current_userdata['name']
            response.append(f"{username}, Choose an Option:")
            response.append("1. Predict Disease")
            response.append("2. Check Disease Symptoms")
            current_userdata['state'] = 2

    elif currentState == 2:
        if '2' in user_message or 'check' in user_message:
            username = current_userdata['name']
            response.append(f"{username}, What is the Disease Name?")
            current_userdata['state'] = 20
        else:
            username = current_userdata['name']
            response.append(f"{username}, What symptoms are you experiencing?")
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
            current_userdata['state'] = 3

    elif currentState == 3:
        current_userdata['symptoms'].extend([s.strip() for s in user_message.split(",")])
        username = current_userdata['name']
        response.append(f"{username}, describing any more symptoms? If done, select Option 1.")
        response.append("1. Check Disease")
        response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
        current_userdata['state'] = 4

    elif currentState in [4, 5, 6, 7, 8]:
        if '1' in user_message or 'disease' in user_message:
            disease_html, disease_name = predict_disease_from_symptom(current_userdata['symptoms'])
            response.append("<b>The following disease may be causing your discomfort:</b>")
            response.append(disease_html)
            if disease_name:
                response.append(f'<a href="https://www.google.com/search?q={disease_name}+disease+hospital+near+me" target="_blank">Search Nearby Hospitals</a>')
            current_userdata['state'] = 10
        else:
            current_userdata['symptoms'].extend([s.strip() for s in user_message.split(",")])
            username = current_userdata['name']
            response.append(f"{username}, any other symptoms you're currently dealing with?")
            response.append("1. Check Disease")
            response.append('<a href="/diseases" target="_blank">Symptoms List</a>')
            current_userdata['state'] += 1

    elif currentState == 10:
        response.append('<a href="/user" target="_blank">Predict Again</a>')

    elif currentState == 20:
        success, data = get_symtoms(user_message)
        if success:
            response.append(f"The symptoms of {user_message} are:")
            for sym in data:
                response.append(sym.strip().capitalize())
        else:
            response.append(data)

        current_userdata['state'] = 2
        response.append("<br>Choose an Option:")
        response.append("1. Predict Disease")
        response.append("2. Check Disease Symptoms")

    return jsonify({'status': 'OK', 'answer': response})



if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=3000)


