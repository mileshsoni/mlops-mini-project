from flask import Flask, render_template, request
from preprocessing_utility import normalize_text
import mlflow
import dagshub
import pickle

mlflow.set_tracking_uri('https://dagshub.com/mileshsoni/mlops-mini-project.mlflow')
dagshub.init(repo_owner='mileshsoni', repo_name='mlops-mini-project', mlflow=True)

app = Flask(__name__)

# load model
model_name = 'my_model'
alias = 'Staging'

# fetch the latest model version dynamically
def get_latest_model(model_name, alias):
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")

model = get_latest_model(model_name, alias)
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
@app.route('/')
def home ():
    return render_template('index.html', result=None)

@app.route('/predict', methods = ['POST'])
def predict():
    text = request.form['text']
    # clean input text
    text = normalize_text(text)
    # apply bow
    features = vectorizer.transform([text])
    # final prediction
    
    result= model.predict(features)
    print(result)
    return render_template('index.html', result=result[0])

app.run(debug=True)