from flask import Flask, render_template, request
from preprocessing_utility import normalize_text
import mlflow
import dagshub
import pickle
import os
import pandas as pd

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "mileshsoni"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

app = Flask(__name__)

# load model
model_name = 'my_model'
alias = 'Staging'

# fetch the latest model version dynamically
def get_latest_model(model_name, alias):
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")

model = get_latest_model(model_name, alias)
vectorizer = pickle.load(open('./models/vectorizer.pkl', 'rb'))

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
    
    # Convert sparse matrix to DataFrame
    features_df = pd.DataFrame.sparse.from_spmatrix(features)
    features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])
    
    # final prediction
    result= model.predict(features)
    
    return render_template('index.html', result=result[0])

if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0')
    
    
