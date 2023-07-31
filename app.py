from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Define the paths to the pickled models
MODEL_PATHS = {
    13: '../models/heart.pkl',
    18: '../models/kidney.pkl',
    10: '../models/liver.pkl'
}

def load_model(num_features):
    # Load the appropriate model based on the number of features
    model_path = MODEL_PATHS.get(num_features)
    if model_path is None:
        raise ValueError("No model found for the specified number of features.")
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def predict(values, num_features):
    model = load_model(num_features)
    values = np.asarray(values)
    return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        num_features = len(to_predict_list)
        pred = predict(to_predict_list, num_features)
    except:
        message = "Please enter valid Data"
        return render_template("home.html", message=message)

    # Redirect the user to the result page with the prediction result
    return redirect(url_for('resultPage', pred=pred))

@app.route("/result")
def resultPage():
    # Get the prediction result from the URL parameter
    pred = request.args.get('pred')

    # Render the result page with the prediction result
    return render_template('predict.html', pred=pred)

if __name__ == '__main__':
    app.run()
