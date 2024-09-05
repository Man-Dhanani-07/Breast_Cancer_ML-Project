import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)
model_path = 'model.pkl'
# model = pickle.load(open('../model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    if os.path.isfile(model_path):
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print(f"File {model_path} does not exist.")
        model = None
    
    if model is None:
        return render_template('index.html', prediction_text='Model is not loaded.')

    try:
        
        input_features = [float(x) for x in request.form.values()]
        features_value = [np.array(input_features)]
        
        features_name = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]
        
    
        df = pd.DataFrame(features_value, columns=features_name)
        
      
        output = model.predict(df)
       
        if output[0] == 0:
            res_val = "** breast cancer **"
        else:
            res_val = "no breast cancer"
        
        return render_template('index.html', prediction_text=f'Patient has {res_val}')
    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('index.html', prediction_text='Error in prediction.')

if __name__ == "__main__":
    app.run(debug=True) 
