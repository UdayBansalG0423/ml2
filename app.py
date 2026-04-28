from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not found. Please run train.py first.", 500

    try:
        # Get values from the form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Predict
        prediction = model.predict(final_features)
        
        # Map prediction to class name
        classes = ['Setosa', 'Versicolor', 'Virginica']
        output = classes[prediction[0]]
        
        return render_template('index.html', prediction_text=f'Predicted Iris Species: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    # Host '0.0.0.0' allows external connections (useful for EC2)
    app.run(host="0.0.0.0", port=5000)
