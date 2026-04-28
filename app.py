from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import sys

app = Flask(__name__)

# --- MODEL LOADING LOGIC ---
MODEL_PATH = "model.pkl"

try:
    if os.path.exists(MODEL_PATH):
        # We use joblib as it is the standard for scikit-learn .pkl files
        model = joblib.load(MODEL_PATH)
        print("--- SUCCESS: Model loaded successfully ---")
    else:
        print(f"--- ERROR: {MODEL_PATH} not found in {os.getcwd()} ---")
        model = None
except Exception as e:
    print(f"--- ERROR: Could not load model file. Detail: {e} ---")
    model = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "ML Model API is running",
        "model_loaded": model is not None,
        "python_version": sys.version
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded on server. Check server logs."}), 500

    try:
        data = request.get_json()
        
        # 1. Extract the 8 features from your JSON input
        # These keys MUST match your curl command exactly
        feature_values = [
            data.get('temperature'),
            data.get('humidity'),
            data.get('vibration'),
            data.get('pressure'),
            data.get('energy_consumption'),
            data.get('machine_status'),
            data.get('anomaly_flag'),
            data.get('predicted_remaining_life')
        ]

        # 2. Check if any value is missing
        if None in feature_values:
            return jsonify({"error": "One or more required fields are missing from the input"}), 400

        # 3. Convert to numpy array and reshape for a single prediction (1, 8)
        input_array = np.array(feature_values).reshape(1, -1)

        # 4. Perform prediction
        prediction = model.predict(input_array)

        return jsonify({
            "status": "success",
            "prediction": prediction.tolist(),
            "input_received": data
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Host 0.0.0.0 is required for EC2 to accept external traffic
    app.run(host='0.0.0.0', port=5000, debug=False)
