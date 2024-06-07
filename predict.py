# predict.py
import joblib
import numpy as np
from flask import Flask, request, jsonify

# Load the trained model
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['features'])])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083)
