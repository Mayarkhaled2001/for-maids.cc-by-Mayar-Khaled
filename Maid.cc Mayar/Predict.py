from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained model (Assuming the model is saved as 'model.pkl')
model = joblib.load('model.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get device specs from the request body (JSON format)
    data = request.get_json()

    # Example: Extracting features from the JSON input
    features = np.array([
        data.get('battery_power'),
        data.get('ram'),
        data.get('px_width'),
        data.get('px_height'),
        data.get('talk_time')
    ]).reshape(1, -1)  # Reshape to a 2D array for prediction

    # Make a prediction using the loaded model
    predicted_price = model.predict(features)[0]

    # Return the prediction as a JSON response
    return jsonify({'predicted_price': str(predicted_price)})

if __name__ == '__main__':
    # Run Flask app on localhost:6000
    app.run(port=6000, debug=True)
