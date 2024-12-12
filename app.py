from flask import Flask, render_template, request
import numpy as np
import joblib  # Changed to joblib

# Load your trained model (make sure the model path is correct)
model_path = 'credit_card_model.pkl'  # Update this path as needed
model = joblib.load(model_path)  # Use joblib to load the model

# Check the model type
print(f"Model type: {type(model)}")  # Should print the classifier type

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect input values from the form
    input_data = request.form['data']  # Expecting a single input for all features
    input_features = list(map(float, input_data.split()))  # Split the input string and convert to float

    # Automatically remove the first and last elements
    input_features = input_features[1:-1]  # Removing the first and last feature
    input_features = np.array(input_features).reshape(1, -1)  # Reshape for prediction

    # Make prediction
    prediction = model.predict(input_features)  # Ensure model is the classifier

    # Determine the result
    result = "Normal Transaction" if prediction[0] == 0 else "Fraud Transaction"
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
