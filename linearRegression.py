import pickle
import numpy as np
from flask import Flask, request, render_template

# Create a Flask web application
app = Flask(__name__)

# Load the pickled model pipeline
# Ensure 'model_pipeline.pkl' is in the same directory as this script
try:
    with open('model_pipeline.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: 'model_pipeline.pkl' not found. Please ensure the file is in the correct directory.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles both displaying the form (GET) and processing the prediction (POST).
    """
    prediction_text = ''
    
    # This block executes when the user submits the form
    if request.method == 'POST':
        if model:
            try:
                # Get data from the form and convert to integers
                # Note: Categorical features are expected as numerical codes
                form_data = [int(request.form[field]) for field in [
                    'yr_mfr', 'fuel_type', 'kms_run', 'body_type', 
                    'transmission', 'registered_state', 'make', 'model', 'total_owners'
                ]]

                # Create a numpy array for the model's input
                features = np.array([form_data])
                
                # Make a prediction using the loaded model pipeline
                prediction = model.predict(features)
                
                # Format the prediction output for display
                prediction_text = f'Predicted Price: ₹ {prediction[0]:,.2f}'

            except Exception as e:
                # Handle potential errors like non-integer input
                prediction_text = f"An error occurred: {e}"
        else:
            prediction_text = "Model could not be loaded. Please check the server logs."

    # Render the HTML page, passing the prediction result to it
    return render_template('index.html', prediction_text=prediction_text)

# This is the standard entry point for a Flask application
if __name__ == '__main__':
    app.run(debug=True)