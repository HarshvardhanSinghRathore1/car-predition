from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model_pip.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get YearsExperience from form
        years_exp = float(request.form['years_exp'])
        prediction = model.predict([[years_exp]])  # Predict salary

        return render_template('index.html',
                               prediction_text=f'Predicted Salary for {years_exp} years experience: ₹{prediction[0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
