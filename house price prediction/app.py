from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def hello_world():
    return render_template('house_price_prediction.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        bdrm = request.form.get('bedrooms')
        btrm = request.form.get('bathrooms')
        sf_liv = request.form.get('sqft_living')
        sf_lot = request.form.get('sqft_lot')
        flr = request.form.get('floors')

        # Set a default value for condition (you might want to get this from the form as well)
        condition = int(3)
    
        # Assuming 'year' is missing, you might want to add it or remove it from the features
        # Add or remove other features as needed
        features = np.array([bdrm, btrm, sf_liv, sf_lot, flr, condition]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Calculate price (adjust this based on your model and application)
        price = prediction

        return render_template('house_price_prediction.html', result=np.round(price, 2))

    # Handle GET requests separately, e.g., redirect to the home page
    return redirect(url_for('hello_world'))

if __name__ == '__main__':
    app.run(debug=True)
