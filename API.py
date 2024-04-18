from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained machine learning model from the pickle file
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

    # Load the one-hot encoder
with open('ct.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

    # Load the label encoder
with open('label.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
    
    

# Define a route for making predictions
@app.route('/')
def fun():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction')
def home():
    return render_template('pred.html')

# yaha pe sare edit 



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        feature1 = int(request.form['feature1'])
        feature2 = request.form['feature2']  # Assuming feature2 is categorical
        feature3 = request.form['feature3']  # Assuming feature3 is categorical
        feature4 = request.form['feature4']  # Assuming feature4 is categorical
        
        input_data = np.array([[feature1, feature2, feature3, feature4]])

        input_data[:, 3] = label_encoder.transform(input_data[:, 3])
        input_data = np.array(one_hot_encoder.transform(input_data))
        # Make prediction using the pre-trained model
        prediction = model.predict(input_data)
        output = int(prediction[0])
        
        return render_template('pred.html', prediction_text='Expected Crowd Density Will be {}'.format(output))
    except Exception as e:
        # If any error occurs, return an error message
        return jsonify({'error': str(e)}), 500
        

if __name__ == '__main__':
    app.run(debug=True, port=8000)
