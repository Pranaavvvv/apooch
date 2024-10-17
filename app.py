from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model (assuming you've saved the pipelines as 'num_pipeline.pkl' and 'cat_pipeline.pkl')
# Save the models beforehand using: 
#   pickle.dump(num_pipeline, open('num_pipeline.pkl', 'wb'))
#   pickle.dump(cat_pipeline, open('cat_pipeline.pkl', 'wb'))
num_pipeline = pickle.load(open('num_pipeline.pkl', 'rb'))
cat_pipeline = pickle.load(open('cat_pipeline.pkl', 'rb'))

# Define the features
features = ['Pet Type', 'Breed', 'Weight (Kg)', 'Gender', 'Age (Years)', 'Activity Level', 'Food Allergies', 'Neutered/Spayed Status']

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.get_json()
    
    # Convert input data into a DataFrame
    input_data = pd.DataFrame([data], columns=features)
    
    # Predict numerical and categorical outputs
    num_prediction = num_pipeline.predict(input_data)
    cat_prediction = cat_pipeline.predict(input_data)
    
    # Prepare the response
    nutrition_plan = {
        'Protein (kg)': num_prediction[0][0],
        'Carbs (kg)': num_prediction[0][1],
        'Fats (kg)': num_prediction[0][2],
        'Best Protein Sources': cat_prediction[0][0],
        'Best Carb Sources': cat_prediction[0][1],
        'Best Fat Sources': cat_prediction[0][2],
        'Special Note': cat_prediction[0][3],
        'Suggested Nutrition': cat_prediction[0][4]
    }
    
    # Return the prediction as a JSON response
    return jsonify(nutrition_plan)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
