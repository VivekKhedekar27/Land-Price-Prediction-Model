from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Flask App Initialization
app = Flask(__name__)
CORS(app)

# Globals
model_pipeline = None
X_train_transformed, y_train, X_train = None, None, None
neighbors_model = None

def load_resources():
    """
    Load model pipeline and training references.
    """
    global model_pipeline, X_train_transformed, y_train, X_train, neighbors_model
    try:
        print("[INFO] Loading model pipeline...")
        model_pipeline = joblib.load("property_price_lgb.pkl")
        
        print("[INFO] Loading training references...")
        X_train_transformed, y_train, X_train = joblib.load("training_references_2.pkl")
        
        print("[INFO] Initializing NearestNeighbors model...")
        neighbors_model = NearestNeighbors(n_neighbors=3)
        neighbors_model.fit(X_train_transformed)
        
        print("[INFO] Resources loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load resources: {e}")
        raise

# Load resources at startup
try:
    load_resources()
except Exception as e:
    print("[ERROR] App initialization failed. Ensure all required files are present.")
    exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict property price and find nearest neighbors.
    """
    try:
        print("[DEBUG] Received request for prediction.")
        data = request.json
        print(f"[DEBUG] Incoming JSON data: {data}")

        # Validate input
        required_fields = ['Collateral Type', 'Pin Code', 'Latitude', 'Longitude', 'City']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({'error': f"Missing fields: {', '.join(missing_fields)}"}), 400

        # Extract inputs
        collateral_subtype = data['Collateral Type']
        pin_code = data['Pin Code']
        latitude = data['Latitude']
        longitude = data['Longitude']
        city = data['City']

        # Prepare input DataFrame
        input_data = pd.DataFrame({
            'Collateral SubType': [collateral_subtype],
            'Pin Code': [int(pin_code)],
            'Latitude': [float(latitude)],
            'Longitude': [float(longitude)],
            'City': [city]
        })
        print(f"[DEBUG] Input DataFrame:\n{input_data}")

        # Transform input for prediction
        transformed_data = model_pipeline.named_steps['preprocessor'].transform(input_data)
        predicted_price = model_pipeline.predict(input_data)
        print(f"[DEBUG] Model prediction: {predicted_price}")

        # Find nearest neighbors
        distances, indices = neighbors_model.kneighbors(transformed_data)
        nearest_references = X_train.iloc[indices.flatten()].copy()

        # Add 'Price Per Sq.Ft' to nearest references
        nearest_references['Price Per Sq.Ft'] = y_train.iloc[indices.flatten()].values

        # Prepare response
        response = {
            'predicted_price': f"{predicted_price[0]:.2f}",
            'nearest_references': nearest_references[['City', 'Latitude', 'Longitude', 'Collateral SubType', 'Price Per Sq.Ft']].to_dict(orient='records')
        }

        return jsonify(response)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    app.run(host='0.0.0.0', port=8000, debug=False)
