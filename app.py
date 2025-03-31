import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from PIL import Image
import json
from keras.layers import TFSMLayer  # Required for Keras 3

app = Flask(__name__)
CORS(app)

# ✅ Load Model using TFSMLayer (since Keras 3 doesn't support SavedModel directly)
MODEL_PATH = "final_model"
model = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")  

# ✅ Load breed names from JSON file
with open("breeds.json", "r") as f:
    class_names = json.load(f)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)  # Save the uploaded image

    # ✅ Preprocess Image
    img = Image.open(file_path).resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # ✅ Make Prediction
    try:
        predictions_dict = model(img_array)  # Returns a dictionary
        print("Raw Model Output:", predictions_dict)  # Debugging

        predictions = list(predictions_dict.values())[0]  # Extract tensor
        predictions = predictions.numpy().flatten()  # Convert to NumPy array

        predicted_breed = class_names[np.argmax(predictions)]

        return jsonify({"breed": predicted_breed, "filename": file.filename, "message": "Prediction successful"})

    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)})

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Dog Breed Classifier API!"})

if __name__ == "__main__":
    app.run(debug=True)
