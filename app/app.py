import os
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
import requests
from PIL import Image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import io

# Initialize Flask app
app = Flask(__name__)

# Load the model (Ensure the model file exists in the specified path)
MODEL_PATH = os.environ.get('MODEL_PATH', './models/efficientnetv2b0.h5')

try:
    model = EfficientNetV2B0(weights='imagenet')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")


# Route for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running"}), 200


# Route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read the uploaded file and 
        # convert it to a format that `load_img` can handle
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]

        # Format response
        predictions = [
            {"class": cls, "description": desc, "confidence": float(conf)}
            for cls, desc, conf in decoded_preds]

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/predict_url', methods=['POST'])
def predict_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400

    image_url = data['url']

    try:
        # Download image
        response = requests.get(image_url, stream=True)
        response.raise_for_status()

        # Convert the image into a format that `load_img` can handle
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))  # Resize based on model input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]

        # Format response
        predictions = [
            {"class": cls, "description": desc, "confidence": float(conf)}
            for cls, desc, conf in decoded_preds
        ]

        return jsonify({"predictions": predictions})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
