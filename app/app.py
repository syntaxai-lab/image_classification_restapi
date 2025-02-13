import os
import numpy as np
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify

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
        # Preprocess the image
        img = image.load_img(file.stream, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        preds = model.predict(img_array)
        decoded_preds = decode_predictions(preds, top=3)[0]

        # Format response
        predictions = [{"class": cls, "description": desc, "confidence": float(conf)}
                        for cls, desc, conf in decoded_preds]

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)