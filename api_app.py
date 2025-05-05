from __future__ import division, print_function
import sys
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# Define a Flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model_resnet152V2.tflite'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0  # Scale the pixel values
    x = np.expand_dims(x, axis=0)
    
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    
    # Map the prediction to label
    if preds == 0:
        return "The leaf is diseased cotton leaf"
    elif preds == 1:
        return "The leaf is diseased cotton plant"
    elif preds == 2:
        return "The leaf is fresh cotton leaf"
    else:
        return "The leaf is fresh cotton plant"

@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(file.filename))
        file.save(file_path)
        
        # Make prediction
        result = model_predict(file_path, model)
        # os.remove(file_path)  # Remove file after prediction
        return jsonify({"prediction": result})

    return jsonify({"error": "File upload failed"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
