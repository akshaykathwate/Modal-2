from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename

# Define a Flask app
app = Flask(__name__)

# Model saved with TFLite
MODEL_PATH = 'model_resnet152V2.tflite'

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def model_predict(img_path, interpreter):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0  # Normalize
    x = np.expand_dims(x, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    preds = np.argmax(output, axis=1)

    if preds == 0:
        return "The leaf is diseased cotton leaf"
    elif preds == 1:
        return "The leaf is diseased cotton plant"
    elif preds == 2:
        return "The leaf is fresh cotton leaf"
    else:
        return "The leaf is fresh cotton plant"

# API endpoint
@app.route('/api/predict', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        basepath = os.path.dirname(__file__)
        uploads_dir = os.path.join(basepath, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        file_path = os.path.join(uploads_dir, secure_filename(file.filename))
        file.save(file_path)

        # Predict
        result = model_predict(file_path, interpreter)
        # os.remove(file_path)  # Optional: clean up after prediction
        return jsonify({"prediction": result})

    return jsonify({"error": "File upload failed"}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
