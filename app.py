from flask import Flask, request, send_from_directory, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="/")
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "model/brain_tumor_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 128  # Image size used in training

# Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img / 255.0

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file=request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path=os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    img=preprocess_image(file_path)
    input_img=np.expand_dims(img, axis=0)
    prediction=model.predict(input_img)
    predicted_class = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor"
    return jsonify({"filename": file.filename, "result": predicted_class})


@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
