import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model("models/fruit_cnn_model(1).h5")

# Class names (must match training folders order)
class_names = ['Apple', 'avocado', 'Banana', 'cherry', 'kiwi', 
               'mango', 'orange', 'pineapple', 'strawberries', 'watermelon']

def predict_fruit(img_path):
    # Load image with same size as used during training
    img = image.load_img(img_path, target_size=(64, 64))  # adjust size if you trained differently
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    fruit_name = class_names[class_index]
    confidence = round(100 * np.max(predictions[0]), 2)

    return fruit_name, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # check if an image was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict
            fruit, confidence = predict_fruit(filepath)

            return render_template('result.html', fruit=fruit, confidence=confidence, user_image=filepath)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
