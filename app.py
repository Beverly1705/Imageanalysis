from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from utils import extract_face

# === CONFIG ===
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = "person_classifier_model.h5"
DATASET_PATH = 'C:/Users/Beverly Lee/Desktop/image_classification_dataset'

# === FLASK SETUP ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === LOAD MODEL ===
model = tf.keras.models.load_model(MODEL_PATH)

# === CLASS NAMES ===
class_names = ['marian rivera', 'klarisse de guzman', 'esnyr', 'brent manalo', 'angel locsin', 'alden richard']
print("Class names loaded:", class_names)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 🧽 Delete all old uploads first
        for f in os.listdir(app.config['UPLOAD_FOLDER']):
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], f))
            except:
                pass  # Ignore if already deleted

        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # 🧠 Extract face from image
            face = extract_face(filepath)
            if face is None:
                return render_template("index.html", result="No face detected.", filename=filename, class_names=class_names)

            # 🔮 Predict
            prediction = model.predict(face)
            predicted_index = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100

            if confidence < 60:
                result = f"Unknown person (Confidence: {confidence:.2f}%)"
            else:
                person = class_names[predicted_index]
                result = f"{person} ({confidence:.2f}%)"

            # ✅ Keep image for preview (don't delete it)
            return render_template("index.html", result=result, filename=filename, class_names=class_names)

    # GET request: show blank form
    return render_template("index.html", result=None, filename=None, class_names=class_names)

if __name__ == "__main__":
    app.run(debug=True)