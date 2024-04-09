from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Load your pre-trained model (replace with your model path)
MODEL = tf.keras.models.load_model("./models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", prediction="No image uploaded")

    image_file = request.files["image"]

    # Check if file is empty
    if image_file.filename == '':
        return render_template("index.html", prediction=f"Please choose an image file.")

    image = image_file.read()

    # Preprocess image
    image_pil = Image.open(BytesIO(image))
    image_pil.thumbnail((300, 300))  # Resize image to fit in template

    # Convert image to base64 for embedding in HTML
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Get the filename
    filename = image_file.filename

    # Add dimension for batch processing
    image = np.array(image_pil)
    img_batch = np.expand_dims(image, 0)

    # Make prediction
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return render_template("index.html", prediction=f"{predicted_class}", image=img_str, filename=filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
