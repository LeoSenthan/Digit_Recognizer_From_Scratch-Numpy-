from flask import Flask, render_template, request, jsonify
import numpy as np
from models.network import NeuralNetwork
from models.layers import Layer_Dense
from models.activations import Activation_ReLU, Activation_Softmax
from utility_functions.save_and_load_model import load_model
from train import create_digit_model

from PIL import Image, ImageOps
import base64
from io import BytesIO

app = Flask(__name__)

# Load the trained neural network
nn = create_digit_model()
load_model(nn, "Saved_Model.npz")


@app.route("/")
def home():
    return render_template("index.html")


def preprocess_canvas_image(img, target_size=28):
    # Ensure grayscale
    img = img.convert("L")

    # Crop to bounding box of the digit
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    else:
        # If empty, return a black image
        img = Image.new("L", (target_size, target_size), color=0)

    # Resize while preserving aspect ratio
    img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

    # Create a new black image and paste the resized digit to center it
    new_img = Image.new("L", (target_size, target_size), color=0)
    paste_x = (target_size - img.width) // 2
    paste_y = (target_size - img.height) // 2
    new_img.paste(img, (paste_x, paste_y))

    # Convert to numpy and normalize
    x = np.array(new_img).reshape(1, target_size * target_size).astype("float32") / 255.0

    return x


@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]
        import base64
        from io import BytesIO
        from PIL import Image

        img_bytes = base64.b64decode(data.split(",")[1])
        img = Image.open(BytesIO(img_bytes))

        # Preprocess the image for the model
        x = preprocess_canvas_image(img, target_size=28)

        # Forward pass
        probs = nn.forward(x)
        prediction = int(np.argmax(probs, axis=1)[0])

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
