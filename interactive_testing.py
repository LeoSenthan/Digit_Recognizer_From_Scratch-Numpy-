from flask import Flask, render_template, request, jsonify
import numpy as np
from models.network import NeuralNetwork
from models.layers import Layer_Dense
from models.activations import Activation_ReLU, Activation_Softmax
from utility_functions.save_and_load_model import load_model
from train import create_digit_model
app = Flask(__name__)

nn = create_digit_model()
load_model(nn, "Saved_Model.npz") 

@app.route("/")
def home():
    return render_template("index.html")  

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expect base64-encoded 28x28 image data from frontend
        data = request.json["image"]
        import base64
        from io import BytesIO
        from PIL import Image

        img_bytes = base64.b64decode(data.split(",")[1])
        img = Image.open(BytesIO(img_bytes)).convert("L")
        img = img.resize((28, 28))
        x = np.array(img).reshape(1, 28*28).astype("float32") / 255.0

        probs = nn.forward(x)
        prediction = int(np.argmax(probs, axis=1)[0])

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
