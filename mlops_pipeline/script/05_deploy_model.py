# 05_deploy_model.py — Mushroom-EfficientNet (multipart form upload)
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import mlflow.pyfunc

# Flask app
app = Flask(__name__)

# โหลดโมเดลจาก MLflow Model Registry
try:
    model = mlflow.pyfunc.load_model("models:/Mushroom-EfficientNet@champion")
    mushroom_classes = [
        "Agaricus", "Amanita", "Boletus", "Cortinarius", "Entoloma",
        "Hygrocybe", "Lactarius", "Russula", "Suillus"
    ]
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    mushroom_classes = []


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize และ Normalize image ก่อนส่งเข้า EfficientNet"""
    img = image.resize((224, 224))  # EfficientNet input size
    img = np.array(img) / 255.0  # normalize
    if img.ndim == 2:  # grayscale → RGB
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:  # RGBA → RGB
        img = img[:, :, :3]
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
        features = preprocess_image(image)

        preds = model.predict(features)
        pred_index = int(np.argmax(preds, axis=1)[0])
        pred_class = mushroom_classes[pred_index]

        result = {
            "predicted_class": pred_class,
            "probabilities": {
                mushroom_classes[i]: float(preds[0][i]) for i in range(len(mushroom_classes))
            }
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    return "<h1>Mushroom EfficientNet API</h1><p>Use POST /predict with form-data (key=file)</p>"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
