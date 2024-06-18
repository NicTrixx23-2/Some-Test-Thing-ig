from flask import Flask, request, jsonify
import base64
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize

app = Flask(__name__)

# Load your machine learning model
model = load_model('path_to_your_model.h5')  # Adjust path as needed
graph = None  # Initialize graph (needed for TensorFlow 1.x)

def load_model():
    global model
    model = load_model('path_to_your_model.h5')  # Adjust path as needed

@app.route('/')
def index():
    return 'Server is running'

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data found'}), 400

    img_data = request.json['image'].split(',')[1]  # Remove 'data:image/png;base64,' part
    img_bytes = base64.b64decode(img_data)
    with open('output.png', 'wb') as f:
        f.write(img_bytes)

    img = Image.open('output.png').convert('L')  # Convert to grayscale
    img = np.invert(img)
    img = resize(img, (28, 28), order=1, mode='constant', cval=0, clip=False, preserve_range=True)
    img = img.astype(int).reshape(1, 28, 28, 1)

    global graph
    with graph.as_default():
        prediction = model.predict(img)

    result = np.argmax(prediction)  # Get the index of the highest probability

    return jsonify({'expression': str(result)}), 200

if __name__ == '__main__':
    load_model()  # Load the model before starting the server
    app.run(debug=True)
