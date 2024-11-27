from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter


# Load the quantized model
interpreter = Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('result.html', prediction="No file part")

    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', prediction="No selected file")

    # Read and preprocess the image
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((128, 128))  # Resize to match model input size
    img = np.array(img, dtype=np.float32)

    # Normalize image to [0, 1] if required (depends on your model's training)
    img = img / 255.0

    # Expand dimensions to match the input shape (batch_size, height, width, channels)
    input_data = np.expand_dims(img, axis=0)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor and return the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=-1)

    # You can map predicted_class to your actual label names (if required)
    result = f"Predicted Class: {predicted_class[0]}"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
