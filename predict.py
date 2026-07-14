import tensorflow as tf
import numpy as np
import cv2
import os
from config import MODEL_PATH, IMAGE_SIZE

# ------------------------------------
# Load Trained Model
# ------------------------------------

print("Loading Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully")

CLASS_NAMES = {
    0: "Benign",
    1: "Malignant"
}

# ------------------------------------
# Image Preprocessing
# ------------------------------------

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("Unable to load image.")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, IMAGE_SIZE)

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image


# ------------------------------------
# Prediction
# ------------------------------------

def predict_image(image_path):

    image = preprocess_image(image_path)

    prediction = model.predict(image, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:

        label = "Malignant"

        confidence = probability * 100

        risk = "HIGH"

    else:

        label = "Benign"

        confidence = (1 - probability) * 100

        risk = "LOW"

    return {

        "label": label,

        "confidence": round(confidence, 2),

        "risk": risk,

        "raw_probability": probability

    }


# ------------------------------------
# Test Prediction
# ------------------------------------

if __name__ == "__main__":

    image_path = input("Enter Image Path : ")

    result = predict_image(image_path)

    print("\nPrediction")

    print("----------------------")

    print("Class :", result["label"])

    print("Confidence :", result["confidence"], "%")

    print("Risk :", result["risk"])