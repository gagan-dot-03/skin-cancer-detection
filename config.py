import os

# -----------------------------
# Project Configuration
# -----------------------------

IMAGE_SIZE = (224,224)

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 0.0001

MODEL_PATH = "models/model.keras"

UPLOAD_FOLDER = "static/uploads"

HEATMAP_FOLDER = "static/heatmaps"

TRAIN_DIR = "dataset/train"

TEST_DIR = "dataset/test"

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}