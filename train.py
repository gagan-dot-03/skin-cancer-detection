import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from config import *

print("="*60)
print(" Skin Cancer Detection Model Training ")
print("="*60)

# ---------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------

train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,

    width_shift_range=0.15,

    height_shift_range=0.15,

    zoom_range=0.20,

    shear_range=0.15,

    horizontal_flip=True,

    fill_mode="nearest",

    validation_split=0.20

)

# Validation Generator

validation_datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.20

)

# ---------------------------------------------------
# Load Training Images
# ---------------------------------------------------

train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="training",

    shuffle=True

)

# ---------------------------------------------------
# Load Validation Images
# ---------------------------------------------------

validation_generator = validation_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="binary",

    subset="validation",

    shuffle=False

)

print()

print("Training Images :",train_generator.samples)

print("Validation Images :",validation_generator.samples)

print()

# ---------------------------------------------------
# Build EfficientNetB0
# ---------------------------------------------------

base_model = EfficientNetB0(

    include_top=False,

    weights="imagenet",

    input_shape=(224,224,3)

)

# Freeze base model

base_model.trainable=False

print()

print("EfficientNetB0 Loaded")

print()

# ---------------------------------------------------
# Add Custom Layers
# ---------------------------------------------------

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(

    256,

    activation="relu"

)(x)

x = Dropout(0.50)(x)

x = Dense(

    128,

    activation="relu"

)(x)

x = Dropout(0.30)(x)

predictions = Dense(

    1,

    activation="sigmoid"

)(x)

model = Model(

    inputs=base_model.input,

    outputs=predictions

)

print()

model.summary()

# ---------------------------------------------------
# Compile Model
# ---------------------------------------------------

model.compile(

    optimizer=Adam(

        learning_rate=LEARNING_RATE

    ),

    loss="binary_crossentropy",

    metrics=[

        "accuracy"

    ]

)

print()

print("Model Compiled Successfully")

print()

print("="*60)

# ---------------------------------------------------
# Callbacks
# ---------------------------------------------------

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1
)

# ---------------------------------------------------
# Train Model
# ---------------------------------------------------

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)

print("\nTraining Completed Successfully!")

# ---------------------------------------------------
# Save Model
# ---------------------------------------------------

model.save(MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")

# ---------------------------------------------------
# Plot Accuracy and Loss
# ---------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.plot(history.history["accuracy"],label="Training")

plt.plot(history.history["val_accuracy"],label="Validation")

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.subplot(1,2,2)

plt.plot(history.history["loss"],label="Training")

plt.plot(history.history["val_loss"],label="Validation")

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()

plt.savefig("graphs/training_graph.png")

plt.show()

# ---------------------------------------------------
# Evaluate Model
# ---------------------------------------------------

test_datagen = ImageDataGenerator(
    rescale=1./255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator)

print("\n==============================")
print(f"Test Accuracy : {accuracy*100:.2f}%")
print(f"Test Loss     : {loss:.4f}")
print("==============================")

# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------

import numpy as np

from sklearn.metrics import classification_report

predictions = model.predict(test_generator)

predicted_classes = np.where(
    predictions > 0.5,
    1,
    0
)

true_classes = test_generator.classes

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=["Benign","Malignant"]
    )
)

# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\nConfusion Matrix")

print(cm)

print("\n========================================")
print(" Skin Cancer Detection Model Ready ")
print("========================================")