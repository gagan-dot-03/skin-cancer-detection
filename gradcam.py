import tensorflow as tf
import numpy as np
import cv2
import os

from config import MODEL_PATH

# -----------------------------------------
# Load Trained Model
# -----------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------------------
# Find Last Convolution Layer
# -----------------------------------------

def get_last_conv_layer(model):

    for layer in reversed(model.layers):

        if len(layer.output_shape) == 4:
            return layer.name

    raise ValueError("No Conv Layer Found")


LAST_CONV_LAYER = get_last_conv_layer(model)

# -----------------------------------------
# Generate GradCAM
# -----------------------------------------

def generate_gradcam(image_path):

    img = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=(224,224)
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = np.expand_dims(img_array,axis=0)

    img_array = img_array/255.0

    grad_model = tf.keras.models.Model(

        [model.inputs],

        [

            model.get_layer(LAST_CONV_LAYER).output,

            model.output

        ]

    )

    with tf.GradientTape() as tape:

        conv_outputs,predictions = grad_model(img_array)

        loss = predictions[:,0]

    grads = tape.gradient(loss,conv_outputs)

    pooled_grads = tf.reduce_mean(

        grads,

        axis=(0,1,2)

    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(

        pooled_grads * conv_outputs,

        axis=-1

    )

    heatmap = np.maximum(

        heatmap,

        0

    )

    heatmap = heatmap / np.max(heatmap)

    original = cv2.imread(image_path)

    original = cv2.resize(

        original,

        (

            original.shape[1],

            original.shape[0]

        )

    )

    heatmap = cv2.resize(

        heatmap,

        (

            original.shape[1],

            original.shape[0]

        )

    )

    heatmap = np.uint8(

        heatmap * 255

    )

    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )

    output = cv2.addWeighted(

        original,

        0.6,

        heatmap,

        0.4,

        0

    )

    filename = os.path.basename(image_path)

    save_path = os.path.join(

        "static",

        "heatmaps",

        filename

    )

    cv2.imwrite(

        save_path,

        output

    )

    return filename


# -----------------------------------------
# Test
# -----------------------------------------

if __name__=="__main__":

    image=input("Image Path : ")

    generate_gradcam(image)

    print("Heatmap Generated Successfully")