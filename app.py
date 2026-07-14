from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
from werkzeug.utils import secure_filename

import os

from predict import predict_image
from gradcam import generate_gradcam

from config import *

# -----------------------------------------
# Flask App
# -----------------------------------------

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -----------------------------------------
# Allowed File
# -----------------------------------------

def allowed_file(filename):

    return "." in filename and \
    filename.rsplit(".",1)[1].lower() in \
    ALLOWED_EXTENSIONS

# -----------------------------------------
# Home Page
# -----------------------------------------

@app.route("/")

def home():

    return render_template("index.html")

# -----------------------------------------
# About Page
# -----------------------------------------

@app.route("/about")

def about():

    return render_template("about.html")

# -----------------------------------------
# Prediction
# -----------------------------------------

@app.route("/predict",methods=["POST"])

def predict():

    if "image" not in request.files:

        return render_template(

            "error.html",

            message="No File Uploaded"

        )

    file=request.files["image"]

    if file.filename=="":

        return render_template(

            "error.html",

            message="Please Select an Image"

        )

    if not allowed_file(file.filename):

        return render_template(

            "error.html",

            message="Only JPG JPEG PNG Allowed"

        )

    filename=secure_filename(file.filename)

    filepath=os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )

    file.save(filepath)

    result=predict_image(filepath)

    heatmap=generate_gradcam(filepath)

    return render_template(

        "result.html",

        image=filename,

        prediction=result["label"],

        confidence=result["confidence"],

        risk=result["risk"],

        heatmap=heatmap

    )

# -----------------------------------------
# Run App
# -----------------------------------------

if __name__=="__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )