import os

def allowed_file(filename):

    return "." in filename and \
    filename.rsplit(".",1)[1].lower() in \
    {"jpg","jpeg","png"}

def create_directories():

    folders = [

        "dataset/train",

        "dataset/test",

        "models",

        "static/uploads",

        "static/heatmaps",

        "graphs",

        "screenshots"

    ]

    for folder in folders:

        os.makedirs(folder,exist_ok=True)

    print("Project folders verified.")