from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import os

global graph
graph = tf.Graph()

# Create flask instance
app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.Graph()

# Function to load and prepare the image in right shape
def read_image(filename):
    # Load the image
    img = cv2.imread(filename, color_mode="rgb", target_size=(80,60))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image into a sample of 3 channel
    img = img.reshape(1, 80, 60, 3)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join('dataset\myntradataset', filename)
        file.save(file_path)
        img = cv2.imread(file_path)
        # Predict the class of an image

        with graph.as_default():
            model1 = load_model('fashion_image_captioning_model1.h5')
            class_prediction = model1.predict_classes([img])
            print(class_prediction)

        if(class_prediction[0] == 0):
            product = "Apparel"
        elif(class_prediction[0] == 1):
            product = "Accessories"
        elif(class_prediction[0] == 2):
            product = "Footwear"
        elif(class_prediction[0] == 3):
            product = "Personal Care"
        elif(class_prediction[0] == 4):
            product = "Free Items"
        elif(class_prediction[0] == 5):
            product = "Sporting Goods"
        else:
            product = "Home"

        return render_template('predict.html', product = product, user_image = file_path)
        
    return render_template('predict.html')

if(__name__ == "__main__"):
    init()
    app.run(debug=True)
