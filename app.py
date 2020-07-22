from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
global graph;
global sess;
tf_config = None
# sess = tf.Session(config=tf_config)

# graph=tf.get_default_graph()
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='transfer_l_resnet.h5'

# Load your trained model
# set_session(sess)

model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x,axis=0)
    x=preprocess_input(x);
    print('x value is {}'.format(x))
    preds=model.predict(x)
    print('prediction value is {}'.format(preds))
    preds=np.argmax(preds, axis=1)
    print('prediction value is {}'.format(preds))
    # console.log(preds)
    if preds==0:
        preds="The Car IS Audi"
    elif preds==1:
        preds="The Car is Lamborghini"
    elif preds==2:
        preds="The Car Is Mercedes"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result

if __name__ == '__main__':
    app.run(debug=True)
