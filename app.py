from flask import Flask, render_template, request
from keras.models import load_model
import os,shutil
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
from werkzeug.utils import secure_filename

model = load_model('keras_model.h5')
dic = {0 : 'red', 1 : 'green'}

app=Flask(__name__)

@app.route('/')
def index():
      return render_template("index.html")


@app.route("/predict", methods=["POST"])
def prediction():

  f= request.files['img']

  folder = r'uploads'
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
    elif os.path.isdir(file_path):
       shutil.rmtree(file_path)

  
  basepath = os.path.dirname(__file__)
  file_path = os.path.join(
      basepath, 'uploads', secure_filename(f.filename))
  f.save(file_path)
  img = tf.keras.utils.load_img(file_path,target_size=(224,224)) ##loading the image
  img = np.asarray(img) ##converting to an array
  img = img / 255 ##scaling by doing a division of 255
  img = img.reshape(1, 224,224,3)
  p = np.argmax(model.predict(img), axis=1)

  print(dic[p[0]])
  return render_template("index.html",prediction=dic[p[0]])



if __name__=="__main__":
      app.run(debug=True)
