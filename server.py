from flask import Flask, request, render_template, flash, redirect, send_from_directory
import sys
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
from segmen import FA, Upsample, FADPNEncoder, FADPNDecoder, segm
import pickle
from pretrainedmodels import dpn92
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.datasets.folder import default_loader
from typing import List
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import os
from img2cap import as_matrix, ScaledDotProductScore, Attention, CaptionNet, BeheadedVGG19, capt


WIDTH = 500
HEIGHT = 400

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()  
set_session(sess)


with open('vocab.bin', 'rb') as fin:
    vocab = pickle.load(fin)
with open('word_to_index.bin', 'rb') as fin:
    word_to_index = pickle.load(fin)

eos_ix = word_to_index['#END#']
unk_ix = word_to_index['#UNK#']
pad_ix = word_to_index['#PAD#']

with open('network.bin', 'rb') as fin:
            network = pickle.load(fin)
with open('features_net.bin', 'rb') as fin:
            features_net = pickle.load(fin)

with open('encoder.bin', 'rb') as fin:
    encoder = pickle.load(fin)
with open('decoder.bin', 'rb') as fin:
    decoder = pickle.load(fin)

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

UPLOAD_FOLDER = 'images'
ALLOWED_EXTENSIONS = {'png', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'super super secret key'

@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route("/")
def index(for_print=0, error=0, cap_print = 0):
    print(for_print)

    images = []
    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith('.jpg') and not filename.endswith('.png') and not filename.endswith('.jpeg') :
                continue
            im = Image.open(filename)
            w, h = im.size
            aspect = 1.0*w/h
            if aspect > 1.0*WIDTH/HEIGHT:
                width = min(w, WIDTH)
                height = width/aspect
            else:
                height = min(h, HEIGHT)
                width = height*aspect
            images.append({
                'width': int(width),
                'height': int(height),
                'src': filename
            })
    #if for_print == 0:
    #    a = 'start.html'
    #elif for_print == 25:
    #    a = 'index.html''''

    return render_template('caption.html', for_print=for_print, error=error, images = images, cap_print = cap_print) #raise_global_error() 

@app.route("/")
def raise_global_error():
    return render_template('error.html')

@app.route("/rnn", methods=['POST'])
def rnn(**args):
    file_names = list(os.walk('./images'))[0][2]
    for fn in file_names:
        os.remove(os.path.join('./images', fn))
    file_extension = ''
    if 'image_file' not in request.files:
        flash('No file part')
    file = request.files['image_file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'input.' + filename.split('.')[-1]))
        file_extension = filename.split('.')[-1]
        # return raise_global_error()

    command = request.form['text1']
    global graph

    # return request.form['text'] + " Command executed via subprocess"
    '''if int(command) > 15000:
            return index(error=2)
        if command3.isdigit():
            if int(command3)>150000:
                return index(error=4)
            if len(command2)>20:
                return index(error=5)'''

    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        for_print = segm(encoder, decoder, './images/input.'+file_extension, command)
        # keras.backend.clear_session()
        for_print = 25
    return index(for_print=for_print, cap_print = 0)
    '''else:
            return index(error=3)
    else:
        return index(error=1)'''
 
@app.route("/caption", methods=['POST'])
def caption(**args):
    file_extension = ''
    file_names = list(os.walk('./images'))[0][2]
    for fn in file_names:
        os.remove(os.path.join('./images', fn))

    if 'image_file' not in request.files:
        flash('No file part')
    file = request.files['image_file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'input.' + filename.split('.')[-1]))
        # return raise_global_error()
        file_extension = filename.split('.')[-1]

    command = request.form['text1']
    global graph

    # return request.form['text'] + " Command executed via subprocess"
    '''if int(command) > 15000:
            return index(error=2)
        if command3.isdigit():
            if int(command3)>150000:
                return index(error=4)
            if len(command2)>20:
                return index(error=5)'''

    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        cap_print = capt(network, features_net, './images/input.'+file_extension, command)
        # keras.backend.clear_session()
        for_print = 25

    return index(for_print=for_print, cap_print = cap_print)
    '''else:
            return index(error=3)
    else:
        return index(error=1)'''
 


@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


if __name__ == "__main__":
    app.run(debug='True')
