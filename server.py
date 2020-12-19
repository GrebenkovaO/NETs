from flask import Flask, request, render_template, flash, redirect, send_from_directory
import sys
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import segmen as s
import pickle
from pretrainedmodels import dpn92
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pickle
from torchvision.datasets.folder import default_loader
from typing import List
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename
import os

WIDTH = 500
HEIGHT = 400

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()  
set_session(sess)

class FA(nn.Module):
    def __init__(self, num_ch):
        super().__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(num_ch, 32, 1),
            nn.BatchNorm2d(32)
        )
        self.K = nn.Sequential(
            nn.Conv2d(num_ch, 32, 1),
            nn.BatchNorm2d(32)
        )
        self.V = nn.Sequential(
            nn.Conv2d(num_ch, num_ch, 1),
            nn.BatchNorm2d(num_ch)
        )
        self.convBNrelu = nn.Sequential(
            nn.Conv2d(num_ch, num_ch, 1),
            nn.BatchNorm2d(num_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V0 = self.V(x)

        N,C,H,W = x.size()

        Q_ = Q.view(N,32,-1).permute(0, 2, 1)
        Q = F.normalize(Q_, p=2, dim=2, eps=1e-12)

        K_   = K.view(N,32,-1)
        K   = F.normalize(K_, p=2, dim=1, eps=1e-12)

        V_ = V0.view(N,C,-1).permute(0, 2, 1)
        V = F.relu(V_)

        f = torch.matmul(K, V)
        y = torch.matmul(Q, f)
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W = self.convBNrelu(y)
        return W + V0
        
class Upsample(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode, align_corners=True)

class FADPNEncoder(nn.Module):
    def __init__(self, num_classes=2, arch="dpn92"):
        super().__init__()
        backbone = dpn92(pretrained='imagenet+5k').features

        self.conv0 = nn.Sequential(backbone[0])
        self.block1 = nn.Sequential(*backbone[1:4])
        self.block2 = nn.Sequential(*backbone[4:8])
        self.block3 = nn.Sequential(*backbone[8:28])
        self.block4 = nn.Sequential(*backbone[28:])

    def forward(self, x):
        acts = []
        x = self.conv0(x)
        for i in range(4):
            x = self.__getattr__(f"block{i + 1}")(x)
            acts.append(x)
        return acts

class FADPNDecoder(nn.Module):    
    def __init__(self):
        super().__init__()   
        self.fa1 = FA(2688)
        self.fa2 = FA(1552)
        self.fa3 = FA(704)
        self.fa4 = FA(336)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(2688, 1552, 1),
            nn.BatchNorm2d(1552),
            Upsample((50, 38), mode='bilinear'),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(1552, 704, 1),
            nn.BatchNorm2d(704),
            Upsample((100, 75), mode='bilinear'),
            nn.ReLU()
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(704, 336, 1),
            nn.BatchNorm2d(336),
            Upsample((200, 150), mode='bilinear'),
            nn.ReLU()
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(336, 1, 1),
            nn.BatchNorm2d(1),
            Upsample((800, 600), mode='bilinear')
        )

    def forward(self, acts):
        x = self.decoder1(self.fa1(acts[3]))
        x = self.decoder2(x + self.fa2(torch.cat(acts[2], dim=1)))
        x = self.decoder3(x + self.fa3(torch.cat(acts[1], dim=1)))
        x = self.decoder4(x + self.fa4(torch.cat(acts[0], dim=1)))
        return x

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
def index(for_print=[], error=0):
    print(for_print)

    images = []
    for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
        for filename in [os.path.join(root, name) for name in files]:
            if not filename.endswith('.jpg'):
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

    return render_template('index.html', for_print=for_print, error=error, images = images) #raise_global_error() 

@app.route("/")
def raise_global_error():
    return render_template('error.html')

@app.route("/rnn", methods=['POST'])
def rnn(**args):
    

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
        for_print = s.segm(encoder, decoder, './images/input.jpg', command)
        # keras.backend.clear_session()
        for_print = 25
    return index(for_print=for_print)
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
