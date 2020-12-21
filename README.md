# Описание проекта

Проект создан для выделения силуэта людей на фотографиях, а также для генерации описания фотографии. Это происходит с помощью многослойных нейронных сетей, которые обучались на различных фото.

# How to start up with server 

1. Create vitrual env with the following command:

python3 -m venv venv/

2. Then activate it:

source venv/bin/activate

3. Load important packages:

pip install flask

pip install sys

pip install tensorflow

pip install segmen

pip install pickle

pip install pretrainedmodels

pip install torch

pip install torchvision

pip install typing

pip install PIL

pip install requests

pip install io

pip install matplotlib

4. Run server

python server.py

5. Find dunno (for me it is 'http://127.0.0.1:5000/'). And remember last number it as local

6. Install, LogIn/Register in this service https://ngrok.com. Make installation in current folder 

7. Run in terminal 
ngrok http %local%

8. From string starting with 'Forward' get an URL to use

