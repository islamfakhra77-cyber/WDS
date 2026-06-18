Weapon Detection System
This project uses YOLOv8 and Django to detect weapons (such as knives) in images and video streams. It provides a web interface to upload images and view detection results.

Features
Real-time weapon detection using YOLOv8.
Django web framework for the user interface.
Image upload and result visualization.
Deep Learning model integration.
Installation
Clone the repository:
git clone https://github.com/islamfakhra77-cyber/WDS.gitcd WDS
Create a virtual environment (venv):
bash

python -m venv venv
For Windows:
bash

venv\Scripts\activate
For Mac/Linux:
bash

source venv/bin/activate
Install requirements:
bash

pip install -r requirements.txt
Run the Django server:
bash

python manage.py runserver
Open your browser and go to http://127.0.0.1:8000/
Project Structure
detector/: Contains the Django app logic.
models/: Stores trained model files (YOLOv8).
dataset/: Contains training data.
train_model.py: Script for training the model.
manage.py: Django management script.
Usage
Run the server using python manage.py runserver.
Open the home page.
Upload an image containing a potential weapon.
View the detection results.
