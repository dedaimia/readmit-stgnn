# Introduction 
This code base is a simple webapp wrapper around the 30-day readmission Mayo preprocessor (Mayo/preprocessing/main.py)
and the 30-day readmission model itself (train.py .... yes, you call train.py to run the trained model!)

# Prerequisites:
Python 3.9 and pip
gcloud command line tools if will be running locally

# Getting Started
Make a python virtual environment and install dependencies.  Recommend doing this at the root of your git clone.
    
    cd  <your git clone path>
    python -m venv ./venv
    source venv/bin/activate      
    pip install -r webapp/requirements.txt

Before running locally, log in to gcloud

    gcloud auth application-default login

Run gunicorn locally set to restart if code changes:

    cd webapp
    gunicorn --bind 0.0.0.0:5000 "app" -w 1 --threads 8 --timeout 0 --reload

Use postman or curl to call the /preprocess endpoint
