# 
# Flask web app microservice wrapping the 30-day readmission model code
#

from flask import Flask
from flask import request, Response
import sys
import json
from types import SimpleNamespace
import os
import pickle
import pandas as pd

# add the code repo folder to your python module search path, so imports will work:
script_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(script_path, "..")
sys.path.append(code_path)

os.environ['DGLBACKEND'] = "pytorch"

import train
import Mayo.preprocessing
from webapp_utils import defaultInferenceArgs


app = Flask(__name__)

@app.route("/")
def instructions():
    return ('Call POST /preprocess for preprocessing'
        'Then call POST /predict to build the graph and preduct'
        'GET /status is to make vertex AI predict'
        'preprocess just expects "input_folder" and "output_folder" in the input json'
        'both can be either local or gs:// paths.'
        'Predict is expected to be called via vertex AI predict so expects input in vertex AIs format '
        )

@app.route("/preprocess", methods = ["POST"])
def run_preprocess():
    data = request.json

    # ensure trailing slashes on both folders
    input_folder = data['input_folder'].rstrip('/') + '/'
    output_folder = data['output_folder'].rstrip('/') + '/'

    # hard code expected file names (inside input folder)
    preprocessor_args = SimpleNamespace()
    preprocessor_args.hosp_file = "hosp.csv"
    preprocessor_args.demo_file = "dem.csv"
    preprocessor_args.cpt_file = "cpt.csv"
    preprocessor_args.icd_file = "icd.csv"
    preprocessor_args.lab_file = "lab.csv"
    preprocessor_args.vit_file = "vit.csv"

    # call the preprocessor
    preprocess.preprocess(preprocessor_args, input_folder, output_folder)
    return "<p>Preprocessing Complete<p>"
    
@app.route('/predict', methods=['POST'])
def predict():

    # request should contain demos filepath, edge_ehr_file path, ehr_feature_file path and output_path
    # Todo: nicer error message if data doesn't contain expected fields
    data = request.json['instances'][0]

    # ensure trailing slashes on both folders
    results_folder = data['output_folder'].rstrip('/') + '/'

    # Create graph application args
    infer_args = defaultInferenceArgs(edge_ehr_files=data['edge_ehr_files'], 
    ehr_feature_files=data['ehr_feature_files'],
    demo_file=data['demo_file']
    )

    # apply graph
    train.main(infer_args)

    with open(os.path.join(infer_args.save_dir, 'test_predictions.pkl'), 'rb') as f:
        predictions = pd.read_pickle(f)
    
    predictions.to_csv(os.path.join(results_folder, 'readmit-preds.csv'))

    return [{'predictions': 'Inference performed succesfully, results logged to Big Query.'}]

@app.route('/status', methods=['GET'])
def health_check():
	status_code = Response(status=200)
	return status_code
