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
import base64
import threading
import traceback
import logging
import gcsfs
from datetime import datetime

# add the code repo folder to your python module search path, so imports will work:
script_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(script_path, "..")
sys.path.append(code_path)


os.environ['DGLBACKEND'] = "pytorch"

import train
import Mayo.preprocessing.main as preprocess
from webapp_utils import defaultInferenceArgs


app = Flask(__name__)

# pick the appropriate open function depending on whether 
# path is local or google cloud storage
# Todo:  this same code is in Mayo/preprocessing/main.py, consider moving to a utils file
cloud_storage_fs = None
def open_local_or_gs(path, flags, mode=0o777):
    if (path.startswith("gs:")):
        global cloud_storage_fs
        if (cloud_storage_fs is None):
            cloud_storage_fs = gcsfs.GCSFileSystem()
        return cloud_storage_fs.open(path, flags, mode)
    else:
        return open(path, flags, mode)

STATUS_COLS = ["Step", "Start Time", "End Time", "Status", "Output"]

@app.route("/")
def instructions():
    return ('Call POST /preprocess for preprocessing'
        'Then call POST /predict to build the graph and preduct'
        'GET /status is to make vertex AI predict happy'
        'preprocess expects "input_folder", "output_folder", and "originals_folder" in the input json'
        'all can be either local or gs:// paths.'
        'Predict is expected to be called via vertex AI predict so expects input in vertex AIs format '
        )

@app.route("/preprocess", methods = ["POST"])
def run_preprocess():
    data = request.json

    # ensure trailing slashes on both folders
    input_folder = data['input_folder'].rstrip('/') + '/'
    output_folder = data['output_folder'].rstrip('/') + '/'
    originals_folder = data['originals_folder'].rstrip('/') + '/'

    step = 'all'
    if('step' in data):
        step = data['step']

    # hard code expected file names (inside input folder)
    preprocessor_args = SimpleNamespace()
    preprocessor_args.input_folder = input_folder
    preprocessor_args.output_folder = output_folder 
    preprocessor_args.orig_folder = originals_folder
    preprocessor_args.hosp_file = "hosp.csv"
    preprocessor_args.demo_file = "dem.csv"
    preprocessor_args.cpt_file = "cpt.csv"
    preprocessor_args.icd_file = "icd.csv"
    preprocessor_args.lab_file = "lab.csv"
    preprocessor_args.med_file = "med.csv"
    preprocessor_args.step = step

    # call the preprocessor
    preprocess.main(preprocessor_args)
    return f"<p>Preprocessing Complete step {step} <p>"
    
@app.route('/predict', methods=['POST'])
def predict():

    # request should contain demos filepath, edge_ehr_file path, ehr_feature_file path and output_path
    # Todo: nicer error message if data doesn't contain expected fields
    data = request.json['instances'][0]

    # ensure trailing slashes on both folders
    results_folder = data['output_folder'].rstrip('/') + '/'

    # write initial status:
    status_file = os.path.join(results_folder, "status.csv")
    status_df = pd.DataFrame(data=None, index=None, columns=STATUS_COLS, dtype=None, copy=None)
    status_df.loc[0] = ["readmit-predict", datetime.now(), None, "Running", None]
    status_df.to_csv(status_file)

    # Create graph application args
    infer_args = defaultInferenceArgs(edge_ehr_files=data['edge_ehr_files'], 
    ehr_feature_files=data['ehr_feature_files'],
    demo_file=data['demo_file']
    )

    # decide whether we're running asynchronously (returning immediately and continuing processing, or synchronously)
    asynchronous = False
    if 'asynchronous' in data:
        asynchronous = data['asynchronous']

    if(asynchronous):
        thread = threading.Thread(target=predict_and_convert_results, kwargs={
            "results_folder": results_folder, 
            "status_file": status_file, 
            "status_df": status_df, 
            "infer_args": infer_args})
        thread.start()
        return [{'predictions': f'Inference started.  Check {status_file} for status.'}]
    else: 
        predict_and_convert_results(results_folder, status_file, status_df, infer_args)
        return [{'predictions': 'Inference performed succesfully, results logged to Big Query.'}]

def predict_and_convert_results(results_folder, status_file, status_df,  infer_args):
    # apply graph
    try:
        train.main(infer_args)

        with open(os.path.join(infer_args.save_dir, 'test_predictions.pkl'), 'rb') as f:
            predictions = pd.DataFrame(pd.read_pickle(f)).set_index('node_indices')
        with open(os.path.join(code_path, 'node_mapping.pkl'), 'rb') as f:
            maps = pd.DataFrame(list(pd.read_pickle(f)))
        final_preds = pd.mrge(predictions, maps, left_index=True, right_index=True)

        final_preds.to_csv(os.path.join(results_folder, 'readmit-preds.csv'))

        status_df.loc[0, ['End Time', 'Status', 'Output']] = [datetime.now(), "Success", None]
   
    except Exception as e:
        logging.exception('unexpected error in readmit predict ')
        status_df.loc[0, ['End Time', 'Status', 'Output']] = [datetime.now(), "Failed", "See Logs"]

    status_df.to_csv(status_file)

@app.route('/status', methods=['GET'])
def health_check():
	status_code = Response(status=200)
	return status_code
