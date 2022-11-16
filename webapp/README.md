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
Once preprocessing has completed successfully use postman or curl to call the /predict endpoint

# Sample POST /preprocess request bodies
Examples use local file paths.  GCS paths may also be used, starting with gs://

Run entire preprocessing (can take 20-25 minutes)
Omitting "step" entirely will also run all steps.  
    { 
        "input_folder": "/Users/m152601/Downloads/readmits-last-two-admits-original-test-set/raw/",
        "output_folder": "/Users/m152601/Downloads/readmits-last-two-admits-original-test-set/processed-stepwise",
        "originals_folder": "/Users/m152601/IdeaProjects/readmit-stgnn-mayo-2/Mayo/ehr/processed/",
        "step": "all"
    }

Run as single step of preprocessing.  There are four steps: 1, 2, ,3, 4
Useful if calling from another application that is subject to network timeouts
Or if testing changes locally and want to skip to the part you are working.
Previous steps must have already been run on the same folder for this to work.
    { 
        "input_folder": "/Users/m152601/Downloads/readmits-last-two-admits-original-test-set/raw/",
        "output_folder": "/Users/m152601/Downloads/readmits-last-two-admits-original-test-set/processed-stepwise",
        "originals_folder": "/Users/m152601/IdeaProjects/readmit-stgnn-mayo-2/Mayo/ehr/processed/",
        "step": 2
    }

# Sample POST /predict request bodies
Examples use local file paths.  GCS paths may also be used, starting with gs://

Run predict and wait for completion before returning (may take 10 minutes or so)
    { "instances": [{ 
        "edge_ehr_files": ["/Users/m152601/Downloads/2022-11-14-test/readmit/processed/ehr_preprocessed_seq_by_day_gnn_appended.pkl"],
        "ehr_feature_files": ["/Users/m152601/Downloads/2022-11-14-test/readmit/processed/ehr_preprocessed_seq_by_day_tabnet_appended.pkl"],
        "demo_file": "/Users/m152601/Downloads/2022-11-14-test/readmit/processed/cohort_file_appended.csv",
        "output_folder": "/Users/m152601/Downloads/2022-11-14-test/readmit/results/",
        "asynchronous": false
        }],
        "parameters": {}
    }

Start predict but return before completion.   Check "status.json" in output folder to determine when complete.
    { "instances": [{ 
        "edge_ehr_files": ["/Users/m152601/Downloads/2022-11-14-test/readmit/processed/ehr_preprocessed_seq_by_day_gnn_appended.pkl"],
        "ehr_feature_files": ["/Users/m152601/Downloads/2022-11-14-test/readmit/processed/ehr_preprocessed_seq_by_day_tabnet_appended.pkl"],
        "demo_file": "/Users/m152601/Downloads/2022-11-14-test/readmit/processed/cohort_file_appended.csv",
        "output_folder": "/Users/m152601/Downloads/2022-11-14-test/readmit/results/",
        "asynchronous": true
        }],
        "parameters": {}
    }

