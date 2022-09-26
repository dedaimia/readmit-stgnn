import pandas as pd
import numpy as np
import pickle as pkl
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

import cohort_selection
from demographic_featurization import demo_featurization
from CPT_featurization import cpt_featurization
from ICD10_featurization import icd_featurization
from lab_featurization import lab_featurization
from med_featurization import med_featurization

header_raw = '../ehr/raw/'
header_proc = '../ehr/processed/'
def main():
    print('building cohort ....')
    sys.stdout.flush()
    df_hosp  = pd.read_csv(header_raw+'transfer_location_file.csv') ##transfer location file form SQL query
    df = cohort_selection.cohort_selection(df_hosp)
    df.to_csv(header_proc+'cohort_file.csv')

    print('cohort demographics ....')
    sys.stdout.flush()
    df = pd.read_csv(header_proc+'cohort_file.csv')
    df_demo = pd.read_csv(header_raw+'patient_demographics.csv') ## patient demographic file from SQL query
    df_demo = demo_featurization(df, df_demo)
    df_demo.to_csv(header_proc+'cohort_file_w_demo.csv')

    print('cohort CPT ....')
    sys.stdout.flush()
    df = pd.read_csv(header_proc+'cohort_file.csv')
    df_cpt= pd.read_csv(header_raw+'EDTWH_FACT_PROCEDURES.csv') ## procedures file from SQL query
    df_cpt = cpt_featurization(df, df_cpt)
    df_cpt.to_csv(header_proc+'cohort_file_w_cpt.csv')

    print('cohort ICD ....')
    sys.stdout.flush()
    df = pd.read_csv(header_proc+'cohort_file.csv')
    df_icd= pd.read_csv(header_raw+'EDTWH_FACT_DIAGNOSIS.csv') ## diagnoses file from SQL query
    df_icd = icd_featurization(df, df_icd)
    df_icd.to_csv(header_proc+'cohort_file_w_icd.csv')

    print('cohort Labs ....')
    sys.stdout.flush()
    df = pd.read_csv(header_proc+'cohort_file.csv')
    df_lab= pd.read_csv(header_raw+'EDTWH_FACT_LAB_TEST.csv') ## lab tests file from SQL query
    df_lab = lab_featurization(df, df_lab)
    df_lab.to_csv(header_proc+'cohort_file_w_lab.csv')

    print('cohort Medications ....')
    sys.stdout.flush()
    df = pd.read_csv(header_proc+'cohort_file.csv')
    df_med= pd.read_csv(header_raw+'EDTWH_FACT_MEDICATIONS.csv') ## medication file from SQL query
    df_med = med_featurization(df, df_med)
    df_med.to_csv(header_proc+'cohort_file_w_med.csv')

if __name__ == "__main__":
    main()
