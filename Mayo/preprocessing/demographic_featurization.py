import pandas as pd
import numpy as np
import pickle as pkl
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
print('all imported')
sys.stdout.flush()

def demo_featurization(df, df_demo):
    """
    df: selected cohort file, one hospitalization per row
    df_demo: Patient demographics file from SQL query - contians gender, race, ethnicity, and date of birth

    df_all: return file with one hospitalization per row with row containing demographics for that hospitalization
    """     
    df_demo = df_demo.drop_duplicates(subset=['PATIENT_DK']) #drop duplicate patients, will stop row explosion in the next step

    df = df.merge(df_demo[['PATIENT_DK', 'PATIENT_RACE_NAME', 'PATIENT_GENDER_NAME', 'PATIENT_ETHNICITY_NAME', 'PATIENT_BIRTH_DATE']], on = 'PATIENT_DK', how='left' )

    #RACE Preparation
    races = df.PATIENT_RACE_NAME.unique()
    df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({'Unknown':'UNKNOWN', 'Other': 'UNKNOWN', 
                                                    'Choose Not to Disclose': 'UNKNOWN',
                                                    'Unable to Provide': 'UNKNOWN', 
                                                            ' ': 'UNKNOWN'})
    for r in races:
        if type(r) == str and r.startswith('Asian'):
            df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({r:'Asian'})

    for r in races:
        if type(r) == str and ('black' in r.lower() or 'african' in r .lower()):
            df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({r:'Black'})
            
    for r in races:
        if type(r) == str and ('american indian' in r.lower() or 'alaskan' in r .lower()):
            df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({r:'American Indian/Alaskan Native'})

            
    for r in races:
        if type(r) == str and ('hawaii' in r.lower() or 'pacific' in r .lower() or 'samoan' in r.lower() or 'guam' in r.lower()):
            df['PATIENT_RACE_NAME'] = df['PATIENT_RACE_NAME'].replace({r:'Native Hawaiian/Pacific Islander'})

    df = df.fillna(value={'PATIENT_RACE_NAME':'UNKNOWN'}).copy()
    


    #ETHNIC Preparation
    races = df.PATIENT_ETHNICITY_NAME.unique()
    df['PATIENT_ETHNICITY_NAME'] = df['PATIENT_ETHNICITY_NAME'].replace({'Unknown':'UNKNOWN', 'Other': 'UNKNOWN', 
                                                    'Choose Not to Disclose': 'UNKNOWN',
                                                    'Unable to Provide': 'UNKNOWN'})

    for r in races:
        if type(r) == str and ('cuba' in r.lower() or 'mexic' in r.lower() or 'puerto' in r .lower() or 'central americ' in r.lower() or 'south americ' in r.lower() or 'spanish' in r.lower()):
            df['PATIENT_ETHNICITY_NAME'] = df['PATIENT_ETHNICITY_NAME'].replace({r:'Hispanic or Latino'})

    df = df.fillna(value={'PATIENT_ETHNICITY_NAME':'UNKNOWN'}).copy()       
    df.PATIENT_ETHNICITY_NAME.value_counts()

    ## AGE PREP
    df.ADMIT_DTM = pd.to_datetime(df.ADMIT_DTM, errors='coerce')
    df.PATIENT_BIRTH_DATE = pd.to_datetime(df.PATIENT_BIRTH_DATE, errors='coerce')

    df['PATIENT_AGE'] = (df['ADMIT_DTM'] - df['PATIENT_BIRTH_DATE']).astype('<m8[Y]')

    def to_bins(x):
        if np.isnan(x):
            return -1
        else:
            if x<100:
                return int(x/10)
            else:
                return 10
    df['PATIENT_AGE_BINNED'] = df['PATIENT_AGE'].apply(to_bins)
    df['PATIENT_AGE_BINNED'].value_counts()

    ## GENDER PREP - already set - only MALE FEMALE UNKNOWN
    return df