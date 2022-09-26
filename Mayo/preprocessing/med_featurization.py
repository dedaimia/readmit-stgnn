#!/usr/bin/env python3

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

def med_featurization(df, df_med):
    """
    df: selected cohort file, one hospitalization per row
    df_med: Medications file from SQL query - one medication recorded per row

    df_all: return file with one hospitalization per row with row containing all med therapeutic classes for that hospitalization
    """     
    
    
    df['ADMISSION_DTM'] = pd.to_datetime(df['ADMISSION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    df_med_status_sel = pd.read_csv('administered_status_counts_marked.csv')
    df_med_status_sel = df_med_status_sel.dropna(subset=['Include'])
    med_status_sel = df_med_status_sel.Status.values
    
    df_med = df_med.loc[(df_med.PATIENT_DK.isin(df.PATIENT_DK.unique())) & (df_med.ADMINISTERED_STATUS.isin(med_status_sel))] #only actually administered medications, not just prescribed, mathcign cohort patients
    
    print('Meds file length:', len(df_med))

    
    
    meds = pd.read_csv('med_selected.csv')
    meds = meds.dropna(subset=['MED_THERAPEUTIC_CLASS_DESCRIPTION'])
    sel_meds = meds.MED_THERAPEUTIC_CLASS_CODE.values
    print('selected medss:', len(meds))
    df_med = df_med.loc[df_med.MED_THERAPEUTIC_CLASS_CODE.isin(sel_meds)]  #keep pnly selected therapeutic classes of meds
    print('meds file length:', len(df_med))
    sys.stdout.flush()
    
    df_med['ADMINISTERED_DTM'] = pd.to_datetime(df_med['ADMINISTERED_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


    print(len(df), len(df_med))
    columns = [c for c in df.columns if c.startswith('Unnamed')==False]
    df_all = pd.DataFrame(columns = columns)
    pd.set_option('mode.chained_assignment', None)
    
    features = sel_meds
    for k in features:
       df_all[k] = 0
    
    idx = 0
    for i,j in df.iterrows():
       pid = df.at[i, 'PATIENT_DK']
       admit_dt = df.at[i, 'ADMISSION_DTM']
       discharge_dt = df.at[i, 'DISCHARGE_DTM']
       invalid =df.at[i, 'INVALID']
       
       if invalid==False:
           st = admit_dt
           day_no = 1
           while st+timedelta(hours=24) < discharge_dt+timedelta(hours=12):
               ed = st+timedelta(hours=24)
               temp = df_med.loc[(df_med.PATIENT_DK==pid) & (df_med.ADMINISTERED_DTM>=st)
                            & (df_med.ADMINISTERED_DTM<=ed)]
               df_all.at[idx, 'Day Nbr'] = day_no
               for c in columns:
                   df_all.at[idx, c] = df.at[i, c]
           
               if len(temp)>0:
                   print(i, 'Day:', day_no)
                   for c in temp.MED_THERAPEUTIC_CLASS_DESCRIPTION.unique():
                       temp2 = temp.loc[temp.MED_THERAPEUTIC_CLASS_DESCRIPTION==c]
                       df_all.at[idx, c] = len(temp2)
               idx+=1
               day_no+=1
               st = ed
          
           ed = discharge_dt
           temp = df_med.loc[(df_med.PATIENT_DK==pid) & (df_med.ADMINISTERED_DTM>=st)
                            & (df_med.ADMINISTERED_DTM<=ed)]
           df_all.at[idx, 'Day Nbr'] = day_no
           for c in columns:
               df_all.at[idx, c] = df.at[i, c]
           if len(temp)>0:
               print(i, 'Day:', day_no)
               for c in temp.MED_THERAPEUTIC_CLASS_DESCRIPTION.unique():
                       temp2 = temp.loc[temp.MED_THERAPEUTIC_CLASS_DESCRIPTION==c]
                       df_all.at[idx, c] = len(temp2)
           idx+=1
       sys.stdout.flush()         
       if i%100==0:
           print('Count:', i, idx)
    return df_all
    
