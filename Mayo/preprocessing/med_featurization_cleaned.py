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

if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    print(sys.argv)
    d, st_idx, ed_idx = sys.argv
    suffix = st_idx+'_'+ed_idx
    st_idx = int(st_idx)
    ed_idx = int(ed_idx)
    print(st_idx, ed_idx, type(st_idx), type(ed_idx), suffix)
    sys.stdout.flush()

 

       
    df = pd.read_csv('/media/Datacenter_storage/Readmission/Amara/Readmission_label_Demo_processed_w_first_xray_loc_expanded_w_MRN.csv', low_memory=False)
    ed_idx = min(ed_idx, len(df))
    df = df.iloc[st_idx:ed_idx].copy()
    print(len(df))
    sys.stdout.flush()
    
    
    df['ADMISSION_DTM'] = pd.to_datetime(df['ADMISSION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    
    '''
    original file should be xlsx with mulitple sheets. convert it to csv with one large sheet.
    Medication_administered_48h_cohort_given.csv filtered such that PATIENT_DK matches our cohort and ADMINSITERED_STATUS matches the ones marked with Y in administered_status_counts_marked.csv 
    '''
    df_med = pd.read_csv('Medication_administered_48h_cohort_given.csv') #only actually administered medications, not just prescribed
    
    print('Meds file length:', len(df_med))

    
    
    meds = pd.read_csv('med_selected.csv')
    meds = meds.dropna(subset=['MED_THERAPEUTIC_CLASS_DESCRIPTION'])
    sel_meds = meds.MED_THERAPEUTIC_CLASS_CODE.values
    print('selected medss:', len(meds))
    df_med = df_med.loc[df_med.MED_THERAPEUTIC_CLASS_CODE.isin(sel_meds)]
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
       pid = df.at[i, 'PATIENT_DK_x']
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
       if i%1000==0:
           df_all.to_csv('Readmission_label_Demo_processed_w_xray_loc_expanded_48h_Med_daily_'+suffix+'.csv')  
       
    df_all.to_csv('Readmission_label_Demo_processed_w_xray_loc_expanded_48h_Med_daily_'+suffix+'.csv') 
    
