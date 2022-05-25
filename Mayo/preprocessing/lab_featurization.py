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


    
    def map_value(val, val_range):
       #print(val, val_range)
        if type(val_range)==str and np.isnan(val)==False:
            if len(val_range.split('-')) == 2:
                lower = float(val_range.split('-')[0])
                upper = float(val_range.split('-')[1])
                if val >= lower and val <= upper:
                    ans = 'NORMAL'
                else:
                   ans = 'ABNORMAL'
            elif '>' in val_range:
                lower = float(''.join(c for c in val_range if (c.isdigit() or c=='.')))
                if val >= lower:
                    ans = 'NORMAL'
                else:
                    ans = 'ABNORMAL'
            elif '<' in val_range:
                upper = float(''.join(c for c in val_range if (c.isdigit() or c=='.')))
                if val <= upper:
                    ans = 'NORMAL'
                else:
                    ans = 'ABNORMAL'
            else:
                ans = 'UNKNOWN'
        elif type(val_range)==str and type(val)==str:
            if val==val_range:
                ans = 'NORMAL'
            else:
                ans = 'ABNORMAL'
        else:
                ans = 'UNKNOWN'
        return ans
    

       
    df = pd.read_csv('Readmission_label_Demo_processed_w_first_xray_loc_expanded_w_MRN.csv', usecols=cols)
    ed_idx = min(ed_idx, len(df))
    df = df.iloc[st_idx:ed_idx].copy()
    df = df.loc[df['Hours in hospital']>=48]
    print(len(df))
    sys.stdout.flush()
    
    
    df['ADMISSION_DTM'] = pd.to_datetime(df['ADMISSION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    
    
    columns = ['PATIENT_DK',
       'LOCATION_FACILITY_DK', 'LOCATION_DEPARTMENT_DK', 'LOCATION_DK',
       'LOCATION_SITE_NAME',  'LAB_ORDER_DTM',  'LAB_COLLECTION_DTM', 'LAB_RESULT_DTM', 
       'LAB_ABNORMAL_CODES_DK', 'LAB_ACCESSION_NBR', 
       'LAB_COMMENTS', 'LAB_STATUS', 'LAB_SERVICE_TYPE_CODE', 'LAB_TYPE_CODE',
       'LAB_SUBTYPE_CODE', 'NORMAL_RANGE_TXT', 'UNIT_OF_MEASURE_TXT',
       'RESULT_TXT', 'RESULT_VAL', 'LAB_COUNT']
    
    '''
    EDTWH_FACT_LAB_TEST.xlsx has multiple sheets of lab test and timestamps pulled for patients
    EDTWH_FACT_LAB_TEST_ALL.csv has all data combined into one large sheet
    EDTWH_FACT_LAB_TEST_ALL_cohort_patients_in_hospital_selected_labs_only.csv has been filtered based on PATIENT_DK matching our cohort and LAB_SUBTYPE_CODE to match seelcted labs only
    '''
    df_lab = pd.read_csv('EDTWH_FACT_LAB_TEST_ALL_cohort_patients_in_hospital_selected_labs_only.csv', usecols=columns)

    print('labs file length:', len(df_lab))
#    df_lab = df_lab.loc[df_lab.PATIENT_DK.isin(df.PATIENT_DK_x.values)]
#    print('labs file length:', len(df_lab))
    
    df_sel_labs = pd.read_csv('selected_labs_expanded_BP.csv', header=None)
    labs = df_sel_labs[1].values[1:]
    print('selected labs:', len(labs))
#    df_lab = df_lab.loc[df_lab.LAB_SUBTYPE_CODE.isin(sel_labs)]
#    print('labs file length:', len(df_lab))
    sys.stdout.flush()
    
    df_lab['LAB_COLLECTION_DTM'] = pd.to_datetime(df_lab['LAB_COLLECTION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    
    df_lab = df_lab.loc[df_lab.PATIENT_DK.isin(df.PATIENT_DK_x.unique())]
    print('labs file length:', len(df_lab))
    sys.stdout.flush()

#    ## LAB Featurization
#    print(len(df), len(df_lab))
#    columns = [c for c in df.columns if c.startswith('Unnamed')==False]
#    df_all = pd.DataFrame(columns = columns)
#    pd.set_option('mode.chained_assignment', None)
#
#    features =labs
#    for k in features:
#     df_all[k] = 'UNKNOWN'
#
#    idx = 0
#    for i,j in df.iterrows():
#     pid = df.at[i, 'PATIENT_DK_x']
#     admit = df.at[i, 'ADMISSION_DTM']
#     discharge = df.at[i, 'DISCHARGE_DTM']
#     invalid =df.at[i, 'INVALID']
#     for c in columns:
#         df_all.at[idx, c] = df.at[i, c]
#     if invalid==False:
#         temp = df_lab.loc[(df_lab.PATIENT_DK==pid) & (df_lab.LAB_COLLECTION_DTM>=admit)
#                      & (df_lab.LAB_COLLECTION_DTM<=discharge)]
#         if len(temp)>0:
#             for c in temp.LAB_SUBTYPE_CODE.unique():
#                 temp2 = temp.loc[temp.LAB_SUBTYPE_CODE==c]
#                 temp2 = temp2.sort_values('LAB_COLLECTION_DTM')
#                 df_all.at[idx, c] = map_value(temp2.at[temp2.index[-1], 'RESULT_VAL'], temp2.at[temp2.index[-1], 'NORMAL_RANGE_TXT'])        
#     if idx%100==0:
#         print('Count:', idx, i, idx)
#     if idx%1000==0:
#         df_all.to_csv('Readmission_label_Demo_processed_w_xray_loc_expanded_48h_Lab_'+suffix+'.csv')
#     idx+=1
#     sys.stdout.flush()
#     
#    df_all.to_csv('Readmission_label_Demo_processed_w_xray_loc_expanded_48h_Lab_'+suffix+'.csv')   
           
     ## LAB Featurization Temporal - by days
    print(len(df), len(df_lab))
    columns = [c for c in df.columns if c.startswith('Unnamed')==False]
    df_all = pd.DataFrame(columns = columns)
    pd.set_option('mode.chained_assignment', None)
    
    features =labs
    for k in features:
       df_all[k] = 'UNKNOWN'
    
    idx = 0
    for i,j in df.iterrows():
       pid = df.at[i, 'PATIENT_DK_x']
       admit_dt = df.at[i, 'ADMISSION_DTM']
       discharge_dt = df.at[i, 'DISCHARGE_DTM']
       invalid =df.at[i, 'INVALID']
       for c in columns:
           df_all.at[idx, c] = df.at[i, c]
       if invalid==False:
           st = admit_dt
           day_no = 1
           while st+timedelta(hours=24) < discharge_dt+timedelta(hours=12):
               ed = st+timedelta(hours=24)
               temp = df_lab.loc[(df_lab.PATIENT_DK==pid) & (df_lab.LAB_COLLECTION_DTM>=st)
                            & (df_lab.LAB_COLLECTION_DTM<=ed)]
               df_all.at[idx, 'Day Nbr'] = day_no
               if len(temp)>0:
                   #print(i, 'Day:', day_no)
                   for c in temp.LAB_SUBTYPE_CODE.unique():
                       temp2 = temp.loc[temp.LAB_SUBTYPE_CODE==c]
                       temp2 = temp2.sort_values('LAB_COLLECTION_DTM')
                       df_all.at[idx, c] = map_value(temp2.at[temp2.index[-1], 'RESULT_VAL'], temp2.at[temp2.index[-1], 'NORMAL_RANGE_TXT'])
               idx+=1
               day_no+=1
               st = ed
          
           ed = discharge_dt
           temp = df_lab.loc[(df_lab.PATIENT_DK==pid) & (df_lab.LAB_COLLECTION_DTM>=st)
                            & (df_lab.LAB_COLLECTION_DTM<=ed)]
           df_all.at[idx, 'Day Nbr'] = day_no
           if len(temp)>0:
               #print(i, 'Day:', day_no)
               for c in temp.LAB_SUBTYPE_CODE.unique():
                       temp2 = temp.loc[temp.LAB_SUBTYPE_CODE==c]
                       temp2 = temp2.sort_values('LAB_COLLECTION_DTM')
                       df_all.at[idx, c] = map_value(temp2.at[temp2.index[-1], 'RESULT_VAL'], temp2.at[temp2.index[-1], 'NORMAL_RANGE_TXT'])
           idx+=1
       #print(i)          
       if i%100==0:
           print('Count:', i, idx)
       if i%1000==0:
           df_all.to_csv('Readmission_label_Demo_processed_w_xray_loc_expanded_48h_Lab_daily_'+suffix+'.csv')  
       sys.stdout.flush()
    df_all.to_csv('Readmission_label_Demo_processed_w_xray_loc_expanded_48h_Lab_daily_'+suffix+'.csv')  
