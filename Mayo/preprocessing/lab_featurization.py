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

   
script_path = os.path.dirname(os.path.realpath(__file__))

def lab_featurization(df, df_lab):
    """
    df: selected cohort file, one hospitalization per row
    df_lab: EDTWH_FACT_LAB_TEST file from SQL query - one lab test recorded per row

    df_all: return file with one hospitalization per row with row containing all labs mapped to ABNOMRAL/NORMAL for that hospitalization
    """     
      
    
    df['ADMIT_DTM'] = pd.to_datetime(df['ADMIT_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    

    print('labs file length:', len(df_lab))
    df_lab = df_lab.loc[df_lab.PATIENT_DK.isin(df.PATIENT_DK.values)]  # keep only lab tests belonging to cohort patients

    
    df_sel_labs = pd.read_csv(os.path.join(script_path, 'selected_labs_expanded_BP.csv'), header=None)
    labs = df_sel_labs[1].values[1:]
    print('selected labs:', len(labs))
    df_lab = df_lab.loc[df_lab.LAB_SUBTYPE_CODE.isin(labs)]   # keep only lab tests of interest - recorded in selected_labs_expanded_BP
    print('labs file length:', len(df_lab))
    sys.stdout.flush()
    
    df_lab['LAB_COLLECTION_DTM'] = pd.to_datetime(df_lab['LAB_COLLECTION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    
    
    df_lab = df_lab.loc[df_lab.PATIENT_DK.isin(df.PATIENT_DK.unique())]
    print('labs file length:', len(df_lab))
    sys.stdout.flush()

    print(len(df), len(df_lab))
    columns = [c for c in df.columns if c.startswith('Unnamed')==False]
    df_all = pd.DataFrame(columns = columns)
    pd.set_option('mode.chained_assignment', None)
    
    def map_value(val, val_range):
        if type(val) is str and (val.isnumeric() or val.replace('.', '').isnumeric()):
            val = float(val)
        if type(val_range)==str and type(val) is not str and np.isnan(val)==False:
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

    features =labs
    for k in features:
       df_all[k] = 'UNKNOWN'
    columns = [c for c in df.columns if c.startswith('Unnamed')==False]
    pd.set_option('mode.chained_assignment', None)
    idx = 0
    for i,j in df.iterrows():
        pid = df.at[i, 'PATIENT_DK']
        admit_dt = df.at[i, 'ADMIT_DTM']
        discharge_dt = df.at[i, 'DISCHARGE_DTM']
        st = admit_dt
        day_no = 1
        while st+timedelta(hours=24) < discharge_dt+timedelta(hours=12):
            ed = st+timedelta(hours=24)
            temp = df_lab.loc[(df_lab.PATIENT_DK==pid) & (df_lab.LAB_COLLECTION_DTM>=st)
                        & (df_lab.LAB_COLLECTION_DTM<=ed)]
            for c in columns:
                df_all.at[idx, c] = df.at[i, c]
            df_all.at[idx, 'Day_Number'] = day_no
            df_all.at[idx, 'Date'] = admit_dt+timedelta(days=day_no-1)
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
        for c in columns:
            df_all.at[idx, c] = df.at[i, c]
        df_all.at[idx, 'Day_Number'] = day_no
        df_all.at[idx, 'Date'] = admit_dt+timedelta(days=day_no-1)
        if len(temp)>0:
            #print(i, 'Day:', day_no)
            for c in temp.LAB_SUBTYPE_CODE.unique():
                    temp2 = temp.loc[temp.LAB_SUBTYPE_CODE==c]
                    temp2 = temp2.sort_values('LAB_COLLECTION_DTM')
                    df_all.at[idx, c] = map_value(temp2.at[temp2.index[-1], 'RESULT_VAL'], temp2.at[temp2.index[-1], 'NORMAL_RANGE_TXT'])
        idx+=1       
        if i%100==0:
            print('Count:', i, idx)
    return df_all
