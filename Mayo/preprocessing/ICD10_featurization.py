#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

script_path = os.path.dirname(os.path.realpath(__file__))

icd = pd.read_csv(os.path.join(script_path, 'ICD10_Groups.csv')) #ICD10 hierarchy

def find_group(code):
    global icd
    group = ''
    letter = code[0]
    number = code[1:].split('.')[0]
    if number.isnumeric():
        number = (float(number))
        icd_sel = icd.loc[icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.isnumeric()) & (icd_sel.END_IDX.str.isnumeric())].copy()
        icd_sel = icd_sel.loc[ (icd_sel.START_IDX.astype(float)<=number) & (icd_sel.END_IDX.astype(float)>=number)].copy()
        if len(icd_sel)>0:
            group = icd_sel.at[icd_sel.index[0], 'SUBGROUP']
        else:
            group = 'UNKNOWN'
    else:
        icd_sel = icd.loc[icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.isnumeric()==False) & (icd_sel.END_IDX.str.isnumeric()==False)].copy()
        numheader = number[:-1]
        icd_sel = icd_sel.loc[(icd_sel.START_IDX.str.startswith(numheader)) & (icd_sel.END_IDX.str.startswith(numheader))].copy()
        if len(icd_sel)>0:
            group = icd_sel.at[icd_sel.index[0], 'SUBGROUP']
        else:
            group = 'UNKNOWN'
    return group
    
def icd_featurization(df, df_icd):
    """
    df: selected cohort file, one hospitalization per row
    df_icd: EDTWH_FACT_DIAGNOSES file from SQL query - one icd code recorded per row

    df_all: return file with one hospitalization per row with row containing all icd SUBGROUP for that hospitalization
    """
    
    df['ADMIT_DTM'] = pd.to_datetime(df['ADMIT_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_icd['DIAGNOSIS_DTM'] = pd.to_datetime(df_icd['DIAGNOSIS_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


    columns = [c for c in df.columns if c.startswith('Unnamed')==False]
    df_all = pd.DataFrame(columns = df.columns)
    pd.set_option('mode.chained_assignment', None)
    idx=0
    features = icd.SUBGROUP.values
    for k in features:
        df_all[k] = 0
    for i,j in df.iterrows():
        pid = df.at[i, 'PATIENT_DK']
        admit_dt = df.at[i, 'ADMIT_DTM']
        discharge_dt = df.at[i, 'DISCHARGE_DTM']

        st = admit_dt
        day_no = 1
        while st+timedelta(hours=24) < discharge_dt+timedelta(hours=12):
            ed = st+timedelta(hours=24)
            temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.DIAGNOSIS_DTM>=st)
                            & (df_icd.DIAGNOSIS_DTM<=ed)]
            for c in df.columns:
                df_all.at[idx, c] = df.at[i, c]
            df_all.at[idx, 'Day_Number'] = day_no
            df_all.at[idx, 'Date'] = admit_dt+timedelta(days=day_no-1)
            if len(temp)>0:
                # print(i, 'Day:', day_no)
                temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
                d = temp.DIAGNOSIS_CODE.value_counts()
                temp2['SUBGROUP'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
                for s in temp2.SUBGROUP.unique():
                    df_all.at[idx, s] = len(temp.loc[temp.DIAGNOSIS_CODE.isin(temp2.loc[temp2.SUBGROUP==s]['DIAGNOSIS_CODE'].unique())])
            idx+=1
            day_no+=1
            st = ed

        ed = discharge_dt
        temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.DIAGNOSIS_DTM>=st)
                            & (df_icd.DIAGNOSIS_DTM<=ed)]
        for c in df.columns:
            df_all.at[idx, c] = df.at[i, c]
        df_all.at[idx, 'Day_Number'] = day_no
        df_all.at[idx, 'Date'] = admit_dt+timedelta(days=day_no-1)
        if len(temp)>0:
            # print(i, 'Day:', day_no)
            temp['SUBGROUP'] = temp.DIAGNOSIS_CODE.apply(find_group)           
            temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
            d = temp.DIAGNOSIS_CODE.value_counts()
            temp2['SUBGROUP'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
            for s in temp2.SUBGROUP.unique():
                df_all.at[idx, s] = len(temp.loc[temp.DIAGNOSIS_CODE.isin(temp2.loc[temp2.SUBGROUP==s]['DIAGNOSIS_CODE'].unique())])
        print(i, 'Days:', day_no)
        idx+=1         
        if i%100==0:
            print('Count:', i, idx)
    return df_all