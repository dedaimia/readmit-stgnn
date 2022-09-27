#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def icd_featurization(df, df_icd):
    """
    df: selected cohort file, one hospitalization per row
    df_icd: EDTWH_FACT_DIAGNOSES file from SQL query - one icd code recorded per row

    df_all: return file with one hospitalization per row with row containing all icd SUBGROUP for that hospitalization
    """     

    icd = pd.read_csv(ICD10_Groups.csv') #ICD10 hierarchy
    ICD = []
    k = 0
    for i,j in icd.iterrows():
        d = dict()
    #     print(icd.at[i, 'START_IDX'], icd.at[i, 'END_IDX'])
        d['letter'] = icd.at[i, 'START_IDX'][0]
        d['start'] = (icd.at[i, 'START_IDX'][1:])
        d['end'] = (icd.at[i, 'END_IDX'][1:])
        d['group'] = icd.at[i, 'SUBGROUP']
        ICD.append(d)

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


    
    df['ADMISSION_DTM'] = pd.to_datetime(df['ADMISSION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_icd['DIAGNOSIS_DTM'] = pd.to_datetime(df_icd['DIAGNOSIS_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


    
    df_all = pd.DataFrame(columns = df.columns)
    pd.set_option('mode.chained_assignment', None)
    idx=0
    features = icd.SUBGROUP.values
    for k in features:
        df_all[k] = 0
    for i,j in df.iterrows():
        pid = df.at[i, 'PATIENT_DK']
        admit_dt = df.at[i, 'ADMISSION_DTM']
        discharge_dt = df.at[i, 'DISCHARGE_DTM']
        invalid =df.at[i, 'INVALID']
        for c in df.columns:
            df_all.at[idx, c] = df.at[i, c]

        st = admit_dt
        day_no = 1
        while st+timedelta(hours=24) < discharge_dt+timedelta(hours=12):
            ed = st+timedelta(hours=24)
            temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.DIAGNOSIS_DTM>=st)
                            & (df_icd.DIAGNOSIS_DTM<=ed)]
            df_all.at[idx, 'Day_Number'] = day_no
            if len(temp)>0:
                print(i, 'Shift:', day_no)
                temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
                d = temp.DIAGNOSIS_CODE.value_counts()
                temp2['SUBGROUP'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
                for ii, jj in temp2.iterrows():
                    df_all.at[idx, temp2.at[ii, 'SUBGROUP']] = d[temp2.at[ii, 'DIAGNOSIS_CODE']]
            idx+=1
            day_no+=1
            st = ed

        ed = discharge_dt
        temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.DIAGNOSIS_DTM>=st)
                            & (df_icd.DIAGNOSIS_DTM<=ed)]
        df_all.at[idx, 'Day_Number'] = day_no
        if len(temp)>0:
            print(i, 'Shift:', day_no)
            temp['SUBGROUP'] = temp.DIAGNOSIS_CODE.apply(find_group)           
            temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
            d = temp.DIAGNOSIS_CODE.value_counts()
            temp2['SUBGROUP'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
            for ii, jj in temp2.iterrows():
                df_all.at[idx, temp2.at[ii, 'SUBGROUP']] = d[temp2.at[ii, 'DIAGNOSIS_CODE']]
        print(i, 'Days:', day_no)
        idx+=1         
        if i%100==0:
            print('Count:', i, idx)
    return df_all