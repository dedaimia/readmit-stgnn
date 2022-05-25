#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
if __name__ == "__main__":
    print(f"Arguments count: {len(sys.argv)}")
    print(sys.argv)
    d, st_idx, ed_idx = sys.argv
    suffix = st_idx+'_'+ed_idx
    st_idx = int(st_idx)
    ed_idx = int(ed_idx)
    print(st_idx, ed_idx, type(st_idx), type(ed_idx), suffix)
    sys.stdout.flush()

    columns = ['PATIENT_DK', 'DIAGNOSIS_DTM', 'DIAGNOSIS_CODE'] 
    '''
    EDTWH_FACT_DIAGNOSIS.xlsx has multiple sheets of ICD10 codes and timestamps pulled for patients
    EDTWH_FACT_DIAGNOSIS.csv has all data combined into one large sheet
    '''
    df_icd = pd.read_csv('../bucket/Readmission_30days/Readmission/EDTWH_FACT_DIAGNOSIS.csv', usecols = columns) #all icd code recorded, one code per row
    print(len(df_icd), len(df_icd.PATIENT_DK.unique()))
    sys.stdout.flush()

    icd = pd.read_csv('../bucket/Readmission_30days/Amara/Readmission_Data/ICD10_Groups.csv') #ICD10 hierarchy
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


    ## use this as base file
    df = pd.read_csv('../bucket/Readmission_30days/Amara/Readmission_Data/Readmission_label_Demo_processed_w_xray_loc_expanded.csv') #base cohort file
    print(len(df),len(df_icd))
    sys.stdout.flush()
    ed_idx = min(ed_idx, len(df))
    df = df.iloc[st_idx:ed_idx].copy()
    df_icd = df_icd.loc[df_icd.PATIENT_DK.isin(df.PATIENT_DK_x.unique())]
    print(len(df),len(df_icd))
    sys.stdout.flush()
    df['ADMISSION_DTM'] = pd.to_datetime(df['ADMISSION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_icd['DIAGNOSIS_DTM'] = pd.to_datetime(df_icd['DIAGNOSIS_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


    ## CPT Featurization Temporal - by days
    df_all = pd.DataFrame(columns = df.columns)
    pd.set_option('mode.chained_assignment', None)
    idx=0
    features = icd.SUBGROUP.values
    for k in features:
        df_all[k] = 0
    for i,j in df.iterrows():
        pid = df.at[i, 'PATIENT_DK_x']
        admit_dt = df.at[i, 'ADMISSION_DTM']
        discharge_dt = df.at[i, 'DISCHARGE_DTM']
        invalid =df.at[i, 'INVALID']
        for c in df.columns:
            df_all.at[idx, c] = df.at[i, c]
        if invalid==False:
            st = admit_dt
            day_no = 1
            while st+timedelta(hours=24) < discharge_dt+timedelta(hours=12):
                ed = st+timedelta(hours=24)
                temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.DIAGNOSIS_DTM>=st)
                             & (df_icd.DIAGNOSIS_DTM<=ed)]
                df_all.at[idx, 'Shift Nbr'] = day_no
                if len(temp)>0:
                    print(i, 'Shift:', day_no)
                    temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
                    d = temp.DIAGNOSIS_CODE.value_counts()
                    temp2['SUBGROUPS'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
                    for ii, jj in temp2.iterrows():
                        df_all.at[idx, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'DIAGNOSIS_CODE']]
                idx+=1
                day_no+=1
                st = ed

            ed = discharge_dt
            temp = df_icd.loc[(df_icd.PATIENT_DK==pid) & (df_icd.DIAGNOSIS_DTM>=st)
                             & (df_icd.DIAGNOSIS_DTM<=ed)]
            df_all.at[idx, 'Shift Nbr'] = day_no
            if len(temp)>0:
                print(i, 'Shift:', day_no)
                temp['SUBGROUPS'] = temp.DIAGNOSIS_CODE.apply(find_group)           
                temp2 = temp.drop_duplicates(subset=['DIAGNOSIS_CODE'])
                d = temp.DIAGNOSIS_CODE.value_counts()
                temp2['SUBGROUPS'] = temp2.DIAGNOSIS_CODE.apply(find_group) 
                for ii, jj in temp2.iterrows():
                    df_all.at[idx, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'DIAGNOSIS_CODE']]
            print(i, 'Shifts:', day_no)
            idx+=1         
        if i%100==0:
            print('Count:', i, idx)
        if i%1000==0:
            df_all.to_csv('../bucket/Readmission_30days/Amara/Readmission_Data/Readmission_label_Demo_processed_w_xray_loc_expanded_ICD10_daily_'+suffix+'.csv')
            sys.stdout.flush()

    df_all.to_csv('../bucket/Readmission_30days/Amara/Readmission_Data/Readmission_label_Demo_processed_w_xray_loc_expanded_ICD10_daily_'+suffix+'.csv')