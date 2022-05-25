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

    '''
    EDTWH_FACT_PROCEDURES.xlsx has multiple sheets of CPT codes and timestamps pulled for patients
    EDTWH_FACT_PROCEDURES_ALL.csv has all data combined into one large sheet
    EDTWH_FACT_PROCEDURES_ALL_selected_patients.csv has been filtered based on PATIENT_DK matching our cohort
    '''
    df_cpt = pd.read_csv('../bucket/Readmission_30days/Readmission/EDTWH_FACT_PROCEDURES_ALL_selected_pateints.csv', error_bad_lines=False)  ## all cpt recorded /one cpt code per row

    print(len(df_cpt), len(df_cpt.PATIENT_DK.unique()))
    sys.stdout.flush()

    dfcpt_groups = pd.read_csv("EHR2VEC/COVID_code/CPT_group_structure.csv") #cpt code structure
    print(len(dfcpt_groups))
    sys.stdout.flush()


    def to_cpt_group(x):
        out=None
        if type(x)==str and x.isnumeric():
            x = int(x)
            temp = dfcpt_groups.loc[(dfcpt_groups['Low']<=x) & (dfcpt_groups['High']>=x) & (dfcpt_groups['Modifier'].isna())]
            if len(temp)>0:
                out = temp.at[temp.index[0], 'Subgroup']
        elif type(x) == str and x[:-1].isnumeric():
            m = x[-1]
            x = int(x[:-1])
            temp = dfcpt_groups.loc[(dfcpt_groups['Low']<=x) & (dfcpt_groups['High']>=x) & (dfcpt_groups['Modifier']==m)]
            if len(temp)>0:
                out = temp.at[temp.index[0], 'Subgroup']
        return out


    ## use this as base file
    df = pd.read_csv('../bucket/Readmission_30days/Amara/Readmission_Data/Readmission_label_Demo_processed_w_xray_loc_expanded.csv') ## base cohort file
    print(len(df),len(df_cpt))
    sys.stdout.flush()
    ed_idx = min(ed_idx, len(df))
    df = df.iloc[st_idx:ed_idx].copy()
    df_cpt = df_cpt.loc[df_cpt.PATIENT_DK.isin(df.PATIENT_DK_x.unique())]
    print(len(df),len(df_cpt))
    sys.stdout.flush()
    df['ADMISSION_DTM'] = pd.to_datetime(df['ADMISSION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_cpt['PROCEDURE_DTM'] = pd.to_datetime(df_cpt['PROCEDURE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


    ## CPT Featurization Temporal - by days
    features = dfcpt_groups.Subgroup.unique()
    pd.set_option('mode.chained_assignment', None)
    df_all = pd.DataFrame(columns = columns)
    idx = 0
    for k in features:
        df_all[k] = 0
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
                temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.PROCEDURE_DTM>=st)
                             & (df_cpt.PROCEDURE_DTM<=ed)]
                df_all.at[idx, 'Shift Nbr'] = day_no
                temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
                d = temp.PROCEDURE_CODE.value_counts()
                if len(temp)>0:
                    temp2['SUBGROUPS'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
                    for ii, jj in temp2.iterrows():
                        df_all.at[idx, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'PROCEDURE_CODE']]
                idx+=1
                day_no+=1
                st = ed

            ed = discharge_dt
            temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.PROCEDURE_DTM>=st)
                             & (df_cpt.PROCEDURE_DTM<=ed)]
            df_all.at[idx, 'Shift Nbr'] = day_no
            temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
            d = temp.PROCEDURE_CODE.value_counts()
            if len(temp)>0:
                temp2['SUBGROUPS'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
                for ii, jj in temp2.iterrows():
                    df_all.at[idx, temp2.at[ii, 'SUBGROUPS']] = d[temp2.at[ii, 'PROCEDURE_CODE']]
            print(i, 'Shifts:', day_no)
            idx+=1         
        if i%100==0:
            print('Count:', i, idx)
        if i%1000==0:
            df_all.to_csv('../bucket/Readmission_30days/Amara/Readmission_Data/Readmission_label_Demo_processed_w_xray_loc_expanded_CPT_daily_'+suffix+'.csv')
            sys.stdout.flush()

    df_all.to_csv('../bucket/Amara/Readmission_30days/Readmission_Data/Readmission_label_Demo_processed_w_xray_loc_expanded_CPT_daily_'+suffix+'.csv')