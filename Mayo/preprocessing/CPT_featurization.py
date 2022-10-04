#!/usr/bin/env python3

import pandas as pd
import numpy as np
import datetime
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def cpt_featurization(df, df_cpt):
    """
    df: selected cohort file, one hospitalization per row
    df_cpt: EDTWH_FACT_PROCEDURES file from SQL query - one cpt recorded per row

    df_all: return file with one hospitalization per row with row containing all cpt SUBGROUP for that hospitalization
    """     

    dfcpt_groups = pd.read_csv("CPT_group_structure.csv") #cpt code structure
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


    
    df['ADMISSION_DTM'] = pd.to_datetime(df['ADMISSION_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_cpt['PROCEDURE_DTM'] = pd.to_datetime(df_cpt['PROCEDURE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


    features = dfcpt_groups.Subgroup.unique()
    pd.set_option('mode.chained_assignment', None)
    df_all = pd.DataFrame(columns = columns)
    idx = 0
    for k in features:
        df_all[k] = 0
    for i,j in df.iterrows():
        pid = df.at[i, 'PATIENT_DK']
        admit_dt = df.at[i, 'ADMISSION_DTM']
        discharge_dt = df.at[i, 'DISCHARGE_DTM']
        
        for c in columns:
            df_all.at[idx, c] = df.at[i, c]

        st = admit_dt
        day_no = 1
        while st+timedelta(hours=24) < discharge_dt+timedelta(hours=12):
            ed = st+timedelta(hours=24)
            temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.PROCEDURE_DTM>=st)
                            & (df_cpt.PROCEDURE_DTM<=ed)]
            df_all.at[idx, 'Day_Number'] = day_no
            df_all.at[idx, 'Date'] = admit_dt+timedelta(days=day_no-1)
            temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
            d = temp.PROCEDURE_CODE.value_counts()
            if len(temp)>0:
                temp2['SUBGROUP'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
                for ii, jj in temp2.iterrows():
                    df_all.at[idx, temp2.at[ii, 'SUBGROUP']] = d[temp2.at[ii, 'PROCEDURE_CODE']]
            idx+=1
            day_no+=1
            st = ed

        ed = discharge_dt
        temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.PROCEDURE_DTM>=st)
                            & (df_cpt.PROCEDURE_DTM<=ed)]
        df_all.at[idx, 'Day_Number'] = day_no
        df_all.at[idx, 'Date'] = admit_dt+timedelta(days=day_no-1)
        temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
        d = temp.PROCEDURE_CODE.value_counts()
        if len(temp)>0:
            temp2['SUBGROUP'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
            for ii, jj in temp2.iterrows():
                df_all.at[idx, temp2.at[ii, 'SUBGROUP']] = d[temp2.at[ii, 'PROCEDURE_CODE']]
        print(i, 'Days:', day_no)
        idx+=1         
        if i%100==0:
            print('Count:', i, idx)
    return df_all