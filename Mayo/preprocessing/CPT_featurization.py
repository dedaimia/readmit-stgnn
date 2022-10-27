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
        # print(x, type(x), end='\t')
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
        # print(out)
        return out


    
    df['ADMIT_DTM'] = pd.to_datetime(df['ADMIT_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
    df['DISCHARGE_DTM'] = pd.to_datetime(df['DISCHARGE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')

    df_cpt['PROCEDURE_DTM'] = pd.to_datetime(df_cpt['PROCEDURE_DTM'],format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


    features = dfcpt_groups.Subgroup.unique()
    pd.set_option('mode.chained_assignment', None)
    columns = [c for c in df.columns if c.startswith('Unnamed')==False]
    df_all = pd.DataFrame(columns = columns)
    idx = 0
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
            temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.PROCEDURE_DTM>=st)
                            & (df_cpt.PROCEDURE_DTM<=ed)]
            for c in columns:
                df_all.at[idx, c] = df.at[i, c]
            df_all.at[idx, 'Day_Number'] = day_no
            df_all.at[idx, 'Date'] = admit_dt+timedelta(days=day_no-1)
            temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
            d = temp.PROCEDURE_CODE.value_counts()
            if len(temp)>0:
                temp2['SUBGROUP'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
                temp2 = temp2.dropna(subset=['SUBGROUP'])
                for s in temp2.SUBGROUP.unique():
                    df_all.at[idx, s] = len(temp.loc[temp.PROCEDURE_CODE.isin(temp2.loc[temp2.SUBGROUP==s]['PROCEDURE_CODE'].unique())])
            idx+=1
            day_no+=1
            st = ed

        ed = discharge_dt
        temp = df_cpt.loc[(df_cpt.PATIENT_DK==pid) & (df_cpt.PROCEDURE_DTM>=st)
                            & (df_cpt.PROCEDURE_DTM<=ed)]
        for c in columns:
                df_all.at[idx, c] = df.at[i, c]
        df_all.at[idx, 'Day_Number'] = day_no
        df_all.at[idx, 'Date'] = admit_dt+timedelta(days=day_no-1)
        temp2 = temp.drop_duplicates(subset=['PROCEDURE_CODE'])
        d = temp.PROCEDURE_CODE.value_counts()
        if len(temp)>0:
            temp2['SUBGROUP'] = temp2.PROCEDURE_CODE.apply(to_cpt_group) 
            temp2 = temp2.dropna(subset=['SUBGROUP'])
            for s in temp2.SUBGROUP.unique():
                df_all.at[idx, s] = len(temp.loc[temp.PROCEDURE_CODE.isin(temp2.loc[temp2.SUBGROUP==s]['PROCEDURE_CODE'].unique())])

        print(i, 'Days:', day_no)
        idx+=1         
        if i%100==0:
            print('Count:', i, idx)
    return df_all