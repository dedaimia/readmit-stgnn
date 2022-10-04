import pandas as pd
import numpy as np
import os
import pickle
import copy
import argparse
from tqdm import tqdm
import sys

sys.path.append("../")

from collections import Counter
from sklearn.preprocessing import LabelEncoder


###### Constants ######

header_proc = '../ehr/processed/'
header_raw = '../ehr/raw/'

IMG_FEATURE_DIM = 1024


LAB_COLS = [
    "Glucose, POCT, B",
    "Hemoglobin",
    "Platelet Count",
    "Hematocrit",
    "Erythrocytes",
    "Leukocytes",
    "Potassium, S",
    "Sodium, S",
    "Creatinine, S",
    "Bicarbonate, S",
    "Chloride, S",
    "Anion Gap",
    "Calcium, Total, S",
    "Glucose, S",
    "Bld Urea Nitrog(BUN), S",
    "Lymphocytes",
    "Monocytes",
    "Neutrophils",
    "Eosinophils",
    "Basophils",
    "Potassium, P",
    "Sodium, P",
    "Chloride, P",
    "Bicarbonate, P",
    "Creatinine, P",
    "Glucose, P",
    "Anion Gap, P",
    "Calcium, Total, P",
    "pH",
    "Hemoglobin, B",
    "pCO2",
    "pO2",
    "Lactate, P",
    "Glucose",
    "Lactate, B",
    "pH, U",
    "Potassium, B",
    "Sodium, B",
    "Troponin T, 6 hr, 5th gen",
    "Troponin T, 2 hr, 5th gen",
    "Troponin T, Baseline, 5th gen",
    "Venous pH",
    "Venous pCO2",
    "Bicarbonate",
    "Sodium",
    "Potassium",
    "Creatinine",
    "Creatinine, B",
    "Chloride",
    "Sodium, U",
    "Chloride, B",
    "Chloride, U",
]

DEMO_COLS = [
    "PATIENT_AGE_NEW",
    "PATIENT_RACE_NAME",
    "PATIENT_GENDER_NAME",
    "PATIENT_ETHNICITY_NAME",
]

MEDICATION_COLS = [
    "ANESTHETICS",
    "ANTINEOPLASTICS",
    "BIOLOGICALS",
    "DIAGNOSTIC",
    "ANTIBIOTICS",
    "CONTRACEPTIVES",
    "ANTICOAGULANTS",
    "CARDIOVASCULAR",
    "PSYCHOTHERAPEUTIC DRUGS",
    "CNS DRUGS",
    "ELECT/CALORIC/H2O",
    "EENT PREPS",
    "BLOOD",
    "CARDIAC DRUGS",
    "GASTROINTESTINAL",
    "UNCLASSIFIED DRUG PRODUCTS",
    "ANTIFUNGALS",
    "AUTONOMIC DRUGS",
    "DIURETICS",
    "SEDATIVE/HYPNOTICS",
    "IMMUNOSUPPRESSANTS",
    "HORMONES",
    "ANTIHYPERGLYCEMICS",
    "ANTIARTHRITICS",
    "COLONY STIMULATING FACTORS",
    "ANTIASTHMATICS",
    "ANALGESICS",
    "VITAMINS",
    "SKIN PREPS",
    "MUSCLE RELAXANTS",
    "ANTIVIRALS",
    "ANTIPARKINSON DRUGS",
    "ANTIHISTAMINES",
    "ANTIPLATELET DRUGS",
    "MISCELLANEOUS MEDICAL SUPPLIES, DEVICES, NON-DRUG",
    "ANTIINFECTIVES/MISCELLANEOUS",
    "SMOKING DETERRENTS",
    "COUGH/COLD PREPARATIONS",
    "HERBALS",
    "ANALGESIC AND ANTIHISTAMINE COMBINATION",
    "ANTIALLERGY",
    "ANTIPARASITICS",
    "ANTIINFLAM.TUMOR NECROSIS FACTOR INHIBITING AGENTS",
    "ANTIHISTAMINE AND DECONGESTANT COMBINATION",
    "PRE-NATAL VITAMINS",
    "ANTI-OBESITY DRUGS",
]



DEMO_FILE = os.path.join(
    header_proc, "cohort_file_w_demo.csv"
)
CPT_FILE = os.path.join(
    header_proc, "cohort_file_w_cpt.csv"
)
ICD_FILE = os.path.join(
    header_proc, "cohort_file_w_icd.csv"
)
LAB_FILE = os.path.join(
    header_proc, 'cohort_file_w_lab.csv'
)
MED_FILE = os.path.join(
    header_proc, 'cohort_file_w_med.csv'
)
# ADMIT_REASON_FILE = os.path.join(EHR_DIR, "l3_diagnosis_counts_dict.pkl")
# with open(ADMIT_REASON_FILE, "rb") as pf:
#     ADMIT_COLS = pickle.load(pf)
ADMIT_COLS = []#list(ADMIT_COLS.keys())



COLS_IRRELEVANT = [
    "PATIENT_DK",
    "ADMISSION_DTM",
    "DISCHARGE_DTM",
    "split",
    "Date",
    "node_name",
]
CAT_COLUMNS = LAB_COLS + [
    "PATIENT_RACE_NAME",
    "PATIENT_GENDER_NAME",
    "PATIENT_ETHNICITY_NAME",
]

SUBGOUPRS_EXCLUDED = [
    "Z00-Z13",
    "Z14-Z15",
    "Z16-Z16",
    "Z17-Z17",
    "Z18-Z18",
    "Z19-Z19",
    "Z20-Z29",
    "Z30-Z39",
    "Z40-Z53",
    "Z55-Z65",
    "Z66-Z66",
    "Z67-Z67",
    "Z68-Z68",
    "Z69-Z76",
    "Z77-Z99",
    "PR RPR MENINGOCELE <5 CM",
    "HC INCISN EXTENSOR TNDN SHTH WRST",
    "PR INCISN EXTENSOR TNDN SHTH WRST",
]


def ehr_bag_of_words(
    df_demo,
    df_ehr,
    ehr_type="cpt",
    time_step_by="day",
    filter_freq=None,
    label_cutoff=1.0,
):
    """
    Get EHR sequence using naive bag-of-words method
    Args:
        df_demo: demographics dataframe
        df_ehr: CPT/ICD dataframe
        ehr_type: 'cpt' or 'icd'
        time_step_by: 'day', what is the time step size?
    Returns:
        ehr_seq_padded: shape (num_admissions, max_seq_len, num_ehr_subgroups),
            short sequences are padded with -1
    """
    SUBGROUP_COL = "SUBGROUP"

    if ehr_type == "cpt":
        col_prefix = "PROCEDURE_"
    elif ehr_type == "icd":
        col_prefix = "DIAGNOSIS_"

    all_subgroups = list(set(df_ehr[SUBGROUP_COL]))
    all_subgroups = [
        val
        for val in all_subgroups
        if isinstance(val, str) and (val not in SUBGOUPRS_EXCLUDED)
    ]

    df_ehr_count = {
        "PATIENT_DK": [],
        "ADMISSION_DTM": [],
        "DISCHARGE_DTM": [],
        "Date": [],
        "target": [],
        "split": [],
        "node_name": [],
    }
    initial_cols = len(df_ehr_count) + len(DEMO_COLS) #+ len(ADMIT_COLS)
    # add demographic columns
    for demo in DEMO_COLS:
        df_ehr_count[demo] = []

    # add principal diagnoses columns
    for col in ADMIT_COLS:
        df_ehr_count[col] = []

    # add subgroup columns
    for subgrp in all_subgroups:
        df_ehr_count[subgrp] = []

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["PATIENT_DK"]
        admit_dt = row["ADMISSION_DTM"]
        discharge_dt = row["DISCHARGE_DTM"]
        # if not (np.isnan(row["Gap in months"])) and (
        #     row["Gap in months"] <= label_cutoff
        # ):
        #     label = 1
        # else:
        #     label = 0
        label = -1 #unknown
        if (str(pat) + "_" + str(admit_dt)) in df_ehr_count["node_name"]:
            continue

        if time_step_by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(admit_dt).date(),
                end=pd.to_datetime(discharge_dt).date(),
            )  # both inclusive
        else:
            raise NotImplementedError
        assert len(dt_range) > 1

        curr_ehr_df = df_ehr[df_ehr["PATIENT_DK"] == pat]

        for dt in dt_range:
            subgroups = curr_ehr_df[
                pd.to_datetime(curr_ehr_df[col_prefix + "DTM"])
                == pd.to_datetime(dt)
            ][SUBGROUP_COL].tolist()
            subgroups = [grp for grp in subgroups if isinstance(grp, str)]

            df_ehr_count["PATIENT_DK"].append(pat)
            df_ehr_count["ADMISSION_DTM"].append(admit_dt)
            df_ehr_count["DISCHARGE_DTM"].append(discharge_dt)
            df_ehr_count["Date"].append(str(dt))
            df_ehr_count["target"].append(label)
            df_ehr_count["split"].append(row["split"])
            df_ehr_count["node_name"].append(str(pat) + "_" + str(admit_dt))
            for demo in DEMO_COLS:
                if (demo in CAT_COLUMNS) and isinstance(row[demo], float):  # nan
                    df_ehr_count[demo].append("UNKNOWN")
                else:
                    df_ehr_count[demo].append(row[demo])

            for col in ADMIT_COLS:
                if np.isnan(row[col]):
                    df_ehr_count[col].append(0)
                else:
                    df_ehr_count[col].append(row[col])

            if len(subgroups) > 0:
                ehr_counts = Counter(subgroups)
                for subgrp in all_subgroups:
                    if subgrp in ehr_counts.keys():
                        df_ehr_count[subgrp].append(ehr_counts[subgrp])
                    else:
                        df_ehr_count[subgrp].append(0)
            else:
                for subgrp in all_subgroups:
                    df_ehr_count[subgrp].append(0)

    df_ehr_count = pd.DataFrame.from_dict(df_ehr_count)

    # drop zero occurrence subgroups
    if filter_freq is not None:
        freq = df_ehr_count[all_subgroups].sum(axis=0)
        drop_col_idxs = freq.values < filter_freq
        df_ehr_count = df_ehr_count.drop(columns=freq.loc[drop_col_idxs].index)
    else:
        freq = df_ehr_count[all_subgroups].sum(axis=0)
        drop_col_idxs = freq.values == 0
        df_ehr_count = df_ehr_count.drop(columns=freq.loc[drop_col_idxs].index)
    print("Final subgroups:", len(df_ehr_count.columns) - initial_cols)

    return df_ehr_count


def lab_one_hot(
    df_demo, df_lab, time_step_by="day", filter_freq=None, label_cutoff=1.0
):
    """
    Get EHR sequence using naive bag-of-words method
    Args:
        df_demo: demographics dataframe
        df_lab: labs dataframe
        ehr_type: 'lab'
        time_step_by: 'day', what is the time step size?
    Returns:
        ehr_seq_padded: shape (num_admissions, max_seq_len, num_ehr_subgroups),
            short sequences are padded with -1
    """

    df_lab_onehot = {
        "PATIENT_DK": [],
        "ADMISSION_DTM": [],
        "DISCHARGE_DTM": [],
        "Date": [],
        "target": [],
        "split": [],
        "node_name": [],
    }
    initial_cols = len(df_lab_onehot) + len(DEMO_COLS)# + len(ADMIT_COLS)
    for lab in LAB_COLS:
        df_lab_onehot[lab] = []
    for demo in DEMO_COLS:
        df_lab_onehot[demo] = []
    for col in ADMIT_COLS:
        df_lab_onehot[col] = []

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["PATIENT_DK"]
        admit_dt = row["ADMISSION_DTM"]
        discharge_dt = row["DISCHARGE_DTM"]
        # if not (np.isnan(row["Gap in months"])) and (
        #     row["Gap in months"] <= label_cutoff
        # ):
        #     label = 1
        # else:
        #     label = 0
        label = -1 #unknown
        if (str(pat) + "_" + str(admit_dt)) in df_lab_onehot["node_name"]:
            continue

        if time_step_by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(admit_dt).date(),
                end=pd.to_datetime(discharge_dt).date(),
            )  # both inclusive
        else:
            raise NotImplementedError

        # get this patient's lab results
        curr_lab_df = pd.DataFrame(columns=df_lab.columns)
        for _, pat_lab in df_lab[df_lab["PATIENT_DK_x"] == pat].iterrows():
            if pd.Timestamp(pat_lab["ADMISSION_DTM"]).strftime(
                "%Y/%m/%d %X"
            ) == pd.Timestamp(admit_dt).strftime("%Y/%m/%d %X"):
                curr_lab_df = curr_lab_df.append(pat_lab)

        for dt in dt_range:
            day_number = (dt.date() - pd.to_datetime(admit_dt).date()).days + 1
            curr_labs = curr_lab_df[curr_lab_df["Day Nbr"] == float(day_number)]

            df_lab_onehot["PATIENT_DK"].append(pat)
            df_lab_onehot["ADMISSION_DTM"].append(admit_dt)
            df_lab_onehot["DISCHARGE_DTM"].append(discharge_dt)
            df_lab_onehot["Date"].append(str(dt))
            df_lab_onehot["target"].append(label)
            df_lab_onehot["split"].append(row["split"])
            df_lab_onehot["node_name"].append(str(pat) + "_" + str(admit_dt))
            for demo in DEMO_COLS:
                if (demo in CAT_COLUMNS) and isinstance(row[demo], float):  # nan
                    df_lab_onehot[demo].append("UNKNOWN")
                else:
                    df_lab_onehot[demo].append(row[demo])

            for col in ADMIT_COLS:
                if np.isnan(row[col]):
                    df_lab_onehot[col].append(0)
                else:
                    df_lab_onehot[col].append(row[col])

            for lab in LAB_COLS:
                if len(curr_labs) == 0:
                    df_lab_onehot[lab].append("UNKNOWN")
                elif isinstance(curr_labs[lab].values[0], float):  # nan
                    df_lab_onehot[lab].append("UNKNOWN")  # map nan to unknown
                else:
                    df_lab_onehot[lab].append(curr_labs[lab].values[0])

    df_lab_onehot = pd.DataFrame.from_dict(df_lab_onehot)

    # drop zero abnormal subgroups
    if filter_freq is not None:
        freq = (df_lab_onehot[LAB_COLS] == "ABNORMAL").sum(
            axis=0
        )  # number of abnormals
        drop_col_idxs = freq.values < filter_freq
        df_lab_onehot = df_lab_onehot.drop(columns=freq.loc[drop_col_idxs].index)
    else:
        freq = (df_lab_onehot[LAB_COLS] == "ABNORMAL").sum(
            axis=0
        )  # number of abnormals
        drop_col_idxs = freq.values == 0
        df_lab_onehot = df_lab_onehot.drop(columns=freq.loc[drop_col_idxs].index)
    print("Final labs:", len(df_lab_onehot.columns) - initial_cols)

    return df_lab_onehot


def medication_bag_of_words(
    df_demo, df_med, time_step_by="day", filter_freq=None, label_cutoff=1.0
):
    """
    Get EHR sequence using naive bag-of-words method
    Args:
        df_demo: demographics dataframe
        df_med: medications dataframe
        ehr_type: 'med'
        time_step_by: 'day', what is the time step size?
    Returns:
        ehr_seq_padded: shape (num_admissions, max_seq_len, num_ehr_subgroups),
            short sequences are padded with -1
    """

    df_med_count = {
        "PATIENT_DK": [],
        "ADMISSION_DTM": [],
        "DISCHARGE_DTM": [],
        "Date": [],
        "target": [],
        "split": [],
        "node_name": [],
    }
    initial_cols = len(df_med_count) + len(DEMO_COLS)# + len(ADMIT_COLS)
    for med in MEDICATION_COLS:
        df_med_count[med] = []
    for demo in DEMO_COLS:
        df_med_count[demo] = []
    for col in ADMIT_COLS:
        df_med_count[col] = []

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["PATIENT_DK"]
        admit_dt = row["ADMISSION_DTM"]
        discharge_dt = row["DISCHARGE_DTM"]
        # if not (np.isnan(row["Gap in months"])) and (
        #     row["Gap in months"] <= label_cutoff
        # ):
        #     label = 1
        # else:
        #     label = 0
        label = -1

        if (str(pat) + "_" + str(admit_dt)) in df_med_count["node_name"]:
            continue

        if time_step_by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(admit_dt).date(),
                end=pd.to_datetime(discharge_dt).date(),
            )  # both inclusive
        else:
            raise NotImplementedError

        # get this patient's lab results
        curr_med_df = pd.DataFrame(columns=df_med.columns)
        for _, pat_med in df_med[df_med["PATIENT_DK_x"] == pat].iterrows():
            if pd.Timestamp(pat_med["ADMISSION_DTM"]).strftime(
                "%Y/%m/%d %X"
            ) == pd.Timestamp(admit_dt).strftime("%Y/%m/%d %X"):
                curr_med_df = curr_med_df.append(pat_med)

        for dt in dt_range:
            day_number = (dt.date() - pd.to_datetime(admit_dt).date()).days + 1
            curr_meds = curr_med_df[curr_med_df["Day Nbr"] == float(day_number)]

            df_med_count["PATIENT_DK"].append(pat)
            df_med_count["ADMISSION_DTM"].append(admit_dt)
            df_med_count["DISCHARGE_DTM"].append(discharge_dt)
            df_med_count["Date"].append(str(dt))
            df_med_count["target"].append(label)
            df_med_count["split"].append(row["split"])
            df_med_count["node_name"].append(str(pat) + "_" + str(admit_dt))
            for demo in DEMO_COLS:
                if (demo in CAT_COLUMNS) and isinstance(row[demo], float):  # nan
                    df_med_count[demo].append("UNKNOWN")
                else:
                    df_med_count[demo].append(row[demo])

            for col in ADMIT_COLS:
                if np.isnan(row[col]):
                    df_med_count[col].append(0)
                else:
                    df_med_count[col].append(row[col])

            for med in MEDICATION_COLS:
                if len(curr_meds) == 0:
                    df_med_count[med].append(0)
                elif np.isnan(curr_meds[med].values[0]):  # nan
                    df_med_count[med].append(0)  # map nan to 0
                else:
                    df_med_count[med].append(curr_meds[med].values[0])

    df_med_count = pd.DataFrame.from_dict(df_med_count)

    # drop zero occurrence subgroups
    if filter_freq is not None:
        freq = df_med_count[MEDICATION_COLS].sum(axis=0, numeric_only=True)
        drop_col_idxs = freq.values < filter_freq
        df_med_count = df_med_count.drop(columns=freq.loc[drop_col_idxs].index)
    else:
        freq = df_med_count[MEDICATION_COLS].sum(axis=0, numeric_only=True)
        drop_col_idxs = freq.values == 0
        df_med_count = df_med_count.drop(columns=freq.loc[drop_col_idxs].index)
    print("Final medications:", len(df_med_count.columns) - initial_cols)

    return df_med_count





def preproc_ehr(X):
    """
    Args:
        X: pandas dataframe
    Returns:
        X_enc: pandas dataframe, with one-hot encoded columns for categorical variables
    """
    train_indices = X[X["split"] == "train"].index

    # encode categorical variables
    X_enc = []
    num_cols = 0
    categorical_columns = []
    categorical_dims = {}
    for col in tqdm(X.columns):
        if col in COLS_IRRELEVANT:
            X_enc.append(X[col])
            num_cols += 1
        elif col in CAT_COLUMNS:
            print(col, X[col].unique())
            curr_enc = pd.get_dummies(
                X[col], prefix=col
            )  # this will transform into one-hot encoder
            X_enc.append(curr_enc)
            num_cols += curr_enc.shape[-1]
            curr_cols = [col + "_" + str(i) for i in range(curr_enc.shape[-1])]
            categorical_columns.extend(curr_cols)
            for c in curr_cols:
                categorical_dims[c] = 1
        else:
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)
            curr_enc = X[col]
            X_enc.append(curr_enc)
            num_cols += 1

    X_enc = pd.concat(X_enc, axis=1)
    assert num_cols == X_enc.shape[-1]

    feature_cols = [
        col
        for col in X_enc.columns
        if (col != "target") and (col not in COLS_IRRELEVANT)
    ]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f]
        for _, f in enumerate(feature_cols)
        if f in categorical_columns
    ]

    return {
        "X": X_enc,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
    }


def ehr2sequence(preproc_dict, df_demo, by="day"):
    """
    Arrange EHR into sequences for temporal models
    """

    X = preproc_dict["X"]
    df = copy.deepcopy(X)
    feature_cols = preproc_dict["feature_cols"]

    print("Rearranging to sequences by {}...".format(by))
    X = X[feature_cols].values

    X_dict = {}
    for i in range(X.shape[0]):
        key = (
            str(df.iloc[i]["PATIENT_DK"])
            + "_"
            + str(pd.to_datetime(df.iloc[i]["Date"]).date())
        )
        X_dict[key] = X[i]

    # _, node_included_files, _, _, _, _ = get_readmission_label(
    #     df_demo, cutoff_months=1, max_seq_len=None
    # )

    # arrange X by day or by cxr
    feat_dict = {}
    for node_name, cxr_files in tqdm(node_included_files.items()):
        # print(node_name)
        ehr_row = df[df["node_name"] == node_name]
        curr_admit = ehr_row["ADMISSION_DTM"].values[0]
        curr_discharge = ehr_row["DISCHARGE_DTM"].values[0]
        curr_pat = ehr_row["PATIENT_DK"].values[0]

        if by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(curr_admit).date(),
                end=pd.to_datetime(curr_discharge).date(),
            )
        else:
            raise NotImplementedError
            # dt_range = [pd.to_datetime(df_demo[df_demo["copied_PNG_location"] == fn]
            #                            ["Study_DTM"].values[0]).date() for fn in cxr_files]
            # assert sorted(dt_range) == dt_range

        curr_features = []
        for dt in dt_range:
            if by == "cxr":
                key = str(curr_pat) + "_" + str(dt)
            else:
                key = str(curr_pat) + "_" + str(dt.date())
            feat = X_dict[key]
            curr_features.append(feat)

        curr_features = np.stack(curr_features)  # (num_days, feature_dim)
        feat_dict[node_name] = curr_features

    if "cat_idxs" in preproc_dict:
        cat_idxs = preproc_dict["cat_idxs"]
        cat_dims = preproc_dict["cat_dims"]
        return {
            "feat_dict": feat_dict,
            "feature_cols": feature_cols,
            "cat_idxs": cat_idxs,
            "cat_dims": cat_dims,
        }
    else:
        return {"feat_dict": feat_dict, "feature_cols": feature_cols}


def main():
    # read csv files
    df_demo = pd.read_csv(DEMO_FILE)
    df_lab = pd.read_csv(LAB_FILE)
    df_lab = df_lab[[col for col in df_lab.columns if "Unnamed" not in col]]
    df_cpt = pd.read_csv(CPT_FILE)
    df_icd = pd.read_csv(ICD_FILE)
    df_med = pd.read_csv(MED_FILE)



    # cpt
    df_cpt_count = ehr_bag_of_words(
        df_demo,
        df_cpt,
        ehr_type="cpt",
        time_step_by="day",
        filter_freq=None,
        label_cutoff=1.0,#args.label_cutoff,
    )
    print("Unique node names:", len(list(set(df_cpt_count["node_name"].tolist()))))

    # icd
    df_icd_count = ehr_bag_of_words(
        df_demo,
        df_icd,
        ehr_type="icd",
        time_step_by="day",
        filter_freq=None,
        label_cutoff=1.0,#args.label_cutoff,
    )
    print("Unique node names:", len(list(set(df_icd_count["node_name"].tolist()))))

    # lab
    df_lab_onehot = lab_one_hot(
        df_demo,
        df_lab,
        time_step_by="day",
        filter_freq=None,
        label_cutoff=1.0,#args.label_cutoff,
    )
    print("Unique node names:", len(list(set(df_lab_onehot["node_name"].tolist()))))

    # medication
    df_med_count = medication_bag_of_words(
        df_demo,
        df_med,
        time_step_by="day",
        filter_freq=None,
        label_cutoff=1.0,#args.label_cutoff,
    )

    # combine
    df_combined = pd.concat(
        [df_cpt_count, df_icd_count, df_lab_onehot, df_med_count], axis=1
    )

    # drop duplicated columns
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    df_combined.to_csv(os.path.join(header_proc, "ehr_combined.csv"), index=False)
    # df_combined = pd.read_csv(os.path.join(args.save_dir, "ehr_combined.csv"))

    # further preprocess it to meet tabnet format
    for format in ["gnn"]:
        preproc_dict = preproc_ehr(df_combined)

        feature_cols = preproc_dict["feature_cols"]
        demo_cols = [
            col for col in feature_cols if any([s for s in DEMO_COLS if s in col])
        ]
        cpt_cols = [
            col for col in feature_cols if col in list(set(df_cpt["SUBGROUP"].tolist()))
        ]
        icd_cols = [
            col for col in feature_cols if col in list(set(df_icd["SUBGROUP"].tolist()))
        ]
        lab_cols = [
            col for col in feature_cols if any([s for s in LAB_COLS if s in col])
        ]
        med_cols = [
            col for col in feature_cols if any([s for s in MEDICATION_COLS if s in col])
        ]
        # admit_cols = [
        #     col for col in feature_cols if any([s for s in ADMIT_COLS if s in col])
        # ]

        preproc_dict["demo_cols"] = demo_cols
        preproc_dict["cpt_cols"] = cpt_cols
        preproc_dict["icd_cols"] = icd_cols
        preproc_dict["lab_cols"] = lab_cols
        preproc_dict["med_cols"] = med_cols
        # preproc_dict["admit_cols"] = admit_cols

        # save
        with open(
            os.path.join(header_proc, "ehr_preprocessed_all_{}.pkl".format(format)),
            "wb",
        ) as pf:
            pickle.dump(preproc_dict, pf)
        print(
            "Saved to {}".format(
                os.path.join(
                    header_proc, "ehr_preprocessed_all_{}.pkl".format(format)
                )
            )
        )

        # also save it into sequences for temporal models
        for by in ["day"]:
            seq_dict = ehr2sequence(preproc_dict, df_demo, by=by)

            seq_dict["demo_cols"] = demo_cols
            seq_dict["cpt_cols"] = cpt_cols
            seq_dict["icd_cols"] = icd_cols
            seq_dict["lab_cols"] = lab_cols
            seq_dict["med_cols"] = med_cols
            with open(
                os.path.join(
                    header_proc,
                    "ehr_preprocessed_seq_by_{}_{}.pkl".format(by, format),
                ),
                "wb",
            ) as pf:
                pickle.dump(seq_dict, pf)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Preprocessing EHR.")

#     parser.add_argument(
#         "--save_dir", type=str, default=header_proc, help="Dir to save preprocessed files."
#     )
#     parser.add_argument(
#         "--label_cutoff",
#         type=float,
#         default=1.0,
#         help="Cut-off point for positive labels in months.",
#     )
#     # parser.add_argument(
#     #     "--format",
#     #     type=str,
#     #     default="tabnet",
#     #     choices=("tabnet", "gnn"),
#     #     help="Format for EHR inputs."
#     # )

#     args = parser.parse_args()
#     main(args)