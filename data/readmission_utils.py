import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import pickle

###### Constants ######
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
# ADMIT_REASON_FILE = os.path.join(
#     "/mnt/storage/Readmission/colab_readmission/Readmission_Anonym/l3_diagnosis_counts_dict.pkl"
# )
# with open(ADMIT_REASON_FILE, "rb") as pf:
#     ADMIT_COLS = pickle.load(pf)
# ADMIT_COLS = list(ADMIT_COLS.keys())

# LAB_COLS_MIMIC = ['pH Other Body Fluid',
#                   'Lymphocytes Joint Fluid',
#                   'Hemoglobin Blood',
#                   'Basophils Joint Fluid',
#                   'Chloride Blood',
#                   'Glucose Blood',
#                   'Monocytes Blood',
#                   'Eosinophils Joint Fluid',
#                   'pH Urine',
#                   'Monocytes Ascites',
#                   'Basophils Cerebrospinal Fluid',
#                   'Hematocrit Urine',
#                   'Bicarbonate Blood',
#                   'Basophils Other Body Fluid',
#                   'Potassium Blood',
#                   'Eosinophils Blood',
#                   'Eosinophils Pleural',
#                   'L Blood',
#                   'Lymphocytes Ascites',
#                   'Creatinine Blood',
#                   'Eosinophils Ascites',
#                   'pO2 Blood',
#                   'Lymphocytes Other Body Fluid',
#                   'Hematocrit Blood',
#                   'Leukocytes Urine',
#                   'Calcium, Total Blood',
#                   'Anion Gap Blood',
#                   'Eosinophils Other Body Fluid',
#                   'Sodium Blood',
#                   'Lymphocytes Blood',
#                   'Basophils Pleural',
#                   'Lactate Blood',
#                   'H Blood',
#                   'Neutrophils Blood',
#                   'Monocytes Cerebrospinal Fluid',
#                   'Troponin T Blood',
#                   'Platelet Count Blood',
#                   'Basophils Blood',
#                   'Monocytes Joint Fluid',
#                   'Eosinophils Urine',
#                   'pH Blood',
#                   'Glucose Urine',
#                   'Eosinophils Cerebrospinal Fluid',
#                   'pCO2 Blood',
#                   'Basophils Ascites',
#                   'Lymphocytes Pleural']

DEMO_COLS_MIMIC = ["age", "gender", "ethnicity"]
###### Constants ######


def get_patient_age(birth_date, study_date):
    if isinstance(birth_date, str):
        birth_date = pd.to_datetime(birth_date, format="%Y%m%d")
    if isinstance(study_date, str):
        study_date = pd.to_datetime(study_date, format="%Y%m%d")

    age = (
        study_date.year
        - birth_date.year
        - ((study_date.month, study_date.day) < (birth_date.month, birth_date.day))
    )
    return age


def cpt_to_subgroup(df_cpt_struc, x):
    out = None
    if type(x) == str and x.isnumeric():
        x = int(x)
        temp = df_cpt_struc.loc[
            (df_cpt_struc["Low"] <= x)
            & (df_cpt_struc["High"] >= x)
            & (df_cpt_struc["Modifier"].isna())
        ]
        if len(temp) > 0:
            out = temp.at[temp.index[0], "Subgroup"]
    elif type(x) == str and x[:-1].isnumeric():
        m = x[-1]
        x = int(x[:-1])
        temp = df_cpt_struc.loc[
            (df_cpt_struc["Low"] <= x)
            & (df_cpt_struc["High"] >= x)
            & (df_cpt_struc["Modifier"] == m)
        ]
        if len(temp) > 0:
            out = temp.at[temp.index[0], "Subgroup"]
    return out


def find_icd_group(df_icd, code):
    group = ""
    letter = code[0]
    number = code[1:].split(".")[0]
    if number.isnumeric():
        number = float(number)
        icd_sel = df_icd.loc[df_icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[
            (icd_sel.START_IDX.str.isnumeric()) & (icd_sel.END_IDX.str.isnumeric())
        ].copy()
        icd_sel = icd_sel.loc[
            (icd_sel.START_IDX.astype(float) <= number)
            & (icd_sel.END_IDX.astype(float) >= number)
        ].copy()
        if len(icd_sel) > 0:
            group = icd_sel.at[icd_sel.index[0], "SUBGROUP"]
        else:
            group = "UNKNOWN"
    else:
        icd_sel = df_icd.loc[df_icd.SUBGROUP.str.startswith(letter)].copy()
        icd_sel = icd_sel.loc[
            (icd_sel.START_IDX.str.isnumeric() == False)
            & (icd_sel.END_IDX.str.isnumeric() == False)
        ].copy()
        numheader = number[:-1]
        icd_sel = icd_sel.loc[
            (icd_sel.START_IDX.str.startswith(numheader))
            & (icd_sel.END_IDX.str.startswith(numheader))
        ].copy()
        if len(icd_sel) > 0:
            group = icd_sel.at[icd_sel.index[0], "SUBGROUP"]
        else:
            group = "UNKNOWN"
    return group


def get_readmission_label(df_demo, cutoff_months=1.0, max_seq_len=None):
    """
    Args:
        df_demo: dataframe with patient readmission info and demographics:
        cutoff_months: within how many months are positive labels?
        max_seq_len: maximum number of cxrs to use, count backwards from last cxr within the hospitalization
            if max_seq_len=None, will use all cxrs
    Returns:
        labels: same order as rows in df_demo
        node_included_files: dict, key is node name, value is list of cxr files
        label_splits: list indicating the split of each datapoint in labels and node_included_files
        time_deltas: dict, key is node name, value is an array of day difference between currenct cxr to previous cxr
        total_stay: dict, key is node name, value is total length of stay (in days)
    """
    labels = []
    node_included_files = {}
    label_splits = []
    time_deltas = {}
    total_stay = {}
    time_idxs = {}

    # take care of deaths
    df_demo["Discharge2DeathDays"] = df_demo["Discharge2DeathDays"].fillna(10000)
    df_demo["Gap in months"] = df_demo["Gap in months"].fillna(10000)

    # treat death within 30 days after discharge as readmitted
    df_demo.loc[df_demo["Discharge2DeathDays"] <= 30, "Gap in months"] = 1

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["PATIENT_DK"]
        admit_dt = row["ADMISSION_DTM"]
        discharge_dt = row["DISCHARGE_DTM"]
        split = row["split"]
        y = float(row["Gap in months"])

        try:
            curr_name = row["node_name"]
        except:
            curr_name = str(pat) + "_" + str(admit_dt)

        if curr_name not in node_included_files:
            node_included_files[curr_name] = []
        else:
            continue

        label_splits.append(split)

        # label
        if float(row["Gap in months"]) <= cutoff_months:
            labels.append(1)
        else:
            labels.append(0)

        # admit_df = df_demo.loc[
        #     (df_demo["ADMISSION_DTM"] == admit_dt)
        #     & (df_demo["PATIENT_DK"] == pat)
        # ]
        admit_df = df_demo[df_demo["node_name"] == curr_name]

        # sort by study datetime
        admit_df = admit_df.sort_values(by="Study_DTM")

        # get list of cxrs sorted by study datetime
        for _, admit_row in admit_df.iterrows():
            node_included_files[curr_name].append(admit_row["copied_PNG_location"])

        # get last max_seq_len cxrs if max_seq_len is specified
        if max_seq_len is not None:
            node_included_files[curr_name] = node_included_files[curr_name][
                -max_seq_len:
            ]

        # time delta & total hospital stay
        if isinstance(discharge_dt, str):
            curr_timedelta = np.zeros((len(node_included_files[curr_name])))

            prev_date = pd.to_datetime(admit_dt)
            addmission_date = pd.to_datetime(admit_dt)
            discharge_date = pd.to_datetime(discharge_dt)

            hospital_stay = (
                discharge_date.date() - pd.to_datetime(addmission_date).date()
            ).days + 1  # in days
            curr_timeidxs = []
            for t, fn in enumerate(node_included_files[curr_name]):
                study_date = pd.to_datetime(
                    df_demo[df_demo["copied_PNG_location"] == fn][
                        "Study_DTM"
                    ].values[0]
                )
                if t > 0:  # first time delta is always 0
                    curr_timedelta[t] = (
                        study_date.date() - prev_date.date()
                    ).days  # in days
                    prev_date = study_date

                time_index = (
                    study_date.date() - pd.to_datetime(addmission_date).date()
                ).days  # index starts from 0
                curr_timeidxs.append(time_index)

            assert hospital_stay != np.nan
            assert len(curr_timeidxs) > 0  # because at least one cxr
            total_stay[curr_name] = hospital_stay
            time_idxs[curr_name] = curr_timeidxs

            # normalize by hospital stay
            time_deltas[curr_name] = curr_timedelta / hospital_stay
        else:
            time_deltas[curr_name] = np.nan
            total_stay[curr_name] = np.nan
            time_idxs[curr_name] = np.nan

    return (
        np.array(labels),
        node_included_files,
        label_splits,
        time_deltas,
        total_stay,
        time_idxs,
    )


def get_feat_seq(
    node_names,
    feature_dict,
    max_seq_len,
    pad_front=False,
    time_deltas=None,
    padding_val=None,
):
    """
    Get feature sequence of length max_seq_len, pad short lengths with last or first timestep
    Args:
        node_names: list, node names
        feature_dict: dict, key is node name, value is preprocessed EHR/imaging feature
        max_seq_len: int, maximum sequence length
        pad_front: if True, will pad to the front of short sequences; default is False, padding
                    to the end of short sequences
        time_deltas: dict, key is node name, value is an array of time differences between CXR
        padding_val: int, padding value; default is None - padding using last available feature
    Returns:
        padded_features: numpy array, shape (sample_size, max_seq_len, feature_dim)
        seq_lengths: original sequence lengths without any padding
    """
    seq_lengths = []
    padded_features = []
    padded_time_deltas = []
    for name in node_names:
        feature = feature_dict[name]
        orig_seq_len = feature.shape[0]
        feature = feature[-max_seq_len:, :]  # get last max_seq_len time steps

        if time_deltas is not None:
            time_dt = time_deltas[name][-max_seq_len:]  # (max_seq_len,)
            # time_dt = time_deltas[name][:max_seq_len]
            # print("time_dt:",time_dt.shape)
            # print("feature:", feature.shape)
            assert len(time_dt) == feature.shape[0]

        if feature.shape[0] < max_seq_len:
            if not pad_front:
                # pad with last timestep or padding_val
                if padding_val is None:
                    padded = np.repeat(
                        feature[-1, :].reshape(1, -1),
                        repeats=max_seq_len - feature.shape[0],
                        axis=0,
                    )
                else:
                    padded = (
                        np.ones((max_seq_len - feature.shape[0], feature.shape[1]))
                        * padding_val
                    )
                feature = np.concatenate([feature, padded], axis=0)
                if time_deltas != None:
                    padded_dt = np.zeros((max_seq_len - time_dt.shape[0]))
                    time_dt = np.concatenate([time_dt, padded_dt], axis=0)
            else:
                # pad with first timestep or padding_val
                if padding_val is None:
                    padded = np.repeat(
                        feature[0, :].reshape(1, -1),
                        repeats=max_seq_len - feature.shape[0],
                        axis=0,
                    )
                else:
                    padded = (
                        np.ones((max_seq_len - feature.shape[0], feature.shape[1]))
                        * padding_val
                    )
                feature = np.concatenate([padded, feature], axis=0)
                if time_deltas != None:
                    padded_dt = np.zeros((max_seq_len - time_dt.shape[0]))
                    time_dt = np.concatenate([padded_dt, time_dt], axis=0)
        padded_features.append(feature)
        if time_deltas != None:
            padded_time_deltas.append(time_dt)
        seq_len = np.minimum(max_seq_len, orig_seq_len)
        seq_lengths.append(seq_len)

    padded_features = np.stack(padded_features)
    seq_lengths = np.stack(seq_lengths)
    if time_deltas != None:
        padded_time_deltas = np.expand_dims(np.stack(padded_time_deltas), axis=-1)
        padded_features = np.concatenate([padded_features, padded_time_deltas], axis=-1)

    return padded_features, seq_lengths


# def get_time_delta(node_included_files, df_demo):
#     time_deltas = {}
#     for name, files in tqdm(node_included_files.items()):
#         curr_timedelta = np.zeros((len(files)))
#         prev_date = pd.to_datetime(df_demo[df_demo["copied_PNG_location"] == files[0]]["ADMISSION_DTM"].values[0]).date()
#         addmission_date = df_demo[df_demo["copied_PNG_location"] == files[0]]["ADMISSION_DTM"].values[0]
#         discharge_date = df_demo[df_demo["copied_PNG_location"] == files[0]]["DISCHARGE_DTM"].values[0]
#         for t, fn in enumerate(files[1:]):
#             study_date = pd.to_datetime(df_demo[df_demo["copied_PNG_location"] == fn]["Study_DTM"].values[0]).date()
#             curr_timedelta[t+1] = ((study_date - prev_date).days)
#             prev_date = study_date

#         if not(isinstance(discharge_date, float) and np.isnan(discharge_date)):
#             hospital_stay = (pd.to_datetime(discharge_date) - pd.to_datetime(addmission_date)).days
#         else:
#             hospital_stay = (study_date - pd.to_datetime(addmission_date).date()).days
#         assert (hospital_stay != np.nan)
#         if hospital_stay == 0:
#             hospital_stay = 1

#         # normalize by hospital stay
#         time_deltas[name] = curr_timedelta / hospital_stay

#     return time_deltas


# def get_img_features(feature_file, node_included_files):
#     """
#     Get imaging features extracted from pretrained CNN
#     Args:
#         feature_dir: dir to imaging features
#         df_demo:
#     Returns:
#         img_feature_dict: dict, key is image path, value is image features within one hospitalization,
#             shape (num_cxrs_per_hospitalization, feature_dim)
#     """
#     df_features = pd.read_csv(feature_file)

#     img_paths = df_features["image_path"].tolist()

#     img_feature_dict = {}
#     for name, files in node_included_files.items():
#         curr_feat = []
#         for img_dir in files:
#             idx_feature = img_paths.index(img_dir)
#             feature = np.array(df_features.iloc[idx_feature][1:].tolist()) # (1024,)
#             curr_feat.append(feature)
#         curr_feat = np.stack(curr_feat, axis=0)
#         img_feature_dict[name] = curr_feat

#     return img_feature_dict


def get_img_features(
    feature_dir, node_included_files, time_idxs=None, hospital_stay=None, by="cxr"
):
    """
    Get imaging features extracted from pretrained CNN
    Args:
        feature_dir: dir to imaging features
        df_demo:
    Returns:
        img_feature_dict: dict, key is image path, value is image features within one hospitalization,
            shape (num_cxrs_per_hospitalization, feature_dim)
    """

    img_feature_dict = {}
    for name, files in tqdm(node_included_files.items()):
        curr_feat = []

        if by == "cxr":
            for img_dir in files:
                with open(
                    os.path.join(feature_dir, img_dir.split("/")[-1] + ".pkl"), "rb"
                ) as pf:
                    feature = pickle.load(pf)
                curr_feat.append(feature)
            curr_feat = np.stack(curr_feat, axis=0)  # (num_cxrs, feature_dim)
        elif by == "day":
            curr_feat = np.zeros((hospital_stay[name], IMG_FEATURE_DIM))
            # NOTE: the images are already sorted, so this will always take the last cxr if there is > 1 cxr on the same day
            for idx_img, img_dir in enumerate(files):
                with open(
                    os.path.join(feature_dir, img_dir.split("/")[-1] + ".pkl"), "rb"
                ) as pf:
                    feature = pickle.load(pf)
                time_ind = time_idxs[name][idx_img]
                curr_feat[time_ind, :] = feature
                # NOTE: for next few days w/o cxrs, fill with this cxr feature
                next_time_ind = (
                    time_idxs[name][idx_img + 1]
                    if (idx_img != (len(files) - 1))
                    else curr_feat.shape[0]
                )
                curr_feat[time_ind + 1 : next_time_ind, :] = feature

        else:
            raise NotImplementedError

        img_feature_dict[name] = curr_feat

    return img_feature_dict


def get_time_varying_edges(
    node_names,
    max_seq_len,
    edge_dict,
    edge_modality,
    hospital_stay,
    cpt_dict=None,
    icd_dict=None,
    lab_dict=None,
    med_dict=None,
    img_dict=None,
    pad_front=False,
    dynamic_graph=False,
    dataset="mayo",
):
    """
    Returns:
        edge_dict_list: list of dict, key is node name, value is cpt and/or icd features
    """
    print('In get_time_varying_edges')
#     print(edge_dict)
    if dynamic_graph:
        edge_dict_list = []
        for t in range(max_seq_len):
            if edge_dict is None:
                edge_dict_list.append({})
            else:
                edge_dict_list.append(edge_dict)
    else:
        if edge_dict is None:
            edge_dict_list = [{}]
        else:
            edge_dict_list = [edge_dict]

    if not dynamic_graph:
        for i, name in enumerate(node_names):
            if name not in edge_dict_list[0]:
                edge_dict_list[0][name] = []
            # NOTE: for cpt or icd or med, we sum over all days & average by length of stay (in days)
            if "cpt" in edge_modality:
                edge_dict_list[0][name] = np.concatenate(
                    [
                        edge_dict_list[0][name],
                        np.sum(cpt_dict[name], axis=0) / hospital_stay[name],
                    ],
                    axis=-1,
                )
            if "icd" in edge_modality:
                if dataset == "mayo":
                    edge_dict_list[0][name] = np.concatenate(
                        [
                            edge_dict_list[0][name],
                            np.sum(icd_dict[name], axis=0) / hospital_stay[name],
                        ],
                        axis=-1,
                    )
                else:
                    # NOTE: MIMIC ICD is non-temporal
                    edge_dict_list[0][name] = np.concatenate(
                        [
                            edge_dict_list[0][name],
                            icd_dict[name][-1, :] / hospital_stay[name],
                        ],
                        axis=-1,
                    )
            if "med" in edge_modality:
                edge_dict_list[0][name] = np.concatenate(
                    [
                        edge_dict_list[0][name],
                        np.sum(med_dict[name], axis=0) / hospital_stay[name],
                    ],
                    axis=-1,
                )
            # NOTE: for lab or imaging, take the last time step
            if "lab" in edge_modality:
                edge_dict_list[0][name] = np.concatenate(
                    [edge_dict_list[0][name], lab_dict[name][-1, :]], axis=-1
                )
            if ("imaging" in edge_modality) and (img_dict is not None):
                edge_dict_list[0][name] = np.concatenate(
                    [edge_dict_list[0][name], img_dict[name][-1, :]], axis=-1
                )
            # print("edge_dict_list[0][name] shape:", edge_dict_list[0][name].shape)
    else:
        if "cpt" in edge_modality:
            cpt_features, _ = get_feat_seq(
                node_names, cpt_dict, max_seq_len, pad_front, time_deltas=None
            )

        if "icd" in edge_modality:
            icd_features, _ = get_feat_seq(
                node_names, icd_dict, max_seq_len, pad_front, time_deltas=None
            )

        if "lab" in edge_modality:
            lab_features, _ = get_feat_seq(
                node_names, lab_dict, max_seq_len, pad_front, time_deltas=None
            )

        if "med" in edge_modality:
            med_features, _ = get_feat_seq(
                node_names, med_dict, max_seq_len, pad_front, time_deltas=None
            )

        if ("imaging" in edge_modality) and (img_dict is not None):
            img_features, _ = get_feat_seq(
                node_names, img_dict, max_seq_len, pad_front, time_deltas=None
            )
        for t in range(max_seq_len):
            for i, name in enumerate(node_names):
                if name not in edge_dict_list[t]:
                    edge_dict_list[t][name] = []
                if "cpt" in edge_modality:
                    edge_dict_list[t][name] = np.concatenate(
                        [edge_dict_list[t][name], cpt_features[i][t, :]], axis=-1
                    )
                if "icd" in edge_modality:
                    edge_dict_list[t][name] = np.concatenate(
                        [edge_dict_list[t][name], icd_features[i][t, :]], axis=-1
                    )
                if "lab" in edge_modality:
                    edge_dict_list[t][name] = np.concatenate(
                        [edge_dict_list[t][name], lab_features[i][t, :]], axis=-1
                    )
                if "med" in edge_modality:
                    edge_dict_list[t][name] = np.concatenate(
                        [edge_dict_list[t][name], med_features[i][t, :]], axis=-1
                    )
                if ("imaging" in edge_modality) and (img_dict is not None):
                    edge_dict_list[t][name] = np.concatenate(
                        [edge_dict_list[t][name], img_features[i][t, :]], axis=-1
                    )
    #                 print("edge_dict_list[t][name] shape:", edge_dict_list[t][name].shape)
    return edge_dict_list


def compute_edges(
    dist_dict,
    node_names,
    sigma=None,
    top_perc=None,
    gauss_kernel=False,
    admit_reason_dict=None,
):
    # if gauss_kernel:
    #     assert (sigma is not None)

    if "CosineSim" in dist_dict:
        cos_sim = np.array(dist_dict["CosineSim"])
        dist = 1 - cos_sim
    else:
        cos_sim = None
        dist = np.array(dist_dict["Distance"])

    if admit_reason_dict is not None:
        print("Edges will be masked by admission reason...")
        edge_mask = np.array(dist_dict["Mask"])
    else:
        edge_mask = None

    # sanity check shape, (num_nodes) * (num_nodes + 1) / 2, if consider self-edges
    print("dist shape:", dist.shape)

    assert len(dist) == (len(node_names) * (len(node_names) + 1) / 2)

    # apply gaussian kernel, use cosine distance instead of cosine similarity
    if gauss_kernel or (cos_sim is None):
        std = dist.std()
        print("dist std:", std)

        edges = np.exp(-np.square(dist / std))
        # edges = np.exp(-np.square(dist) / (2 * sigma**2))
    else:
        edges = cos_sim

    # mask the edges
    if (edge_mask is None) and (top_perc is not None):
        num = len(edges)
        num_to_keep = int(num * top_perc)
        sorted_dist = np.sort(edges)[::-1]  # descending order
        thresh = sorted_dist[:num_to_keep][-1]
        mask = edges >= thresh
        mask[edges < 0] = 0  # no edge for negative "distance"
        print("Number of non-zero edges:", np.sum(mask))
        edges = edges * mask
    elif edge_mask is not None:
        edges = edges * edge_mask

    return edges


def get_readmission_label_mimic(df_demo, max_seq_len=None):
    """
    Args:
        df_demo: dataframe with patient readmission info and demographics:
        cutoff_months: within how many months are positive labels?
        max_seq_len: maximum number of cxrs to use, count backwards from last cxr within the hospitalization
            if max_seq_len=None, will use all cxrs
    Returns:
        labels: same order as rows in df_demo
        node_included_files: dict, key is node name, value is list of cxr files
        label_splits: list indicating the split of each datapoint in labels and node_included_files
        time_deltas: dict, key is node name, value is an array of day difference between currenct cxr to previous cxr
        total_stay: dict, key is node name, value is total length of stay (in days)
    """
    labels = []
    node_included_files = {}
    label_splits = []
    time_deltas = {}
    total_stay = {}
    time_idxs = {}

    for _, row in tqdm(df_demo.iterrows(), total=len(df_demo)):
        pat = row["subject_id"]
        admit_dt = row["admittime"]
        discharge_dt = row["dischtime"]
        admit_id = row["hadm_id"]
        split = row["splits"]

        curr_name = str(pat) + "_" + str(admit_id)
        if curr_name not in node_included_files:
            node_included_files[curr_name] = []
        else:
            continue

        label_splits.append(split)

        # label
        if str(row["readmitted_within_30days"]).lower() == "true":
            labels.append(1)
        else:
            labels.append(0)

        admit_df = df_demo[df_demo["hadm_id"] == admit_id]

        # sort by study datetime
        admit_df = admit_df.sort_values(
            by=["StudyDate", "StudyTime"], ascending=[True, True]
        )

        # get list of cxrs sorted by study datetime
        for _, admit_row in admit_df.iterrows():
            node_included_files[curr_name].append(admit_row["image_path"])

        # get last max_seq_len cxrs if max_seq_len is specified
        if max_seq_len is not None:
            node_included_files[curr_name] = node_included_files[curr_name][
                -max_seq_len:
            ]

        # time delta & total hospital stay
        curr_timedelta = np.zeros((len(node_included_files[curr_name])))

        prev_date = pd.to_datetime(admit_dt)
        addmission_date = pd.to_datetime(admit_dt)
        discharge_date = pd.to_datetime(discharge_dt)

        hospital_stay = (
            discharge_date.date() - pd.to_datetime(addmission_date).date()
        ).days + 1  # in days
        curr_timeidxs = []
        for t, fn in enumerate(node_included_files[curr_name]):
            study_date = pd.to_datetime(
                df_demo[df_demo["image_path"] == fn]["StudyDate"].values[0],
                format="%Y%m%d",
            )
            if t > 0:  # first time delta is always 0
                curr_timedelta[t] = (
                    study_date.date() - prev_date.date()
                ).days  # in days
                prev_date = study_date

            time_index = (
                study_date.date() - pd.to_datetime(addmission_date).date()
            ).days  # index starts from 0
            curr_timeidxs.append(time_index)

        assert hospital_stay != np.nan
        assert len(curr_timeidxs) > 0  # because at least one cxr
        total_stay[curr_name] = hospital_stay
        time_idxs[curr_name] = curr_timeidxs

        # normalize by hospital stay
        time_deltas[curr_name] = curr_timedelta / hospital_stay

    return (
        np.array(labels),
        node_included_files,
        label_splits,
        time_deltas,
        total_stay,
        time_idxs,
    )


def load_clinical_score_subset(demo_file, merge_train_val=True, score_name="lace"):
    df_demo = pd.read_csv(demo_file)

    # subset of patients with LACE+
    if score_name == "lace":
        df_clin_score = df_demo[~df_demo["LACE"].isnull()].copy()
    else:
        df_clin_score = df_demo[~df_demo["EPIC_SCORE"].isnull()].copy()

    # add node names for convenience
    if "node_name" not in df_clin_score.columns:
        node_names = []
        for _, row in df_clin_score.iterrows():
            pat = str(row["PATIENT_DK"])
            admit_dt = str(row["ADMISSION_DTM"])
            node_names.append(pat + "_" + admit_dt)
        df_clin_score["node_name"] = node_names

    labels, node_included_files, label_splits, _, _, _ = get_readmission_label(
        df_clin_score, cutoff_months=1.0, max_seq_len=None
    )

    # X_train = []
    # y_train = []
    # nodes_train = []
    # X_val = []
    # y_val = []
    # nodes_val = []
    X_test = []
    y_test = []
    nodes_test = []

    i = 0
    for node, _ in node_included_files.items():
        if score_name == "lace":
            clin_score = df_clin_score[df_clin_score["node_name"] == node][
                "LACE"
            ].values[0]
        else:
            clin_score = df_clin_score[df_clin_score["node_name"] == node][
                "EPIC_SCORE"
            ].values[0]
        # if label_splits[i] == "train":
        #     X_train.append(lace)
        #     y_train.append(labels[i])
        #     nodes_train.append(node)
        # elif label_splits[i] == "val":
        #     X_val.append(lace)
        #     y_val.append(labels[i])
        #     nodes_val.append(node)
        # else:
        if label_splits[i] == "test":
            X_test.append(clin_score)
            y_test.append(labels[i])
            nodes_test.append(node)

        i += 1

    # X_train = np.stack(X_train).reshape(-1, 1)
    # y_train = np.stack(y_train)
    # X_val = np.stack(X_val).reshape(-1, 1)
    # y_val = np.stack(y_val)
    X_test = np.stack(X_test).reshape(-1, 1)
    y_test = np.stack(y_test)

    data_dict = {
        "X_test": X_test,
        "y_test": y_test,
        "nodes_test": nodes_test,
    }

    # if merge_train_val:
    #     # no need validation...
    #     X_train = np.concatenate([X_train, X_val]).reshape(-1, 1)
    #     y_train = np.concatenate([y_train, y_val])
    #     nodes_train = nodes_train + nodes_val

    #     print("TRAIN: {}; TEST: {}".format(X_train.shape, X_test.shape))

    #     data_dict = {
    #         "X_train": X_train,
    #         "y_train": y_train,
    #         "nodes_train": nodes_train,
    #         "X_test": X_test,
    #         "y_test": y_test,
    #         "nodes_test": nodes_test,
    #     }
    # else:
    #     print(
    #         "TRAIN: {}; VAL: {}; TEST: {}".format(
    #             X_train.shape, X_val.shape, X_test.shape
    #         )
    #     )
    #     data_dict = {
    #         "X_train": X_train,
    #         "y_train": y_train,
    #         "nodes_train": nodes_train,
    #         "X_val": X_val,
    #         "y_val": y_val,
    #         "nodes_val": nodes_val,
    #         "X_test": X_test,
    #         "y_test": y_test,
    #         "nodes_test": nodes_test,
    #     }

    return data_dict
