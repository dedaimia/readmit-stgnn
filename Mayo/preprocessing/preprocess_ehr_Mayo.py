import pandas as pd
import numpy as np
import os
import pickle
import copy
import argparse
from tqdm import tqdm
import datetime
from datetime import timedelta
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_path, "../"))

from collections import Counter
from sklearn.preprocessing import LabelEncoder


###### Constants ######

#header_proc = '../ehr/processed/'
#header_raw = '../ehr/raw/'

IMG_FEATURE_DIM = 1024

ICD_COLS = ['I10-I16', 'H00-H05', 'L40-L45', 'L60-L75', 'A20-A28', 'S90-S99', 'C73-C75', 'E50-E64', 'N25-N29', 'D65-D69', 'C00-C14', 'R30-R39', 'D60-D64', 'D70-D77', 'H55-H57', 'J00-J06', 'G70-G73', 'F90-F98', 'G30-G32', 'L00-L08', 'L49-L54', 'G10-G14', 'P84-P84', 'I26-I28', 'M80-M85', 'T36-T50', 'H43-H44', 'N30-N39', 'S60-S69', 'S80-S89', 'K35-K38', 'O80-O82', 'F50-F59', 'Q50-Q56', 'R47-R49', 'B15-B19', 'L80-L99', 'UNKNOWN', 'M50-M54', 'I30-I52', 'O60-O77', 'H60-H62', 'Q80-Q89', 'R50-R69', 'E15-E16', 'A15-A19', 'A70-A74', 'M00-M02', 'M20-M25', 'F01-F09', 'S50-S59', 'H49-H52', 'O20-O29', 'W20-W49', 'T14-T14', 'R99-R99', 'I60-I69', 'C51-C58', 'B95-B97', 'J96-J99', 'Q10-Q18', 'B25-B34', 'M26-M27', 'H40-H42', 'R10-R19', 'S40-S49', 'Q30-Q34', 'M05-M14', 'C76-C80', 'N17-N19', 'F30-F39', 'M60-M63', 'J60-J70', 'B00-B09', 'B65-B83', 'P19-P29', 'N40-N53', 'J30-J39', 'D37-D48', 'B35-B49', 'D55-D59', 'S70-S79', 'D78-D78', 'P00-P04', 'C60-C63', 'L10-L14', 'F70-F79', 'H30-H36', 'J20-J22', 'C69-C72', 'R90-R94', 'C45-C49', 'W00-W19', 'D00-D09', 'P70-P74', 'T07-T07', 'S20-S29', 'Y83-Y84', 'J40-J47', 'V10-V19', 'K00-K14', 'N99-N99', 'T26-T28', 'E00-E07', 'M15-M19', 'R70-R79', 'T66-T78', 'P50-P61', 'P05-P08', 'M70-M79', 'J09-J18', 'K80-K87', 'I20-I25', 'H15-H22', 'I00-I02', 'Q00-Q07', 'A90-A99', 'G35-G37', 'Q90-Q99', 'L55-L59', 'A80-A89', 'D49-D49', 'Y70-Y82', 'H53-H54', 'F60-F69', 'Q65-Q79', 'P35-P39', 'S00-S09', 'K50-K52', 'H25-H28', 'D3A-D3A', 'J95-J95', 'M45-M49', 'M40-M43', 'Q60-Q64', 'T33-T34', 'Y90-Y99', 'A00-A09', 'Q20-Q28', 'G40-G47', 'F80-F89', 'H90-H94', 'S10-S19', 'D10-D36', 'B90-B94', 'J80-J84', 'N10-N16', 'B50-B64', 'T15-T19', 'G60-G65', 'F40-F48', 'I80-I89', 'I95-I99', 'K55-K64', 'T80-T88', 'M95-M95', 'E20-E35', 'P90-P96', 'J90-J94', 'A50-A64', 'K20-K31', 'E70-E88', 'L76-L76', 'R00-R09', 'C30-C39', 'H10-H11', 'B20-B20', 'Y21-Y33', 'G80-G83', 'O30-O48', 'E89-E89', 'C7A-C7A', 'I05-I09', 'B85-B89', 'K70-K77', 'C64-C68', 'M97-M97', 'B99-B99', 'D80-D89', 'R97-R97', 'M91-M94', 'Q38-Q45', 'R20-R23', 'M65-M67', 'N20-N23', 'O00-O08', 'A75-A79', 'K90-K95', 'C15-C26', 'T79-T79', 'T30-T32', 'I70-I79', 'R40-R46', 'R25-R29', 'G50-G59', 'M96-M96', 'A65-A69', 'E40-E46', 'F10-F19', 'F20-F29', 'K65-K68', 'H65-H75', 'N00-N08', 'P10-P15', 'G00-G09', 'P80-P83', 'R80-R82', 'N60-N65', 'N70-N77', 'E08-E13', 'C43-C44', 'V80-V89', 'X71-X83', 'S30-S39', 'Q35-Q37', 'C40-C41', 'M30-M36', 'C81-C96', 'V20-V29', 'H80-H83', 'L20-L30', 'K40-K46', 'D50-D53', 'O10-O16', 'O85-O92', 'E65-E68', 'H46-H47', 'O09-O09', 'A30-A49', 'T20-T25', 'C50-C50', 'H95-H95', 'M99-M99', 'P76-P78', 'M86-M90', 'T51-T65', 'J85-J86', 'G89-G99', 'V40-V49', 'B10-B10', 'R83-R89', 'N80-N98', 'G20-G26']

CPT_COLS = ['Physical Medicine and Rehabilitation Evaluations', 'Care Management Evaluation and Management Services', 'Psychiatry Services and Procedures', 'Therapeutic Drug Assays', 'Microbiology Procedures', 'Drug Assay Procedures', 'Medical Nutrition Therapy Procedures', 'Breast, Mammography', 'Organ or Disease Oriented Panels', 'Surgical Procedures on the Mediastinum and Diaphragm', 'Remote Real-Time Interactive Video-conferenced Critical Care Services and Other Undefined Category Codes', 'Radiologic Guidance', 'Cellular Therapy Procedures', 'Pulmonary Procedures', 'Dialysis Services and Procedures', 'Neurology and Neuromuscular Procedures', 'Surgical Procedures on the Respiratory System', 'Surgical Pathology Procedures', 'General Surgical Procedures', 'Transfusion Medicine Procedures', 'Central Nervous System Assessments/Tests (eg, Neuro-Cognitive, Mental Status, Speech Testing)', 'Surgical Procedures on the Integumentary System', 'Surgical Procedures on the Male Genital System', 'Gastroenterology Procedures', 'Cytogenetic Studies', 'Acupuncture Procedures', 'Surgical Procedures on the Female Genital System', 'Reproductive Medicine Procedures', 'Various Services - Category III Codes', 'Cardiovascular Procedures', 'Special Otorhinolaryngologic Services and Procedures', 'Immunization Administration for Vaccines/Toxoids', 'Special Dermatological Procedures', 'Radiation Oncology Treatment', 'Immunology Procedures', 'Moderate (Conscious) Sedation', 'Cytopathology Procedures', 'Surgical Procedures on the Hemic and Lymphatic Systems', 'Medical Genetics and Genetic Counseling Services', 'Surgical Procedures on the Digestive System', 'Other Evaluation and Management Services', 'Operating Microscope Procedures', 'Hydration, Therapeutic, Prophylactic, Diagnostic Injections and Infusions, and Chemotherapy and Other Highly Complex Drug or Highly Complex Biologic Agent Administration', 'Surgical Procedures on the Auditory System', 'Surgical Procedures for Maternity Care and Delivery', 'Other Medicine Services and Procedures', 'Hematology and Coagulation Procedures', 'Diagnostic Ultrasound Procedures', 'Ophthalmology Services and Procedures', 'Anesthesia for Other Procedures', 'Proprietary Laboratory Analyses', 'Qualifying Circumstances for Anesthesia', 'Allergy and Clinical Immunology Procedures', 'Surgical Procedures on the Eye and Ocular Adnexa', 'Other Pathology and Laboratory Procedures', 'Non-Face-to-Face Evaluation and Management Services', 'Surgical Procedures on the Musculoskeletal System', 'Special Services, Procedures and Reports', 'Pacemaker - Leadless and Pocketless System', 'Surgical Procedures on the Nervous System', 'Diagnostic Radiology (Diagnostic Imaging) Procedures', 'Phrenic Nerve Stimulation System Procedures', 'Nuclear Medicine Procedures', 'Advance Care Planning Evaluation and Management Services', 'Non-Invasive Vascular Diagnostic Studies', 'Bone/Joint Studies', 'Atherectomy (Open or Percutaneous) for Supra-Inguinal Arteries and Other Undefined Category Codes', 'Patient History', 'Chemistry Procedures', 'Surgical Procedures on the Cardiovascular System', 'Surgical Procedures on the Urinary System', 'Surgical Procedures on the Endocrine System']

LAB_COLS = ['Glucose, POCT, B', 'Hemoglobin', 'Platelet Count', 'Hematocrit', 'Erythrocytes', 'Leukocytes', 'Potassium, S', 'Sodium, S', 'Creatinine, S', 'Bicarbonate, S', 'Chloride, S', 'Anion Gap', 'Calcium, Total, S', 'Glucose, S', 'Bld Urea Nitrog(BUN), S', 'Lymphocytes', 'Monocytes', 'Neutrophils', 'Eosinophils', 'Basophils', 'Potassium, P', 'Sodium, P', 'Chloride, P', 'Bicarbonate, P', 'Creatinine, P', 'Glucose, P', 'Anion Gap, P', 'Calcium, Total, P', 'pH', 'Hemoglobin, B', 'pCO2', 'pO2', 'Lactate, P', 'Glucose', 'Lactate, B', 'pH, U', 'Potassium, B', 'Sodium, B', 'Troponin T, 6 hr, 5th gen', 'Troponin T, 2 hr, 5th gen', 'Troponin T, Baseline, 5th gen', 'Venous pH', 'Venous pCO2', 'Bicarbonate', 'Sodium', 'Potassium', 'Creatinine', 'Creatinine, B', 'Chloride', 'Sodium, U', 'Chloride, B']


DEMO_COLS = [
    "PATIENT_AGE_NEW",
    "PATIENT_RACE_NAME",
    "PATIENT_GENDER_NAME",
    "PATIENT_ETHNICITY_NAME",
]

MEDICATION_COLS = ['ANESTHETICS', 'ANTINEOPLASTICS', 'BIOLOGICALS', 'DIAGNOSTIC', 'ANTIBIOTICS', 'CONTRACEPTIVES', 'ANTICOAGULANTS', 'CARDIOVASCULAR', 'PSYCHOTHERAPEUTIC DRUGS', 'CNS DRUGS', 'ELECT/CALORIC/H2O', 'EENT PREPS', 'BLOOD', 'CARDIAC DRUGS', 'GASTROINTESTINAL', 'UNCLASSIFIED DRUG PRODUCTS', 'ANTIFUNGALS', 'AUTONOMIC DRUGS', 'DIURETICS', 'SEDATIVE/HYPNOTICS', 'IMMUNOSUPPRESSANTS', 'HORMONES', 'ANTIHYPERGLYCEMICS', 'ANTIARTHRITICS', 'COLONY STIMULATING FACTORS', 'ANTIASTHMATICS', 'ANALGESICS', 'VITAMINS', 'SKIN PREPS', 'MUSCLE RELAXANTS', 'ANTIVIRALS', 'ANTIPARKINSON DRUGS', 'ANTIHISTAMINES', 'ANTIPLATELET DRUGS', 'MISCELLANEOUS MEDICAL SUPPLIES, DEVICES, NON-DRUG', 'ANTIINFECTIVES/MISCELLANEOUS', 'SMOKING DETERRENTS', 'COUGH/COLD PREPARATIONS', 'HERBALS', 'ANALGESIC AND ANTIHISTAMINE COMBINATION', 'ANTIALLERGY', 'ANTIPARASITICS', 'ANTIINFLAM.TUMOR NECROSIS FACTOR INHIBITING AGENTS', 'ANTIHISTAMINE AND DECONGESTANT COMBINATION', 'PRE-NATAL VITAMINS']

ADMIT_COLS = ['vascular disease', 'heart disease', 'disease of metabolism', 'kidney disease', 'sleep disorder', 'cancer', 'disease of mental health', 'overnutrition', 'uterine cancer', 'intestinal disease', 'anemia', 'bacterial infectious disease', 'disease of cellular proliferation', 'aortic valve prolapse', 'mitral valve stenosis', 'blood coagulation disease', 'neuropathy', 'aortic valve stenosis', 'respiratory system disease', 'bone disease', 'brain disease', 'lung disease', 'substance-related disorder', 'nervous system disease', 'genetic disease', 'nutrition disease', 'peripheral nervous system disease', 'cellulitis', 'aortic disease', 'renal tubular transport disease', 'autoimmune disease of the nervous system', 'substance abuse', 'immune system disease', 'thoracic disease', 'autoimmune disease', 'carcinoma', 'bacterial pneumonia', 'disease by infectious agent', 'liver disease', 'muscular disease', 'parathyroid gland disease', 'urinary tract obstruction', 'B-cell lymphoma', 'pancreatitis', 'physical disorder', 'mouth disease', 'prostatic hypertrophy', 'syndrome', 'hepatobiliary disease', 'inherited metabolic disorder', 'gonadal disease', 'endocrine gland cancer', 'hepatic vascular disease', 'viral infectious disease', 'kidney cancer', 'cardiovascular system disease', 'bone marrow disease', 'central nervous system disease', 'leukopenia', 'aphasia', 'connective tissue cancer', 'vein disease', 'pupil disease', 'acanthoma', 'bicuspid aortic valve disease', 'pneumonia', 'mucositis', 'upper respiratory tract disease', 'reproductive organ cancer', 'bladder disease', 'lymph node disease', 'peripheral vascular disease', 'movement disease', 'lower urinary tract calculus', 'female reproductive system disease', 'familial atrial fibrillation', 'uterine disease', 'arthropathy', 'bone cancer', 'hypothyroidism', 'tuberculosis', 'cerebrum cancer', 'gallbladder disease', 'myasthenia gravis', 'gastric outlet obstruction', 'adrenal gland disease', 'colonic benign neoplasm', 'senile cataract', 'bacteriuria', 'hypereosinophilic syndrome', 'central nervous system mesenchymal non-meningothelial tumor', 'exocrine pancreatic insufficiency', 'hemangiopericytoma', 'penile disease', 'esophageal disease', 'vaginal disease', 'colon cancer', 'tauopathy', 'urethral disease', 'diarrhea', 'dermatitis', 'mitral valve insufficiency', 'urolithiasis', 'eyelid disease', 'thyroiditis', 'cartilage disease', 'fasciitis', 'peroxisomal disease', 'angioedema', 'amyotrophic lateral sclerosis', 'liver sarcoma', 'goiter', 'laryngeal disease', 'leukocyte disease', 'male urethral cancer', 'myelitis', 'urticaria', 'leg dermatosis', 'calcinosis', 'intestinal infectious disease', 'hydrocele', 'primary immunodeficiency disease', 'endometriosis', 'mitochondrial metabolism disease', 'adenoma', 'lipomatosis', 'ureteral disease', 'hiatus hernia', 'ischemia', 'congenital hemolytic anemia', 'myopathy', 'splenic disease', 'external ear disease', 'interstitial keratitis', 'panniculitis', 'myocarditis', 'epididymis disease', 'exanthem', 'hyperthyroidism', 'vascular cancer', 'liver benign neoplasm', 'hepatocellular carcinoma', 'nephrolithiasis', 'retinoblastoma', 'hand dermatosis', 'phobic disorder', 'gastric dilatation', 'sex cord-gonadal stromal tumor', 'heart cancer', 'orbital disease', 'mature B-cell neoplasm', 'iris disease', 'strabismus', 'lipodystrophy', 'thyrotoxicosis', 'diabetic cataract', 'motor neuron disease', 'auditory system cancer', 'dystonia', 'neurofibroma', 'gastritis', 'T-cell lymphoblastic leukemia/lymphoma', 'annular pancreas', 'synovitis', 'ocular hypotension', 'peritonitis', 'extrinsic allergic alveolitis', 'hypopituitarism', 'myositis', 'proteinuria', 'varicose veins', 'lymphedema', 'endocrine pancreas disease', 'mononeuritis of upper limb and mononeuritis multiplex', 'blood protein disease', 'adenocarcinoma', 'vasculitis', 'sebaceous gland disease', 'thyroid crisis', 'extracranial neuroblastoma', 'mutism', 'motor neuritis', 'specific language impairment', 'Dieulafoy lesion', 'nail disease', 'neuromuscular disease', 'lacrimal system cancer', 'placenta disease', 'lens disease', 'tenosynovitis', 'cleft palate', 'gastroenteritis', 'sweat gland disease', 'muscle benign neoplasm', 'cavernous hemangioma', 'vascular skin disease', 'parasitic helminthiasis infectious disease', 'distal arthrogryposis', 'pancreatic agenesis', 'appendix disease', 'gastrointestinal tuberculosis', 'necrosis of pituitary', 'trigeminal nerve disease', 'cranial nerve neoplasm', 'histiocytosis', 'severe combined immunodeficiency', 'posterior polar cataract', 'peptic ulcer disease', 'mastocytosis', 'female reproductive organ cancer', 'urethral benign neoplasm', 'prion disease', 'systemic scleroderma', 'vaginal carcinosarcoma', 'male infertility', 'neuroblastoma', 'spinal cord disease', 'infertility', 'pericardium cancer', 'pancreatic steatorrhea', 'thyroid malformation', 'metal metabolism disorder', 'pigmentation disease', 'lymphangitis', 'leiomyoma', 'choroidal sclerosis', 'mixed cell type cancer']


# DEMO_FILE = os.path.join(
#     header_proc, "cohort_file_w_demo.csv"
# )
# CPT_FILE = os.path.join(
#     header_proc, "cohort_file_w_cpt.csv"
# )
# ICD_FILE = os.path.join(
#     header_proc, "cohort_file_w_icd.csv"
# )
# LAB_FILE = os.path.join(
#     header_proc, 'cohort_file_w_lab.csv'
# )
# MED_FILE = os.path.join(
#     header_proc, 'cohort_file_w_med.csv'
# )




COLS_IRRELEVANT = [
    "PATIENT_DK",
    "ADMIT_DTM",
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

encoders_dct = pickle.load(open(os.path.join(script_path, 'categorical_encoders.pkl'), 'rb'))
encoders_dct['Chloride, U'] = encoders_dct['Chloride']
pickle.dump(encoders_dct, open(os.path.join(script_path, 'categorical_encoders.pkl'), 'wb'))

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
        "ADMIT_DTM": [],
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
        admit_dt = row["ADMIT_DTM"]
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
            df_ehr_count["ADMIT_DTM"].append(admit_dt)
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
        "ADMIT_DTM": [],
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
        admit_dt = row["ADMIT_DTM"]
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
            if pd.Timestamp(pat_lab["ADMIT_DTM"]).strftime(
                "%Y/%m/%d %X"
            ) == pd.Timestamp(admit_dt).strftime("%Y/%m/%d %X"):
                curr_lab_df = curr_lab_df.append(pat_lab)

        for dt in dt_range:
            day_number = (dt.date() - pd.to_datetime(admit_dt).date()).days + 1
            curr_labs = curr_lab_df[curr_lab_df["Day Nbr"] == float(day_number)]

            df_lab_onehot["PATIENT_DK"].append(pat)
            df_lab_onehot["ADMIT_DTM"].append(admit_dt)
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
        "ADMIT_DTM": [],
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
        admit_dt = row["ADMIT_DTM"]
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
            if pd.Timestamp(pat_med["ADMIT_DTM"]).strftime(
                "%Y/%m/%d %X"
            ) == pd.Timestamp(admit_dt).strftime("%Y/%m/%d %X"):
                curr_med_df = curr_med_df.append(pat_med)

        for dt in dt_range:
            day_number = (dt.date() - pd.to_datetime(admit_dt).date()).days + 1
            curr_meds = curr_med_df[curr_med_df["Day Nbr"] == float(day_number)]

            df_med_count["PATIENT_DK"].append(pat)
            df_med_count["ADMIT_DTM"].append(admit_dt)
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


def preproc_ehr4tabnet(X):

    train_indices = X[X["split"] == "train"].index
    target = "target"

    types = X.dtypes

    # encode categorical variables
    categorical_columns = []
    categorical_dims = {}

    categorical_encoders = {}
    for col in tqdm(X.columns):
        if col in COLS_IRRELEVANT:
            continue
        if col in CAT_COLUMNS:
            l_enc = encoders_dct[col]#LabelEncoder() - use pretrained
            print(col, X[col].unique())
            X[col] = X[col].fillna("VV_likely")
            X[col] = X[col].replace(
                {0.1590516924884353: "UNKNOWN"}
            )  # handle this weirdness
            # print(col, X[col].unique())
            X[col] = l_enc.transform(X[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

            categorical_encoders[col] = l_enc
        else:
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)

    feature_cols = [
        col for col in X.columns if (col != target) and (col not in COLS_IRRELEVANT)
    ]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f]
        for i, f in enumerate(feature_cols)
        if f in categorical_columns
    ]

    return {
        "X": X,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
        "cat_encoders": categorical_encoders,
    }


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
            
            # print(col, np.unique(X[col].values), encoders_dct[col].classes_)
            
            # curr_enc = pd.get_dummies(
            #     X[col], prefix=col
            # )  # this will transform into one-hot encoder
            # print(curr_enc.shape)

            vals = encoders_dct[col].transform(X[col].values)
            curr_enc = pd.DataFrame(columns=[col+'_'+str(c) for c in range(len(encoders_dct[col].classes_))])
            # print(curr_enc.columns)
            for i in range(len(vals)):
               curr_enc.at[i, col+'_'+str(vals[i])] = 1
            # print(curr_enc.columns)
            curr_enc = curr_enc.fillna(0)
            # print(curr_enc.shape)

            X_enc.append(curr_enc)
            num_cols += curr_enc.shape[-1]
            curr_cols = [col + "_" + str(i) for i in range(curr_enc.shape[-1])]
            categorical_columns.extend(curr_cols)
            for c in curr_cols:
                categorical_dims[c] = 1
            # print('catgeorical', len(X_enc))
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
    print(categorical_columns)
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    print(cat_idxs)
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


def ehr2sequence(preproc_dict,  by="day"):#df_demo,
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
    for node_name in df["node_name"].unique():#, cxr_files in tqdm(node_included_files.items()):
        # print(node_name)
        ehr_row = df[df["node_name"] == node_name]
        curr_admit = ehr_row["ADMIT_DTM"].values[0]
        curr_discharge = ehr_row["DISCHARGE_DTM"].values[0]
        curr_pat = ehr_row["PATIENT_DK"].values[0]

        if by == "day":
            dt_range = pd.date_range(
                start=pd.to_datetime(curr_admit).date(),
                end=pd.to_datetime(curr_discharge).date()-timedelta(days=1),
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


def main(df_combined, header_proc):
    if 'split' not in df_combined.columns:
        df_combined['split'] = 'Test'
        
    cols = COLS_IRRELEVANT+DEMO_COLS+ADMIT_COLS+CPT_COLS+ICD_COLS+LAB_COLS+MEDICATION_COLS
    cols = [c for c in cols if c not in SUBGOUPRS_EXCLUDED]
    df_combined = df_combined[cols].copy()

    # # cpt
    # df_cpt_count = ehr_bag_of_words(
    #     df_demo,
    #     df_cpt,
    #     ehr_type="cpt",
    #     time_step_by="day",
    #     filter_freq=None,
    #     label_cutoff=1.0,#args.label_cutoff,
    # )
    # print("Unique node names:", len(list(set(df_cpt_count["node_name"].tolist()))))

    # # icd
    # df_icd_count = ehr_bag_of_words(
    #     df_demo,
    #     df_icd,
    #     ehr_type="icd",
    #     time_step_by="day",
    #     filter_freq=None,
    #     label_cutoff=1.0,#args.label_cutoff,
    # )
    # print("Unique node names:", len(list(set(df_icd_count["node_name"].tolist()))))

    # # lab
    # df_lab_onehot = lab_one_hot(
    #     df_demo,
    #     df_lab,
    #     time_step_by="day",
    #     filter_freq=None,
    #     label_cutoff=1.0,#args.label_cutoff,
    # )
    # print("Unique node names:", len(list(set(df_lab_onehot["node_name"].tolist()))))

    # # medication
    # df_med_count = medication_bag_of_words(
    #     df_demo,
    #     df_med,
    #     time_step_by="day",
    #     filter_freq=None,
    #     label_cutoff=1.0,#args.label_cutoff,
    # )

    # # combine
    # df_combined = pd.concat(
    #     [df_cpt_count, df_icd_count, df_lab_onehot, df_med_count], axis=1
    # )

    # # drop duplicated columns
    # df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    # df_combined.to_csv(os.path.join(header_proc, "ehr_combined.csv"), index=False)
    # # df_combined = pd.read_csv(os.path.join(args.save_dir, "ehr_combined.csv"))

    # df_combined = pd.read_csv(header_proc, "ehr_combined.csv")
    # further preprocess it to meet tabnet format
    
    for format in ["gnn", "tabnet"]:#"tabnet", "gnn"]:
        if format == "tabnet":
            preproc_dict = preproc_ehr4tabnet(df_combined)
        else:
            preproc_dict = preproc_ehr(df_combined)

        feature_cols = preproc_dict["feature_cols"]
        demo_cols = DEMO_COLS
        cpt_cols = CPT_COLS
        icd_cols = ICD_COLS
        med_cols = MEDICATION_COLS
        lab_cols = LAB_COLS
        admit_cols = ADMIT_COLS
        # demo_cols = [
        #     col for col in feature_cols if any([s for s in DEMO_COLS if s in col])
        # ]
        # cpt_cols = [
        #     col for col in feature_cols if col in CPT_COLS
        # ]
        # icd_cols = [
        #     col for col in feature_cols if col in ICD_COLS
        # ]
        # lab_cols = [
        #     col for col in feature_cols if any([s for s in LAB_COLS if s in col])
        # ]
        # med_cols = [
        #     col for col in feature_cols if any([s for s in MEDICATION_COLS if s in col])
        # ]
        # admit_cols = [
        #     col for col in feature_cols if any([s for s in ADMIT_COLS if s in col])
        # ]

        preproc_dict["demo_cols"] = demo_cols
        preproc_dict["cpt_cols"] = cpt_cols
        preproc_dict["icd_cols"] = icd_cols
        preproc_dict["lab_cols"] = lab_cols
        preproc_dict["med_cols"] = med_cols
        preproc_dict["admit_cols"] = admit_cols

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
            seq_dict = ehr2sequence(preproc_dict,  by=by)#df_demo,

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