import pandas as pd
import numpy as np
import pickle as pkl
import os
import sys
import shutil
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

import cohort_selection
from demographic_featurization import demo_featurization
from CPT_featurization import cpt_featurization
from ICD10_featurization import icd_featurization
from lab_featurization import lab_featurization
from med_featurization import med_featurization

import  preprocess_ehr_Mayo
header_raw = '../ehr/raw/'
header_proc = '../ehr/processed/'


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

ALL_COLS = ['PATIENT_AGE_NEW', 'PATIENT_RACE_NAME', 'PATIENT_GENDER_NAME', 'PATIENT_ETHNICITY_NAME', 'vascular disease', 'heart disease', 'disease of metabolism', 'kidney disease', 'sleep disorder', 'cancer', 'disease of mental health', 'overnutrition', 'uterine cancer', 'intestinal disease', 'anemia', 'bacterial infectious disease', 'disease of cellular proliferation', 'aortic valve prolapse', 'mitral valve stenosis', 'blood coagulation disease', 'neuropathy', 'aortic valve stenosis', 'respiratory system disease', 'bone disease', 'brain disease', 'lung disease', 'substance-related disorder', 'nervous system disease', 'genetic disease', 'nutrition disease', 'peripheral nervous system disease', 'cellulitis', 'aortic disease', 'renal tubular transport disease', 'autoimmune disease of the nervous system', 'substance abuse', 'immune system disease', 'thoracic disease', 'autoimmune disease', 'carcinoma', 'bacterial pneumonia', 'disease by infectious agent', 'liver disease', 'muscular disease', 'parathyroid gland disease', 'urinary tract obstruction', 'B-cell lymphoma', 'pancreatitis', 'physical disorder', 'mouth disease', 'prostatic hypertrophy', 'syndrome', 'hepatobiliary disease', 'inherited metabolic disorder', 'gonadal disease', 'endocrine gland cancer', 'hepatic vascular disease', 'viral infectious disease', 'kidney cancer', 'cardiovascular system disease', 'bone marrow disease', 'central nervous system disease', 'leukopenia', 'aphasia', 'connective tissue cancer', 'vein disease', 'pupil disease', 'acanthoma', 'bicuspid aortic valve disease', 'pneumonia', 'mucositis', 'upper respiratory tract disease', 'reproductive organ cancer', 'bladder disease', 'lymph node disease', 'peripheral vascular disease', 'movement disease', 'lower urinary tract calculus', 'female reproductive system disease', 'familial atrial fibrillation', 'uterine disease', 'arthropathy', 'bone cancer', 'hypothyroidism', 'tuberculosis', 'cerebrum cancer', 'gallbladder disease', 'myasthenia gravis', 'gastric outlet obstruction', 'adrenal gland disease', 'colonic benign neoplasm', 'senile cataract', 'bacteriuria', 'hypereosinophilic syndrome', 'central nervous system mesenchymal non-meningothelial tumor', 'exocrine pancreatic insufficiency', 'hemangiopericytoma', 'penile disease', 'esophageal disease', 'vaginal disease', 'colon cancer', 'tauopathy', 'urethral disease', 'diarrhea', 'dermatitis', 'mitral valve insufficiency', 'urolithiasis', 'eyelid disease', 'thyroiditis', 'cartilage disease', 'fasciitis', 'peroxisomal disease', 'angioedema', 'amyotrophic lateral sclerosis', 'liver sarcoma', 'goiter', 'laryngeal disease', 'leukocyte disease', 'male urethral cancer', 'myelitis', 'urticaria', 'leg dermatosis', 'calcinosis', 'intestinal infectious disease', 'hydrocele', 'primary immunodeficiency disease', 'endometriosis', 'mitochondrial metabolism disease', 'adenoma', 'lipomatosis', 'ureteral disease', 'hiatus hernia', 'ischemia', 'congenital hemolytic anemia', 'myopathy', 'splenic disease', 'external ear disease', 'interstitial keratitis', 'panniculitis', 'myocarditis', 'epididymis disease', 'exanthem', 'hyperthyroidism', 'vascular cancer', 'liver benign neoplasm', 'hepatocellular carcinoma', 'nephrolithiasis', 'retinoblastoma', 'hand dermatosis', 'phobic disorder', 'gastric dilatation', 'sex cord-gonadal stromal tumor', 'heart cancer', 'orbital disease', 'mature B-cell neoplasm', 'iris disease', 'strabismus', 'lipodystrophy', 'thyrotoxicosis', 'diabetic cataract', 'motor neuron disease', 'auditory system cancer', 'dystonia', 'neurofibroma', 'gastritis', 'T-cell lymphoblastic leukemia/lymphoma', 'annular pancreas', 'synovitis', 'ocular hypotension', 'peritonitis', 'extrinsic allergic alveolitis', 'hypopituitarism', 'myositis', 'proteinuria', 'varicose veins', 'lymphedema', 'endocrine pancreas disease', 'mononeuritis of upper limb and mononeuritis multiplex', 'blood protein disease', 'adenocarcinoma', 'vasculitis', 'sebaceous gland disease', 'thyroid crisis', 'extracranial neuroblastoma', 'mutism', 'motor neuritis', 'specific language impairment', 'Dieulafoy lesion', 'nail disease', 'neuromuscular disease', 'lacrimal system cancer', 'placenta disease', 'lens disease', 'tenosynovitis', 'cleft palate', 'gastroenteritis', 'sweat gland disease', 'muscle benign neoplasm', 'cavernous hemangioma', 'vascular skin disease', 'parasitic helminthiasis infectious disease', 'distal arthrogryposis', 'pancreatic agenesis', 'appendix disease', 'gastrointestinal tuberculosis', 'necrosis of pituitary', 'trigeminal nerve disease', 'cranial nerve neoplasm', 'histiocytosis', 'severe combined immunodeficiency', 'posterior polar cataract', 'peptic ulcer disease', 'mastocytosis', 'female reproductive organ cancer', 'urethral benign neoplasm', 'prion disease', 'systemic scleroderma', 'vaginal carcinosarcoma', 'male infertility', 'neuroblastoma', 'spinal cord disease', 'infertility', 'pericardium cancer', 'pancreatic steatorrhea', 'thyroid malformation', 'metal metabolism disorder', 'pigmentation disease', 'lymphangitis', 'leiomyoma', 'choroidal sclerosis', 'mixed cell type cancer', 'Physical Medicine and Rehabilitation Evaluations', 'Care Management Evaluation and Management Services', 'Psychiatry Services and Procedures', 'Therapeutic Drug Assays', 'Microbiology Procedures', 'Drug Assay Procedures', 'Medical Nutrition Therapy Procedures', 'Breast, Mammography', 'Organ or Disease Oriented Panels', 'Surgical Procedures on the Mediastinum and Diaphragm', 'Remote Real-Time Interactive Video-conferenced Critical Care Services and Other Undefined Category Codes', 'Radiologic Guidance', 'Cellular Therapy Procedures', 'Pulmonary Procedures', 'Dialysis Services and Procedures', 'Neurology and Neuromuscular Procedures', 'Surgical Procedures on the Respiratory System', 'Surgical Pathology Procedures', 'General Surgical Procedures', 'Transfusion Medicine Procedures', 'Central Nervous System Assessments/Tests (eg, Neuro-Cognitive, Mental Status, Speech Testing)', 'Surgical Procedures on the Integumentary System', 'Surgical Procedures on the Male Genital System', 'Gastroenterology Procedures', 'Cytogenetic Studies', 'Acupuncture Procedures', 'Surgical Procedures on the Female Genital System', 'Reproductive Medicine Procedures', 'Various Services - Category III Codes', 'Cardiovascular Procedures', 'Special Otorhinolaryngologic Services and Procedures', 'Immunization Administration for Vaccines/Toxoids', 'Special Dermatological Procedures', 'Radiation Oncology Treatment', 'Immunology Procedures', 'Moderate (Conscious) Sedation', 'Cytopathology Procedures', 'Surgical Procedures on the Hemic and Lymphatic Systems', 'Medical Genetics and Genetic Counseling Services', 'Surgical Procedures on the Digestive System', 'Other Evaluation and Management Services', 'Operating Microscope Procedures', 'Hydration, Therapeutic, Prophylactic, Diagnostic Injections and Infusions, and Chemotherapy and Other Highly Complex Drug or Highly Complex Biologic Agent Administration', 'Surgical Procedures on the Auditory System', 'Surgical Procedures for Maternity Care and Delivery', 'Other Medicine Services and Procedures', 'Hematology and Coagulation Procedures', 'Diagnostic Ultrasound Procedures', 'Ophthalmology Services and Procedures', 'Anesthesia for Other Procedures', 'Proprietary Laboratory Analyses', 'Qualifying Circumstances for Anesthesia', 'Allergy and Clinical Immunology Procedures', 'Surgical Procedures on the Eye and Ocular Adnexa', 'Other Pathology and Laboratory Procedures', 'Non-Face-to-Face Evaluation and Management Services', 'Surgical Procedures on the Musculoskeletal System', 'Special Services, Procedures and Reports', 'Pacemaker - Leadless and Pocketless System', 'Surgical Procedures on the Nervous System', 'Diagnostic Radiology (Diagnostic Imaging) Procedures', 'Phrenic Nerve Stimulation System Procedures', 'Nuclear Medicine Procedures', 'Advance Care Planning Evaluation and Management Services', 'Non-Invasive Vascular Diagnostic Studies', 'Bone/Joint Studies', 'Atherectomy (Open or Percutaneous) for Supra-Inguinal Arteries and Other Undefined Category Codes', 'Patient History', 'Chemistry Procedures', 'Surgical Procedures on the Cardiovascular System', 'Surgical Procedures on the Urinary System', 'Surgical Procedures on the Endocrine System', 'I10-I16', 'H00-H05', 'L40-L45', 'L60-L75', 'A20-A28', 'S90-S99', 'C73-C75', 'E50-E64', 'N25-N29', 'D65-D69', 'C00-C14', 'R30-R39', 'D60-D64', 'D70-D77', 'H55-H57', 'J00-J06', 'G70-G73', 'F90-F98', 'G30-G32', 'L00-L08', 'L49-L54', 'G10-G14', 'P84-P84', 'I26-I28', 'M80-M85', 'T36-T50', 'H43-H44', 'N30-N39', 'S60-S69', 'S80-S89', 'K35-K38', 'O80-O82', 'F50-F59', 'Q50-Q56', 'R47-R49', 'B15-B19', 'L80-L99', 'UNKNOWN', 'M50-M54', 'I30-I52', 'O60-O77', 'H60-H62', 'Q80-Q89', 'R50-R69', 'E15-E16', 'A15-A19', 'A70-A74', 'M00-M02', 'M20-M25', 'F01-F09', 'S50-S59', 'H49-H52', 'O20-O29', 'W20-W49', 'T14-T14', 'R99-R99', 'I60-I69', 'C51-C58', 'B95-B97', 'J96-J99', 'Q10-Q18', 'B25-B34', 'M26-M27', 'H40-H42', 'R10-R19', 'S40-S49', 'Q30-Q34', 'M05-M14', 'C76-C80', 'N17-N19', 'F30-F39', 'M60-M63', 'J60-J70', 'B00-B09', 'B65-B83', 'P19-P29', 'N40-N53', 'J30-J39', 'D37-D48', 'B35-B49', 'D55-D59', 'S70-S79', 'D78-D78', 'P00-P04', 'C60-C63', 'L10-L14', 'F70-F79', 'H30-H36', 'J20-J22', 'C69-C72', 'R90-R94', 'C45-C49', 'W00-W19', 'D00-D09', 'P70-P74', 'T07-T07', 'S20-S29', 'Y83-Y84', 'J40-J47', 'V10-V19', 'K00-K14', 'N99-N99', 'T26-T28', 'E00-E07', 'M15-M19', 'R70-R79', 'T66-T78', 'P50-P61', 'P05-P08', 'M70-M79', 'J09-J18', 'K80-K87', 'I20-I25', 'H15-H22', 'I00-I02', 'Q00-Q07', 'A90-A99', 'G35-G37', 'Q90-Q99', 'L55-L59', 'A80-A89', 'D49-D49', 'Y70-Y82', 'H53-H54', 'F60-F69', 'Q65-Q79', 'P35-P39', 'S00-S09', 'K50-K52', 'H25-H28', 'D3A-D3A', 'J95-J95', 'M45-M49', 'M40-M43', 'Q60-Q64', 'T33-T34', 'Y90-Y99', 'A00-A09', 'Q20-Q28', 'G40-G47', 'F80-F89', 'H90-H94', 'S10-S19', 'D10-D36', 'B90-B94', 'J80-J84', 'N10-N16', 'B50-B64', 'T15-T19', 'G60-G65', 'F40-F48', 'I80-I89', 'I95-I99', 'K55-K64', 'T80-T88', 'M95-M95', 'E20-E35', 'P90-P96', 'J90-J94', 'A50-A64', 'K20-K31', 'E70-E88', 'L76-L76', 'R00-R09', 'C30-C39', 'H10-H11', 'B20-B20', 'Y21-Y33', 'G80-G83', 'O30-O48', 'E89-E89', 'C7A-C7A', 'I05-I09', 'B85-B89', 'K70-K77', 'C64-C68', 'M97-M97', 'B99-B99', 'D80-D89', 'R97-R97', 'M91-M94', 'Q38-Q45', 'R20-R23', 'M65-M67', 'N20-N23', 'O00-O08', 'A75-A79', 'K90-K95', 'C15-C26', 'T79-T79', 'T30-T32', 'I70-I79', 'R40-R46', 'R25-R29', 'G50-G59', 'M96-M96', 'A65-A69', 'E40-E46', 'F10-F19', 'F20-F29', 'K65-K68', 'H65-H75', 'N00-N08', 'P10-P15', 'G00-G09', 'P80-P83', 'R80-R82', 'N60-N65', 'N70-N77', 'E08-E13', 'C43-C44', 'V80-V89', 'X71-X83', 'S30-S39', 'Q35-Q37', 'C40-C41', 'M30-M36', 'C81-C96', 'V20-V29', 'H80-H83', 'L20-L30', 'K40-K46', 'D50-D53', 'O10-O16', 'O85-O92', 'E65-E68', 'H46-H47', 'O09-O09', 'A30-A49', 'T20-T25', 'C50-C50', 'H95-H95', 'M99-M99', 'P76-P78', 'M86-M90', 'T51-T65', 'J85-J86', 'G89-G99', 'V40-V49', 'B10-B10', 'R83-R89', 'N80-N98', 'G20-G26', 'Glucose, POCT, B', 'Hemoglobin', 'Platelet Count', 'Hematocrit', 'Erythrocytes', 'Leukocytes', 'Potassium, S', 'Sodium, S', 'Creatinine, S', 'Bicarbonate, S', 'Chloride, S', 'Anion Gap', 'Calcium, Total, S', 'Glucose, S', 'Bld Urea Nitrog(BUN), S', 'Lymphocytes', 'Monocytes', 'Neutrophils', 'Eosinophils', 'Basophils', 'Potassium, P', 'Sodium, P', 'Chloride, P', 'Bicarbonate, P', 'Creatinine, P', 'Glucose, P', 'Anion Gap, P', 'Calcium, Total, P', 'pH', 'Hemoglobin, B', 'pCO2', 'pO2', 'Lactate, P', 'Glucose', 'Lactate, B', 'pH, U', 'Potassium, B', 'Sodium, B', 'Troponin T, 6 hr, 5th gen', 'Troponin T, 2 hr, 5th gen', 'Troponin T, Baseline, 5th gen', 'Venous pH', 'Venous pCO2', 'Bicarbonate', 'Sodium', 'Potassium', 'Creatinine', 'Creatinine, B', 'Chloride', 'Sodium, U', 'Chloride, B', 'ANESTHETICS', 'ANTINEOPLASTICS', 'BIOLOGICALS', 'DIAGNOSTIC', 'ANTIBIOTICS', 'CONTRACEPTIVES', 'ANTICOAGULANTS', 'CARDIOVASCULAR', 'PSYCHOTHERAPEUTIC DRUGS', 'CNS DRUGS', 'ELECT/CALORIC/H2O', 'EENT PREPS', 'BLOOD', 'CARDIAC DRUGS', 'GASTROINTESTINAL', 'UNCLASSIFIED DRUG PRODUCTS', 'ANTIFUNGALS', 'AUTONOMIC DRUGS', 'DIURETICS', 'SEDATIVE/HYPNOTICS', 'IMMUNOSUPPRESSANTS', 'HORMONES', 'ANTIHYPERGLYCEMICS', 'ANTIARTHRITICS', 'COLONY STIMULATING FACTORS', 'ANTIASTHMATICS', 'ANALGESICS', 'VITAMINS', 'SKIN PREPS', 'MUSCLE RELAXANTS', 'ANTIVIRALS', 'ANTIPARKINSON DRUGS', 'ANTIHISTAMINES', 'ANTIPLATELET DRUGS', 'MISCELLANEOUS MEDICAL SUPPLIES, DEVICES, NON-DRUG', 'ANTIINFECTIVES/MISCELLANEOUS', 'SMOKING DETERRENTS', 'COUGH/COLD PREPARATIONS', 'HERBALS', 'ANALGESIC AND ANTIHISTAMINE COMBINATION', 'ANTIALLERGY', 'ANTIPARASITICS', 'ANTIINFLAM.TUMOR NECROSIS FACTOR INHIBITING AGENTS', 'ANTIHISTAMINE AND DECONGESTANT COMBINATION', 'PRE-NATAL VITAMINS']

def main():
    # print('building cohort ....')
    # sys.stdout.flush()
    # df_hosp  = pd.read_csv(header_raw+'transfer_location_file.csv') ##transfer location file form SQL query
    # df = cohort_selection.cohort_selection(df_hosp)
    # df.to_csv(header_proc+'cohort_file.csv')

    # print('cohort demographics ....')
    # sys.stdout.flush()
    # df = pd.read_csv(header_proc+'cohort_file.csv')
    # df_demo = pd.read_csv(header_raw+'patient_demographics.csv') ## patient demographic file from SQL query
    # df_demo = demo_featurization(df, df_demo)
    # df_demo.to_csv(header_proc+'cohort_file_w_demo.csv')

    # print('cohort CPT ....')
    # sys.stdout.flush()
    # df = pd.read_csv(header_proc+'cohort_file.csv')
    # df_cpt= pd.read_csv(header_raw+'EDTWH_FACT_PROCEDURES.csv') ## procedures file from SQL query
    # df_cpt = cpt_featurization(df, df_cpt)
    # df_cpt.to_csv(header_proc+'cohort_file_w_cpt.csv')

    # print('cohort ICD ....')
    # sys.stdout.flush()
    # df = pd.read_csv(header_proc+'cohort_file.csv')
    # df_icd= pd.read_csv(header_raw+'EDTWH_FACT_DIAGNOSIS.csv') ## diagnoses file from SQL query
    # df_icd = icd_featurization(df, df_icd)
    # df_icd.to_csv(header_proc+'cohort_file_w_icd.csv')

    # print('cohort Labs ....')
    # sys.stdout.flush()
    # df = pd.read_csv(header_proc+'cohort_file.csv')
    # df_lab= pd.read_csv(header_raw+'EDTWH_FACT_LAB_TEST.csv') ## lab tests file from SQL query
    # df_lab = lab_featurization(df, df_lab)
    # df_lab.to_csv(header_proc+'cohort_file_w_lab.csv')

    # print('cohort Medications ....')
    # sys.stdout.flush()
    # df = pd.read_csv(header_proc+'cohort_file.csv')
    # df_med= pd.read_csv(header_raw+'EDTWH_FACT_MEDICATIONS.csv') ## medication file from SQL query
    # df_med = med_featurization(df, df_med)
    # df_med.to_csv(header_proc+'cohort_file_w_med.csv')

    # print('merging files ....')
    # sys.stdout.flush()
    # df_demo = pd.read_csv(header_proc+'cohort_file_w_demo.csv')
    # df_demo['PATIENT_AGE_NEW'] = df_demo['PATIENT_AGE'].copy()
    # for c in ADMIT_COLS:
    #     df_demo[c] = 0
    # df_cpt = pd.read_csv(header_proc+'cohort_file_w_cpt.csv')
    # for c in CPT_COLS:
    #     if c not in df_cpt.columns:
    #         df_cpt[c] = None
    # df_icd = pd.read_csv(header_proc+'cohort_file_w_icd.csv')
    # for c in ICD_COLS:
    #     if c not in df_icd.columns:
    #         df_icd[c] = None
    # df_lab = pd.read_csv(header_proc+'cohort_file_w_lab.csv')
    # for c in LAB_COLS:
    #     if c not in df_lab.columns:
    #         df_lab[c] = 'UNKNOWN'
    # df_med = pd.read_csv(header_proc+'cohort_file_w_med.csv')
    # for c in MEDICATION_COLS:
    #     if c not in df_med.columns:
    #         df_med[c] = None

    # merge_cols = ['PATIENT_DK', 'ADMIT_DTM']
    # df_cpt[CPT_COLS] = df_cpt[CPT_COLS].fillna(0)
    # df_combined = df_cpt.merge(df_demo[merge_cols+DEMO_COLS+ADMIT_COLS], on = merge_cols, how='left')
    # merge_cols = ['PATIENT_DK', 'ADMIT_DTM', 'Day_Number']
    # df_icd[ICD_COLS] = df_icd[ICD_COLS].fillna(0)
    # df_combined = df_combined.merge(df_icd[merge_cols+ICD_COLS], on = merge_cols, how='left')
    # df_lab[LAB_COLS] = df_lab[LAB_COLS].fillna('UNKNOWN')
    # df_combined = df_combined.merge(df_lab[merge_cols+LAB_COLS], on = merge_cols, how='left')
    # df_med[MEDICATION_COLS] = df_med[MEDICATION_COLS].fillna(0)
    # df_combined = df_combined.merge(df_med[merge_cols+MEDICATION_COLS], on = merge_cols, how='left')


    # def node_name(pid, dt):
    #     return str(pid)+'_'+str(dt)

    # df_combined["node_name"] = df_combined.apply(lambda x: node_name(x.PATIENT_DK, x.ADMIT_DTM), axis=1)
    # df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    # df_combined.to_csv(header_proc+'ehr_combined.csv')

    # print('processing EHR for sequence creation ...')
    # sys.stdout.flush()
    # df_combined = pd.read_csv(header_proc+'ehr_combined.csv')
    # preprocess_ehr_Mayo.main(df_combined)

    # print('combining sequences with pre-made sequences')

    # fname = '../ehr/processed/ehr_preprocessed_seq_by_day_tabnet'
    # dct_org = pkl.load(open(fname+'_org.pkl', 'rb'))
    # dct = pkl.load(open(fname+'.pkl', 'rb'))
    # for k in dct['feat_dict'].keys():
    #     dct_org['feat_dict'][k] = dct['feat_dict'][k].copy()
    # pkl.dump(dct_org, open(fname+'_appended.pkl', 'wb'))

    # fname = '../ehr/processed/ehr_preprocessed_seq_by_day_gnn'
    # dct_org = pkl.load(open(fname+'_org.pkl', 'rb'))
    # dct = pkl.load(open(fname+'.pkl', 'rb'))
    # for k in dct['feat_dict'].keys():
    #     dct_org['feat_dict'][k] = dct['feat_dict'][k].copy()
    # pkl.dump(dct_org, open(fname+'_appended.pkl', 'wb'))

    print('adding rows to base cohort file')
    df = pd.read_csv(header_proc+'cohort_file_org.csv')
    df_combined = pd.read_csv(header_proc+'ehr_combined.csv')
    df_combined['split'] = "test"
    df_combined['ADMISSION_DTM'] = df_combined["ADMIT_DTM"].copy()
    df_combined['copied_PNG_location'] = "dummy.png"
    df_combined['Study_DTM'] = df_combined["ADMIT_DTM"].copy()
    df_combined['Discharge2DeathDays'] = 100000
    df_combined['Gap in months'] = 100000
    cols = list(set(df.columns).intersection(set(df_combined.columns)))
    df = pd.concat([df, df_combined[cols]])
    df.to_csv(header_proc+'cohort_file_appended.csv')


if __name__ == "__main__":
    main()



