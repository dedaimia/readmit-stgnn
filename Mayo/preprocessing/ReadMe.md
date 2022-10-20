# Cohort selectipon

cohort_selection.py - keep all unique hospitaliztions longer than 2 days and less than 180 days

# Five EHR modalities

* demographics
* CPT/procedures - mapped to CPT subgroups
* ICD/diagnoses - mapped to ICD10 subgroups
* Lab test - selected labs, values mapped to ABNORMAL/NORMAL
* Medications - mapped to MED_THERAPEUTIC_CLASS


### run main.py for all pre-processing step including cohort selection
python3 main.py --hosp_file <name/of/transfer/location/file> --demo_file <name/of/demographics/file> --cpt_file <name/of/cpt/file> --icd_file <name/of/icd/file> --lab_file <name/of/labs/file> --med_file <name/of/medications/file>


