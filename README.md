# Multimodal spatiotemporal graph neural networks for improved prediction of 30-day all-cause hospital readmission

Siyi Tang, Amara Tariq, Jared Dunnmon, Umesh Sharma, Praneetha Elugunti, Daniel Rubin, Bhavik N. Patel, Imon Banerjee, *arXiv*, 2022. http://arxiv.org/abs/2204.06766.

## Background
Measures to predict 30-day readmission are considered an important quality factor for hospitals as they can reduce the overall cost of care through identification of high risk patients and allow allocation of resources accordingly. In this study, we propose a multimodal spatiotemporal graph neural network (MM-STGNN) for prediction of 30-day all-cause hospital readmission by fusing longitudinal chest radiographs and electronic health records (EHR) during hospitalizations.


## Mayo data preprocessing
Run the following command from Mayo/preprocessing/ folder after placing data in Mayo/ehr/ folder
```
python3 main.py --hosp_file <name/of/transfer/location/file> --demo_file <name/of/demographics/file> --cpt_file <name/of/cpt/file> --icd_file <name/of/icd/file> --lab_file <name/of/labs/file> --med_file <name/of/medications/file>
```

## Inference
After placing best.pth.tar in pretrained/ folder, run the run.sh file.


## Reference
If you use this codebase, or otherwise find our work valuable, please cite:
```
@ARTICLE{Tang2022-jt,
   title         = "Multimodal spatiotemporal graph neural networks for improved
                    prediction of 30-day all-cause hospital readmission",
   author        = "Tang, Siyi and Tariq, Amara and Dunnmon, Jared and Sharma,
                    Umesh and Elugunti, Praneetha and Rubin, Daniel and Patel,
                    Bhavik N and Banerjee, Imon",
   month         =  apr,
   year          =  2022,
   archivePrefix = "arXiv",
   primaryClass  = "cs.LG",
   eprint        = "2204.06766"
 }
```
