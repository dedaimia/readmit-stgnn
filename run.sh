#!/bin/bash

edge_modality='cpt'
pos_weight=3
feature_type=non-imaging
ehr_types='demo cpt icd lab med'
demo_file='Mayo/ehr/processed/cohort_file_appended.csv'

max_seq_len_ehr=8
cat_emb_dim=3
dropout=0.3
edge_top_perc=0.05
hidden_dim=256
joint_hidden=256
lr=0.001
num_rnn_layers=1

python3 train.py \
    --save_dir 'results/temporal_graph_'$feature_type \
    --label_cutoff 1 \
    --edge_modality $edge_modality \
    --feature_type $feature_type \
    --demo_file $demo_file \
    --ehr_feature_files 'Mayo/ehr/processed/ehr_preprocessed_seq_by_day_tabnet_appended.pkl' \
    --edge_ehr_files 'Mayo/ehr/processed/ehr_preprocessed_seq_by_day_gnn_appended.pkl' \
    --load_model_path 'pretrained/best.pth.tar' \
    --ehr_encoder_name 'embedder' \
    --cat_emb_dim $cat_emb_dim \
    --ehr_types $ehr_types \
    --edge_top_perc $edge_top_perc \
    --dist_measure euclidean \
    --use_gauss_kernel True \
    --max_seq_len_ehr $max_seq_len_ehr \
    --hidden_dim $hidden_dim \
    --num_gcn_layers 1 \
    --num_rnn_layers $num_rnn_layers \
    --add_bias True \
    --g_conv graphsage \
    --aggregator_type mean \
    --num_classes 1 \
    --dropout $dropout \
    --activation_fn elu \
    --eval_every 1 \
    --metric_name F1 \
    --lr $lr \
    --l2_wd 5e-4 \
    --rand_seed 123 \
    --do_train False \
    --thresh_search True \
    --patience 10 \
    --pos_weight $pos_weight \
    --loss_func binary_cross_entropy \
    --num_epochs 100 \
    --final_pool last \
    --model_name stgcn \
    --t_model gru \
    --data_augment False \
    --standardize True \
    --dataset mayo \
    --gpu_id 1
    
    
 
    
