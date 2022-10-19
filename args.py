import argparse


def str2bool(x):
    if x == "true" or x == "True":
        return True
    else:
        return False


def get_args():
    parser = argparse.ArgumentParser(description="COVID spatiotemporal GNN")

    # general args
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the outputs and checkpoints.",
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Model checkpoint to start training/testing from.",
    )
    parser.add_argument(
        "--do_train",
        default=False,
        type=str2bool,
        # action="store_true",
        help="Whether perform training.",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="Which GPU to use? If None, use default GPU.",
    )
    parser.add_argument("--rand_seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of patience epochs before early stopping.",
    )
    parser.add_argument(
        "--extract_embeddings",
        default=False,
        action="store_true",
        help="Whether or not to extract node embeddings after training.",
    )

    # graph args
    # parser.add_argument(
    #     "--edge_thresh",
    #     default=None,
    #     type=float,
    #     help="Edge threshold",
    # )
    parser.add_argument(
        "--edge_top_perc",
        default=None,
        type=float,
        help="Top percentage edges to be kept.",
    )
    parser.add_argument(
        "--edge_sigma", default=None, type=float, help="Kernel size for edge weight."
    )
    parser.add_argument(
        "--use_gauss_kernel",
        default=False,
        # action="store_true",
        type=str2bool,
        help="Whether or not to use thresholded Gaussian kernel for edges",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=None,
        help="Maximum sequence length (num of study dates).",
    )
    parser.add_argument(
        "--max_seq_len_img",
        type=int,
        default=None,
        help="Maximum sequence length for images.",
    )
    parser.add_argument(
        "--max_seq_len_ehr",
        type=int,
        default=None,
        help="Maximum sequence length for ehr.",
    )
    parser.add_argument(
        "--label_cutoff", type=int, default=7, help="Cutoff days for binary labels."
    )
    parser.add_argument(
        "--dist_measure",
        type=str,
        default="cosine",
        choices=("cosine", "hamming", "euclidean", "correlation", "jaccard"),
        help="Which distance measure? cosine, distance, or correlation.",
    )
    # parser.add_argument(
    #     "--by_study_date",
    #     default=False,
    #     action='store_true',
    #     help='Whether or not use one study date as one time step.'
    # )
    parser.add_argument(
        "--img_by",
        type=str,
        default="cxr",
        choices=("day", "cxr"),
        help="Each time step is by one day or by one cxr.",
    )
    parser.add_argument(
        "--label_name",
        default="hospital_stay",
        choices=("hospital_stay", "icu_admission", "multiclass"),
        type=str,
        help="Label name. hospital_stay or icu_admission.",
    )
    parser.add_argument(
        "--node_by",
        type=str,
        default="hospital_admission",
        choices=("patient", "icu_admission", "hospital_admission"),
        help="Each node is one patient, one hospital admission, or one ICU admission.",
    )
    parser.add_argument(
        "--filter_short_stay",
        action="store_true",
        default=False,
        help="Whether or not to filter out hospital stays <= 1 day.",
    )
    parser.add_argument(
        "--filter_preadmit",
        action="store_true",
        default=False,
        help="Whehter or not to filter out preadmission CXRs.",
    )
    parser.add_argument(
        "--edge_modality",
        type=str,
        nargs="+",
        default=["demo", "cpt", "icd", "lab", "hospital_stay", "med", "admit_reason"],
        help="Modalities used for constructing edges.",
    )
    # parser.add_argument(
    #     "--add_pxs",
    #     default=False,
    #     action='store_true',
    #     help='Whether to add PXS score when computing edges.'
    # )
    # parser.add_argument(
    #     "--add_cpt",
    #     default=False,
    #     action='store_true',
    #     help='Whether to add CPT when computing edges.'
    # )
    parser.add_argument(
        "--dynamic_graph",
        default=False,
        action="store_true",
        help="Whether to have one unique graph at each time step.",
    )
    # parser.add_argument(
    #     '--comor_dict_dir',
    #     default=None,
    #     type=str,
    #     help='Path to precomputed demo/comorbidity dictionary.'
    # )
    # parser.add_argument(
    #     '--demo_dict_dir',
    #     default=None,
    #     type=str,
    #     help='Path to precomputed demo dictionary.'
    # )
    # parser.add_argument(
    #     '--cpt_dict_dir',
    #     default=None,
    #     type=str,
    #     help='Path to precomputed cpt dictionary.'
    # )
    # parser.add_argument(
    #     '--icd_dict_dir',
    #     default=None,
    #     type=str,
    #     help='Path to precomputed icd dictionary.'
    # )
    # parser.add_argument(
    #     '--lab_dict_dir',
    #     default=None,
    #     type=str,
    #     help='Path to precomputed lab dictionary.'
    # )
    # parser.add_argument(
    #     '--admit_reason_dict_dir',
    #     default=None,
    #     type=str,
    #     help="Path to precomputed admission reason dictionary."
    # )
    parser.add_argument(
        "--mask_by_admit_reason",
        type=str2bool,
        default=False,
        help="Whether to mask the edges by admission reason.",
    )
    parser.add_argument(
        "--feature_type",
        default="imaging",
        choices=("imaging", "non-imaging", "pxs", "multimodal", "raw_images"),
        #         choices=('imaging', 'pxs', 'cpt', 'comor', 'combined', 'node_embedding'),
        type=str,
        help="Options: imaging, pxs, cpt etc.",
    )
    parser.add_argument(
        "--ehr_types",
        default=["demo", "cpt", "icd", "lab", "med"],
        nargs="+",
        type=str,
        help="Types of EHR for node features.",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=128,
        help="Embedding dimension for early fusion model.",
    )
    # parser.add_argument(
    #     '--cat_emb_dim',
    #     type=int,
    #     default=None,
    #     help='Embedding dim for categorical variables'
    # )
    # parser.add_argument(
    #     "--feature_dim",
    #     type=int,
    #     default=1024,
    #     help="Image feature dim."
    # )
    # parser.add_argument(
    #     "--multimodal_in_dims",
    #     type=int,
    #     default=None,
    #     nargs="+",
    #     help="List of dims for multimodal inputs. Must be the same order as the inputs."
    # )
    # parser.add_argument(
    #     "--multimodal_emb_dim",
    #     type=int,
    #     default=None,
    #     help="Linear embedding dimension for multimodal inputs."
    # )
    # parser.add_argument(
    #     "--ehr_feature_dim",
    #     type=int,
    #     default=358,
    #     help="EHR feature dim."
    # )
    parser.add_argument(
        "--demo_file",
        type=str,
        default=None,
        help="Path to csv file containing patient demographics etc.",
    )
    parser.add_argument(
        "--img_feature_files",
        type=str,
        nargs="+",
        default=None,
        help="Dir to imaging feature files (csv files).",
    )
    parser.add_argument(
        "--ehr_feature_files",
        type=str,
        nargs="+",
        default=None,
        help="Dir to EHR feature files (csv files).",
    )
    parser.add_argument(
        "--edge_ehr_files",
        type=str,
        nargs="+",
        default=None,
        help="Dir to EHR feature files (csv files) for graph edges.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mayo",
        choices=("mayo", "emory", "stanford", "mimic"),
        help="Name of dataset.",
    )
    parser.add_argument(
        "--pack_padded_seq",
        action="store_true",
        default=False,
        help="Whether to pack sequence for GRU/LSTM.",
    )
    parser.add_argument(
        "--add_timedelta",
        action="store_true",
        default=False,
        help="Whether to append timedelta to features.",
    )
    parser.add_argument(
        "--standardize",
        type=str2bool,
        default=True,
        help="Whether to standardize input to zero mean and unit variance.",
    )
    parser.add_argument(
        "--tabnet_pretrain",
        type=str2bool,
        default=False,
        help="Whether to pretrain TabNet.",
    )

    # model args
    parser.add_argument(
        "--model_name",
        type=str,
        default="stgcn",
        choices=(
            "stgcn",
            "gat",
            "multihead_gat",
            "graphsage",
            "gcn",
            "densenet",
            "mlp",
            "gru",
            "lstm",
            "gaan",
            "brits",
            "graph_transformer",
            "gin",
            "transformer",
            "tabnet",
            "hippo",
            "joint_fusion",
            "early_fusion",
            "late_fusion",
            "joint_fusion_nontemporal",
            "lstm_fusion",
            "gru_fusion",
        ),
        help="Name of the model.",
    )
    # parser.add_argument(
    #     "--img_encoder_name",
    #     type=str,
    #     default='stgcn',
    #     choices=('stgcn'),
    #     help="Name of image encoder."
    # )
    parser.add_argument(
        "--ehr_encoder_name",
        type=str,
        default=None,
        choices=("tabnet", "embedder", None),
        help="Name of ehr encoder.",
    )
    parser.add_argument(
        "--freeze_pretrained",
        type=str2bool,
        default="False",
        help="Whether to freeze pretrained model.",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden size of GCN layers."
    )
    parser.add_argument(
        "--joint_hidden",
        nargs="+",
        type=int,
        default=[128],
        help="List of hidden dims for joint fusion model.",
    )
    parser.add_argument(
        "--gaan_map_feats",
        type=int,
        default=128,
        help="Hidden size of intermediate mapping layer for GaAN only.",
    )
    parser.add_argument(
        "--num_gcn_layers", type=int, default=1, help="Number of GCN layers."
    )
    parser.add_argument(
        "--g_conv",
        type=str,
        default="graphsage",
        choices=("gcn", "graphsage", "gat", "multihead_gat", "gaan", "gin"),
        help="Type of GRU layers.",
    )
    parser.add_argument(
        "--num_rnn_layers", type=int, default=1, help="Number of RNN (GRU) layers."
    )
    parser.add_argument(
        "--rnn_hidden_dim", type=int, default=64, help="Hidden size of RNN layers."
    )
    parser.add_argument(
        "--add_bias",
        # action='store_true',
        type=str2bool,
        default=False,
        help="Whether to add bias to GraphGRU cell.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of output class. 1 for binary classification.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout proba.")
    parser.add_argument(
        "--activation_fn",
        type=str,
        choices=("relu", "elu"),
        default="relu",
        help="Activation function name.",
    )
    parser.add_argument(
        "--norm",
        type=str,
        choices=("", "batch", "layer"),
        default="",
        help="Normalization layer name.",
    )
    parser.add_argument(
        "--aggregator_type",
        type=str,
        default="mean",
        choices=("mean", "gcn", "pool", "lstm"),
        help="Aggregator type. For GraphSAGE only.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=3,
        help="Number of GAT heads. For Multihead GAT only.",
    )
    parser.add_argument(
        "--num_mlp_layers", type=int, default=2, help="Number of MLP layers in GIN."
    )
    parser.add_argument(
        "--learn_eps",
        action="store_true",
        default=False,
        help="Whether to learn eps or keep it a constant.",
    )
    # parser.add_argument(
    #     "--multihead_merge",
    #     type=str,
    #     default="cat",
    #     choices=("mean", "cat"),
    #     help="How to merge multihead outputs. For Multihead GAT only."
    # )
    parser.add_argument(
        "--final_pool",
        type=str,
        default="last",
        choices=("last", "mean", "max", "cat"),
        help="How to pool time step results?",
    )
    parser.add_argument(
        "--t_model",
        type=str,
        default="gru",
        choices=("gru", "lstm", "mgu", "rnn", "minimalrnn", "hippo-legs"),
        help="Which temporal model to use?",
    )
    parser.add_argument(
        "--memory_size", type=int, default=1, help="Memory size for HiPPO."
    )
    parser.add_argument(
        "--memory_order", type=int, default=-1, help="Memory order for HiPPO."
    )
    parser.add_argument(
        "--tcn_kernel_size", type=int, default=2, help="Kernel size for TCN."
    )
    parser.add_argument(
        "--negative_slope",
        type=float,
        default=0.2,
        help="Negative slope for LeakyReLU.",
    )
    parser.add_argument(
        "--gat_residual",
        action="store_true",
        default=False,
        help="Whether to add residual connection for GAT.",
    )

    # training args
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate on dev set every x epoch."
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="F1",
        choices=("F1", "acc", "loss", "auroc", "aupr"),
        help="Name of dev metric to determine best checkpoint.",
    )
    parser.add_argument("--l2_wd", type=float, default=5e-4, help="L2 weight decay.")
    parser.add_argument(
        "--pos_weight",
        type=float,
        nargs="+",
        default=1,
        help="Positive class weight or list of class weights to weigh the loss function.",
    )
    parser.add_argument(
        "--thresh_search",
        # action='store_true',
        type=str2bool,
        default=False,
        help="Whether or not to perform threshold search on dev set.",
    )
    # parser.add_argument(
    #     '--undersample',
    #     action='store_true',
    #     default=False,
    #     help='Whether or not to subsample majority class.'
    # )
    parser.add_argument(
        "--use_sampler",
        action="store_true",
        default=False,
        help="Whether or not to upsample or undersample majority class.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples for random sampler.",
    )
    parser.add_argument(
        "--data_augment",
        # action='store_true',
        type=str2bool,
        default=False,
        help="Whether to perform data augmentation.",
    )
    parser.add_argument(
        "--feature_mask_prob",
        type=float,
        default=0.2,
        help="Proba for masking node features.",
    )
    parser.add_argument(
        "--impute_weight", type=float, default=0.3, help="Imputation loss weight."
    )
    # parser.add_argument(
    #     '--up_scale',
    #     type=float,
    #     default=0,
    #     help="Upsampling scale for minority class. For GraphSMOTE only."
    # )
    # parser.add_argument(
    #     '--opt_new_G',
    #     action='store_true',
    #     default=False,
    #     help='Whether or not to update the decoder with classification loss. \
    #         For GraphSMOTE only.'
    # )
    # parser.add_argument(
    #     '--graphsmote_setting',
    #     type=str,
    #     default='recon_newG',
    #     choices=('recon', 'newG_cls', 'recon_newG'),
    #     help='Choose recon for pretraining feature extractor. \
    #         newG_cls for node classification using pretrained model. \
    #         recon_newG for fine-tuning on pretrained feature extractor for node classification.'
    # )
    # parser.add_argument(
    #     '--rec_weight',
    #     type=float,
    #     default=1e-6,
    #     help="Weighting of L_edge. For GraphSMOTE only."
    # )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=64, help="Test batch size."
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers.")
    parser.add_argument(
        "--which_img",
        type=str,
        default="last",
        choices=("last", "mean", "all"),
        help="Which image to use for the patient for non-temporal models.",
    )
    parser.add_argument(
        "--cnn_finetune",
        default=False,
        action="store_true",
        help="Whether or not only fine-tune classifier for CNN.",
    )
    parser.add_argument(
        "--loss_func",
        type=str,
        default="binary_cross_entropy",
        choices=("binary_cross_entropy", "cross_entropy", "focal_loss"),
        help="Loss function to use.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Alpha hyperparam for focal loss.",
    )
    parser.add_argument(
        "--focal_gamma", type=float, default=2, help="Gamma hyperparam for focal loss."
    )
    parser.add_argument(
        "--fanout", type=int, default=4, help="Neighbors to sample for each edge."
    )

    #### HiPPO args ####
    parser.add_argument(
        "--hippo_memory_size", type=int, default=1, help="Memory size for HiPPO model."
    )
    parser.add_argument(
        "--hippo_memory_order",
        type=int,
        default=-1,
        help="Memory order for HiPPO model. If -1, equal to hidden size.",
    )

    #### TabNet args ####
    parser.add_argument(
        "--pretraining_ratio",
        type=float,
        default=0.2,
        help="Ratio to mask out for pretraining.",
    )
    parser.add_argument(
        "--n_d", type=int, default=8, help="Dimension of prediction layer."
    )
    parser.add_argument(
        "--n_a", type=int, default=8, help="Dimension of attention layer."
    )
    parser.add_argument(
        "--n_steps", type=int, default=8, help="Number of successive steps in network."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.3,
        help="Scaling factor for attention updates (typically between 1 and 2).",
    )
    parser.add_argument(
        "--cat_emb_dim",
        type=int,
        # nargs="+",
        default=1,
        help="Size of the embedding of categorical features \
            if int, all categorical features will have same embedding size \
            if list of int, every corresponding feature will have specific size.",
    )
    parser.add_argument(
        "--n_independent",
        type=int,
        default=2,
        help="Number of independent GLU layer in each GLU block (default 2).",
    )
    parser.add_argument(
        "--n_shared",
        type=int,
        default=2,
        help="Number of shared GLU layer in each GLU block (default 2).",
    )
    parser.add_argument(
        "--virtual_batch_size",
        type=int,
        default=128,
        help="Batch size for Ghost Batch Normalization.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.02,
        help="Float value between 0 and 1 which will be used for momentum in all batch norm.",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="sparsemax",
        choices=("sparsemax", "entmax"),
        help="Either 'sparsemax' or 'entmax' : this is the masking function to use",
    )
    parser.add_argument(
        "--ehr_pretrain_path", default=None, type=str, help="Path to pretrained model."
    )
    #### End of TabNet args ####

    #### GraphTransformer args ####
    parser.add_argument(
        "--trans_nhead", type=int, default=8, help="Number of transformer heads."
    )
    parser.add_argument(
        "--trans_dim_feedforward",
        type=int,
        default=128,
        help="Feedforward dim for transformer.",
    )
    parser.add_argument(
        "--trans_activation",
        type=str,
        default="relu",
        choices=("relu", "gelu"),
        help="Activation function in transformer layers.",
    )
    parser.add_argument(
        "--att_neighbor",
        action="store_true",
        default=False,
        help="Whether to attend to node's neighbors in graph transformer.",
    )
    #### End of GraphTransformer args ####

    #### BGRL args ####
    #     parser.add_argument(
    #         "--bgrl_aug_proba",
    #         type=float,
    #         nargs='+',
    #         help='List of probability for masking node features and dropping edges for BGRL. In the order of \
    #             p_f1, p_f2, p_e1, p_e2',
    #         default=[0.2, 0.1, 0.2, 0.3]
    #     )
    #     parser.add_argument(
    #         '--bgrl_pred_hidden',
    #         type=int,
    #         default=128,
    #         help='Predictor hidden size for BGRL.'
    #     )
    #     parser.add_argument(
    #         '--bgrl_ema_decay',
    #         type=float,
    #         default=0.99,
    #         help='Decay rate for moving average.'
    #     )
    #     parser.add_argument(
    #         "--bgrl_pretrain_only",
    #         action='store_true',
    #         default=False,
    #         help='Whether to only pretrain on BGRL self-supervised task.'
    #     )
    #     parser.add_argument(
    #         '--bgrl_loss_weight',
    #         type=float,
    #         default=1.,
    #         help='Weight for BGRL self-supervised loss.'
    #     )
    #     parser.add_argument(
    #         '--bgrl_finetune',
    #         action='store_true',
    #         default=False,
    #         help='Whether to only finetune on BGRL node embeddings.'
    #     )
    #     parser.add_argument(
    #         '--bgrl_node_embedding_dir',
    #         type=str,
    #         default=None,
    #         help='Dir to trained BGRL node embeddings.'
    #     )

    args = parser.parse_args()

    #     if args.bgrl_pretrain_only:
    #         args.metric_name = 'loss'
    #         args.maximize_metric = False

    # which metric to maximize
    if args.metric_name == "loss":
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ("F1", "acc", "auroc", "aupr"):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError('Unrecognized metric name: "{}"'.format(args.metric_name))

    # must provide load_model_path if testing only
    if (args.load_model_path is None) and (not (args.do_train)):
        raise ValueError(
            "For prediction only, please provide trained model checkpoint in argument load_model_path."
        )
    #     if (args.load_model_path is None) and args.bgrl_finetune:
    #         raise ValueError(
    #             "For finetuning, please provide pretrained BGRL checkpoint in argument load_model_path."
    #         )

    return args