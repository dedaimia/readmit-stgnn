import os
from types import SimpleNamespace

script_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(script_path, '..')

class defaultInferenceArgs(SimpleNamespace):
    def __init__ (self, edge_ehr_files=None, ehr_feature_files=None, demo_file=None):
        super().__init__(edge_ehr_files=edge_ehr_files, ehr_feature_files=ehr_feature_files, demo_file=demo_file)

        # Researcher set args - leaving separate to match formatting of run.sh
        edge_modality = 'cpt'
        pos_weight = 3
        feature_type = 'non-imaging'
        ehr_types = 'demo cpt icd lab med'
        max_seq_len_ehr = 8
        cat_emb_dim = 3
        dropout = 0.3
        edge_top_perc = 0.05
        hidden_dim = 256
        joint_hidden = 256 # TODO: verify run.sh having this set, but not used
        lr = 0.001
        num_rnn_layers = 1

        # Likely to never change for inference
        self.save_dir = os.path.join(code_path, 'temp_storage')
        self.label_cutoff = 1
        self.edge_modality = edge_modality
        self.feature_type = feature_type
        self.load_model_path = 'pretrained/best.pth.tar'
        self.ehr_encoder_name = 'embedder'
        self.cat_emb_dim = cat_emb_dim
        self.ehr_types = ehr_types
        self.edge_top_perc = edge_top_perc
        self.dist_measure = 'euclidean'
        self.use_gauss_kernel = True
        self.max_seq_len_ehr = max_seq_len_ehr
        self.max_seq_len_img = None
        self.max_seq_len = None
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = 1
        self.num_rnn_layers = num_rnn_layers
        self.add_bias = True
        self.g_conv = 'graphsage'
        self.aggregator_type = 'mean'
        self.num_classes = 1
        self.dropout = dropout
        self.activation_fn = 'elu'
        self.eval_every = 1
        self.metric_name = 'F1'
        self.lr = lr
        self.l2_wd = 5e-4
        self.rand_seed = 123
        self.do_train = False
        self.thresh_search = True
        self.patience = 10
        self.pos_weight = pos_weight
        self.loss_func = 'binary_cross_entropy'
        self.num_epochs = 100
        self.final_pool = 'last'
        self.model_name = 'stgcn'
        self.t_model = 'gru'
        self.data_augment = False
        self.standardize = True
        self.dataset = 'mayo'
        self.gpu_id = 0

        # Additional args used for cli but not in run.sh
        # using defaults from args.py
        self.wandb_mode = 'offline'
        self.extract_embeddings = False
        self.edge_sigma = None
        self.img_by = 'cxr'
        self.label_name = 'hospital_stay'
        self.node_by = 'hospital_admission'
        self.filter_short_stay = False
        self.filter_preadmit = False
        self.dynamic_graph = False
        self.mask_by_admit_reason = False
        self.emb_dim = 128
        self.img_feature_files = None
        self.pack_padded_seq = False
        self.add_timedelta = False
        self.tabnet_pretrain = False
        self.freeze_pretrained = False
        self.joint_hidden = joint_hidden
        self.gaan_map_feats = 128
        self.rnn_hidden_dim = 64
        self.norm = ''
        self.num_heads = 3
        self.num_mlp_layers = 2
        self.learn_eps = False
        self.memory_size = 1
        self.memory_order = -1
        self.tcn_kernel_size = 2
        self.negative_slope = 0.2
        self.gat_residual = False
        self.use_sampler = False
        self.num_samples = None
        self.feature_mask_prob = 0.2
        self.impute_weight = 0.3
        self.train_batch_size = 64
        self.test_batch_size = 64
        self.num_workers = 8
        self.which_img = 'last'
        self.cnn_finetune = False
        self.focal_alpha = 0.25
        self.focal_gamma = 2
        self.fanout = 4
        self.pretraining_ratio = 0.2
        self.n_d = 8
        self.n_a = 8
        self.n_steps = 8
        self.gamma = 1.3
        self.n_independent = 2
        self.n_shared = 2
        self.virtual_batch_size = 128
        self.momentum = 0.02
        self.mask_type = 'sparsemax'
        self.ehr_pretrain_path = None
        self.trans_nhead = 8
        self.trans_dim_feedforward = 128
        self.trans_activation = 'relu'
        self.att_neighbor = False