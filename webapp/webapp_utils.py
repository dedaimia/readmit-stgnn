import os
from types import SimpleNamespace

script_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(script_path, '..')

class defaultInferenceArgs(SimpleNamespace):
    def __init__ (self, edge_ehr_file=None, ehr_feature_file=None, demo_file=None):
        super().__init__(edge_ehr_file=edge_ehr_file, ehr_feature_file=ehr_feature_file, demo_file=demo_file)

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
        joint_hidden = 256
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