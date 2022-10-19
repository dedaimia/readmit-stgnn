import sys
import torch
import torch.nn as nn

sys.path.append("../")
from model.model import *
import copy

class LateFusionModel(nn.Module):
    """
    Joint fusion model combining imaging and EHR data
    """

    def __init__(
        self,
        img_in_dim,
        ehr_in_dim,
        img_config,
        ehr_config,
        ehr_encoder_config=None,
        cat_idxs=None,
        cat_dims=None,
        ehr_encoder_name="embedder",
        ehr_checkpoint_path=None,
        cat_emb_dim=1,
        num_classes=1,
        freeze_pretrained=False,
        dropout=0.0,
        device="cpu",
    ):
        """
        Args:
            img_encoder: e.g. a lightweight CNN or linear layer
            ehr_encoder: e.g. RNN/GRU/LSTM or linear layer
            joint_hidden: list of hidden sizes for joint layer
        """
        super(LateFusionModel, self).__init__()

        if ehr_encoder_name not in ["embedder", "tabnet"]:
            raise NotImplementedError

        # image encoder
        self.img_model = GraphRNN(
            in_dim=img_in_dim,
            n_classes=num_classes,
            device=device,
            is_classifier=True,
            **img_config
        )

        # ehr encoder
        self.ehr_model = GraphRNN(
            in_dim=ehr_in_dim,
            n_classes=num_classes,
            device=device,
            is_classifier=True,
            ehr_encoder_name=ehr_encoder_name,
            ehr_config=ehr_encoder_config,  # only used if ehr_encoder_name=='tabnet'
            ehr_checkpoint_path=ehr_checkpoint_path,
            freeze_pretrained=freeze_pretrained,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            **ehr_config
        )

        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, g, img_inputs, ehr_inputs):
        """
        Args:
            g: list of dgl graph
            img_inputs: shape (batch, max_seq_len, img_input_dim)
            ehr_inputs: shape (batch, max_seq_len, ehr_input_dim)
        """
        img_inputs, _ = self.img_model(g, img_inputs)

        ehr_inputs, _ = self.ehr_model(g, ehr_inputs)

        probs = (
            torch.sigmoid(img_inputs.reshape(-1))
            + torch.sigmoid(ehr_inputs.reshape(-1))
        ) / 2

        # convert to logits for BCEwithLogitsLoss
        logits = torch.log(probs) - torch.log(1 - probs)

        return logits
    
class JointFusionNonTemporalModel(nn.Module):
    """
    Joint fusion model combining imaging and EHR data
    """

    def __init__(
        self,
        img_in_dim,
        ehr_in_dim,
        img_config,
        ehr_config,
        cat_idxs=None,
        cat_dims=None,
        ehr_encoder_name="embedder",
        cat_emb_dim=1,
        joint_hidden=[128],
        num_classes=1,
        dropout=0.0,
        device="cpu",
    ):
        """
        Args:
            img_encoder: e.g. a lightweight CNN or linear layer
            ehr_encoder: e.g. RNN/GRU/LSTM or linear layer
            joint_hidden: list of hidden sizes for joint layer
        """
        super(JointFusionNonTemporalModel, self).__init__()

        if ehr_encoder_name not in ["embedder", "tabnet"]:
            raise NotImplementedError

        if img_config["hidden_dim"] != ehr_config["hidden_dim"]:
            raise ValueError(
                "hidden_dim for img_config and ehr_config must be the same!"
            )

        # image encoder
        self.img_model = GConvLayers(
            in_dim=img_in_dim, device=device, is_classifier=False, **img_config
        )

        # ehr encoder
        self.ehr_model = GConvLayers(
            in_dim=ehr_in_dim,
            device=device,
            is_classifier=False,
            ehr_encoder_name=ehr_encoder_name,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            **ehr_config
        )

        self.joint_hidden = joint_hidden
        self.num_classes = num_classes
        self.dropout = dropout

        # joint MLP layer
        self.mlp = []
        self.mlp.append(nn.Linear(img_config["hidden_dim"] * 2, joint_hidden[0]))
        self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Dropout(p=dropout))
        for idx_hid in range(1, len(joint_hidden)):
            self.mlp.append(nn.Linear(joint_hidden[idx_hid - 1], joint_hidden[idx_hid]))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout))
        self.mlp.append(nn.Linear(joint_hidden[-1], num_classes))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, g, img_inputs, ehr_inputs):
        """
        Args:
            g: list of dgl graph
            img_inputs: shape (batch, max_seq_len, img_input_dim)
            ehr_inputs: shape (batch, max_seq_len, ehr_input_dim)
        """
        img_inputs = self.img_model(g, img_inputs)

        ehr_inputs = self.ehr_model(g, ehr_inputs)

        h = torch.cat([img_inputs, ehr_inputs], dim=-1)  # (batch, hidden_dim*2)

        logits = self.mlp(h)  # (batch, num_classes)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        return logits
class JointFusionModel(nn.Module):
    """
    Joint fusion model combining imaging and EHR data
    """

    def __init__(
        self,
        img_in_dim,
        ehr_in_dim,
        img_config,
        ehr_config,
        cat_idxs=None,
        cat_dims=None,
        ehr_encoder_name="embedder",
        ehr_checkpoint_path=None,
        cat_emb_dim=1,
        joint_hidden=[128],
        num_classes=1,
        freeze_pretrained=False,
        dropout=0.0,
        device="cpu",
    ):
        """
        Args:
            img_encoder: e.g. a lightweight CNN or linear layer
            ehr_encoder: e.g. RNN/GRU/LSTM or linear layer
            joint_hidden: list of hidden sizes for joint layer
        """
        super(JointFusionModel, self).__init__()

        if img_config["hidden_dim"] != ehr_config["hidden_dim"]:
            raise ValueError(
                "hidden_dim for img_config and ehr_config must be the same!"
            )

        # image encoder
        self.img_model = GraphRNN(
            in_dim=img_in_dim,
            n_classes=num_classes,
            device=device,
            is_classifier=False,
            **img_config
        )

        # ehr encoder
        self.ehr_model = GraphRNN(
            in_dim=ehr_in_dim,
            n_classes=num_classes,
            device=device,
            is_classifier=False,
            ehr_encoder_name=ehr_encoder_name,
            ehr_checkpoint_path=ehr_checkpoint_path,
            freeze_pretrained=freeze_pretrained,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            **ehr_config
        )

        self.joint_hidden = joint_hidden
        self.num_classes = num_classes
        self.dropout = dropout

        # joint MLP layer
        self.mlp = []
        self.mlp.append(nn.Linear(img_config["hidden_dim"] * 2, joint_hidden[0]))
        self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Dropout(p=dropout))
        for idx_hid in range(1, len(joint_hidden)):
            self.mlp.append(nn.Linear(joint_hidden[idx_hid - 1], joint_hidden[idx_hid]))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.Dropout(p=dropout))
        self.mlp.append(nn.Linear(joint_hidden[-1], num_classes))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, g, img_inputs, ehr_inputs):
        """
        Args:
            g: list of dgl graph
            img_inputs: shape (batch, max_seq_len, img_input_dim)
            ehr_inputs: shape (batch, max_seq_len, ehr_input_dim)
        """
        img_inputs = self.img_model(g, img_inputs)

        ehr_inputs = self.ehr_model(g, ehr_inputs)

        h = torch.cat([img_inputs, ehr_inputs], dim=-1)  # (batch, hidden_dim*2)

        logits = self.mlp(h)  # (batch, num_classes)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        return logits
    
class EarlyFusionModel(nn.Module):
    """
    Joint fusion model combining imaging and EHR data
    """

    def __init__(
        self,
        img_in_dim,
        ehr_in_dim,
        emb_dim,
        config,
        num_classes=1,
        device=None,
        add_timedelta=False,
        ehr_config=None,
        ehr_encoder_name=None,
        ehr_checkpoint_path=None,
        freeze_ehr_encoder=False,
        cat_idxs=None,
        cat_dims=None,
        cat_emb_dim=1,
    ):
        """
        Args:
            img_encoder: e.g. a lightweight CNN or linear layer
            ehr_encoder: e.g. RNN/GRU/LSTM or linear layer
            joint_hidden: list of hidden sizes for joint layer
        """
        super(EarlyFusionModel, self).__init__()

        self.add_timedelta = add_timedelta
        self.ehr_encoder_name = ehr_encoder_name

        if self.ehr_encoder_name is not None:
            if self.ehr_encoder_name == "tabnet":
                self.ehr_encoder = tab_network.TabNet(
                    input_dim=ehr_in_dim,
                    output_dim=num_classes,  # dummy
                    cat_idxs=cat_idxs,
                    cat_dims=cat_dims,
                    **ehr_config
                )
                if ehr_checkpoint_path is not None:
                    update_state_dict = copy.deepcopy(self.ehr_encoder.state_dict())
                    ckpt = torch.load(ehr_checkpoint_path)

                    for param, weights in ckpt["state_dict"].items():
                        if param.startswith("encoder"):
                            # Convert encoder's layers name to match
                            new_param = "tabnet." + param
                        else:
                            new_param = param
                        if self.ehr_encoder.state_dict().get(new_param) is not None:
                            # update only common layers
                            update_state_dict[new_param] = weights
                    self.ehr_encoder.load_state_dict(update_state_dict)
                    print("Loaded pretrained TabNet...")
                    if freeze_ehr_encoder:
                        for param in self.ehr_encoder.parameters():
                            param.requires_grad = False
                        print("Tabnet params frozen...")
                emb_dim = ehr_config["n_d"]
                print(
                    "Setting emb_dim to {} (same as n_d in TabNet)...".format(emb_dim)
                )

            elif ehr_encoder_name == "embedder":
                print("Using embedder to embed ehr data...")
                self.embedder = tab_network.EmbeddingGenerator(
                    input_dim=ehr_in_dim,
                    cat_dims=cat_dims,
                    cat_idxs=cat_idxs,
                    cat_emb_dim=cat_emb_dim,
                )
                ehr_in_dim = (ehr_in_dim - len(cat_idxs)) + len(cat_idxs) * cat_emb_dim

            else:
                raise NotImplementedError

        # projection layers
        if not add_timedelta:
            self.img_proj = nn.Linear(img_in_dim, emb_dim)
            if self.ehr_encoder_name != "tabnet":
                self.ehr_proj = nn.Linear(ehr_in_dim, emb_dim)

            # gnn
            self.gnn = GraphRNN(
                in_dim=emb_dim * 2,
                n_classes=num_classes,
                device=device,
                is_classifier=True,
                **config
            )
        else:
            self.img_proj = nn.Linear(img_in_dim - 1, emb_dim)
            if self.ehr_encoder_name != "tabnet":
                self.ehr_proj = nn.Linear(ehr_in_dim - 1, emb_dim)

            self.gnn = GraphRNN(
                in_dim=emb_dim * 2 + 1,
                n_classes=num_classes,
                device=device,
                is_classifier=True,
                **config
            )

    def forward(self, g, img_inputs, ehr_inputs):
        """
        Args:
            g: list of dgl graph
            img_inputs: shape (batch, max_seq_len, img_input_dim)
            ehr_inputs: shape (batch, max_seq_len, ehr_input_dim)
        """
        if not self.add_timedelta:
            # project imaging
            img_h = self.img_proj(img_inputs)  # (batch, max_seq_len, emb_dim)

            # project ehr
            if self.ehr_encoder_name is None:
                ehr_h = self.ehr_proj(ehr_inputs)  # (batch, max_seq_len, emb_dim)
            elif self.ehr_encoder_name == "tabnet":
                batch, seq_len, _ = ehr_inputs.shape
                ehr_inputs = ehr_inputs.reshape(batch * seq_len, -1)
                x = self.ehr_encoder.embedder(ehr_inputs)
                steps_output, _ = self.ehr_encoder.tabnet.encoder(x)
                ehr_h = torch.sum(
                    torch.stack(steps_output, dim=0), dim=0
                )  # (batch*seq_len, n_d)
                ehr_h = ehr_h.reshape(batch, seq_len, -1)
            else:
                batch, seq_len, _ = ehr_inputs.shape
                ehr_inputs = ehr_inputs.reshape(batch * seq_len, -1)
                ehr_inputs = self.embedder(ehr_inputs).reshape(batch, seq_len, -1)
                ehr_h = self.ehr_proj(ehr_inputs)
            h = torch.cat([img_h, ehr_h], dim=-1)
        else:
            time_delta = img_inputs[:, :, -1].unsqueeze(-1)  # (batch, max_seq_len, 1)

            # project imaging
            img_h = self.img_proj(
                img_inputs[:, :, :-1]
            )  # (batch, max_seq_len, emb_dim)

            # project ehr
            if self.ehr_encoder_name is None:
                ehr_h = self.ehr_proj(
                    ehr_inputs[:, :, :-1]
                )  # (batch, max_seq_len, emb_dim)
            else:
                batch, seq_len, _ = ehr_inputs.shape
                ehr_inputs = ehr_inputs.reshape(batch * seq_len, -1)
                x = self.ehr_encoder.embedder(ehr_inputs[:, :, :-1])
                steps_output, _ = self.ehr_encoder.tabnet.encoder(x)
                ehr_h = torch.sum(
                    torch.stack(steps_output, dim=0), dim=0
                )  # (batch*seq_len, n_d)
                ehr_h = ehr_h.reshape(batch, seq_len, -1)
            h = torch.cat([img_h, ehr_h, time_delta], dim=-1)

        logits, _ = self.gnn(g, h)

        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)

        return logits
