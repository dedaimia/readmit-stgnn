import numpy as np
import os
import sys
import pickle
import torch
import json
from argparse import Namespace
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import math
import utils
import dgl
from data.mayo_dataset import ReadmissionDataset
from args import get_args
from collections import OrderedDict, defaultdict
from json import dumps
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from model.model import GraphRNN, GraphTransformer

from model.model import GConvLayers
from model.fusion import (
    JointFusionModel,
    EarlyFusionModel,
    LateFusionModel,
    JointFusionNonTemporalModel,
)
from constants import *
from dotted_dict import DottedDict


def evaluate(
    args,
    model,
    graph,
    features,
    labels,
    nid,
    loss_fn,
    best_thresh=0.5,
    save_file=None,
    thresh_search=False,
    img_features=None,
    ehr_features=None,
):
    model.eval()
    with torch.no_grad():
        if "fusion" in args.model_name:
            if "nontemporal" not in args.model_name:
                logits = model(graph, img_features, ehr_features)
            else:
                img_features_avg = img_features[:, -1, :]
                ehr_features_avg = ehr_features[:, -1, :]
                logits = model(graph[0], img_features_avg, ehr_features_avg)
        elif (args.model_name != "stgcn") and (args.model_name != "graph_transformer"):
            assert len(graph) == 1
            features_avg = features[:, -1, :]
            logits, _ = model(graph[0], features_avg)
        else:
            logits, _ = model(graph, features)
        logits = logits[nid]

        if logits.shape[-1] == 1:
            logits = logits.view(-1)  # (batch_size,)

        labels = labels[nid]
        loss = loss_fn(logits, labels)

        if not (isinstance(loss_fn, nn.CrossEntropyLoss)):
            logits = logits.view(-1)  # (batch_size,)
            probs = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
            preds = (probs >= best_thresh).astype(int)  # (batch_size, )
        else:
            # (batch_size, num_classes)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1).reshape(-1)  # (batch_size,)
            probs = np.max(probs, axis=1).reshape(-1)

        eval_results = utils.eval_dict(
            y=labels.data.cpu().numpy(),
            y_pred=preds,
            y_prob=probs,
            average="micro" if args.label_name == "multiclass" else "binary",
            thresh_search=thresh_search,
            best_thresh=best_thresh,
        )
        eval_results["loss"] = loss.item()

    if save_file is not None:
        with open(save_file, "wb") as pf:
            pickle.dump(
                {
                    "probs": probs,
                    "labels": labels.cpu().numpy(),
                    "preds": preds,
                    "node_indices": nid,
                },
                pf,
            )

    return eval_results


def main(args):
    print('in train.py')
    sys.stdout.flush()
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"
    print("Using device", device)

    # set random seed
    utils.seed_torch(seed=args.rand_seed)

    # get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False
    )

    # save args
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    logger = utils.get_logger(args.save_dir, "train")
    logger.info("Args: {}".format(dumps(vars(args), indent=4, sort_keys=True)))

    # wandb
    wandb.init(project="covid-gnn", entity="siyitang")
    wandb.init(config=args)
    wandb.run.name = "{}_{}_edge_{}_edgeThresh_{}_hidden={}_rnnLayers={}_{}".format(
        args.model_name,
        args.g_conv,
        args.edge_modality,
        args.edge_top_perc,
        args.hidden_dim,
        args.num_rnn_layers,
        args.save_dir.split("/")[-1],
    )

    # build dataset
    logger.info("Building dataset...")
    if args.dataset == "mayo":
        data = ReadmissionDataset(
            demo_file=args.demo_file,
            edge_ehr_file=args.edge_ehr_files[0],
            ehr_feature_file=args.ehr_feature_files[0]
            if (args.feature_type != "imaging")
            else None,
            edge_modality=args.edge_modality,
            feature_type=args.feature_type,
            img_feature_dir=args.img_feature_files[0]
            if (args.feature_type != "non-imaging")
            else None,
            top_perc=args.edge_top_perc,
            sigma=args.edge_sigma,
            gauss_kernel=args.use_gauss_kernel,
            max_seq_len_img=args.max_seq_len_img,
            max_seq_len_ehr=args.max_seq_len_ehr,
            label_cutoff=args.label_cutoff,
            sim_measure=args.dist_measure,
            dynamic_graph=args.dynamic_graph,
            standardize=args.standardize,
            pad_front=False,
            add_timedelta=args.add_timedelta,
            mask_by_admit_reason=args.mask_by_admit_reason,
            #   ehr_types=args.ehr_types,
            ehr_types=["demo", "cpt", "icd", "lab", "med"]
            if args.dataset == "mayo"
            else ["demo", "icd", "lab", "med"],
            img_by=args.img_by,
            dataset_name=args.dataset,
        )
    else:
        raise NotImplementedError

    g = data[0]  # g now is a list
    if args.feature_type != "multimodal":
        features = g[0].ndata[
            "feat"
        ]  # features for each graph are the same, including temporal info
        img_features = None
        ehr_features = None
        cat_idxs = data.cat_idxs
        cat_dims = data.cat_dims
    else:
        img_features = g[0].ndata["img_feat"]
        ehr_features = g[0].ndata["ehr_feat"]
        features = None
        cat_idxs = data.cat_idxs
        cat_dims = data.cat_dims
    labels = g[0].ndata["label"]  # labels are the same
    if args.loss_func == "cross_entropy":
        labels = labels.long()
    train_mask = g[0].ndata["train_mask"]
    val_mask = g[0].ndata["val_mask"]
    test_mask = g[0].ndata["test_mask"]

    # save graph for post analyses
    with open(os.path.join(args.save_dir, "node_labels.pkl"), "wb") as pf:
        pickle.dump(labels.data.cpu().numpy(), pf)

    # ensure self-edges
    for idx_g in range(len(g)):
        g[idx_g] = dgl.remove_self_loop(g[idx_g])
        g[idx_g] = dgl.add_self_loop(g[idx_g])
        n_edges = g[idx_g].number_of_edges()
        n_nodes = g[idx_g].number_of_nodes()
        logger.info(
            """----Graph %d------
                    # Nodes %d
                    # Undirected edges %d
                    # Average degree %d """
            % (
                idx_g,
                n_nodes,
                int(n_edges / 2),
                g[idx_g].in_degrees().float().mean().item(),
            )
        )

    train_nid = torch.nonzero(train_mask).squeeze().to(device)
    val_nid = torch.nonzero(val_mask).squeeze().to(device)
    test_nid = torch.nonzero(test_mask).squeeze().to(device)

    train_labels = labels[train_nid]
    val_labels = labels[val_nid]
    test_labels = labels[test_nid]

    if not (args.label_name == "multiclass"):
        logger.info(
            "#Train samples: {}; positive percentage :{:.2f}".format(
                train_mask.int().sum().item(),
                (train_labels == 1).sum().item() / len(train_labels) * 100,
            )
        )
        logger.info(
            "#Val samples: {}; positive percentage :{:.2f}".format(
                val_mask.int().sum().item(),
                (val_labels == 1).sum().item() / len(val_labels) * 100,
            )
        )
        logger.info(
            "#Test samples: {}; positive percentage :{:.2f}".format(
                test_mask.int().sum().item(),
                (test_labels == 1).sum().item() / len(test_labels) * 100,
            )
        )
    else:
        logger.info(
            "#Train samples: {}; Class 0 vs Class 1 vs Class 2 : {} vs {} vs {}".format(
                train_mask.int().sum().item(),
                (train_labels == 0).sum().item(),
                (train_labels == 1).sum().item(),
                (train_labels == 2).sum().item(),
            )
        )
        logger.info(
            "#Val samples: {}; Class 0 vs Class 1 vs Class 2 : {} vs {} vs {}".format(
                val_mask.int().sum().item(),
                (val_labels == 0).sum().item(),
                (val_labels == 1).sum().item(),
                (val_labels == 2).sum().item(),
            )
        )
        logger.info(
            "#Test samples: {}; Class 0 vs Class 1 vs Class 2 : {} vs {} vs {}".format(
                test_mask.int().sum().item(),
                (test_labels == 0).sum().item(),
                (test_labels == 1).sum().item(),
                (test_labels == 2).sum().item(),
            )
        )

    if args.cuda:
        if args.feature_type != "multimodal":
            features = features.to(device)
            print("features shape:", features.shape)
        else:
            img_features = img_features.to(device)
            ehr_features = ehr_features.to(device)
            print("img_features shape:", img_features.shape) 
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        for idx_g in range(len(g)):
            g[idx_g] = g[idx_g].int().to(device)
    

    if args.model_name == "stgcn":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        if args.ehr_encoder_name == "tabnet":
            if args.ehr_pretrain_path is not None:
                with open(
                    os.path.join(
                        args.ehr_pretrain_path.split("best.pth.tar")[0], "args.json"
                    ),
                    "r",
                ) as jf:
                    args_pretrained = json.load(jf)
                args_pretrained = DottedDict(args_pretrained)
                ehr_config = utils.get_config("tabnet", args_pretrained)
            else:
                ehr_config = utils.get_config("tabnet", args)
        else:
            ehr_config = None
        model = GraphRNN(
            in_dim=in_dim,
            n_classes=args.num_classes,
            device=device,
            is_classifier=True,
            ehr_encoder_name=args.ehr_encoder_name
            if args.feature_type != "imaging"
            else None,
            ehr_config=ehr_config,
            ehr_checkpoint_path=args.ehr_pretrain_path,
            freeze_pretrained=args.freeze_pretrained,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=args.cat_emb_dim,
            **config
        )

    elif args.model_name == "graph_transformer":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        model = GraphTransformer(
            seq_len=args.max_seq_len,
            in_dim=in_dim,
            num_nodes=g[0].number_of_nodes(),
            n_classes=args.num_classes,
            device=device,
            is_classifier=True,
            **config
        )
    elif args.model_name == "joint_fusion":
        img_config = utils.get_config("stgcn", args)
        ehr_config = utils.get_config("stgcn", args)
        if args.ehr_encoder_name == "tabnet":
            if args.ehr_pretrain_path is not None:
                with open(
                    os.path.join(
                        args.ehr_pretrain_path.split("best.pth.tar")[0], "args.json"
                    ),
                    "r",
                ) as jf:
                    args_pretrained = json.load(jf)
                args_pretrained = DottedDict(args_pretrained)
                ehr_encoder_config = utils.get_config("tabnet", args_pretrained)
            else:
                ehr_encoder_config = utils.get_config("tabnet", args)
        else:
            ehr_encoder_config = None
        img_in_dim = img_features.shape[-1]
        ehr_in_dim = ehr_features.shape[-1]
        model = JointFusionModel(
            img_in_dim=img_in_dim,
            ehr_in_dim=ehr_in_dim,
            img_config=img_config,
            ehr_config=ehr_config,
            ehr_encoder_config=ehr_encoder_config,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            ehr_encoder_name=args.ehr_encoder_name,
            ehr_checkpoint_path=args.ehr_pretrain_path,
            cat_emb_dim=args.cat_emb_dim,
            freeze_pretrained=args.freeze_pretrained,
            joint_hidden=args.joint_hidden,
            num_classes=args.num_classes,
            dropout=args.dropout,
            device=device,
        )
    elif args.model_name == "joint_fusion_nontemporal":
        img_config = utils.get_config(args.g_conv, args)
        ehr_config = utils.get_config(args.g_conv, args)
        img_in_dim = img_features.shape[-1]
        ehr_in_dim = ehr_features.shape[-1]
        model = JointFusionNonTemporalModel(
            img_in_dim=img_in_dim,
            ehr_in_dim=ehr_in_dim,
            img_config=img_config,
            ehr_config=ehr_config,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            ehr_encoder_name=args.ehr_encoder_name,
            cat_emb_dim=args.cat_emb_dim,
            joint_hidden=args.joint_hidden,
            num_classes=args.num_classes,
            dropout=args.dropout,
            device=device,
        )
    elif args.model_name == "late_fusion":
        img_config = utils.get_config("stgcn", args)
        ehr_config = utils.get_config("stgcn", args)
        if args.ehr_encoder_name == "tabnet":
            if args.ehr_pretrain_path is not None:
                with open(
                    os.path.join(
                        args.ehr_pretrain_path.split("best.pth.tar")[0], "args.json"
                    ),
                    "r",
                ) as jf:
                    args_pretrained = json.load(jf)
                args_pretrained = DottedDict(args_pretrained)
                ehr_encoder_config = utils.get_config("tabnet", args_pretrained)
            else:
                ehr_encoder_config = utils.get_config("tabnet", args)
        else:
            ehr_encoder_config = None
        img_in_dim = img_features.shape[-1]
        ehr_in_dim = ehr_features.shape[-1]
        model = LateFusionModel(
            img_in_dim=img_in_dim,
            ehr_in_dim=ehr_in_dim,
            img_config=img_config,
            ehr_config=ehr_config,
            ehr_encoder_config=ehr_encoder_config,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            ehr_encoder_name=args.ehr_encoder_name,
            ehr_checkpoint_path=args.ehr_pretrain_path,
            cat_emb_dim=args.cat_emb_dim,
            freeze_pretrained=args.freeze_pretrained,
            num_classes=args.num_classes,
            dropout=args.dropout,
            device=device,
        )
    elif args.model_name == "early_fusion":
        config = utils.get_config("stgcn", args)  # hard-coded stgcn
        img_in_dim = img_features.shape[-1]
        ehr_in_dim = ehr_features.shape[-1]
        print("img_in_dim:", img_in_dim)
        print("ehr_in_dim:", ehr_in_dim)
        if args.ehr_encoder_name == "tabnet":
            if args.ehr_pretrain_path is not None:
                with open(
                    os.path.join(
                        args.ehr_pretrain_path.split("best.pth.tar")[0], "args.json"
                    ),
                    "r",
                ) as jf:
                    args_pretrained = json.load(jf)
                args_pretrained = DottedDict(args_pretrained)
                ehr_config = utils.get_config("tabnet", args_pretrained)
            else:
                ehr_config = utils.get_config("tabnet", args)
        else:
            ehr_config = None
        model = EarlyFusionModel(
            img_in_dim=img_in_dim,
            ehr_in_dim=ehr_in_dim,
            emb_dim=args.emb_dim,
            config=config,
            num_classes=args.num_classes,
            device=device,
            add_timedelta=args.add_timedelta,
            ehr_config=ehr_config,
            ehr_encoder_name=args.ehr_encoder_name,
            ehr_checkpoint_path=args.ehr_pretrain_path,
            freeze_ehr_encoder=args.freeze_pretrained,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=args.cat_emb_dim,
        )
    else:
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        model = GConvLayers(
            in_dim=in_dim,
            num_classes=args.num_classes,
            is_classifier=True,
            device=device,
            ehr_encoder_name=args.ehr_encoder_name,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=args.cat_emb_dim,
            **config
        )

    

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_wd
    )

    # load model checkpoint
    if args.load_model_path is not None:
#         model, optimizer = utils.load_model_checkpoint(
#             args.load_model_path, model, optimizer
#         )
        model = utils.load_model_checkpoint(
            args.load_model_path, model
        )
    
    model.to(device)
    # count params
    params = utils.count_parameters(model)
    logger.info("Trainable parameters: {}".format(params))

    # loss func
    if args.loss_func == "binary_cross_entropy":
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.FloatTensor(args.pos_weight)
        ).to(device)
    elif args.loss_func == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.pos_weight)).to(
            device
        )
    #         loss_fn = nn.CrossEntropyLoss().to(device)
    elif args.loss_func == "focal_loss" and args.label_name == "multiclass":
        loss_fn = utils.FocalLoss(
            alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="mean"
        ).to(device)
    elif args.loss_func == "focal_loss":
        loss_fn = utils.BinaryFocalLossWithLogits(
            alpha=args.focal_alpha, gamma=args.focal_gamma, reduction="mean"
        ).to(device)
    else:
        raise NotImplementedError

    # checkpoint saver
    saver = utils.CheckpointSaver(
        save_dir=args.save_dir,
        metric_name=args.metric_name,
        maximize_metric=args.maximize_metric,
        log=logger,
    )

    # scheduler
    logger.info("Using cosine annealing scheduler...")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    if args.do_train:
        # if undersampling majority class during training
        if args.use_sampler:
            if (args.label_name != "multiclass") and (args.dataset == "emory"):
                logger.info("Undersampling...")
                num_pos = int((labels[train_nid] == 1).sum())
                sampler = utils.ImbalancedDatasetSampler(
                    data, indices=train_nid, num_samples=num_pos * 2
                )
            elif (args.label_name == "multiclass") and (args.dataset == "emory"):
                logger.info("Balanced sampling...")
                sampler = utils.ImbalancedDatasetSampler(
                    data, indices=train_nid, num_samples=args.num_samples
                )
            else:
                logger.info("Upsampling...")
                num_neg = int((labels[train_nid] == 0).sum())
                sampler = utils.ImbalancedDatasetSampler(
                    data, indices=train_nid, num_samples=num_neg * 2
                )
            train_idxs = []
            for idx in sampler:
                train_idxs.append(idx)
            train_idxs = torch.tensor(train_idxs, dtype=torch.int64)
        else:
            train_idxs = train_nid

        # Train
        logger.info("Training...")
        model.train()
        epoch = 0
        prev_val_loss = 1e10
        patience_count = 0
        early_stop = False
        wandb.watch(model)
        while (epoch != args.num_epochs) and (not early_stop):

            epoch += 1
            logger.info("Starting epoch {}...".format(epoch))

            # augment training nodes' features using masking
            if args.data_augment:
                if args.feature_type != "multimodal":
                    features_aug = utils.feature_masking(
                        features, args.feature_mask_prob, device, train_mask.to(device)
                    )
                else:
                    img_features_aug = utils.feature_masking(
                        img_features,
                        args.feature_mask_prob,
                        device,
                        train_mask.to(device),
                    )
                    img_features_aug = utils.feature_masking(
                        ehr_features,
                        args.feature_mask_prob,
                        device,
                        train_mask.to(device),
                    )
            else:
                if args.feature_type != "multimodal":
                    features_aug = features
                else:
                    img_features_aug = img_features
                    ehr_features_aug = ehr_features

                    # forward
                    # if no temporal dim
            if "fusion" in args.model_name:
                if "nontemporal" not in args.model_name:
                    logits = model(g, img_features_aug, ehr_features_aug)
                else:
                    img_features_avg = img_features_aug[:, -1, :]
                    ehr_features_avg = ehr_features_aug[:, -1, :]
                    logits = model(g[0], img_features_avg, ehr_features_avg)
            elif (args.model_name != "stgcn") and (
                args.model_name != "graph_transformer"
            ):
                assert len(g) == 1
                features_avg = features_aug[:, -1, :]
                logits, _ = model(g[0], features_avg)
            else:
                logits, _ = model(g, features_aug)

            if logits.shape[-1] == 1:
                logits = logits.view(-1)
            loss = loss_fn(logits[train_idxs], labels[train_idxs])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"train_loss": loss.item(), "epoch": epoch})

            # evaluate on val set
            if epoch % args.eval_every == 0:
                logger.info("Evaluating at epoch {}...".format(epoch))
                eval_results = evaluate(
                    args=args,
                    model=model,
                    graph=g,
                    features=features,
                    labels=labels,
                    nid=val_nid,
                    loss_fn=loss_fn,
                    img_features=img_features,
                    ehr_features=ehr_features,
                )
                model.train()
                saver.save(epoch, model, optimizer, eval_results[args.metric_name])
                wandb.run.summary["best_{}".format(args.metric_name)] = saver.best_val
                # accumulate patience for early stopping
                if eval_results["loss"] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results["loss"]

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Log to console
                results_str = ", ".join(
                    "{}: {:.4f}".format(k, v) for k, v in eval_results.items()
                )
                logger.info("VAL - {}".format(results_str))

                for k, v in eval_results.items():
                    wandb.log({"val_{}".format(k): v, "epoch": epoch})

            # step lr scheduler
            scheduler.step()

        logger.info("Training DONE.")
        best_path = os.path.join(args.save_dir, "best.pth.tar")
        model = utils.load_model_checkpoint(best_path, model)
        model.to(device)

#     # evaluate
#     print('load from ', 'results/temporal_graph_non-imaging/train/train-03/best.pth.tar')
#     model = utils.load_model_checkpoint(
#         'results/temporal_graph_non-imaging/train/train-03/best.pth.tar', model
#     ) 
    model.eval()
    train_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=train_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "train_predictions.pkl"),
        thresh_search=args.thresh_search,
        img_features=img_features,
        ehr_features=ehr_features,
    )
    train_results_str = ", ".join(
        "{}: {:.4f}".format(k, v) for k, v in train_results.items()
    )
    logger.info("TRAIN - {}".format(train_results_str))
    wandb.log({"best_train_{}".format(args.metric_name): train_results[args.metric_name]})

    
    val_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=val_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "val_predictions.pkl"),
        thresh_search=args.thresh_search,
        img_features=img_features,
        ehr_features=ehr_features,
    )
    val_results_str = ", ".join(
        "{}: {:.4f}".format(k, v) for k, v in val_results.items()
    )
    logger.info("VAL - {}".format(val_results_str))
    wandb.log({"best_val_{}".format(args.metric_name): val_results[args.metric_name]})

    # eval on test set
    test_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=test_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "test_predictions.pkl"),
        best_thresh=val_results["best_thresh"],
        img_features=img_features,
        ehr_features=ehr_features,
    )
    test_results_str = ", ".join(
        "{}: {:.4f}".format(k, v) for k, v in test_results.items()
    )
    logger.info("TEST - {}".format(test_results_str))
    wandb.log({"best_test_{}".format(args.metric_name): test_results[args.metric_name]})

    logger.info("Results saved to {}".format(args.save_dir))

    if args.extract_embeddings:
        _, node_emb = model(graph=g, inputs=features)
        with open(os.path.join(args.save_dir, "node_embeddings.pkl"), "wb") as pf:
            pickle.dump(node_emb.detach().cpu().numpy(), pf)

    return val_results[args.metric_name]


if __name__ == "__main__":
    main(get_args())
