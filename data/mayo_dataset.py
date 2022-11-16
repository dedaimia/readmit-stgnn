import torch
import dgl
import os
import pickle
import pandas as pd
import numpy as np
from dgl.data import DGLDataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys
import gcsfs


from scipy.spatial.distance import cosine, hamming, euclidean, jaccard
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity

sys.path.append("../")
# from data.emory_dataset import compute_cos_sim_mat, compute_dist_mat
from data.readmission_utils import *

# READMIT_DEMO_FILE = "/home/siyi/data_readmission/Readmission_label_Demo_48hr_lowQualRemoved.csv"
# READMIT_DEMO_FILE = "/home/siyi/data_readmission/Readmission_mini.csv"

script_path = os.path.dirname(os.path.realpath(__file__))
code_path = os.path.join(script_path, "..")

cloud_storage_fs = None

def open_local_or_gs(path, flags, mode=0o777):
    if (path.startswith("gs:")):
        global cloud_storage_fs
        if (cloud_storage_fs is None):
            cloud_storage_fs = gcsfs.GCSFileSystem()
        return cloud_storage_fs.open(path, flags, mode)
    else:
        return open(path, flags, mode)

def compute_dist_mat(demo_dict, scale=False, admit_reason_dict=None):
    """
    NOTE: Only for Euclidean distance
    Args:
        demo_dict: dict, key is node name, value is demo/comor/pxs/cpt array
    Returns:
        dist_dict: dict of pairwise distances between nodes
    """

    demo_arr = []
    for _, arr in demo_dict.items():
        demo_arr.append(arr)
    demo_arr = np.stack(demo_arr, axis=0)

    # Scaler to scale each continuous variable to be between 0 and 1
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(demo_arr)
        demo_arr = scaler.transform(demo_arr)
    print(demo_arr)
    demo_arr[np.isnan(demo_arr)] = 0
    distances = euclidean_distances(X=demo_arr, Y=demo_arr)

    dist_dict = {"From": [], "To": [], "Distance": [], "Mask": []}
    node_names = list(demo_dict.keys())
    for idx_node1 in tqdm(range(len(node_names))):
        for idx_node2 in range(idx_node1, len(node_names)):
            node1 = node_names[idx_node1]
            node2 = node_names[idx_node2]

            dist_dict["From"].append(node1)
            dist_dict["To"].append(node2)
            dist_dict["Distance"].append(distances[idx_node1, idx_node2])

            # admission reason
            if admit_reason_dict is not None:
                reason1 = admit_reason_dict[node1]
                reason2 = admit_reason_dict[node2]

                if idx_node1 == idx_node2:
                    dist_dict["Mask"].append(1)
                else:
                    # same reason, mask 1
                    if np.all(reason1 == reason2):
                        dist_dict["Mask"].append(1)
                    else:
                        dist_dict["Mask"].append(0)
                    # if (reason1 == reason2) and \
                    #     not(isinstance(reason1, float) and np.isnan(reason1)):
                    #     dist_dict["Mask"].append(1)
                    # else:
                    #     dist_dict["Mask"].append(0)

    return dist_dict


def compute_cos_sim_mat(demo_dict, scale=False, admit_reason_dict=None):
    """
    NOTE: Only implemented for cosine similarity
    Args:
        demo_dict: key is patient, value is demographics/comorbidity array
    Returns:
        cos_sim_dict: dict, containing source patient & destination patient & their cosine sim
    """

    # Scaler to scale each variable to be between 0 and 1
    demo_arr = []
    for _, arr in demo_dict.items():
        demo_arr.append(arr)
    demo_arr = np.stack(demo_arr, axis=0)

    # Scaler to scale each continuous variable to be between 0 and 1
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(demo_arr)
        demo_arr = scaler.transform(demo_arr)

    cos_sim = cosine_similarity(demo_arr, demo_arr)

    cos_sim_dict = {"From": [], "To": [], "CosineSim": [], "Mask": []}
    node_names = list(demo_dict.keys())
    for idx_node1 in tqdm(range(len(node_names))):
        for idx_node2 in range(idx_node1, len(node_names)):
            node1 = node_names[idx_node1]
            node2 = node_names[idx_node2]

            cos_sim_dict["From"].append(node_names[idx_node1])
            cos_sim_dict["To"].append(node_names[idx_node2])
            cos_sim_dict["CosineSim"].append(cos_sim[idx_node1, idx_node2])

            # admission reason
            if admit_reason_dict is not None:
                reason1 = admit_reason_dict[node1]
                reason2 = admit_reason_dict[node2]

                if idx_node1 == idx_node2:
                    cos_sim_dict["Mask"].append(1)
                else:
                    # same reason, mask 1
                    if np.all(reason1 == reason2):
                        cos_sim_dict["Mask"].append(1)
                    else:
                        cos_sim_dict["Mask"].append(0)
                    # if (reason1 == reason2) and \
                    #         not(isinstance(reason1, float) and np.isnan(reason1)):
                    #     cos_sim_dict["Mask"].append(1)
                    # else:
                    #     cos_sim_dict["Mask"].append(0)

    return cos_sim_dict

def construct_graph_readmission(
    df_demo,
    ehr_feature_file=None,
    edge_ehr_file=None,
    edge_modality=["demo"],
    feature_type="imaging",
    img_feature_dir=None,
    top_perc=None,
    sigma=None,
    gauss_kernel=False,
    standardize=True,
    max_seq_len_img=5,
    max_seq_len_ehr=8,
    label_cutoff=1,
    sim_measure="cosine",
    dynamic_graph=False,
    pad_front=False,
    padding_val=None,
    add_timedelta=False,
    mask_by_admit_reason=False,
    ehr_types=["demo", "cpt", "icd", "lab", "med"],
    img_by="cxr",
    dataset_name="mayo",
):
    """
    Construct an entire graph for transductive learning
    Args:
    Returns:
        edge_dict: dict that defines the graph,
            {"From": list of node idxs for start of the edge,
            "To": list of node idxs for end of the edge,
            "Weight": edge weight, float between 0 to 1
            }
        dgl_G: dgl graph
    """
    if feature_type not in ["imaging", "non-imaging", "multimodal"]:
        raise NotImplementedError

    if sim_measure not in ["cosine", "euclidean"]:
        raise NotImplementedError

    if (feature_type == "multimodal") and (dynamic_graph):
        raise ValueError("Dynamic graph for multimodal features not supported.")

    if dataset_name not in ["mayo", "mimic"]:
        raise NotImplementedError

    # node labels
    if dataset_name == "mayo":
        (
            labels,
            node_included_files,
            splits,
            time_deltas,
            hospital_stays,
            time_idxs,
        ) = get_readmission_label(df_demo, cutoff_months=label_cutoff, max_seq_len=None)
    else:
        (
            labels,
            node_included_files,
            splits,
            time_deltas,
            hospital_stays,
            time_idxs,
        ) = get_readmission_label_mimic(df_demo, max_seq_len=None)
    train_idxs = np.array([ind for ind in range(len(splits)) if splits[ind] == "train"])
    node_names = list(node_included_files.keys())

    if not (add_timedelta):
        time_deltas = None

    # node name to node index dict
    node2idx = {}
    for idx, name in enumerate(node_names):
        node2idx[name] = idx

    # save node mapping to join with predictions for patient info
    with open(os.path.join(code_path, 'node_mapping.pkl'), 'wb') as f:
        pickle.dump(node2idx, f)

    # node features (i.e. imaging features) to ndata
    if feature_type == "imaging":
        assert img_feature_dir is not None
        # img_feat_dict = get_img_features(img_feature_dir, node_included_files)
        img_feat_dict = get_img_features(
            img_feature_dir,
            node_included_files,
            time_idxs=time_idxs,
            hospital_stay=hospital_stays,
            by=img_by,
        )
        node_features, seq_lengths = get_feat_seq(
            node_names,
            img_feat_dict,
            max_seq_len_img,
            pad_front,
            time_deltas,
            padding_val=padding_val,
        )
        cat_idxs = []
        cat_dims = []

    elif feature_type == "non-imaging":
        assert ehr_feature_file is not None
        with open_local_or_gs(ehr_feature_file, "rb") as pf:
            raw_feat_dict = pickle.load(pf)
        feat_dict = raw_feat_dict["feat_dict"]
        feat_cols = raw_feat_dict["feature_cols"]
        cols_to_keep = []

        for ehr_name in ehr_types:
            cols_to_keep = cols_to_keep + raw_feat_dict["{}_cols".format(ehr_name)]

        col_idxs = np.array(
            [feat_cols.index(col) for col in cols_to_keep]
        )  # wrt original cols
        feat_dict = {
            name: feat_dict[name][:, col_idxs] for name in node_names
        }  # get relevant cols
        node_features, seq_lengths = get_feat_seq(
            node_names,
            feat_dict,
            max_seq_len_ehr,
            pad_front,
            time_deltas,
            padding_val=padding_val,
        )

        if "cat_idxs" in raw_feat_dict:
            cat_col2dim = {
                feat_cols[raw_feat_dict["cat_idxs"][ind]]: raw_feat_dict["cat_dims"][
                    ind
                ]
                for ind in range(len(raw_feat_dict["cat_dims"]))
            }

            # reindex categorical variables
            cat_cols = [
                col
                for col in cols_to_keep
                if (feat_cols.index(col) in raw_feat_dict["cat_idxs"])
            ]
            cat_idxs = [cols_to_keep.index(col) for col in cat_cols]
            cat_dims = [cat_col2dim[col] for col in cat_cols]
        else:
            cat_idxs = []
            cat_dims = []

        assert np.all(node_features != -1)
        assert node_features.shape[1] == max_seq_len_ehr
        del feat_dict

        img_feat_dict = None

    elif feature_type == "multimodal":
        assert img_feature_dir is not None
        assert ehr_feature_file is not None

        # imaging features
        # img_feat_dict = get_img_features(img_feature_dir, node_included_files)
        img_feat_dict = get_img_features(
            img_feature_dir,
            node_included_files,
            time_idxs=time_idxs,
            hospital_stay=hospital_stays,
            by=img_by,
        )
        img_node_features, seq_lengths = get_feat_seq(
            node_names,
            img_feat_dict,
            max_seq_len_img,
            pad_front,
            time_deltas,
            padding_val=padding_val,
        )

        # ehr features
        with open_local_or_gs(ehr_feature_file, "rb") as pf:
            raw_ehr_dict = pickle.load(pf)
        ehr_feat_dict = raw_ehr_dict["feat_dict"]
        ehr_feat_cols = raw_ehr_dict["feature_cols"]
        cols_to_keep = []

        for ehr_name in ehr_types:
            cols_to_keep = cols_to_keep + raw_ehr_dict["{}_cols".format(ehr_name)]

        col_idxs = np.array(
            [ehr_feat_cols.index(col) for col in cols_to_keep]
        )  # wrt original cols
        ehr_feat_dict = {
            name: ehr_feat_dict[name][:, col_idxs] for name in node_names
        }  # get relevant cols
        ehr_node_features, seq_lengths = get_feat_seq(
            node_names,
            ehr_feat_dict,
            max_seq_len_ehr,
            pad_front,
            time_deltas,
            padding_val=padding_val,
        )

        if "cat_idxs" in raw_ehr_dict:
            cat_col2dim = {
                ehr_feat_cols[raw_ehr_dict["cat_idxs"][ind]]: raw_ehr_dict["cat_dims"][
                    ind
                ]
                for ind in range(len(raw_ehr_dict["cat_dims"]))
            }

            # reindex categorical variables
            cat_cols = [
                col
                for col in cols_to_keep
                if (ehr_feat_cols.index(col) in raw_ehr_dict["cat_idxs"])
            ]
            cat_idxs = [cols_to_keep.index(col) for col in cat_cols]
            cat_dims = [cat_col2dim[col] for col in cat_cols]
        else:
            cat_idxs = []
            cat_dims = []

        del ehr_feat_dict
        del raw_ehr_dict

        node_features = None
    else:
        raise NotImplementedError

    # standardize
    if standardize:
        if feature_type == "non-imaging":
            seq_len = max_seq_len_ehr
            scaler = StandardScaler()
            train_feat = node_features[train_idxs].reshape(
                (len(train_idxs) * seq_len, -1)
            )
            # NOTE: only standardize non-categorical features
            continuous_cols = np.array(
                [ind for ind in range(train_feat.shape[-1]) if ind not in cat_idxs]
            )
            scaler.fit(train_feat[:, continuous_cols])
            continuous_node_features = scaler.transform(
                node_features[:, :, continuous_cols].reshape(
                    (len(node_names) * seq_len, -1)
                )
            )
            node_features[:, :, continuous_cols] = continuous_node_features.reshape(
                (len(node_names), seq_len, -1)
            )
            del train_feat
        elif feature_type == "multimodal":
            # img_scaler = StandardScaler()
            # img_train_feat = img_node_features[train_idxs].reshape((len(train_idxs) * max_seq_len_img, -1))
            # img_scaler.fit(img_train_feat)
            # img_node_features = img_scaler.transform(img_node_features.reshape((len(node_names) * max_seq_len_img, -1)))
            # img_node_features = img_node_features.reshape((len(node_names), max_seq_len_img, -1))
            # del img_train_feat

            ehr_scaler = StandardScaler()
            ehr_train_feat = ehr_node_features[train_idxs].reshape(
                (len(train_idxs) * max_seq_len_ehr, -1)
            )
            # NOTE: only standardize non-categorical features
            continuous_cols = np.array(
                [ind for ind in range(ehr_train_feat.shape[-1]) if ind not in cat_idxs]
            )
            ehr_scaler.fit(ehr_train_feat[:, continuous_cols])
            continuous_ehr_node_features = ehr_scaler.transform(
                ehr_node_features[:, :, continuous_cols].reshape(
                    (len(node_names) * max_seq_len_ehr, -1)
                )
            )
            ehr_node_features[
                :, :, continuous_cols
            ] = continuous_ehr_node_features.reshape(
                (len(node_names), max_seq_len_ehr, -1)
            )
            del ehr_train_feat
        else:
            scaler = None

    # edges
    node_edge_dict = {}
    if "hospital_stay" in edge_modality:
        for name in node_names:
            if name not in node_edge_dict:
                node_edge_dict[name] = []
            node_edge_dict[name] = np.concatenate(
                [node_edge_dict[name], [hospital_stays[name]]], axis=-1
            )

    if (
        ("demo" in edge_modality)
        or ("cpt" in edge_modality)
        or ("icd" in edge_modality)
        or ("lab" in edge_modality)
        or ("imaging" in edge_modality)
        or ("med" in edge_modality)
        or ("admit_reason" in edge_modality)
    ):
        assert edge_ehr_file is not None
        with open_local_or_gs(edge_ehr_file, "rb") as pf:
            raw_ehr_dict = pickle.load(pf)
        feat_cols = raw_ehr_dict["feature_cols"]
        node_edge_dict = {}

        if "demo" in edge_modality:
            demo_col_idxs = np.array(
                [feat_cols.index(col) for col in raw_ehr_dict["demo_cols"]]
            )  # wrt original cols
            demo_dict = {
                name: raw_ehr_dict["feat_dict"][name][0, demo_col_idxs]
                for name in node_names
            }
            for name in node_names:
                if name not in node_edge_dict:
                    node_edge_dict[name] = []
                node_edge_dict[name] = np.concatenate(
                    [node_edge_dict[name], demo_dict[name]], axis=-1
                )

        if "med" in edge_modality:
            med_col_idxs = np.array(
                [feat_cols.index(col) for col in raw_ehr_dict["med_cols"]]
            )  # wrt original cols
            med_dict = {
                name: raw_ehr_dict["feat_dict"][name][0, med_col_idxs]
                for name in node_names
            }
            for name in node_names:
                if name not in node_edge_dict:
                    node_edge_dict[name] = []
                node_edge_dict[name] = np.concatenate(
                    [node_edge_dict[name], med_dict[name]], axis=-1
                )

        if "admit_reason" in edge_modality:
            admit_cols_idxs = np.array([feat_cols.index(col) for col in ADMIT_COLS])
            admit_dict = {
                name: raw_ehr_dict["feat_dict"][name][0, admit_cols_idxs]
                for name in node_names
            }
            for name in node_names:
                if name not in node_edge_dict:
                    node_edge_dict[name] = []
                node_edge_dict[name] = np.concatenate(
                    [node_edge_dict[name], admit_dict[name]], axis=-1
                )

        if mask_by_admit_reason:
            admit_cols_idxs = np.array([feat_cols.index(col) for col in ADMIT_COLS])
            admit_dict = {
                name: raw_ehr_dict["feat_dict"][name][0, admit_cols_idxs]
                for name in node_names
            }
        else:
            admit_dict = None

        # time varying edges
        if (
            ("cpt" in edge_modality)
            or ("icd" in edge_modality)
            or ("lab" in edge_modality)
            or ("imaging" in edge_modality)
            or ("med" in edge_modality)
        ):
            if "cpt" in edge_modality:
                cpt_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["cpt_cols"]]
                )  # wrt original cols
                cpt_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, cpt_col_idxs]
                    for name in node_names
                }
            else:
                cpt_dict = None
            if "icd" in edge_modality:
                icd_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["icd_cols"]]
                )  # wrt original cols
                icd_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, icd_col_idxs]
                    for name in node_names
                }
            else:
                icd_dict = None
            if "lab" in edge_modality:
                lab_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["lab_cols"]]
                )  # wrt original cols
                lab_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, lab_col_idxs]
                    for name in node_names
                }
            else:
                lab_dict = None
            if "med" in edge_modality:
                med_col_idxs = np.array(
                    [feat_cols.index(col) for col in raw_ehr_dict["med_cols"]]
                )  # wrt original cols
                med_dict = {
                    name: raw_ehr_dict["feat_dict"][name][:, med_col_idxs]
                    for name in node_names
                }
            else:
                med_dict = None
            ehr_edge_dict_list = get_time_varying_edges(
                node_names=node_names,
                max_seq_len=max_seq_len_img
                if feature_type == "imaging"
                else max_seq_len_ehr,
                edge_dict=node_edge_dict,
                edge_modality=edge_modality,
                hospital_stay=hospital_stays,
                cpt_dict=cpt_dict,
                icd_dict=icd_dict,
                lab_dict=lab_dict,
                med_dict=med_dict,
                img_dict=None,
                pad_front=pad_front,
                dynamic_graph=dynamic_graph,
                dataset=dataset_name,
            )
        else:
            ehr_edge_dict_list = [node_edge_dict]
    else:
        ehr_edge_dict_list = []

    if "imaging" in edge_modality:
        img_edge_dict_list = get_time_varying_edges(
            node_names=node_names,
            max_seq_len=max_seq_len_img
            if feature_type == "imaging"
            else max_seq_len_ehr,
            edge_dict={},
            edge_modality="imaging",
            cpt_dict=None,
            icd_dict=None,
            lab_dict=None,
            img_dict=img_feat_dict,
            pad_front=pad_front,
            dynamic_graph=dynamic_graph,
            dataset=dataset_name,
        )
    else:
        img_edge_dict_list = []

    print("Using {} for similarity/distance measure...".format(sim_measure))
    ehr_dist_dict_list = []
    img_dist_dict_list = []
#     print(ehr_edge_dict_list)
    if sim_measure == "cosine":
        for new_edge_dict in ehr_edge_dict_list:
            ehr_dist_dict_list.append(
                compute_cos_sim_mat(
                    new_edge_dict, scale=True, admit_reason_dict=admit_dict
                )
            )
        for new_edge_dict in img_edge_dict_list:
            img_dist_dict_list.append(
                compute_cos_sim_mat(
                    new_edge_dict, scale=True, admit_reason_dict=admit_dict
                )
            )
    elif sim_measure == "euclidean":
        for new_edge_dict in ehr_edge_dict_list:
            ehr_dist_dict_list.append(
                compute_dist_mat(
                    new_edge_dict, scale=True, admit_reason_dict=admit_dict
                )
            )
        for new_edge_dict in img_edge_dict_list:
            img_dist_dict_list.append(
                compute_dist_mat(
                    new_edge_dict, scale=True, admit_reason_dict=admit_dict
                )
            )
    else:
        raise NotImplementedError

    # Construct graphs
    g_list = []
    num_graphs = np.maximum(len(ehr_dist_dict_list), len(img_dist_dict_list))
    for idx_g in range(num_graphs):
        # both ehr & imaging are used as edges, edge_weight = ehr_edge_weight * img_edge_weight
        if len(ehr_dist_dict_list) > 0 and len(img_dist_dict_list) > 0:
            assert len(ehr_dist_dict_list) == len(img_dist_dict_list)
            ehr_edges = compute_edges(
                ehr_dist_dict_list[idx_g],
                node_names,
                sigma=sigma,
                top_perc=None,
                gauss_kernel=gauss_kernel,
                admit_reason_dict=admit_dict,
            )  # keep all edges first
            img_edges = compute_edges(
                img_dist_dict_list[idx_g],
                node_names,
                sigma=sigma,
                top_perc=None,
                gauss_kernel=gauss_kernel,
                admit_reason_dict=admit_dict,
            )  # keep all edges first
            edges = ehr_edges * img_edges  # element wise multiplication of edge weight

            if (top_perc is not None) and (admit_dict is None):
                num = len(edges)
                num_to_keep = int(num * top_perc)
                sorted_dist = np.sort(edges)[::-1]  # descending order
                thresh = sorted_dist[:num_to_keep][-1]
                mask = edges >= thresh
                mask[edges < 0] = 0  # no edge for negative "distance"
                edges = edges * mask

            edge_dict = {
                "From": ehr_dist_dict_list[idx_g]["From"],
                "To": ehr_dist_dict_list[idx_g]["To"],
                "Weight": [],
            }

        else:
            curr_dict = (
                img_dist_dict_list[idx_g]
                if len(img_dist_dict_list) > 0
                else ehr_dist_dict_list[idx_g]
            )
            edges = compute_edges(
                curr_dict,
                node_names,
                sigma=sigma,
                top_perc=top_perc,
                gauss_kernel=gauss_kernel,
                admit_reason_dict=admit_dict,
            )

            if len(img_dist_dict_list) > 0:
                edge_dict = {
                    "From": img_dist_dict_list[idx_g]["From"],
                    "To": img_dist_dict_list[idx_g]["To"],
                    "Weight": [],
                }
            else:
                edge_dict = {
                    "From": ehr_dist_dict_list[idx_g]["From"],
                    "To": ehr_dist_dict_list[idx_g]["To"],
                    "Weight": [],
                }

        src_nodes = []
        dst_nodes = []
        weights = []
        for idx in range(len(edge_dict["From"])):
            from_node_name = edge_dict["From"][idx]
            to_node_name = edge_dict["To"][idx]

            if (from_node_name not in node2idx) or (to_node_name not in node2idx):
                raise ValueError

            from_node = node2idx[from_node_name]
            to_node = node2idx[to_node_name]

            if edges[idx] == 0:
                edge_dict["Weight"].append(0)  # no edge
            else:
                edge_dict["Weight"].append(edges[idx])
                src_nodes.append(from_node)
                dst_nodes.append(to_node)
                weights.append(edges[idx])
        src_nodes = torch.tensor(src_nodes)
        dst_nodes = torch.tensor(dst_nodes)

        del edge_dict

        g_directed = dgl.graph((src_nodes, dst_nodes), idtype=torch.int32)
        g_directed.edata["weight"] = torch.FloatTensor(weights)

        # NOTE: to_bidirected is needed for undirected graphs in DGL!!
        # This will also copy over edata, we need edge weight
        dgl_G = dgl.add_reverse_edges(g_directed, copy_ndata=True, copy_edata=True)
        dgl_G = dgl.to_simple(
            dgl_G, return_counts=None, copy_ndata=True, copy_edata=True
        )

        num_nodes = dgl_G.num_nodes()
        train_masks = torch.zeros(num_nodes, dtype=torch.int32)
        val_masks = torch.zeros(num_nodes, dtype=torch.int32)
        test_masks = torch.zeros(num_nodes, dtype=torch.int32)

        train_ind = torch.LongTensor(
            [ind for ind in range(len(splits)) if splits[ind] == "train"]
        )
        val_ind = torch.LongTensor(
            [ind for ind in range(len(splits)) if splits[ind] == "val"]
        )
        test_ind = torch.LongTensor(
            [ind for ind in range(len(splits)) if splits[ind] == "test"]
        )
        train_masks[train_ind] = 1
        val_masks[val_ind] = 1
        test_masks[test_ind] = 1

        dgl_G.ndata["train_mask"] = train_masks
        dgl_G.ndata["val_mask"] = val_masks
        dgl_G.ndata["test_mask"] = test_masks

        dgl_G.ndata["label"] = torch.FloatTensor(labels)
        dgl_G.ndata["seq_lengths"] = torch.FloatTensor(seq_lengths)

        if feature_type == "multimodal":
            dgl_G.ndata["img_feat"] = torch.FloatTensor(img_node_features)
            dgl_G.ndata["ehr_feat"] = torch.FloatTensor(ehr_node_features)
        else:
            dgl_G.ndata["feat"] = torch.FloatTensor(node_features)
        g_list.append(dgl_G)

    return node2idx, g_list, cat_idxs, cat_dims, scaler


class ReadmissionDataset(DGLDataset):
    def __init__(
        self,
        demo_file,
        edge_ehr_file=None,
        ehr_feature_file=None,
        edge_modality=["demo"],
        feature_type="imaging",
        img_feature_dir=None,
        top_perc=None,
        sigma=None,
        gauss_kernel=False,
        max_seq_len_img=6,
        max_seq_len_ehr=8,
        label_cutoff=1,
        sim_measure="hamming",
        dynamic_graph=False,
        standardize=True,
        pad_front=False,
        add_timedelta=False,
        mask_by_admit_reason=False,
        ehr_types=["demo", "cpt", "icd", "lab", "med"],
        img_by="cxr",
        dataset_name="mayo",
    ):
        """
        Args:
            dist_dir: full path to re-computed cosine similarity / Euclidean distances (pkl file)
            pat2node_dir: full path to pat to node idx dict (pkl file)
            thresh: float
            split_dir: full path to train/val/test split pkl file
            max_seq_len: int, maximum sequence length
            label_cutoff: int, cutoff hospital stay days to get binary labels
            gauss_kernel: if true, will applly Gaussian kernel to cosine distance measure
        """
        self.demo_file = demo_file
        self.edge_modality = edge_modality
        self.feature_type = feature_type
        self.img_feature_dir = img_feature_dir
        self.ehr_feature_file = ehr_feature_file
        self.edge_ehr_file = edge_ehr_file
        self.top_perc = top_perc
        self.sigma = sigma
        self.gauss_kernel = gauss_kernel
        self.max_seq_len_img = max_seq_len_img
        self.max_seq_len_ehr = max_seq_len_ehr
        self.label_cutoff = label_cutoff
        self.sim_measure = sim_measure
        self.dynamic_graph = dynamic_graph
        self.standardize = standardize
        self.pad_front = pad_front
        self.add_timedelta = add_timedelta
        self.mask_by_admit_reason = mask_by_admit_reason
        self.ehr_types = ehr_types
        self.img_by = img_by
        self.dataset_name = dataset_name

        if sim_measure not in ["cosine", "euclidean"]:
            raise NotImplementedError

        print("Edge modality:", edge_modality)

        # get patients
        self.df_all = pd.read_csv(demo_file)

        # if admit_reason_dict_dir is not None:
        #     with open(admit_reason_dict_dir, 'rb') as pf:
        #         self.admit_reason_dict = pickle.load(pf)
        # else:
        #     self.admit_reason_dict = None

        super().__init__(name="readmission")

    def process(self):
        (
            self.node2idx,
            self.graphs,
            self.cat_idxs,
            self.cat_dims,
            self.scaler,
        ) = construct_graph_readmission(
            df_demo=self.df_all,
            ehr_feature_file=self.ehr_feature_file,
            edge_ehr_file=self.edge_ehr_file,
            edge_modality=self.edge_modality,
            feature_type=self.feature_type,
            img_feature_dir=self.img_feature_dir,
            top_perc=self.top_perc,
            sigma=self.sigma,
            gauss_kernel=self.gauss_kernel,
            standardize=self.standardize,
            max_seq_len_img=self.max_seq_len_img,
            max_seq_len_ehr=self.max_seq_len_ehr,
            label_cutoff=self.label_cutoff,
            sim_measure=self.sim_measure,
            dynamic_graph=self.dynamic_graph,
            pad_front=self.pad_front,
            add_timedelta=self.add_timedelta,
            mask_by_admit_reason=self.mask_by_admit_reason,
            ehr_types=self.ehr_types,
            img_by=self.img_by,
            dataset_name=self.dataset_name,
        )

        self.targets = self.graphs[0].ndata["label"].cpu().numpy()

    def __getitem__(self, i):
        return self.graphs

    def __len__(self):
        return len(self.graphs)