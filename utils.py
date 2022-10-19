import logging
import numpy as np
import os
import pickle
import sys
import torch
import json
import time
import random
import queue
import shutil
import tqdm
import math
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# from dgl.data import DGLDataset
import scipy.sparse as sp
import pandas as pdƒ

# import sklearn
from scipy.sparse import linalg
from collections import defaultdict

# from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict, defaultdict
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.metrics import precision_recall_curve, average_precision_score


def last_relevant_pytorch(output, lengths, batch_first=False):
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.type(torch.int64)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output


def seed_torch(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.
    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.
    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.
        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, "log.txt")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%m.%d.%y %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%m.%d.%y %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_save_dir(base_dir, training, id_max=5000):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = "train" if training else "test"
        save_dir = os.path.join(base_dir, subdir, "{}-{:02d}".format(subdir, uid))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError(
        "Too many save directories created with the same name. \
                       Delete old save directories or use another name."
    )


class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, metric_name, maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(
            "Saver will {}imize {}...".format(
                "max" if maximize_metric else "min", metric_name
            )
        )

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return (self.maximize_metric and self.best_val <= metric_val) or (
            not self.maximize_metric and self.best_val >= metric_val
        )

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, epoch, model, optimizer, metric_val):
        """Save model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(self.save_dir, "last.pth.tar")
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ""
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, "best.pth.tar")
            shutil.copy(checkpoint_path, best_path)
            self._print("New best checkpoint at epoch {}...".format(epoch))

    def save_multi(self, epoch, model_dict, optimizer_dict, metric_val):
        """Save multiple model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """

        ckpt_dict = {
            "epoch": epoch,
            # 'model_state': model.state_dict(),
            # 'optimizer_state': optimizer.state_dict()
        }

        for model_name, model in model_dict.items():
            ckpt_dict[model_name + "_model_state"] = model.state_dict()
        for optimizer_name, optimizer in optimizer_dict.items():
            ckpt_dict[optimizer_name + "_optimizer_state"] = optimizer.state_dict()

        checkpoint_path = os.path.join(self.save_dir, "last.pth.tar")
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ""
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, "best.pth.tar")
            shutil.copy(checkpoint_path, best_path)
            self._print("New best checkpoint at epoch {}...".format(epoch))


def load_model_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint["model_state"])
    except:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return model, optimizer

    return model


def load_model_checkpoint_multi(checkpoint_file, model_dict, optimizer_dict=None):
    checkpoint = torch.load(checkpoint_file)
    for model_name, model in model_dict.items():
        model.load_state_dict(checkpoint[model_name + "_model_state"])
        model_dict[model_name] = model
    if optimizer_dict is not None:
        for optimizer_name, optimizer in optimizer_dict.items():
            optimizer.load_state_dict(checkpoint[optimizer_name + "_optimizer_state"])
            optimizer_dict[optimizer_name] = optimizer
        return model_dict, optimizer_dict

    return model_dict


def count_parameters(model):
    """
    Counter total number of parameters, for Pytorch
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Keep track of average values over time.
    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Adapted from https://github.com/ufoym/imbalanced-dataset-sampler
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        if isinstance(dataset, Dataset):  # torch dataset
            # if indices is not provided,
            # all elements in the dataset will be considered
            self.indices = list(range(len(dataset))) if indices is None else indices
        else:  # DGL dataset
            self.indices = (
                list(range(dataset.graph[0].num_nodes()))
                if indices is None
                else indices
            )

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.targets[idx]

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class ImbalancedNodeSampler(torch.utils.data.sampler.Sampler):
    """
    Adapted from https://github.com/ufoym/imbalanced-dataset-sampler
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(graph.num_nodes())) if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.targets[idx]

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


def eval_dict(
    y, y_pred, y_prob, average="binary", thresh_search=False, best_thresh=0.5
):
    """
    Args:
        y : labels, shape (num_examples, num_classes)
        y_pred: per-time-step predictions, shape (num_examples, num_classes)
        y-prob: per-time-step probabilities, shape (num_examples, num_classes)
        average: 'weighted', 'micro', 'macro' etc. to compute F1 score etc.
    Returns:
        scores_dict: Dictionary containing scores such as F1, acc etc.
    """
    if thresh_search:
        best_thresh = thresh_max_f1(y_true=y, y_prob=y_prob)
        y_pred = (y_prob >= best_thresh).astype(int)

    scores_dict = {}
    if len(np.unique(y)) == 2:  # binary case
        scores_dict["auroc"] = roc_auc_score(y_true=y, y_score=y_prob)
        scores_dict["aupr"] = average_precision_score(y, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_pred).ravel()
        scores_dict["specificity"] = tn / (tn + fp)

    scores_dict["acc"] = accuracy_score(y_true=y, y_pred=y_pred)
    scores_dict["F1"] = f1_score(y_true=y, y_pred=y_pred, average=average)
    scores_dict["precision"] = precision_score(
        y_true=y, y_pred=y_pred, average=average, zero_division=0
    )
    scores_dict["recall"] = recall_score(y_true=y, y_pred=y_pred, average=average)

    scores_dict["best_thresh"] = best_thresh

    return scores_dict


def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    if len(np.unique(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


#### Copied from https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py ####
from typing import Optional


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.
    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.
    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,
    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])
    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            "Input labels type is not a torch.Tensor. Got {}".format(type(labels))
        )

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype)
        )

    if num_classes < 1:
        raise ValueError(
            "The number of classes must be bigger than one."
            " Got: {}".format(num_classes)
        )

    shape = labels.shape
    one_hot = torch.zeros(
        (shape[0], num_classes) + shape[1:], device=device, dtype=dtype
    )

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.
    Return:
        the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))

    if not len(input.shape) >= 2:
        raise ValueError(
            "Invalid input shape, we expect BxCx*. Got: {}".format(input.shape)
        )

    if input.size(0) != target.size(0):
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(
                input.size(0), target.size(0)
            )
        )

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(
            "Expected target size {}, got {}".format(out_size, target.size())
        )

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".format(
                input.device, target.device
            )
        )

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1], device=input.device, dtype=input.dtype
    )

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(reduction))
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(
        self,
        alpha: float,
        gamma: float = 2.0,
        reduction: str = "none",
        eps: float = 1e-8,
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(
            input, target, self.alpha, self.gamma, self.reduction, self.eps
        )


def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: Optional[float] = None,
) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input: input data tensor of arbitrary shape.
        target: the target tensor with shape matching input.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.
    Returns:
        the computed loss.
    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[6.325]],[[5.26]],[[87.49]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(21.8725)
    """

    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`binary_focal_loss_with_logits` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(
            f"Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)})."
        )

    probs_pos = torch.sigmoid(input)
    probs_neg = torch.sigmoid(-input)
    loss_tmp = -alpha * torch.pow(probs_neg, gamma) * target * F.logsigmoid(input) - (
        1 - alpha
    ) * torch.pow(probs_pos, gamma) * (1.0 - target) * F.logsigmoid(-input)

    if reduction == "none":
        loss = loss_tmp
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha): Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
    Shape:
        - Input: :math:`(N, *)`.
        - Target: :math:`(N, *)`.
    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = BinaryFocalLossWithLogits(**kwargs)
        >>> input = torch.randn(1, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(
        self, alpha: float, gamma: float = 2.0, reduction: str = "none"
    ) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(
            input.reshape(-1, 1),
            target.reshape(-1, 1),
            self.alpha,
            self.gamma,
            self.reduction,
        )


#### Focal loss copied from https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py ####


def get_effective_sample_weights(labels, beta=0.999):
    """
    Weights for balanced loss from paper: https://arxiv.org/pdf/1901.05555.
    Authors suggest varying beta from 0.9, 0.99, 0.999, 0.9999
    """
    weights = []
    num_classes = len(np.unique(labels))
    for c in range(num_classes):
        num_samples = np.sum(labels == c)
        w = (1 - beta) / (1 - np.power(beta, num_samples))
        weights.append(w)
    return weights


def add_masked_gaussian_noise(x, train_idxs, device, std=0.1):
    """
    Adds Gaussian noise with zero mean and <std> standard deviation to training points in tensor x
    Args:
        x: tensor, first dim is batch, (batch, ...)
        std: standard deviation of the Gaussian
    Returns:
        augmented x
    """
    x_shape = x.shape
    noise = (torch.rand(x_shape) * std).to(device)
    mask = torch.zeros(x_shape).to(device)
    mask[train_idxs] = 1
    return x + noise * mask


def feature_masking(features, p, device, train_mask=None):
    """
    Mask all training datapoints in the same way to preserve graph topology
    """
    if len(features.shape) == 3:
        feat_mask = (
            torch.FloatTensor(features.shape[1], features.shape[2]).uniform_() > p
        )
    else:
        feat_mask = torch.FloatTensor(features.shape[1]).uniform_() > p
    feat_mask = feat_mask.to(device)
    feat_aug = features * feat_mask
    if train_mask is not None:
        # do not mask val/test nodes
        feat_aug[train_mask != 1] = features[train_mask != 1]
    return feat_aug


class Augmentation:
    """
    My own augmentation function for DGL graphs
    """

    def __init__(self, p_f1=0.2, p_f2=0.1, p_e1=0.2, p_e2=0.3, device="cpu"):
        """
        two simple graph augmentation functions --> "Node feature masking" and "Edge masking"
        Random binary node feature mask following Bernoulli distribution with parameter p_f
        Random binary edge mask following Bernoulli distribution with parameter p_e
        """
        self.p_f1 = p_f1
        self.p_f2 = p_f2
        self.p_e1 = p_e1
        self.p_e2 = p_e2
        self.method = "BGRL"
        self.device = device

    def _feature_masking(self, graph):
        feat = graph.ndata["feat"]
        if len(feat.shape) == 3:
            feat_mask1 = (
                torch.FloatTensor(feat.shape[1], feat.shape[2]).uniform_() > self.p_f1
            )
            feat_mask2 = (
                torch.FloatTensor(feat.shape[1], feat.shape[2]).uniform_() > self.p_f2
            )
        else:
            feat_mask1 = torch.FloatTensor(feat.shape[1]).uniform_() > self.p_f1
            feat_mask2 = torch.FloatTensor(feat.shape[1]).uniform_() > self.p_f2
        feat_mask1, feat_mask2 = feat_mask1.to(self.device), feat_mask2.to(self.device)
        x1, x2 = feat.clone(), feat.clone()
        x1, x2 = x1 * feat_mask1, x2 * feat_mask2

        new_graph1 = self._drop_edges(graph, p=self.p_e1)
        new_graph2 = self._drop_edges(graph, p=self.p_e2)
        new_graph1.ndata["feat"] = x1
        new_graph2.ndata["feat"] = x2

        for key in graph.ndata.keys():
            if key != "feat":
                new_graph1.ndata[key] = graph.ndata[key]
                new_graph2.ndata[key] = graph.ndata[key]

        return new_graph1, new_graph2

    def _drop_edges(self, graph, p):
        src, dst = graph.all_edges()
        weight = graph.edata["weight"]

        # Randomly select edges with a probability of p
        mask = torch.zeros_like(src).bernoulli_(p).bool()
        self_edges = src == dst
        mask[self_edges] = 1  # keep self-edges

        src = src[mask]
        dst = dst[mask]
        weight = weight[mask]
        # Return a new graph with the same nodes as the original graph
        new_graph = dgl.graph((src, dst), num_nodes=graph.number_of_nodes())

        new_graph.edata["weight"] = weight
        return new_graph

    def __call__(self, graph):

        return self._feature_masking(graph)


def get_config(model_name, args):
    if model_name == "stgcn":
        config = {
            "hidden_dim": args.hidden_dim,
            "num_gcn_layers": args.num_gcn_layers,
            "g_conv": args.g_conv,
            "num_gru_layers": args.num_rnn_layers,
            "rnn_hidden_dim": args.rnn_hidden_dim,
            "add_bias": True,
            "dropout": args.dropout,
            "activation_fn": args.activation_fn,
            # "norm": args.norm,
            "aggregator_type": args.aggregator_type,
            "num_heads": args.num_heads,
            "num_mlp_layers": args.num_mlp_layers,
            "learn_eps": args.learn_eps,
            "final_pool": args.final_pool,
            "t_model": args.t_model,
            "negative_slope": args.negative_slope,
            "gat_residual": args.gat_residual,
            "neighbor_pooling_type": args.aggregator_type,
            "memory_size": args.memory_size,
            "memory_order": args.memory_order,
        }
    elif model_name in ["gcn", "gat", "gin", "graphsage", "gaan"]:
        config = {
            "hidden_dim": args.hidden_dim,
            "num_gcn_layers": args.num_gcn_layers,
            "g_conv": args.g_conv,
            "num_gru_layers": args.num_rnn_layers,
            "rnn_hidden_dim": args.rnn_hidden_dim,
            "add_bias": True,
            "dropout": args.dropout,
            "activation_fn": args.activation_fn,
            # "norm": args.norm,
            "aggregator_type": args.aggregator_type,
            "num_heads": args.num_heads,
            "num_mlp_layers": args.num_mlp_layers,
            "learn_eps": args.learn_eps,
            "negative_slope": args.negative_slope,
            "gat_residual": args.gat_residual,
            "neighbor_pooling_type": args.aggregator_type,
        }
    elif model_name in ["lstm", "gru"]:
        config = {
            "hidden_size": args.hidden_dim,
            "num_rnn_layers": args.num_rnn_layers,
            "num_classes": args.num_classes,
            "model_name": model_name,
            "dropout": args.dropout,
            "final_pool": args.final_pool,
            "pack_padded_seq": args.pack_padded_seq,
        }
    elif model_name == "tabnet_temporal":
        config = {
            "hidden_size": args.hidden_dim,
            "num_rnn_layers": args.num_rnn_layers,
            "t_model": args.t_model,
            "ehr_checkpoint_path": args.ehr_pretrain_path,
            "n_d": args.n_d,
            "n_a": args.n_a,
            "n_steps": args.n_steps,
            "gamma": args.gamma,
            "cat_emb_dim": args.cat_emb_dim,
            "n_independent": args.n_independent,
            "n_shared": args.n_shared,
            "epsilon": 1e-15,
            "virtual_batch_size": args.virtual_batch_size,
            "momentum": args.momentum,
            "mask_type": args.mask_type,
            "dropout": args.dropout,
            "final_pool": args.final_pool,
        }
    elif model_name == "tabnet":
        config = {
            "n_d": args.n_d,
            "n_a": args.n_a,
            "n_steps": args.n_steps,
            "gamma": args.gamma,
            "cat_emb_dim": args.cat_emb_dim,
            "n_independent": args.n_independent,
            "n_shared": args.n_shared,
            "epsilon": 1e-15,
            "virtual_batch_size": args.virtual_batch_size,
            "momentum": args.momentum,
            "mask_type": args.mask_type,
        }
    elif model_name == "graph_transformer":
        config = {
            "hidden_dim": args.hidden_dim,
            "num_gcn_layers": args.num_gcn_layers,
            "g_conv": args.g_conv,
            "dropout": args.dropout,
            "activation_fn": args.activation_fn,
            "final_pool": args.final_pool,
            "trans_nhead": args.trans_nhead,
            "trans_dim_feedforward": args.trans_dim_feedforward,
            "trans_activation": args.trans_activation,
            "att_neighbor": args.att_neighbor,
            "aggregator_type": args.aggregator_type,
            "init_eps": 0,
            "learn_eps": args.learn_eps,
            "negative_slope": args.negative_slope,
            "gat_residual": args.gat_residual,
            "gaan_map_feats": args.gaan_map_feats,
            "num_mlp_layers": args.num_mlp_layers,
            "neighbor_pooling_type": args.aggregator_type,
        }
    else:
        raise NotImplementedError

    return config