
import torch
import numpy as np
from typing import Union
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
# Extra metrics that require all the data...
from sklearn.metrics import precision_recall_curve, auc, mean_squared_error, mean_absolute_error

Tensor = Union[torch.Tensor, np.array]

def user_accuracy(cm:Tensor) -> float:
    """ TP / (TP + FP) """
    return precision(cm)

def producer_accuracy(cm:Tensor) -> float:
    """ TP / (TP + FN) """
    return recall(cm)

def TPR(cm:Tensor) -> float:
    """ TP / (TP + FN) """
    return recall(cm)

def precision(cm:Tensor) -> float:
    """ TP / (TP + FP) """
    assert cm.shape == (2, 2), f"Expected binary found {cm.shape}"

    return cm[1,1] / (cm[1,1] + cm[0,1])

def recall(cm:Tensor) -> float:
    """ TP / (TP + FN) """
    assert cm.shape == (2, 2), f"Expected binary found {cm.shape}"

    return cm[1, 1] / (cm[1, 1] + cm[1, 0])

def f1score(cm:Tensor) -> float:
    prec = precision(cm)
    rec = recall(cm)
    return 2 * (prec * rec) / (prec + rec)

def FPR(cm:Tensor) -> float:
    """ FP / (FP + TN)"""
    return cm[0, 1] / (cm[0, 1] + cm[0, 0])

def iou(cm:Tensor) -> float:
    """ TP / (TP + FN + FP) """
    assert cm.shape == (2, 2), f"Expected binary found {cm.shape}"

    return cm[1, 1] / (cm[1, 1] + cm[1, 0] + cm[0,1])

def accuracy(cm:Tensor) -> float:
    """ (TP + TN) / (TP + FN + FP + TN) """
    assert cm.shape == (2, 2), f"Expected binary found {cm.shape}"

    return (cm[1, 1] + cm[0, 0]) / cm.sum()

def cohen_kappa(cm:Tensor) -> float:
    confmat = cm.float() if not cm.is_floating_point() else cm
    sum0 = confmat.sum(dim=0, keepdim=True)
    sum1 = confmat.sum(dim=1, keepdim=True)
    expected = sum1 @ sum0 / sum0.sum()

    w_mat = torch.ones_like(confmat).flatten()
    w_mat[:: 2 + 1] = 0
    w_mat = w_mat.reshape(2, 2)

    k = torch.sum(w_mat * confmat) / torch.sum(w_mat * expected)
    return 1 - k

def balanced_accuracy(cm:Tensor) -> float:
    """ 0.5 (PA + TN /(TN + FP))"""

    PA = recall(cm)
    TNR = cm[0, 0] /(cm[0, 0] + cm[0, 1])
    return 0.5 * (PA + TNR)

def TP(cm:Tensor) -> float:
    return cm[1, 1]

def TN(cm:Tensor) -> float:
    return cm[0, 0]

def FP(cm:Tensor) -> float:
    return cm[0, 1]

def FN(cm:Tensor) -> float:
    return cm[1, 0]

METRICS_CONFUSION_MATRIX = [precision, recall, f1score]

def precision_recall(true_changes, pred_change_scores, mask=True):
    # convert to numpy arrays and mask invalid
    # After these lines the changes and scores are 1D
    if mask:
      invalid_masks = [c==2 for c in true_changes]
      true_changes = np.concatenate(
          [c[~m] for m,c in zip(invalid_masks, true_changes)],
          axis=0
      )
      pred_change_scores = np.concatenate(
          [c[~m] for m,c in zip(invalid_masks, pred_change_scores)],
          axis=0
      )
    else:
      # else just flatten
      true_changes = true_changes.flatten()
      pred_change_scores = pred_change_scores.flatten()

    precision, recall, thresholds = precision_recall_curve(
        true_changes,
        pred_change_scores
    )
    return precision, recall, thresholds

def auprc(true_changes, pred_change_scores, mask=False):
    true_changes = true_changes.detach().cpu()
    pred_change_scores = pred_change_scores.detach().cpu()

    try:
        # note: in theory expansive, call sporadically... (at the end of training for example)
        precision, recall, thresholds = precision_recall(true_changes, pred_change_scores, mask=False)
        area_under_precision_curve = auc(recall, precision)
        return area_under_precision_curve

    except:
        # if nans in the data, return nan
        return np.nan

def mse(y, pred, mask=False):
    mse = F.mse_loss(y, pred) # can be not on cpu, can still be 4D
    return mse.detach().cpu()

def mae(y, pred, mask=False):
    mae = F.l1_loss(y, pred)
    return mae.detach().cpu()

def prep_data_multishot(y_long, pred_binary):
    y = y_long.detach().cpu()
    pred = pred_binary.detach().cpu()
    assert len(y.shape) == 4 # [10, 3, 512, 512] like [batch, classes, W, H]

    # consider batch just as another dim
    y = np.transpose(y, (1, 0, 2, 3)) # B, C, W, H -> C, B, W, H
    pred = np.transpose(pred, (1, 0, 2, 3)) # B, C, W, H -> C, B, W, H
    # flatten?
    y = y.flatten(start_dim=1)
    pred = pred.flatten(start_dim=1)

    # These still keep the different multi-hot classes
    y = np.transpose(y, (1, 0)) # C, N -> N, C
    pred = np.transpose(pred, (1, 0)) # C, N -> N, C
    return y, pred

def multihot_f1score(y_long, pred_binary, together=False):
    y, pred = prep_data_multishot(y_long, pred_binary)

    if together:
        weighted_f1score = f1_score(y_true=y, y_pred=pred, average='weighted')
        """
        Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). 
        This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
        """
        return weighted_f1score
    else:
        f1score_per_class = f1_score(y_true=y, y_pred=pred, average=None)
        return f1score_per_class

def multihot_precision(y_long, pred_binary, together=False):
    y, pred = prep_data_multishot(y_long, pred_binary)
    if together:
        weighted_precision = precision_score(y_true=y, y_pred=pred, average='weighted')
        return weighted_precision
    else:
        precision_per_class = precision_score(y_true=y, y_pred=pred, average=None)
        return precision_per_class

def multihot_recall(y_long, pred_binary, together=False):
    y, pred = prep_data_multishot(y_long, pred_binary)
    if together:
        weighted_recall = recall_score(y_true=y, y_pred=pred, average='weighted')
        return weighted_recall
    else:
        recall_per_class = recall_score(y_true=y, y_pred=pred, average=None)
        return recall_per_class

