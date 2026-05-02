import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F

def calc_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    is_labeled = y_true[:] == y_true[:]
    correct = (y_true[is_labeled] == y_pred[is_labeled])

    acc_list.append(float(np.sum(correct)) / len(correct))
    return sum(acc_list) / len(acc_list)

def calc_binary_metrics(y_true, y_pred, logits=None):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    is_labeled = y_true == y_true
    y_true = y_true[is_labeled]
    y_pred = y_pred[is_labeled]

    logits = logits.detach().cpu()[is_labeled]
    probs = F.softmax(logits, dim=1).numpy()
    auc = roc_auc_score(y_true, probs[:, 1])

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    total = len(y_true)
    acc = (TP + TN) / total if total > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": acc,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc
    }


def calc_hits_at_k(pos_scores, neg_scores, k: int):
    if pos_scores.numel() == 0:
        return 0.0
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    _, indices = torch.sort(scores, descending=True)
    topk = labels[indices[:k]]
    hits = topk.sum().item() / pos_scores.numel()
    return hits


def calc_link_metrics(pos_scores, neg_scores, k: int):
    pos_scores = pos_scores.detach().cpu()
    neg_scores = neg_scores.detach().cpu()
    hits = calc_hits_at_k(pos_scores, neg_scores, k)
    all_scores = torch.cat([pos_scores, neg_scores]).numpy()
    all_labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]).numpy()
    auc = roc_auc_score(all_labels, all_scores) if len(all_labels) > 0 else 0.0
    return {"hits": hits, "auc": auc}
