#!/usr/bin/env python3
import numpy as np
from math import sqrt

def _safe_div(a, b):
    return a / b if b != 0 else 0.0

def _fbeta(precision, recall, beta=1.0):
    beta2 = beta * beta
    denom = (beta2 * precision + recall)
    return (1 + beta2) * _safe_div(precision * recall, denom) if denom != 0 else 0.0

def classification_report_from_confmat(cm):
    """
    cm: 2x2 numpy array [[TN, FP],
                         [FN, TP]]
    Returns dict of metrics.
    """
    if not isinstance(cm, np.ndarray):
        cm = np.array(cm, dtype=float)
    if cm.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2: [[TN, FP],[FN, TP]]")

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    N = TN + FP + FN + TP

    precision = _safe_div(TP, TP + FP)
    recall    = _safe_div(TP, TP + FN)
    specificity = _safe_div(TN, TN + FP)
    f1 = _fbeta(precision, recall, beta=1.0)
    f2 = _fbeta(precision, recall, beta=2.0)
    accuracy = _safe_div(TP + TN, N)
    balanced_accuracy = (recall + specificity) / 2
    npv = _safe_div(TN, TN + FN)
    fpr = _safe_div(FP, FP + TN)
    fnr = _safe_div(FN, FN + TP)
    prevalence = _safe_div(TP + FN, N)

    denom = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = _safe_div(TP * TN - FP * FN, denom)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "Balanced Acc": balanced_accuracy,
        "F1": f1,
        "F2": f2,
        "NPV": npv,
        "FPR": fpr,
        "FNR": fnr,
        "Prevalence": prevalence,
        "MCC": mcc,
    }

def print_classification_reports(conf_mats, model_names, aucs):
    """
    Print a clean text classification report per model.
    """
    for name, cm, auc in zip(model_names, conf_mats, aucs):
        m = classification_report_from_confmat(cm)
        print("=" * 60)
        print(f"Model: {name}")
        print("-" * 60)
        print(f"Confusion Matrix (rows=true, cols=pred):\n{np.array(cm)}")
        print()
        print(f"AUC:                {auc:.4f}")
        print(f"Accuracy:           {m['Accuracy']:.4f}")
        print(f"Precision:          {m['Precision']:.4f}")
        print(f"Recall (Sensitivity): {m['Recall']:.4f}")
        print(f"Specificity:        {m['Specificity']:.4f}")
        print(f"Balanced Accuracy:  {m['Balanced Acc']:.4f}")
        print(f"F1 Score:           {m['F1']:.4f}")
        print(f"F2 Score (FN-averse): {m['F2']:.4f}")
        print(f"NPV:                {m['NPV']:.4f}")
        print(f"FPR:                {m['FPR']:.4f}")
        print(f"FNR:                {m['FNR']:.4f}")
        print(f"MCC:                {m['MCC']:.4f}")
        print("=" * 60 + "\n")

# -------- Example usage --------
if __name__ == "__main__":
    cms = [
        [[109, 141],[69, 317]], [[3, 10],[3, 17]], [[124, 126],[62, 324]],
        [[16, 8],[4, 5]], [[96, 154],[58, 328]], [[4, 9],[3, 17]],
        [[117, 133],[91, 295]], [[6, 7],[3, 17]], [[143, 107],[112, 274]],
        [[6, 7],[2, 18]], [[99, 151],[57, 329]], [[5, 8],[2, 18]],
    ]
    names = [
        "LogReg_all_val", "LogReg_all_test", "LogReg_topk_val", "LogReg_topk_test",
        "LogReg_L1_val", "LogReg_L1_test", "LightGBM_all_val", "LightGBM_all_test",
        "LightGBM_topk_val", "LightGBM_topk_test", "LightGBM_L1_val", "LightGBM_L1_test",
    ]

    aucs = [0.6781, 0.5538, 0.7581, 0.6500, 0.7075, 0.6615, 0.6563, 0.6462, 0.6627, 0.6692, 0.6705, 0.6269]

    print_classification_reports(cms, names, aucs)

