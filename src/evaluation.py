import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    average_precision_score, confusion_matrix, classification_report
)

def evaluate_and_plot(all_labels, all_preds, out_dir="results"):
    """
    Computes key performance metrics and generates ROC and Precision-Recall 
    curves critical for showcasing generalizability on clustered unbalanced data.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    binary_preds = (all_preds > 0.5).astype(int)

    # Classification Report
    print("Classification Report:")
    report = classification_report(all_labels, binary_preds, target_names=["Non-Binding", "Binding"])
    print(report)
    
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(report)
        
    # Plot Configuration
    sns.set_theme(style="whitegrid")
    
    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300)
    plt.close()
    
    # --- Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPRC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=300)
    plt.close()

    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, binary_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Binding', 'Binding'], 
                yticklabels=['Non-Binding', 'Binding'])
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"Evaluation plots saved to '{out_dir}/'")
    return roc_auc, pr_auc
