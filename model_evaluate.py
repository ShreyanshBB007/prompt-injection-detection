"""Run full post-training evaluation for the prompt injection classifier."""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models", "distilbert-injection")
DATA_DIR = os.path.join(BASE, "data")
MAX_LENGTH = 128
BATCH_SIZE = 32


def load_model_and_tokenizer(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def predict_batch(texts, tokenizer, model, device, max_length=MAX_LENGTH):
    enc = tokenizer(
        list(texts), truncation=True, max_length=max_length,
        padding=True, return_tensors='pt'
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)
    return preds, probs


def run_evaluation(model_dir=MODEL_DIR, data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    print("Loading model from:", model_dir)
    model, tokenizer, device = load_model_and_tokenizer(model_dir)
    print(f"Device: {device}")

    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}. Run train.py first.")

    test_df = pd.read_csv(test_path)
    print(f"Test samples: {len(test_df)}")

    all_preds, all_probs = [], []
    for i in range(0, len(test_df), batch_size):
        batch = test_df['text'].iloc[i:i + batch_size]
        preds, probs = predict_batch(batch.values, tokenizer, model, device)
        all_preds.extend(preds)
        all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    true_labels = test_df['label'].values

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        true_labels, all_preds,
        target_names=['SAFE', 'INJECTION'], digits=4
    )
    print(report)

    report_path = os.path.join(data_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Prompt Injection Classifier - Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    print(f"  Report saved -> {report_path}")

    injection_probs = all_probs[:, 1]
    auc = roc_auc_score(true_labels, injection_probs)
    print(f"\nROC-AUC Score: {auc:.4f}")
    fpr, tpr, _ = roc_curve(true_labels, injection_probs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Prompt Injection Classifier - Evaluation", fontsize=15, fontweight='bold')

    cm = confusion_matrix(true_labels, all_preds)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
        xticklabels=['SAFE', 'INJECTION'], yticklabels=['SAFE', 'INJECTION']
    )
    axes[0].set_title(f"Confusion Matrix\n(Total: {len(test_df)})", fontsize=12)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    tn, fp, fn, tp = cm.ravel()
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    axes[0].text(
        0.5, -0.18,
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n"
        f"False Negative Rate (missed attacks) = {fnr:.3f}",
        transform=axes[0].transAxes, ha='center', fontsize=10, color='darkred'
    )

    axes[1].plot(fpr, tpr, color='#2980b9', lw=2, label=f'AUC = {auc:.4f}')
    axes[1].plot([0, 1], [0, 1], '--', color='gray', lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='#2980b9')
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate (Recall)")
    axes[1].set_title("ROC Curve", fontsize=12)
    axes[1].legend(loc='lower right')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1.01])

    safe_conf = all_probs[all_preds == 0, 0]
    inject_conf = all_probs[all_preds == 1, 1]
    axes[2].hist(safe_conf, bins=30, alpha=0.7, color='#27ae60', label='Predicted SAFE')
    axes[2].hist(inject_conf, bins=30, alpha=0.7, color='#e74c3c', label='Predicted INJECTION')
    axes[2].set_title("Prediction Confidence Distribution", fontsize=12)
    axes[2].set_xlabel("Confidence Score")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    plt.tight_layout()
    eval_path = os.path.join(data_dir, "evaluation_plots.png")
    plt.savefig(eval_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Evaluation plots saved -> {eval_path}")

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    correct_mask = all_preds == true_labels
    print("\nCorrectly identified injections (sample):")
    correct_indices = np.where(correct_mask & (true_labels == 1))[0]
    for idx in correct_indices[:3]:
        conf = all_probs[idx, 1]
        print(f"  [conf={conf:.3f}] {str(test_df['text'].iloc[idx])[:100]}")

    print("\nMissed injections (False Negatives - most dangerous):")
    wrong_indices = np.where(~correct_mask & (true_labels == 1))[0]
    for idx in wrong_indices[:3]:
        print(f"  {str(test_df['text'].iloc[idx])[:100]}")

    print("\n  Evaluation complete.")
    return {
        "report_path": report_path,
        "evaluation_plot_path": eval_path,
        "roc_auc": float(auc),
        "n_test": int(len(test_df)),
    }


if __name__ == "__main__":
    run_evaluation()
