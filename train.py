"""
train.py — Fine-tunes DistilBERT for Prompt Injection Detection
"""

import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import evaluate

# ─── CONFIGURATION ───────────────────────────────────────────
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH       = 128
BATCH_SIZE       = 16
LEARNING_RATE    = 2e-5
NUM_EPOCHS       = 2     # reduced for speed
SEED             = 42


BASE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "models", "distilbert-injection")
DATA_DIR   = os.path.join(BASE, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

ID2LABEL = {0: "SAFE", 1: "INJECTION"}
LABEL2ID = {"SAFE": 0, "INJECTION": 1}

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required for training, but no CUDA device was detected."
    )

gpu_name = torch.cuda.get_device_name(0)
print(f"Using GPU: {gpu_name}")

# ─── STEP 1: LOAD DATASET ────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Loading Dataset from HuggingFace")
print("="*60)

raw = load_dataset("jayavibhav/prompt-injection")
print(f"Splits available: {list(raw.keys())}")
print(f"Features: {raw['train'].features}")

# Detect column names robustly
def get_col(split, candidates):
    for c in candidates:
        if c in raw[split].features:
            return c
    raise ValueError(f"None of {candidates} found in split '{split}'")

# Aggregate all splits into one DataFrame
all_texts, all_labels = [], []
for split in raw.keys():
    tc = get_col(split, ['text', 'prompt', 'query'])
    lc = get_col(split, ['label', 'target'])
    all_texts  += list(raw[split][tc])
    all_labels += list(raw[split][lc])

df = pd.DataFrame({"text": all_texts, "label": all_labels})
df.dropna(inplace=True)
df['label'] = df['label'].astype(int)
print(f"\nFull dataset size: {len(df)}")
print(df['label'].value_counts().rename(index={0: "SAFE(0)", 1: "INJECTION(1)"}))

# ─── EDA VISUALISATIONS ──────────────────────────────────────
print("\nGenerating EDA visualisations...")

tokenizer_eda = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
tqdm.pandas(desc="EDA token length")
df['token_len'] = df['text'].progress_apply(
    lambda t: len(tokenizer_eda.encode(str(t), truncation=False))
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Prompt Injection Dataset — EDA Overview", fontsize=15, fontweight='bold')

counts = df['label'].value_counts().sort_index()
axes[0].bar(['SAFE (0)', 'INJECTION (1)'], counts.values,
            color=['#27ae60', '#e74c3c'], edgecolor='black', linewidth=0.8)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + max(counts) * 0.01, str(v),
                 ha='center', fontweight='bold', fontsize=13)
axes[0].set_title('Label Distribution', fontsize=13)
axes[0].set_ylabel('Count')

axes[1].pie(counts.values, labels=['SAFE', 'INJECTION'],
            colors=['#27ae60', '#e74c3c'], autopct='%1.1f%%',
            startangle=140, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Balance', fontsize=13)

for label, color, name in [(0, '#27ae60', 'SAFE'), (1, '#e74c3c', 'INJECTION')]:
    s = df[df['label'] == label]['token_len']
    axes[2].hist(s, bins=40, alpha=0.7, color=color,
                 label=f'{name} (mean={s.mean():.0f})')
axes[2].axvline(MAX_LENGTH, color='navy', linestyle='--',
                linewidth=1.5, label=f'max_len={MAX_LENGTH}')
axes[2].set_title('Token Length Distribution', fontsize=13)
axes[2].set_xlabel('Token Count')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
eda_path = os.path.join(DATA_DIR, "eda_overview.png")
plt.savefig(eda_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {eda_path}")

print("\nSample SAFE queries:")
for t in df[df['label'] == 0]['text'].head(3):
    print(f"  > {str(t)[:110]}")
print("Sample INJECTION queries:")
for t in df[df['label'] == 1]['text'].head(3):
    print(f"  > {str(t)[:110]}")

# ─── STEP 2: SPLIT & TOKENISE ────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Split & Tokenise (80/10/10, stratified)")
print("="*60)

train_df, temp_df = train_test_split(
    df[['text', 'label']], test_size=0.2,
    stratify=df['label'], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5,
    stratify=temp_df['label'], random_state=SEED
)

for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
    d.to_csv(os.path.join(DATA_DIR, f"{name}.csv"), index=False)
    print(f"  {name:5s}: {len(d)} samples | {d['label'].value_counts().to_dict()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def make_hf(df_):
    ds = Dataset.from_pandas(df_.reset_index(drop=True))
    ds = ds.map(
        lambda b: tokenizer(b['text'], truncation=True, max_length=MAX_LENGTH),
        batched=True,
        remove_columns=['text']
    )
    ds = ds.rename_column('label', 'labels')
    ds.set_format('torch')
    return ds

hf_train = make_hf(train_df)
hf_val   = make_hf(val_df)
hf_test  = make_hf(test_df)

# ─── CLASS WEIGHTS ───────────────────────────────────────────
cw = compute_class_weight(
    'balanced', classes=np.array([0, 1]), y=train_df['label'].values
)
weight_tensor = torch.tensor(cw, dtype=torch.float)
print(f"\nClass weights → SAFE={cw[0]:.3f}  INJECTION={cw[1]:.3f}")

# ─── STEP 3: MODEL ───────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3 — Loading DistilBERT")
print("="*60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params / 1e6:.1f}M")

# ─── CUSTOM TRAINER WITH WEIGHTED LOSS ───────────────────────
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.CrossEntropyLoss(
            weight=weight_tensor.to(outputs.logits.device)
        )(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ─── METRICS ─────────────────────────────────────────────────
acc_metric = evaluate.load("accuracy")
f1_metric  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = acc_metric.compute(predictions=preds, references=labels)
    f1  = f1_metric.compute(predictions=preds, references=labels, average='macro')
    return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

# ─── STEP 4: TRAIN ───────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4 — Training")
print("="*60)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_strategy="steps",
    logging_steps=10,
    logging_first_step=True,
    disable_tqdm=False,
    fp16=True,
    no_cuda=False,
    report_to="none",
    seed=SEED,
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=hf_train,
    eval_dataset=hf_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n  Best model saved → {OUTPUT_DIR}")

# ─── STEP 5: EVALUATE ────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5 — Test Set Evaluation")
print("="*60)

preds_out   = trainer.predict(hf_test)
pred_labels = np.argmax(preds_out.predictions, axis=-1)
true_labels = preds_out.label_ids

print("\nClassification Report:")
print(classification_report(
    true_labels, pred_labels,
    target_names=['SAFE', 'INJECTION'], digits=4
))

cm = confusion_matrix(true_labels, pred_labels)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    cm, display_labels=['SAFE', 'INJECTION']
).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(DATA_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {cm_path}")

# Mislabelled examples
wrong_mask = pred_labels != true_labels
test_texts = test_df['text'].values
wrong_df = pd.DataFrame({
    'text':       test_texts[wrong_mask],
    'true_label': [ID2LABEL[int(l)] for l in true_labels[wrong_mask]],
    'pred_label': [ID2LABEL[int(l)] for l in pred_labels[wrong_mask]],
})
wrong_df.to_csv(os.path.join(DATA_DIR, "mislabelled.csv"), index=False)
print(f"  Mislabelled examples: {len(wrong_df)} saved")

if len(wrong_df) > 0:
    print("\n  Sample errors:")
    for _, r in wrong_df.head(5).iterrows():
        print(f"    TRUE={r['true_label']} PRED={r['pred_label']} | {r['text'][:90]}")

print("\n" + "="*60)
print("✅  Training pipeline complete!")
print(f"    Model saved : {OUTPUT_DIR}")
print(f"    Charts/data : {DATA_DIR}")
print("="*60)