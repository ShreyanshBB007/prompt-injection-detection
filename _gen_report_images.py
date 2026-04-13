#!/usr/bin/env python
"""
Generate 4 report-ready images:
1. report_roc_curve.png - ROC curve
2. report_confusion_matrix.png - Confusion matrix
3. report_eda_graphs.png - 3-panel EDA (label dist, class pie, token length histogram)
4. report_word_cloud.png - Word cloud from injection class
"""

import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Try to use wordcloud, fallback to manual implementation
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = "data"
MODEL_DIR = "models/distilbert-injection"
OUTPUT_DIR = "data"

print(f"[*] Loading test data from {DATA_DIR}/test.csv")
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
print(f"[*] Loaded test data: {test_df.shape}")

print(f"[*] Loading train data from {DATA_DIR}/train.csv")
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
print(f"[*] Loaded train data: {train_df.shape}")

print(f"[*] Loading model from {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
model.eval()

# ============================================================================
# 1. GET PREDICTIONS AND PROBABILITIES
# ============================================================================
print("[*] Generating predictions on test set...")

texts = test_df['text'].tolist()
true_labels = test_df['label'].values

probs_list = []
preds_list = []

with torch.no_grad():
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, truncation=True, max_length=512, return_tensors='pt', padding=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        for j, prob in enumerate(probs):
            probs_list.append(prob.cpu().numpy())
            preds_list.append(np.argmax(prob.cpu().numpy()))
        
        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} samples...")

probs_array = np.array(probs_list)
preds_array = np.array(preds_list)
probs_injection = probs_array[:, 1]  # probability of class 1 (injection)

# ============================================================================
# 2. GENERATE ROC CURVE
# ============================================================================
print("[*] Generating ROC curve...")

fpr, tpr, _ = roc_curve(true_labels, probs_injection)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
ax.plot(fpr, tpr, color='#2E86AB', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='#A23B72', lw=2, linestyle='--', label='Random Classifier')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve - Prompt Injection Detection', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, 'report_roc_curve.png')
plt.savefig(roc_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[✓] Saved: {roc_path}")

# ============================================================================
# 3. GENERATE CONFUSION MATRIX
# ============================================================================
print("[*] Generating confusion matrix...")

cm = confusion_matrix(true_labels, preds_array)
fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

# Plot confusion matrix as heatmap
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax, label='Count')

classes = ['Clean (0)', 'Injection (1)']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes, fontsize=11)
ax.set_yticklabels(classes, fontsize=11)

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight='bold')

ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Prompt Injection Detection', fontsize=14, fontweight='bold')
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, 'report_confusion_matrix.png')
plt.savefig(cm_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[✓] Saved: {cm_path}")

# ============================================================================
# 4. GENERATE EDA GRAPHS (3-panel)
# ============================================================================
print("[*] Generating EDA graphs...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=200)

# Panel 1: Label distribution
label_counts = train_df['label'].value_counts().sort_index()
colors_dist = ['#2E86AB', '#A23B72']
axes[0].bar(['Clean (0)', 'Injection (1)'], label_counts.values, color=colors_dist, alpha=0.8, edgecolor='black', linewidth=1.5)
axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[0].set_title('Label Distribution (Train)', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3, linestyle='--')
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# Panel 2: Class proportion pie
sizes = label_counts.values
labels_pie = [f"Clean\n({sizes[0]} samples)", f"Injection\n({sizes[1]} samples)"]
axes[1].pie(sizes, labels=labels_pie, colors=colors_dist, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
axes[1].set_title('Class Proportion (Train)', fontsize=12, fontweight='bold')

# Panel 3: Token length histogram by class
train_df['token_length'] = train_df['text'].apply(lambda x: len(tokenizer.encode(x, truncation=False)))
clean_lengths = train_df[train_df['label'] == 0]['token_length']
injection_lengths = train_df[train_df['label'] == 1]['token_length']

axes[2].hist(clean_lengths, bins=30, alpha=0.6, label='Clean', color='#2E86AB', edgecolor='black')
axes[2].hist(injection_lengths, bins=30, alpha=0.6, label='Injection', color='#A23B72', edgecolor='black')
axes[2].set_xlabel('Token Length', fontsize=11, fontweight='bold')
axes[2].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[2].set_title('Token Length Distribution (Train)', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
eda_path = os.path.join(OUTPUT_DIR, 'report_eda_graphs.png')
plt.savefig(eda_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"[✓] Saved: {eda_path}")

# ============================================================================
# 5. GENERATE WORD CLOUD (or frequency-weighted visualization)
# ============================================================================
print("[*] Generating word cloud...")

# Get all injection class texts
injection_texts = ' '.join(train_df[train_df['label'] == 1]['text'].tolist()).lower()

# Simple stopwords
stopwords = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
    'did', 'will', 'would', 'should', 'could', 'may', 'might', 'can', 'it', 'its', 'this',
    'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'what', 'which', 'who',
    'where', 'when', 'why', 'how', 'as', 'from', 'up', 'about', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'now', 'no', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'just', 'don', 'now', 'being', 'has',
    'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'him', 'his', 'himself', 'her', 'hers', 'herself', 'them',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'why', 'how'
])

# Extract words (only keep alphanumeric sequences of 3+ chars)
words = re.findall(r'\b[a-z]{3,}\b', injection_texts)
words = [w for w in words if w not in stopwords and len(w) > 2]

# Count frequencies
from collections import Counter
word_freq = Counter(words)
top_words = word_freq.most_common(50)

if HAS_WORDCLOUD:
    print("  [+] Using wordcloud package")
    # Use wordcloud library
    wordcloud = WordCloud(width=1200, height=600, background_color='white',
                          colormap='viridis', relative_scaling=0.5, min_font_size=10).generate_from_frequencies(word_freq)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud - Prompt Injection Class', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
else:
    print("  [+] Using manual frequency-weighted visualization")
    # Manual word cloud using matplotlib text placement
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200, facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Normalize frequencies for sizing
    max_freq = max(freq for _, freq in top_words)
    min_freq = min(freq for _, freq in top_words)
    freq_range = max_freq - min_freq if max_freq > min_freq else 1
    
    # Color map
    colors_list = plt.cm.viridis(np.linspace(0, 1, len(top_words)))
    
    # Scatter plot words with size proportional to frequency
    np.random.seed(42)
    positions = []
    for idx, (word, freq) in enumerate(top_words):
        # Calculate font size (10 to 40)
        font_size = 10 + 30 * (freq - min_freq) / freq_range if freq_range > 0 else 20
        
        # Random position with collision avoidance
        attempts = 0
        while attempts < 20:
            x = np.random.uniform(0.5, 9.5)
            y = np.random.uniform(0.5, 9.5)
            
            # Simple collision check
            collision = False
            for px, py in positions:
                if np.sqrt((x - px)**2 + (y - py)**2) < 0.8:
                    collision = True
                    break
            
            if not collision:
                positions.append((x, y))
                ax.text(x, y, word, fontsize=int(font_size), color=colors_list[idx],
                        fontweight='bold', ha='center', va='center', alpha=0.8)
                break
            attempts += 1
    
    ax.set_title('Word Cloud - Prompt Injection Class', fontsize=16, fontweight='bold', pad=20)

wordcloud_path = os.path.join(OUTPUT_DIR, 'report_word_cloud.png')
plt.savefig(wordcloud_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[✓] Saved: {wordcloud_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("REPORT IMAGE GENERATION COMPLETE")
print("="*70)

generated_files = [
    roc_path,
    cm_path,
    eda_path,
    wordcloud_path
]

for fpath in generated_files:
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        print(f"[✓] {os.path.basename(fpath):40s} ({size:,} bytes)")
    else:
        print(f"[✗] {os.path.basename(fpath):40s} - NOT FOUND")

print("="*70)
