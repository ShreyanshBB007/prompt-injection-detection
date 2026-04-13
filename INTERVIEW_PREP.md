# Prompt Injection and Jailbreak Detection - Interview Prep

## 1) Executive Summary
This project is a binary NLP security classifier that detects whether a user prompt is:
- SAFE (label 0)
- INJECTION (label 1)

It fine-tunes DistilBERT on a labeled prompt dataset and exposes the model through:
- training and evaluation scripts
- attention-based interpretability charts
- a reusable Python inference API
- a Streamlit dashboard for live testing and artifact review

The system is intended to be used as a pre-filter before sending user prompts to an LLM.

## 2) What Problem It Solves
### Threat model
The project targets prompt-based adversarial inputs such as:
- instruction override attempts ("ignore previous instructions")
- jailbreak role-play prompts ("act as DAN")
- hidden exfiltration requests ("reveal your system prompt/training data")

### Security objective
Minimize false negatives on INJECTION class (missed attacks), while keeping false positives manageable for normal traffic.

## 3) How It Is Made (System Design)
### Core stack
- Model framework: Hugging Face Transformers + PyTorch
- Data/metrics: pandas, scikit-learn, evaluate
- Visualization: matplotlib, seaborn
- UI: Streamlit

### Data flow
1. Load and normalize dataset from Hugging Face: `jayavibhav/prompt-injection`.
2. Build a unified dataframe with `text` and `label`.
3. Perform EDA and save plots.
4. Stratified split into train/val/test.
5. Tokenize text for DistilBERT.
6. Train sequence classifier with weighted cross-entropy.
7. Evaluate on held-out test set and save report/plots.
8. Serve predictions through API or Streamlit app.
9. (Optional) Generate adversarial paraphrases with Anthropic API for data augmentation.

## 4) Which Algorithm Is Used
### Learning algorithm
- Supervised binary text classification.
- Backbone: DistilBERT transformer encoder.
- Head: sequence classification layer (`AutoModelForSequenceClassification`).
- Objective: class-weighted cross-entropy.

### Why weighted loss?
Class-weighted loss increases penalty for mistakes on underrepresented or security-critical class. In security classification, this helps prioritize catching injection attempts.

### Prediction algorithm
- Convert logits to probabilities via softmax.
- Predicted class = argmax(probabilities).
- Risk tier derived from injection probability:
  - LOW: < 0.30
  - MEDIUM: 0.30 to < 0.70
  - HIGH: >= 0.70

## 5) Which Model Is Used and Why Not Others
### Chosen model
- `distilbert-base-uncased`
- DistilBERT config from model artifact:
  - layers: 6
  - attention heads: 12
  - hidden size: 768
  - FFN hidden dim: 3072
  - vocab size: 30522
  - max positions: 512

### Why DistilBERT for this project
- Good latency/accuracy trade-off for real-time filtering.
- Much lighter than BERT-base/large while retaining strong language understanding.
- Strong ecosystem support in Hugging Face.
- Easy deployment for API and dashboard use.

### Why not simpler models (TF-IDF + Logistic Regression)
- Prompt injection is semantic and compositional, not only keyword-based.
- Attackers can evade lexical models using paraphrase/obfuscation.
- Transformers model context and intent better.

### Why not larger models (RoBERTa-large, DeBERTa-large, LLM classifiers)
- Higher inference latency and memory costs.
- More expensive for always-on pre-filtering.
- DistilBERT is typically sufficient for binary safety gate use cases.

## 6) Exact Parameters and Training Settings
From `train.py` and artifacts:

### Data / tokenization
- `MAX_LENGTH = 128`
- tokenizer/model checkpoint = `distilbert-base-uncased`

### Optimization and trainer
- `BATCH_SIZE = 16`
- `LEARNING_RATE = 2e-5`
- `NUM_EPOCHS = 2`
- `weight_decay = 0.01`
- `warmup_ratio = 0.1`
- mixed precision: `fp16 = True`
- early stopping patience: `2`
- best model metric: `f1` (macro F1 from `evaluate`)
- random seed: `42`
- training device policy: hard fail if CUDA unavailable

### Label mapping
- `0 -> SAFE`
- `1 -> INJECTION`

### Evaluation settings
- `model_evaluate.py` batch size: `32`
- computes classification report, confusion matrix, ROC-AUC, confidence histograms

## 7) Actual Observed Data and Metrics in This Workspace
### Split sizes and label counts
- train.csv: label 0 = 132502, label 1 = 129221
- val.csv: label 0 = 16563, label 1 = 16152
- test.csv: label 0 = 16563, label 1 = 16153

### Reported evaluation metrics (`data/classification_report.txt`)
- SAFE: precision 0.9944, recall 0.9943, f1 0.9943
- INJECTION: precision 0.9941, recall 0.9942, f1 0.9942
- Accuracy: 0.9943

Interpretation for interview:
- Very high performance on current test distribution.
- Must still discuss distribution shift risk and adversarial adaptation in production.

## 8) End-to-End Workflow (How It Actually Works)
### Offline workflow
1. Run `train.py`.
2. It pulls dataset, runs EDA, splits data, tokenizes, trains DistilBERT, evaluates test split, saves model and artifacts.
3. Run `model_evaluate.py` for extended evaluation plots/report.
4. Run `attention_viz.py` to generate attention explainability visuals.

### Online workflow (runtime)
1. App/API receives user prompt.
2. `inference.py` lazily loads tokenizer/model once.
3. Prompt is tokenized and scored.
4. Output includes class label, confidence, injection probability, and risk level.
5. Upstream gateway can block/highlight requests based on risk threshold.

## 9) In-Depth Use of Each File
### Root files
- `train.py`
  - Full pipeline script.
  - Loads Hugging Face dataset.
  - EDA plotting and sample inspection.
  - Stratified split to train/val/test CSVs.
  - Tokenization with max length 128.
  - Computes class weights and uses custom `WeightedTrainer`.
  - Trains DistilBERT and saves model/tokenizer.
  - Produces confusion matrix and mislabelled examples.

- `model_evaluate.py`
  - Standalone post-training evaluation.
  - Batched prediction on `data/test.csv`.
  - Saves `classification_report.txt` and `evaluation_plots.png`.
  - Calculates ROC-AUC and prints sample TP/FN behavior.

- `inference.py`
  - Production-style API:
    - `detect_injection(text)` for single input.
    - `detect_batch(texts)` for bulk scoring.
  - Singleton model loading for speed.
  - Converts probability to LOW/MEDIUM/HIGH risk tiers.

- `attention_viz.py`
  - Loads model with `output_attentions=True`.
  - Uses final-layer, mean-head attention.
  - Uses CLS-row attention to estimate token contribution.
  - Generates per-example heatmaps and aggregated top tokens.

- `app.py`
  - Streamlit multipage dashboard.
  - Live detector page (single + batch samples).
  - Data artifact pages (EDA, evaluation, attention, report, dataset viewer).
  - Supports local model load or remote Hugging Face model via env vars.

- `adversarial.py`
  - Optional data augmentation using Anthropic API.
  - Creates paraphrased malicious prompts with preserved attack intent.
  - Writes `adversarial_injections.csv` and `train_augmented.csv`.

- `README.md`
  - Project documentation and conceptual architecture.

- `working-explanations.md`
  - Detailed implementation narrative and audit notes.

- `requirements.txt`, `runtime.txt`
  - Dependency and runtime configuration.

### Notebooks
- `01_eda.ipynb`
  - Dataset loading, class distribution, token length analysis, top words, class weight demo.

- `02_training.ipynb`
  - Training walkthrough that calls `train.py` from notebook.
  - Confirms generated artifacts.

- `03_evaluation.ipynb`
  - Runs `run_evaluation()`.
  - Demonstrates `detect_injection()` API.
  - Generates attention comparison and custom heatmap.

### Data directory (`data/`)
- `train.csv`, `val.csv`, `test.csv`: training/validation/test splits.
- `mislabelled.csv`: false positives/negatives from training pipeline test pass.
- `classification_report.txt`: persisted classification report.
- `eda_overview.png`, `confusion_matrix.png`, `evaluation_plots.png`, attention images: visual artifacts.

### Model directory (`models/distilbert-injection/`)
- `model.safetensors`: trained weights.
- `config.json`: architecture and label maps.
- `tokenizer.json`, `vocab.txt`, tokenizer configs: text preprocessing artifacts.
- `checkpoint-*`: intermediate checkpoints and trainer states.

## 10) Important Interview Talking Points (High Value)
### A) Why this is not just a keyword detector
- It uses contextual embeddings and self-attention to capture intent and instruction hierarchy.
- Better resistance to paraphrased attacks than bag-of-words baselines.

### B) Why recall on INJECTION matters most
- False negatives are security failures.
- Design choices (weighted loss + thresholded risk levels) reflect that priority.

### C) How interpretability is addressed
- Attention maps expose influential tokens and support debugging and trust.
- Use explainability to inspect false positives/false negatives.

### D) How it fits into production
- Works as a pre-LLM safety gate.
- Can route MEDIUM-risk requests to second-stage checks/human review.
- Singleton loading reduces repeated startup overhead.

## 11) Missing Topics Added for Interview Readiness
### A) Threshold calibration strategy
- Risk cutoffs should be tuned on validation data for desired security/UX trade-off.
- Use PR curve or cost-sensitive objective to choose thresholds.

### B) Robustness strategy
- Periodic retraining with newly discovered jailbreak patterns.
- Add adversarial augmentation and possibly hard-negative mining.
- Evaluate with out-of-distribution prompts and multilingual attacks.

### C) Security hardening beyond classifier
- Defense-in-depth:
  - input classifier
  - rule-based guardrails
  - output moderation
  - audit logging + abuse analytics
- Never rely on a single model for critical security controls.

### D) MLOps and governance
- Track model/data versioning.
- Maintain confusion matrix drift monitoring in production.
- Add canary deployment before full rollout.

### E) Ethical and legal concerns
- False positives can suppress benign users.
- Need appeal/review flow and transparent moderation policy.

## 12) Known Inconsistencies You Should Acknowledge If Asked
- README includes some conceptual statements not fully aligned with current root-level execution flow.
- `app.py` risk thresholds use strict `>` comparisons while `inference.py` uses `<` boundaries; edge case at exactly 0.70 differs.
- `train.py` enforces CUDA-only training, while docs may imply CPU fallback in some places.
- Artifact freshness can differ by run date; always reference latest regenerated reports.

## 13) Interview Q and A Cheat Sheet
### Q1. "Explain your architecture in 60 seconds."
A: "I built a binary transformer classifier using DistilBERT fine-tuned on prompt injection data. The training pipeline performs EDA, stratified splitting, tokenization, and weighted-loss fine-tuning, then saves reusable model artifacts. At runtime, a lightweight inference API scores prompts and returns both class and risk tier so a gateway can block high-risk prompts or route medium-risk prompts to secondary checks. I also added attention visualizations for interpretability and a Streamlit dashboard for operational analysis."

### Q2. "Why DistilBERT and not a larger model?"
A: "This is an always-on security pre-filter, so latency and cost are core constraints. DistilBERT gives strong contextual understanding with significantly lower inference cost than larger encoders, which is practical for real-time protection."

### Q3. "How do you handle class imbalance and security cost asymmetry?"
A: "I use class-weighted cross-entropy and evaluate class-wise metrics, especially recall on the injection class. Missing an attack is costlier than a false alarm, so optimization and thresholding are tuned around that risk profile."

### Q4. "How do you explain model decisions?"
A: "I enable transformer attentions, aggregate last-layer attention across heads, and inspect CLS-token attention toward input tokens. This helps verify whether the model attends to adversarial trigger semantics and helps debug misclassifications."

### Q5. "What are the biggest limitations?"
A: "Distribution shift, evolving jailbreak tactics, language/domain mismatch, and adversarial evasion against known boundaries. That is why I recommend continuous retraining and defense-in-depth rather than one-model reliance."

### Q6. "How would you improve this project next?"
A: "I would add threshold calibration tooling, OOD test suites, multilingual support, hard-negative mining, drift monitoring, and a two-stage ensemble gate for higher assurance in production."

## 14) Suggested Demo Script (Interview)
1. Show `app.py` live detector with one safe and one injection prompt.
2. Show risk-level output and explain decision thresholding.
3. Show confusion matrix + classification report.
4. Open attention heatmap and explain why highlighted tokens make sense.
5. Explain how this slots in front of any LLM endpoint.

## 15) One-Line Pitch
"This project is a production-oriented transformer safety gate that detects prompt injections in real time, explains model focus via attention visualization, and provides both research and deployment pathways through scripts, notebooks, and a Streamlit dashboard."
