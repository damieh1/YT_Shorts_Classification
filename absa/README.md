# ABSA Fine-Tuning Scripts

This directory contains all Aspect-Based Sentiment Analysis (ABSA) fine-tuning
scripts used in our submission. Each script takes the same input
format (raw `.raw` triplets: sentence, aspect, label) but implements a
different model architecture.

## Input Format
All scripts expect a `.raw` file with entries in the format:
- sentence_1
- aspect_term_1
- label_1

- sentence_2
- aspect_term_2
- label_2

---

Labels may be variations of "positive", "neutral", "negative". They are
normalized internally.

## Scripts

### 1. `finetune_roberta.py`  (RoBERTa-base)
Handles:
- tokenization via HF AutoTokenizer
- sequence classification using `AutoModelForSequenceClassification`
- macro-F1 evaluation
- 6 epochs, LR=3e-5

Source: `finetune_roberta.py`  

---

### 2. `finetune_bert.py` (BERT-base-uncased)
Handles:
- BERT baseline
- same pipeline as RoBERTa except tokenizer/model
- 7 epochs, LR=2e-5

Source: `finetune_bert.py`  

---

### 3. `finetune_deberta.py` (DeBERTa-v3-base / -large, Yang variant)
Handles:
- DeBERTa-v3 baseline and tuned large variants
- slow tokenizer support
- cosine scheduler + warmup
- up to 5 epochs

Source: `finetune_deberta.py`  

---

### 4. `finetune_qwen_qlora.py` (Qwen2.5-7B-Instruct + QLoRA)
Handles:
- QLoRA 4-bit quantization
- instruction-style prompts (“You are a domain expert…”)
- generation-based evaluation
- PEFT LoraConfig integration

Source: `finetune_qwen_qlora.py`  

---

## Output Structure

Each run produces:
 - runs/
 - <model_name>/
 - config.json
 - pytorch_model.bin
 - tokenizer/
 - val_metrics.json
 - test_metrics.json
