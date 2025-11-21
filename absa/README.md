# ABSA Fine-Tuning Scripts

This directory contains all Aspect-Based Sentiment Analysis (ABSA) fine-tuning
scripts used in our submission. Each script takes the same input
format (raw `.raw` triplets: sentence, aspect, label) but implements a
different model architecture.

## Input Format
All scripts expect a `.raw` file with triplet entries in the format (see /data/`Ground_truth_APC.raw`):
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

### 3. `finetune_deberta.py` (DeBERTa-v3-base / large / DeBERT-v3-large-absa-v1.1)
Handles:
- DeBERTa-v3 baseline and tuned large variants
- slow tokenizer support
- cosine scheduler + warmup
- up to 5 epochs

Source: `finetune_deberta.py`  

### finetune `deberta-v3-base`
```bash
cd ~/absa-ft/absa-ft-venv/bin/python \
source absa-ft-venv/bin/activate
  scripts/finetune_deberta.py \
  --model_name_or_path microsoft/deberta-v3-base \
  --raw_file data/Ground_truth_APC.raw \
  --output_dir runs/deberta_v3_base \
  --num_train_epochs 6 \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 16 \
  --max_length 256
```

### finetune `deberta-v3-lage`
```bash
cd ~/absa-ft/absa-ft-venv/bin/python \
source absa-ft-venv/bin/activate
  scripts/finetune_deberta.py \
  --model_name_or_path microsoft/deberta-v3-lage \
  --raw_file data/Ground_truth_APC.raw \
  --output_dir runs/deberta_v3_large \
  --num_train_epochs 4 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 8 \
  --max_length 256
```

### finetune `deberta-v3-large-absa-v1.1`
```bash
cd ~/absa-ft/absa-ft-venv/bin/python \
source absa-ft-venv/bin/activate
  scripts/finetune_deberta.py \
  --model_name_or_path yangheng/deberta-v3-large-absa-v1.1 \
  --raw_file data/Ground_truth_APC.raw \
  --output_dir runs/yang_deberta_v3_large_absa_v1.1_lr5e-6_ep5 \
  --num_train_epochs 5 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 8 \
  --max_length 256
```
---

### 4. `finetune_qwen_qlora.py` (Qwen2.5-7B-Instruct + QLoRA)
Handles:
- QLoRA 4-bit quantization
- instruction-style prompts
```bash
"You are a domain expert in political communication performing aspect-based sentiment analysis.\n"
        f"Sentence: {text}\n"
        f"Aspect: {aspect}\n"
        "Task: What is the sentiment towards this aspect? "
        "Answer with exactly one word: negative, neutral, or positive.\n"
        "Answer:"
```
- generation-based evaluation
- PEFT LoraConfig integration

Source: `finetune_qwen_qlora.py`  
```bash
cd ~/absa/qlora
source absa-qwen-venv/bin/activate
python /absa/qlora/finetune_qwen_qlora.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --raw_file /data/Ground_truth_APC.raw \
  --output_dir runs/qwen2_5_7b_qlora \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 4 \
  --max_length 256 \
  --load_in_4bit
```
 - config.json
 - pytorch_model.bin
 - tokenizer/
 - val_metrics.json
 - test_metrics.json

---

### 5. `predict_absa.py` (deberta-v3-base)

Source: `predict_absa.py`  
```bash
cd ~/absa-ft/absa-ft-venv/bin/python \
python /absa-ft/scripts/predict_absa.py \
  --model /absa-ft/runs/deberta_v3_base \
  --data /absa-ft/data/ \
  --out /absa-ft/predictions
```
