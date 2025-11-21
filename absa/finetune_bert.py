import os, sys, json
sys.path.insert(0, "/absa-ft-venv/lib/python3.10/site-packages")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# label mapping
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}



# label normalization
def normalize_label(x):
    x = str(x).strip().lower()
    if "neg" in x:
        return "negative"
    if "pos" in x:
        return "positive"
    return "neutral"

# load raw triplets
def load_raw_triplets(path):
    texts, aspects, labels = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(f"{path} is not valid: must contain 3 lines per entry.")

    for i in range(0, len(lines), 3):
        text = lines[i]
        aspect = lines[i + 1]
        label = normalize_label(lines[i + 2])

        texts.append(text)
        aspects.append(aspect)
        labels.append(label)

    df = pd.DataFrame({"text": texts, "aspect": aspects, "label": labels})
    df["label_id"] = df["label"].map(LABEL2ID)
    return df

# metrics
def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)[2]

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
    }

# main
def main():

    RAW_FILE = "data/Ground_truth_APC.raw"
    OUTPUT_DIR = "runs/bert_base"

    # load data
    df = load_raw_triplets(RAW_FILE)

    # split data
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42)

    def to_ds(d):
        return Dataset.from_pandas(d[["text", "aspect", "label_id"]], preserve_index=False)

    dataset = DatasetDict(
        train=to_ds(train_df),
        validation=to_ds(val_df),
        test=to_ds(test_df),
    )

    # load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def tok(batch):
        enc = tokenizer(
            batch["text"],
            batch["aspect"],
            truncation=True,
            padding="max_length",
            max_length=256
        )
        enc["labels"] = batch["label_id"]
        return enc

    encoded = dataset.map(tok, batched=True, remove_columns=["text", "aspect", "label_id"])

    # load BERT model
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # training args
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=7,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train
    trainer.train()

    # eval
    test_metrics = trainer.evaluate(encoded["test"])
    print("Test Metrics:", test_metrics)

    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("training complete and saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

