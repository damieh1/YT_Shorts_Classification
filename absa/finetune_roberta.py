import os, json, sys
sys.path.insert(0, "/home/treeves/projects/absa-ft/absa-ft-venv/lib/python3.10/site-packages")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

def normalize_label(x):
    x = str(x).strip().lower()
    if "neg" in x: return "negative"
    if "pos" in x: return "positive"
    return "neutral"

def load_raw_triplets(path):
    texts, aspects, labels = [], [], []
    lines = [l.strip() for l in open(path) if l.strip()]
    for i in range(0, len(lines), 3):
        texts.append(lines[i])
        aspects.append(lines[i+1])
        labels.append(normalize_label(lines[i+2]))
    df = pd.DataFrame({"text": texts, "aspect": aspects, "label": labels})
    df["label_id"] = df["label"].map(LABEL2ID)
    return df

def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    f1_macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)[2]
    return {"f1_macro": f1_macro}

def main():
    raw = "data/Ground_truth_APC.raw"
    out = "runs/roberta_base"

    df = load_raw_triplets(raw)
    train_df, tmp = train_test_split(df, test_size=0.2, stratify=df["label_id"])
    val_df, test_df = train_test_split(tmp, test_size=0.5, stratify=tmp["label_id"])

    def to_ds(d): return Dataset.from_pandas(d[["text","aspect","label_id"]], preserve_index=False)
    ds = DatasetDict(train=to_ds(train_df), validation=to_ds(val_df), test=to_ds(test_df))

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

    def tok(b):
        e = tokenizer(b["text"], b["aspect"], truncation=True, padding="max_length", max_length=256)
        e["labels"] = b["label_id"]
        return e

    enc = ds.map(tok, batched=True, remove_columns=["text","aspect","label_id"])

    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    args = TrainingArguments(
        output_dir=out,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=6,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(model=model, args=args,
                      train_dataset=enc["train"], eval_dataset=enc["validation"],
                      compute_metrics=compute_metrics, tokenizer=tokenizer)
    trainer.train()
    m = trainer.evaluate(enc["test"])
    print(m)
    with open(os.path.join(out, "test_metrics.json"), "w") as f: json.dump(m, f)

if __name__ == "__main__":
    main()

