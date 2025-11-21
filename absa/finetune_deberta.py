import sys
sys.path.insert(0, "/absa-ft-venv/lib/python3.10/site-packages")

import os
import argparse
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import transformers
from transformers import (
    DebertaV2Tokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import torch

# cuda ceck
print("DEBUG: transformers version:", transformers.__version__)
print("DEBUG: transformers file:", transformers.__file__)
print("DEBUG: CUDA available:", torch.cuda.is_available())

# label maps
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def normalize_label(x):
    x = str(x).strip().lower()
    if "neg" in x:
        return "negative"
    if "pos" in x:
        return "positive"
    return "neutral"

# load triplets
def load_raw_triplets(path):
    texts = []
    aspects = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(f"Raw file malformed: {len(lines)} lines (not divisible by 3)")

    for i in range(0, len(lines), 3):
        texts.append(lines[i])
        aspects.append(lines[i + 1])
        labels.append(normalize_label(lines[i + 2]))

    df = pd.DataFrame({"text": texts, "aspect": aspects, "label": labels})
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


# metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, zero_division=0, average="macro"
    )
    return {"accuracy": acc, "f1_macro": f1}


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--raw_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--per_device_train_batch_size", default=16, type=int)
    parser.add_argument("--max_length", default=256, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load dataset
    df = load_raw_triplets(args.raw_file)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label_id"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42
    )

    def to_ds(x):
        return Dataset.from_pandas(x[["text", "aspect", "label_id"]], preserve_index=False)

    dataset = DatasetDict(
        train=to_ds(train_df),
        validation=to_ds(val_df),
        test=to_ds(test_df),
    )

    # tokenzing using slow tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False
    )

    # model 
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # use CUDA
    if torch.cuda.is_available():
        print("Using device:", torch.cuda.get_device_name(0))
        model = model.to("cuda")
    else:
        print("warning: CUDA not available")

    # tokenization
    def tok(batch):
        enc = tokenizer(
            batch["text"],
            batch["aspect"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )
        enc["labels"] = batch["label_id"]
        return enc

    encoded = dataset.map(tok, batched=True, remove_columns=["text", "aspect", "label_id"])
    
    training_args = TrainingArguments(
    	output_dir=args.output_dir,
    	learning_rate=args.learning_rate,      # 1e-5 from CLI
	num_train_epochs=args.num_train_epochs,
    	per_device_train_batch_size=args.per_device_train_batch_size,
    	per_device_eval_batch_size=32,
    	evaluation_strategy="epoch",
    	save_strategy="epoch",
    	metric_for_best_model="f1_macro",
    	load_best_model_at_end=True,
    	logging_steps=50,
    	warmup_ratio=0.1,                      
    	lr_scheduler_type="cosine",
    	weight_decay=0.01,                     
    	report_to=["none"],
    	)


    # old training args
#    training_args = TrainingArguments(
#        output_dir=args.output_dir,
#        learning_rate=args.learning_rate,
#        num_train_epochs=args.num_train_epochs,
#        per_device_train_batch_size=args.per_device_train_batch_size,
#        per_device_eval_batch_size=32,
#        evaluation_strategy="epoch",
#        save_strategy="epoch",
#        metric_for_best_model="f1_macro",
#        load_best_model_at_end=True,
#        logging_steps=50,
#        warmup_ratio=0.06,
#        lr_scheduler_type="cosine",
#        report_to=["none"],
#    )


    # trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # train
    trainer.train()

    # eval on test
    test_metrics = trainer.evaluate(encoded["test"])
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("DONE. Metrics:", test_metrics)


if __name__ == "__main__":
    main()

