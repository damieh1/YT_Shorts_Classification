import os
import sys
import argparse
import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def normalize_label(x: str) -> str:
    x = str(x).strip().lower()
    if "neg" in x:
        return "negative"
    if "pos" in x:
        return "positive"
    return "neutral"


def load_raw_triplets(path: str) -> pd.DataFrame:
    texts, aspects, labels = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(lines) % 3 != 0:
        raise ValueError(f"Raw file malformed: {len(lines)} lines (not % 3 == 0)")

    for i in range(0, len(lines), 3):
        texts.append(lines[i])
        aspects.append(lines[i + 1])
        labels.append(normalize_label(lines[i + 2]))

    df = pd.DataFrame({"text": texts, "aspect": aspects, "label": labels})
    df["label_id"] = df["label"].map(LABEL2ID)
    return df


def build_prompt(text: str, aspect: str) -> str:
    return (
        "You are a domain expert in political communication performing aspect-based sentiment analysis.\n"
        f"Sentence: {text}\n"
        f"Aspect: {aspect}\n"
        "Task: What is the sentiment towards this aspect? "
        "Answer with exactly one word: negative, neutral, or positive.\n"
        "Answer:"
    )


@dataclass
class QwenCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        prompts = [e["prompt"] for e in examples]
        answers = [e["label_text"] for e in examples]

        full_texts = [p + " " + a for p, a in zip(prompts, answers)]
        prompt_lens = [
            len(self.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in prompts
        ]

        tokenized = self.tokenizer(
            full_texts,
            max_length=self.max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )

        labels = tokenized["input_ids"].clone()
        for i, plen in enumerate(prompt_lens):
            plen = min(plen, self.max_length)
            labels[i, :plen] = -100

        tokenized["labels"] = labels
        return tokenized


def extract_label_from_output(text: str) -> str:
    t = text.lower()
    for lab in ["negative", "neutral", "positive"]:
        if lab in t:
            return lab
    return "neutral"


def compute_metrics_from_generations(
    tokenizer, model, dataset: Dataset, max_length: int, num_samples: int = None
) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []

    loader_indices = list(range(len(dataset)))
    if num_samples is not None:
        loader_indices = loader_indices[:num_samples]

    device = model.device

    for idx in loader_indices:
        ex = dataset[idx]
        prompt = ex["prompt"]
        label_id = ex["label_id"]

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                num_beams=1,
            )

        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_label = extract_label_from_output(gen)
        preds.append(LABEL2ID[pred_label])
        gts.append(label_id)

    preds = np.array(preds)
    gts = np.array(gts)
    acc = accuracy_score(gts, preds)
    f1 = precision_recall_fscore_support(gts, preds, average="macro", zero_division=0)[2]
    return {"accuracy": acc, "f1_macro": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Qwen model (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument("--raw_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_raw_triplets(args.raw_file)

    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label_id"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label_id"], random_state=42
    )

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df["prompt"] = [
            build_prompt(t, a) for t, a in zip(split_df["text"], split_df["aspect"])
        ]
        split_df["label_text"] = [ID2LABEL[i] for i in split_df["label_id"]]

    def to_ds(d: pd.DataFrame) -> Dataset:
        return Dataset.from_pandas(
            d[["prompt", "label_text", "label_id"]], preserve_index=False
        )

    dataset = DatasetDict(
        train=to_ds(train_df),
        validation=to_ds(val_df),
        test=to_ds(test_df),
    )

    print("Loading tokenizer:", args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print("loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = QwenCollator(tokenizer=tokenizer, max_length=args.max_length)
	
	#training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=4,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        gradient_accumulation_steps=1,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        report_to=["none"],
        load_best_model_at_end=True,
        bf16=torch.cuda.is_available(),
        remove_unused_columns=False,

    )
	#trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    trainer.train()

    print("evaluating on validation via generation...")
    val_metrics = compute_metrics_from_generations(
        tokenizer, model, dataset["validation"], max_length=args.max_length
    )
    print("VAL metrics:", val_metrics)

    print("Evaluating on test via generation...")
    test_metrics = compute_metrics_from_generations(
        tokenizer, model, dataset["test"], max_length=args.max_length
    )
    print("TEST metrics:", test_metrics)

    with open(os.path.join(args.output_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(args.output_dir, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
