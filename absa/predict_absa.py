import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0: "negative", 1: "neutral", 2: "positive"}

# tokenizer & model 
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"Loaded model on device: {device}")
    return tokenizer, model

# predict in safe mini-batches on 256 max tokens 
def predict_batch(tokenizer, model, texts, aspects, max_len=256, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_labels = []
    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_aspects = aspects[i:i + batch_size]

        enc = tokenizer(
            list(batch_texts),
            list(batch_aspects),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()

        preds = probs.argmax(axis=1)
        labels = [LABELS[p] for p in preds]

        all_labels.extend(labels)
        all_probs.extend(probs)

        # prevent CUDA fragmentation
        del enc, logits
        torch.cuda.empty_cache()

    return all_labels, all_probs


# file inputs | DataFrame structure | file outputs 
def main(args):
    tokenizer, model = load_model(args.model)

    os.makedirs(args.out, exist_ok=True)

    input_files = [
        "BBC_absa_from_parses.csv",
        "AJ_absa_from_parses.csv",
        "TRT_absa_from_parses.csv",
        "DW_absa_from_parses.csv",
    ]

    for fname in input_files:
        path = os.path.join(args.data, fname)
        print(f"\nProcessing: {path}")

        df = pd.read_csv(path)

        labels, probs = predict_batch(
            tokenizer, model,
            df["sentence"].astype(str),
            df["aspect"].astype(str),
            max_len=256,
            batch_size=4 ### Seems to be the most safe approach to avoiod OOM for our GPU
        )

        # convert list to 2D-array
        probs = np.vstack(probs)

        df["predicted_sentiment"] = labels
        df["neg_score"] = probs[:, 0]
        df["neutral_score"] = probs[:, 1]
        df["positive_score"] = probs[:, 2]

        out_path = os.path.join(args.out, fname.replace(".csv", "_with_sentiment.csv"))
        df.to_csv(out_path, index=False)

        print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned DeBERTa-v3-base model")
    parser.add_argument("--data", required=True, help="Folder containing input CSVs")
    parser.add_argument("--out", required=True, help="Output folder")
    args = parser.parse_args()

    main(args)

