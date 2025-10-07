#!/usr/bin/env python3
import argparse, csv, numpy as np, torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Trainer, TrainingArguments

LABELS = ['Speech','Shout_Scream','Siren','Chant','Music_FG','Music_BG','Crowd_noise']
L2I = {l:i for i,l in enumerate(LABELS)}
SR = 16000

def load_rows(csv_path):
    rows=[]
    with open(csv_path,'r',encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            labs = [x.strip() for x in (row.get('labels','') or '').split(',') if x.strip()]
            y = np.zeros(len(LABELS), dtype=np.float32)
            for l in labs:
                if l in L2I: y[L2I[l]] = 1.0
            rows.append((row['path'], float(row['start']), float(row['end']), y))
    return rows

class WinDataset(Dataset):
    def __init__(self, rows, feat, pad_sec=5.0, sr=SR):
        self.rows=rows; self.feat=feat; self.pad_n=int(round(pad_sec*sr)); self.sr=sr
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        path,t0,t1,y = self.rows[i]
        import soundfile as sf, numpy as np
        x, sr = sf.read(path, dtype='float32', always_2d=False)
        if x.ndim>1: x=x.mean(axis=1)
        i0=int(round(t0*sr)); i1=int(round(t1*sr)); seg=x[i0:i1]
        if sr!=self.sr:
            n=max(1,int(round(len(seg)*self.sr/max(1,sr))))
            xp=np.linspace(0,1,len(seg),endpoint=True); xq=np.linspace(0,1,n,endpoint=True)
            seg=np.interp(xq,xp,seg).astype(np.float32)
        if len(seg)<self.pad_n: seg=np.pad(seg,(0,self.pad_n-len(seg)))
        elif len(seg)>self.pad_n: seg=seg[:self.pad_n]
        inp=self.feat(seg, sampling_rate=self.sr, return_tensors='pt')
        return {'input_values': inp['input_values'][0],
                'labels': torch.tensor(y, dtype=torch.float32)}

def freeze_backbone(model):
    for p in model.parameters(): p.requires_grad=False
    for p in model.classifier.parameters(): p.requires_grad=True

class WeightedTrainer(Trainer):
    def __init__(self, pos_weight=None, **kw):
        super().__init__(**kw)
        self.loss_fn = (torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                        if pos_weight is not None else torch.nn.BCEWithLogitsLoss())
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--val-csv', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--model-id', default='MIT/ast-finetuned-audioset-10-10-0.4593')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--pad-sec', type=float, default=10.0)
    args=ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feat = AutoFeatureExtractor.from_pretrained(args.model_id)
    model = AutoModelForAudioClassification.from_pretrained(
        args.model_id,
        problem_type='multi_label_classification',
        num_labels=len(LABELS),
        label2id={l:i for i,l in enumerate(LABELS)},
        id2label={i:l for i,l in enumerate(LABELS)},
        ignore_mismatched_sizes=True
    ).to(device)
    freeze_backbone(model)

    tr_rows = load_rows(args.train_csv)
    va_rows = load_rows(args.val_csv)
    if len(tr_rows)==0 or len(va_rows)==0:
        raise SystemExit("No training/validation rows found. Check your CSVs.")
    print(f"Train rows: {len(tr_rows)} | Val rows: {len(va_rows)}")
    print("Using device:", device, "| fp16:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    Y = np.stack([y for _,_,_,y in tr_rows])
    P = Y.sum(axis=0); N = len(Y) - P
    pos_weight = np.where(P>0, (N / np.maximum(P,1e-6)).astype(np.float32), np.ones_like(P,dtype=np.float32))
    print("pos_weight:", pos_weight.tolist())

    ds_tr = WinDataset(tr_rows, feat, pad_sec=args.pad_sec)
    ds_va = WinDataset(va_rows, feat, pad_sec=args.pad_sec)

    def collate(batch):
        return {'input_values': torch.stack([b['input_values'] for b in batch]),
                'labels': torch.stack([b['labels'] for b in batch])}

    args_tr = TrainingArguments(
        output_dir=args.outdir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        dataloader_num_workers=4
    )

    trainer = WeightedTrainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tr,
        eval_dataset=ds_va,
        data_collator=collate,
        pos_weight=torch.tensor(pos_weight, device=device)
    )
    trainer.train()
    trainer.save_model(args.outdir)
    print("Saved fine-tuned model to", args.outdir)

if __name__=='__main__': main()
