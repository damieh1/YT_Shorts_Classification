# Reads LS JSON, matches basenames to the resolved mirror, recenters spans by clip start from filename, windows --> train/val CSV.

#!/usr/bin/env python3
import argparse, csv, json, math, os, random, re, ntpath, urllib.parse
from pathlib import Path
random.seed(13)

LABELS = ["Speech","Shout_Scream","Siren","Chant","Music_BG","Music_FG","Crowd_noise"]
HEX_PREFIX  = re.compile(r'^[0-9a-f]{8}-', re.I)
PAIR_RE     = re.compile(r'((?:\d+(?:\.\d+)?|\.\d+))_((?:\d+(?:\.\d+)?|\.\d+))(?=\.wav$)', re.I)

def win_base_noext(s: str) -> str:
    b = ntpath.basename(s)
    if b == s: b = os.path.basename(s)
    b = urllib.parse.unquote(b)
    b = HEX_PREFIX.sub('', b)
    return b[:-4] if b.lower().endswith('.wav') else b

def parse_bounds_from_stem(stem: str):
    m = PAIR_RE.search(stem + ".wav")
    if not m: return None
    return (float(m.group(1)), float(m.group(2)))

def overlap(a,b):
    (s1,e1),(s2,e2) = a,b
    return max(0.0, min(e1,e2) - max(s1,s2))

def make_wins(L, w):
    n = max(1, math.ceil((L - 1e-9) / w))
    return [(i*w, min((i+1)*w, L)) for i in range(n)]

def load_ls(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tasks" in data: data = data["tasks"]
    if not isinstance(data, list): raise ValueError("Unexpected LS JSON format")
    return data

def collect_spans(tasks, labels_set):
    spans = {}; nseg = 0
    for t in tasks:
        raw = t.get("file_upload") or (t.get("data",{}) or {}).get("audio","")
        base = win_base_noext(raw)
        anns = t.get("annotations") or t.get("completions") or []
        for ann in anns:
            for r in ann.get("result", []):
                if r.get("to_name") not in ("audio","Audio","waveform"): continue
                if r.get("from_name") not in ("event","labels"): continue
                v = r.get("value", {})
                labs = v.get("labels") or ([v.get("label")] if v.get("label") else [])
                if not labs or "start" not in v or "end" not in v: continue
                try: s = float(v["start"]); e = float(v["end"])
                except: continue
                if e <= s: continue
                for lab in labs:
                    if lab in labels_set:
                        spans.setdefault(base, {}).setdefault(lab, []).append((s,e)); nseg += 1
    return spans, nseg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ls-json", required=True)
    ap.add_argument("--audio-root", required=True)  # data/ls_resolved
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--labels", nargs="+", default=LABELS)
    ap.add_argument("--clip-sec", type=float, default=10.0)
    ap.add_argument("--win-sec",  type=float, default=0.25)
    ap.add_argument("--ov-frac",  type=float, default=0.10)
    ap.add_argument("--neg-pos-ratio", type=float, default=3.0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(); random.seed(args.seed)

    tasks = load_ls(Path(args.ls_json))
    spans, nseg = collect_spans(tasks, set(args.labels))
    print(f"Collected spans: {nseg} across LS clips: {len(spans)}")

    root = Path(args.audio_root)
    idx = {p.stem: p for p in root.glob("*.wav")}
    print(f"Resolved audio available (flat): {len(idx)}")

    have = sum(1 for b in spans.keys() if b in idx)
    print(f"Basename matches (JSON->resolved): {have} / {len(spans)}")

    rows_by_clip = {}; pos_tot = neg_tot = ms_tot = shift_tot = drop_tot = 0; missed=[]

    for base, bylab in spans.items():
        p = idx.get(base)
        if p is None: missed.append(base); continue

        bounds = parse_bounds_from_stem(p.stem)
        if bounds:
            clip_start, clip_end = bounds
            L = max(0.5, clip_end - clip_start)
        else:
            clip_start, L = 0.0, args.clip_sec

        norm = {}
        for lab, segs in bylab.items():
            keep = []
            for (s, e) in segs:
                ms = False; shifted = False
                if (e - s) > 5 * L: s/=1000.0; e/=1000.0; ms = True
                if s >= L or e > L: s -= clip_start; e -= clip_start; shifted = True
                s2 = max(0.0, s); e2 = min(L, e)
                if e2 > s2: keep.append((s2,e2))
                else: drop_tot += 1
                ms_tot += int(ms); shift_tot += int(shifted)
            if keep: norm[lab] = keep

        W = make_wins(L, args.win_sec)
        pos_here = 0
        for ws, we in W:
            need = args.ov_frac * (we - ws)
            labs = set(l for l, segs in norm.items() if any(overlap((ws,we), seg) >= need for seg in segs))
            if labs:
                rows_by_clip.setdefault(base, []).append((str(p.resolve()), ws, we, ",".join(sorted(labs))))
                pos_here += 1
        pos_tot += pos_here

        neg = []
        for ws, we in W:
            need = args.ov_frac * (we - ws)
            active = any(any(overlap((ws,we), seg) >= need for seg in segs) for segs in norm.values())
            if not active: neg.append((str(p.resolve()), ws, we, ""))
        keep = min(len(neg), int(math.ceil(args.neg_pos_ratio * max(1, pos_here))))
        random.shuffle(neg)
        rows_by_clip.setdefault(base, []).extend(neg[:keep])
        neg_tot += min(len(neg), keep)

    if missed:
        print("Examples of JSON bases not in resolved (first 5):")
        for m in missed[:5]: print("  -", m)

    print(f"Pos windows: {pos_tot} | Neg windows kept: {neg_tot} | ms->s: {ms_tot} | shifted: {shift_tot} | dropped: {drop_tot}")

    bases = list(rows_by_clip.keys()); random.shuffle(bases)
    nval = max(1, int(round(len(bases) * args.val_frac)))
    val_set = set(bases[:nval]); tr, va = [], []
    for b, rows in rows_by_clip.items():
        (va if b in val_set else tr).extend(rows)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_train,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["path","start","end","labels"])
        for pth,s,e,lab in tr: w.writerow([pth,f"{s:.3f}",f"{e:.3f}",lab])
    with open(args.out_val,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["path","start","end","labels"])
        for pth,s,e,lab in va: w.writerow([pth,f"{s:.3f}",f"{e:.3f}",lab])
    print("Wrote:", args.out_train, "rows:", len(tr), " |  ", args.out_val, "rows:", len(va))

if __name__ == "__main__":
    main()
