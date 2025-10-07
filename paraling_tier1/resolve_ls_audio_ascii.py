# Robust LS-JSON -->> WAV resolver (handles Windows paths, %XX, 8-hex prefixes, time suffixes like .500_12.500, fuzzy matching, mapping CSV).

#!/usr/bin/env python3
import argparse, json, os, re, urllib.parse, ntpath, csv, shutil
from pathlib import Path
from difflib import SequenceMatcher

HEX_PREFIX = re.compile(r'^[0-9a-f]{8}-', re.I)
PAIR_RE    = re.compile(r'((?:\d+(?:\.\d+)?|\.\d+))_((?:\d+(?:\.\d+)?|\.\d+))(?=\.wav$)', re.I)

def win_basename(s: str) -> str:
    b = ntpath.basename(s)
    return b if b != s else os.path.basename(s)

def tokens_relaxed(s: str) -> str:
    s = urllib.parse.unquote(s)
    s = HEX_PREFIX.sub('', s)
    s = s.lower().strip()
    return re.sub(r'[^a-z0-9]+', '', s)

def parse_times_any(stem: str):
    m = PAIR_RE.search(stem + ".wav")
    if not m: return None
    try:
        return (float(m.group(1)), float(m.group(2)))
    except ValueError:
        return None

def gather_wavs(roots):
    wavs=[]
    for r in roots:
        r=Path(r)
        if r.is_file() and r.suffix.lower()=='.wav': wavs.append(r)
        elif r.is_dir(): wavs += list(r.rglob('*.wav'))
    return wavs

def best_fuzzy_path(target_token_str: str, candidate_paths):
    best = (None, 0.0)
    for p in candidate_paths:
        cand_tok = tokens_relaxed(p.name)
        score = SequenceMatcher(a=target_token_str, b=cand_tok).ratio()
        if score > best[1]: best = (p, score)
    return best

def time_match_with_tolerance(target, wavs, tolerances):
    if target is None: return (None, None)
    ts, te = target
    parsed = []
    for p in wavs:
        be = parse_times_any(p.stem)
        if be: parsed.append((p, be[0], be[1]))
    for tol in tolerances:
        cands = [p for (p, s, e) in parsed if abs(s - ts) <= tol and abs(e - te) <= tol]
        if len(cands) == 1: return (cands[0], tol)
        if len(cands) > 1:
            tgt_tok = tokens_relaxed(f"{ts:.3f}_{te:.3f}")
            winner, score = best_fuzzy_path(tgt_tok, cands)
            if winner: return (winner, tol)
    return (None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ls-json', required=True)
    ap.add_argument('--audio-roots', nargs='+', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--mapcsv', required=True)
    ap.add_argument('--min-fuzzy', type=float, default=0.65)
    args = ap.parse_args()

    tasks = json.loads(Path(args.ls_json).read_text(encoding='utf-8'))
    if isinstance(tasks, dict) and 'tasks' in tasks: tasks = tasks['tasks']

    wavs = gather_wavs(args.audio_roots)
    by_name = {p.name: p for p in wavs}
    by_tok  = {}
    for p in wavs: by_tok.setdefault(tokens_relaxed(p.name), []).append(p)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    mapped = []; ok=miss=0; show=[]

    for t in tasks:
        raw = t.get('file_upload') or (t.get('data',{}) or {}).get('audio','')
        base = win_basename(raw); base = urllib.parse.unquote(base); base = HEX_PREFIX.sub('', base)
        if not base.lower().endswith('.wav'): base += '.wav'

        method=""; score=""
        cand = by_name.get(base)
        if not cand:
            target_times = parse_times_any(Path(base).stem)
            cand, tol_used = time_match_with_tolerance(target_times, wavs, [0.02, 0.10, 0.50, 1.00])
            if cand: method=f"time±{tol_used:.2f}s"; score="1.0"
        if not cand:
            tok = tokens_relaxed(base)
            cands = by_tok.get(tok)
            if cands and len(cands)==1:
                cand = cands[0]; method="tokens-exact"; score="1.0"
            else:
                cand, s = best_fuzzy_path(tok, wavs)
                if cand and s >= args.min_fuzzy: method="global-fuzzy"; score=f"{s:.3f}"
                else: cand=None

        if cand:
            dst = outdir / base
            if not dst.exists():
                try: dst.symlink_to(cand.resolve())
                except Exception: shutil.copy2(cand, dst)
            mapped.append((base, str(cand), method, score)); ok += 1
        else:
            mapped.append((base, "", "UNMATCHED", "")); miss += 1
            if len(show) < 10: show.append(base)

    with open(args.mapcsv,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["ls_name","matched_path","method","score"]); w.writerows(mapped)

    print(f"Resolved {ok} / {ok+miss}  ->  {outdir}")
    if show:
        print("Unmatched examples:"); [print("  -", s) for s in show]

if __name__=='__main__': main()
