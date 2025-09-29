"""
Convert Label Studio Tier-1 event spans -> 1s windows CSV with label collapsing and attributes.

Expected LS labels:
  Speech, Shout, Screaming, Siren, Chant, Music, Crowd_noise

Per-region attribute (Choices):
  Music_FG or Music_BG (set only when Music present)

Output labels (default):
  Speech, Shout_Scream, Siren, Chant, Music_FG, Music_BG, Crowd_noise

Usage:
  python ls_tier1_spans_to_windows.py \
      --ls-json export.json \
      --out windows_tier1.csv \
      --window 1.0 --hop 1.0 \
      --combine majority \
      --min-overlap 0.02 \
      --durations-csv durations.csv

Options:
  --keep-shout-scream-separate   # instead of collapsing, output Shout and Screaming separately
  --labels ...                   # optional explicit output label list
  --pad-seconds 0.0              # add padding when inferring duration (if durations.csv missing)
"""

import argparse, csv, json, math, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

DEFAULT_OUTPUT_LABELS = ['Speech','Shout_Scream','Siren','Chant','Music_FG','Music_BG','Crowd_noise']

def stem_from_uri(uri: str) -> str:
    for sep in ['?', '#']:
        if sep in uri:
            uri = uri.split(sep, 1)[0]
    base = os.path.basename(uri)
    return '.'.join(base.split('.')[:-1]) if '.' in base else base

def load_durations_csv(path: Path) -> Dict[str, float]:
    mapping = {}
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            s = row.get('stem'); d = row.get('duration')
            if s and d:
                try: mapping[s] = float(d)
                except: pass
    return mapping

def overlap_seconds(a0,a1,b0,b1): 
    return max(0.0, min(a1,b1) - max(a0,b0))

def ceil_to(x: float, step: float) -> float:
    k = math.ceil(x / step)
    return k * step

def collect_regions(tasks: List[Dict[str,Any]]):
    """
    Return dict: spans[stem][ann_id] = list of (start,end,primary_label,choices[])
    Tries to attach per-region Choices via region_id -> labels result id mapping.
    """
    spans = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        uri = task.get('data',{}).get('audio') or task.get('data',{}).get('url')
        if not uri:
            continue
        stem = stem_from_uri(uri)
        for ann_idx, ann in enumerate(task.get('annotations') or []):
            results = ann.get('result') or []
            # First collect choices linked to a region (by region_id)
            choices_by_region = defaultdict(list)
            for res in results:
                if res.get('type') == 'choices':
                    # Try both top-level region_id and nested value.region_id (LS exports vary)
                    reg_id = res.get('region_id') or (res.get('value') or {}).get('region_id')
                    ch = (res.get('value') or {}).get('choices') or []
                    if reg_id and ch:
                        choices_by_region[reg_id].extend(ch)
            # Then collect labels, attach choices if region ids match
            for res in results:
                if res.get('type') != 'labels':
                    continue
                val = res.get('value') or {}
                start, end = val.get('start'), val.get('end')
                labels = val.get('labels') or []
                if start is None or end is None or not labels:
                    continue
                try:
                    s = float(start); e = float(end)
                except:
                    continue
                if e <= s:
                    continue
                # direct choices if present on same object; else via region id
                direct_choices = list(val.get('choices') or [])
                region_id = res.get('id')  # labels result id
                linked_choices = choices_by_region.get(region_id, [])
                all_choices = direct_choices or linked_choices
                for lab in labels:
                    spans[stem][ann_idx].append((s, e, lab, all_choices))
    return spans

def map_output_labels(primary: str, choices: List[str], keep_separate: bool) -> List[str]:
    """Map LS labels + choices to output label(s)."""
    if primary == 'Music':
        out = []
        if 'Music_FG' in choices: out.append('Music_FG')
        if 'Music_BG' in choices: out.append('Music_BG')
        if not out:
            # If annotator forgot the role, default to BG to be conservative
            out = ['Music_BG']
        return out
    if primary in ('Shout','Screaming'):
        return [primary] if keep_separate else ['Shout_Scream']
    # pass-through
    return [primary]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ls-json', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--window', type=float, default=1.0)
    ap.add_argument('--hop', type=float, default=1.0)
    ap.add_argument('--combine', choices=['union','majority','intersection'], default='majority')
    ap.add_argument('--min-overlap', type=float, default=0.02)
    ap.add_argument('--labels', nargs='*', default=None, help='Override output label set and order')
    ap.add_argument('--keep-shout-scream-separate', action='store_true')
    ap.add_argument('--durations-csv', type=str, default=None)
    ap.add_argument('--pad-seconds', type=float, default=0.0)
    args = ap.parse_args()

    with open(args.ls_json, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        print('Expected a list of tasks in LS export JSON.', file=sys.stderr); sys.exit(1)

    spans = collect_regions(tasks)
    durations = {}
    if args.durations_csv:
        durations.update(load_durations_csv(Path(args.durations_csv)))

    # determine output label set
    if args.labels:
        out_labels = args.labels
    else:
        out_labels = (['Speech','Shout','Screaming','Siren','Chant','Music_FG','Music_BG','Crowd_noise']
                      if args.keep_shout_scream_separate else DEFAULT_OUTPUT_LABELS)

    # fallback durations when missing
    for stem, annmap in spans.items():
        if stem in durations:
            continue
        mx = 0.0
        for segs in annmap.values():
            for (s,e,lab,choices) in segs:
                if e > mx: mx = e
        durations[stem] = mx + float(args.pad_seconds)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['stem','start','end','label','gold'])
        for stem, annmap in spans.items():
            n_annot = len(annmap) if annmap else 0
            if n_annot == 0:
                continue
            dur = float(durations.get(stem, 0.0))
            win = float(args.window); hop = float(args.hop)
            T = max(win, ceil_to(dur, hop))
            t = 0.0
            while t < T - 1e-9:
                t0, t1 = t, min(t + win, T)
                for lab in out_labels:
                    votes = 0
                    for segs in annmap.values():
                        pos = 0
                        for (s,e,primary,choices) in segs:
                            for mapped in map_output_labels(primary, choices, args.keep_shout_scream_separate):
                                if mapped != lab:
                                    continue
                                if overlap_seconds(s,e,t0,t1) >= args.min_overlap:
                                    pos = 1; break
                            if pos: break
                        votes += pos
                    if args.combine == 'union':
                        gold = 1 if votes > 0 else 0
                    elif args.combine == 'majority':
                        need = (n_annot + 1)//2
                        gold = 1 if votes >= need else 0
                    else:
                        gold = 1 if votes == n_annot and n_annot > 0 else 0
                    w.writerow([stem, f"{t0:.3f}", f"{t1:.3f}", lab, gold])
                t += hop
    print(f"Wrote {args.out}")

if __name__ == '__main__':
    main()
