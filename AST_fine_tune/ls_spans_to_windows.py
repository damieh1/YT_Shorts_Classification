"""
´This script converts the Label Studio event-span annotations to window-level CSV for AST tuning.

Input: Label Studio JSON export (list of tasks). Each task should have data.audio (string URL/path)
and one or more annotations with results of type "labels" over the audio region.

Output CSV columns: stem,start,end,label,gold
 - stem: base name of the audio file (no extension)
 - start,end: window boundaries in seconds
 - label: one of the target labels
 - gold: 1 (present) or 0 (absent)

Windows cover [0, duration] with step=hop. Duration is taken from:
  1) --durations-csv (recommended): CSV with columns stem,duration
  2) fallback: max end time across all annotated regions for that task, optionally + --pad-seconds

Multiple annotators are combined via --combine:
  - union: window positive if ANY annotator overlapped the window for that label
  - majority: positive if >= ceil(n_annotators/2)
  - intersection: positive if ALL annotators overlapped

Overlap rule: a region counts for a window if overlap_seconds >= --min-overlap (default 0.05s).

Usage:
  python ls_spans_to_windows.py \
      --ls-json export.json \
      --out windows.csv \
      --window 1.0 --hop 1.0 \
      --combine majority \
      --durations-csv durations.csv

"""

import argparse, csv, json, math, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

def stem_from_uri(uri: str) -> str:
    # Strip querystrings and fragments
    for sep in ['?', '#']:
        if sep in uri:
            uri = uri.split(sep, 1)[0]
    # Handle file:// or http(s):// by taking path part
    base = os.path.basename(uri)
    # Remove extension(s)
    if '.' in base:
        return '.'.join(base.split('.')[:-1])
    return base

def load_durations_csv(path: Path) -> Dict[str, float]:
    mapping = {}
    with path.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            s = row.get('stem')
            d = row.get('duration')
            if s is None or d is None:
                continue
            try:
                mapping[s] = float(d)
            except ValueError:
                pass
    return mapping

def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def ceil_to(x: float, step: float) -> float:
    k = math.ceil(x / step)
    return k * step

def collect_spans_by_annotator(tasks: List[Dict[str, Any]], target_labels: List[str]) -> Dict[str, Dict[int, Dict[str, List[Tuple[float, float]]]]]:
    """
    Returns: spans[stem][annotator_id][label] = list[(start,end)]
    Annotator ID is index in task['annotations'] list.
    """
    spans = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for task in tasks:
        uri = task.get('data', {}).get('audio') or task.get('data', {}).get('url')
        if not uri:
            # Skip tasks without audio reference
            continue
        stem = stem_from_uri(uri)
        anns = task.get('annotations') or []
        for ann_idx, ann in enumerate(anns):
            results = ann.get('result') or []
            for res in results:
                if res.get('type') != 'labels':
                    continue
                val = res.get('value') or {}
                start = val.get('start')
                end = val.get('end')
                labels = val.get('labels') or []
                if start is None or end is None:
                    continue
                try:
                    s = float(start)
                    e = float(end)
                except Exception:
                    continue
                if e <= s:
                    continue
                for lab in labels:
                    if lab in target_labels:
                        spans[stem][ann_idx][lab].append((s, e))
    return spans

def infer_duration_fallback(task: Dict[str, Any]) -> Tuple[str, float]:
    """Try to infer duration from max 'end' in any result."""
    uri = task.get('data', {}).get('audio') or task.get('data', {}).get('url')
    stem = stem_from_uri(uri) if uri else None
    max_end = 0.0
    for ann in task.get('annotations') or []:
        for res in ann.get('result') or []:
            if res.get('type') != 'labels':
                continue
            val = res.get('value') or {}
            end = val.get('end')
            if end is None: 
                continue
            try:
                e = float(end)
                if e > max_end:
                    max_end = e
            except Exception:
                pass
    return stem, max_end

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ls-json', required=True, help='Label Studio JSON export')
    parser.add_argument('--out', required=True, help='Output CSV path')
    parser.add_argument('--window', type=float, default=1.0, help='Window size in seconds')
    parser.add_argument('--hop', type=float, default=1.0, help='Hop size in seconds')
    parser.add_argument('--combine', choices=['union','majority','intersection'], default='majority')
    parser.add_argument('--min-overlap', type=float, default=0.05, help='Minimum overlap (s) for a span to count for a window')
    parser.add_argument('--labels', nargs='*', default=[
        'Speech','Shout','Screaming','Laughter','Crying_and_sobbing','Siren','Alarm','Explosion','Engine','Helicopter'
    ], help='Target label set (exactly as in LS config)')
    parser.add_argument('--durations-csv', type=str, default=None, help='Optional CSV with columns stem,duration')
    parser.add_argument('--pad-seconds', type=float, default=0.0, help='Pad fallback duration by this many seconds')
    args = parser.parse_args()

    # Load JSON
    with open(args.ls_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print('Expected a list of tasks in the LS export.', file=sys.stderr)
        sys.exit(1)

    # Build per-stem annotator spans
    spans = collect_spans_by_annotator(data, args.labels)

    # Determine durations
    durations = {}
    if args.durations_csv:
        durations.update(load_durations_csv(Path(args.durations_csv)))

    # Fallback durations from tasks if missing
    by_stem_tasks = defaultdict(list)
    for t in data:
        uri = t.get('data', {}).get('audio') or t.get('data', {}).get('url')
        if not uri: 
            continue
        stem = stem_from_uri(uri)
        by_stem_tasks[stem].append(t)

    for stem in spans.keys():
        if stem in durations:
            continue
        # infer from any task of that stem
        any_task = by_stem_tasks.get(stem, [None])[0]
        if any_task is None:
            continue
        s, max_end = infer_duration_fallback(any_task)
        if s is None:
            continue
        durations[stem] = max(0.0, float(max_end) + float(args.pad_seconds))

    # Build windows and vote
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['stem','start','end','label','gold'])

        for stem, ann_spans in spans.items():
            # ann_spans: annotator_id -> label -> [(s,e)]
            n_annotators = len(ann_spans) if ann_spans else 0
            if n_annotators == 0:
                continue

            duration = durations.get(stem)
            if duration is None or duration <= 0:
                # If unknown, set to max region end across all annotators/labels
                mx = 0.0
                for labmap in ann_spans.values():
                    for segs in labmap.values():
                        for s,e in segs:
                            if e > mx: mx = e
                duration = mx

            win = float(args.window)
            hop = float(args.hop)
            t = 0.0
            # Round duration up to hop multiple so last partial window included
            T = ceil_to(duration, hop)
            while t < T - 1e-9:
                t0, t1 = t, min(t + win, T)
                for lab in args.labels:
                    # votes per annotator
                    votes = 0
                    for ann_id, labmap in ann_spans.items():
                        pos = 0
                        for (s,e) in labmap.get(lab, []):
                            if overlap_seconds(s,e,t0,t1) >= args.min_overlap:
                                pos = 1
                                break
                        votes += pos
                    # combine
                    gold = 0
                    if args.combine == 'union':
                        gold = 1 if votes > 0 else 0
                    elif args.combine == 'majority':
                        needed = math.ceil(n_annotators/2.0)
                        gold = 1 if votes >= needed else 0
                    else:  # intersection
                        gold = 1 if votes == n_annotators and n_annotators > 0 else 0

                    writer.writerow([stem, f"{t0:.3f}", f"{t1:.3f}", lab, gold])
                t += hop

    print(f"Wrote window CSV to {args.out}")
    print("Extra-Info: pass --durations-csv to include full negative windows, not just up to the last span.")
if __name__ == '__main__':
    main()
