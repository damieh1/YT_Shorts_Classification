"""
Uses Whisper captions from stitched sentences and parse aspect rows (1 per aspect occurrence)

Inputs: Whisper .json with {"video_id":..., "segments":[{"id":..., "start":..., "end":..., "text":...}, ...]}
Entity dict: Excel, columns = canonical categories (e.g., Israel, Hamas), cells = surface forms
Output CSV columns:
 video_id, seg_group_id, seg_ids, start, end, sent_ix, sentence,
 aspect, aspect_surface, dep_rel, head_form, dep_form
"""

import os, re, csv, json, argparse
from typing import Dict, List, Tuple, Any
import torch
import pandas as pd
from tqdm import tqdm
from supar import Parser

# text normalization
def normalize_caption_text(s: str) -> str:
    if not s:
        return ""
    # join hard wraps/newlines and collapse whitespace
    s = re.sub(r"\s*\n+\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# entity dict
def load_entity_xlsx(path_xlsx: str) -> Dict[str, str]:
    df = pd.read_excel(path_xlsx)
    surface_to_cat: Dict[str, str] = {}
    for col in df.columns:
        for v in df[col].dropna():
            surf = str(v).strip().lower()
            if surf and surf != "nan":
                surface_to_cat[surf] = col
    return surface_to_cat

# read Whisper segments
def gather_segments(input_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    returns: dict video_id --> list of {id,start,end,text} (normalized)
    """
    vids: Dict[str, List[Dict[str, Any]]] = {}
    for entry in os.scandir(input_dir):
        if not entry.is_file() or not entry.name.endswith(".json"):
            continue
        try:
            with open(entry.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Cannot read {entry.path}: {e}")
            continue
        video_id = data.get("video_id") or os.path.splitext(entry.name)[0]
        segs = []
        for seg in data.get("segments", []):
            txt = normalize_caption_text(seg.get("text") or "")
            if not txt:
                continue
            segs.append({
                "id": seg.get("id"),
                "start": seg.get("start"),
                "end": seg.get("end"),
                "text": txt
            })
        if segs:
            # sort by start or id to be safe
            segs.sort(key=lambda s: (s.get("start") if s.get("start") is not None else 0, s.get("id") or 0))
            vids.setdefault(video_id, []).extend(segs)
    return vids

# stitching heuristics
# sentence enders (., !, ?, …) possibly followed by a closing quote/paren/bracket
_SENT_END_GROUP = r'([.!?…][\"\'”’\)\]]?)'   # capture the terminator
_SENT_END_SPLIT = re.compile(_SENT_END_GROUP + r'\s+')  # used for marker insertion

# continuation detectors
_CONT_START_RE = re.compile(r'^[,;:)\]]')     # clearly a continuation
_LOWER_START_RE = re.compile(r'^[a-z]')       # lowercase start, likely continuation
_LINKERS = {"and","or","but","so","because","though","although","however","whereas",
            "while","yet","nor","than","that","which","who","whom","whose","when",
            "if","then"}

def looks_like_sentence_end(text: str) -> bool:
    return bool(re.search(_SENT_END_GROUP + r'\s*$', text.strip()))

def looks_like_continuation(next_text: str) -> bool:
    t = next_text.strip()
    if not t:
        return False
    if _CONT_START_RE.match(t) or _LOWER_START_RE.match(t):
        return True
    first = t.split(None, 1)[0].strip(",.;:").lower()
    return first in _LINKERS

def should_stitch(prev_text: str, next_text: str) -> bool:
    # stitch if previous segment doesn't look sentence-final, or next starts like a continuation
    if not looks_like_sentence_end(prev_text):
        return True
    if looks_like_continuation(next_text):
        return True
    return False

def split_into_sentences(text: str) -> List[str]:
    """
    split by sentence terminators WITHOUT look-behind:
    insert a marker after ([.!?…]["'”’)]?) then split on the marker.
    """
    if not text:
        return []
    marked = _SENT_END_SPLIT.sub(r'\1<SENT_SPLIT> ', text)
    parts = [p.strip() for p in marked.split('<SENT_SPLIT>') if p.strip()]
    return parts or [text.strip()]

def stitch_segments(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    input: list of seg dicts (id,start,end,text) for a video (sorted)
    output: list of stitched "units":
      {
        "seg_group_id": "id" or "id1+id2+...",
        "seg_ids": [id1,id2,...],
        "start": min start,
        "end": max end,
        "sentences": [full_sentence1, full_sentence2, ...]
      }
    We build units by merging adjacent segments until we have a clear sentence boundary.
    """
    units = []
    curr_ids, curr_start, curr_end, curr_text = [], None, None, ""

    def flush_unit():
        nonlocal curr_ids, curr_start, curr_end, curr_text
        if not curr_ids:
            return
        text = curr_text.strip()
        sentences = split_into_sentences(text)
        units.append({
            "seg_group_id": str(curr_ids[0]) if len(curr_ids)==1 else "+".join(str(i) for i in curr_ids),
            "seg_ids": curr_ids[:],
            "start": curr_start,
            "end": curr_end,
            "sentences": sentences
        })
        curr_ids, curr_start, curr_end, curr_text = [], None, None, ""

    for s in segs:
        sid, st, en, txt = s["id"], s["start"], s["end"], s["text"]
        if not curr_ids:
            curr_ids = [sid]
            curr_start = st
            curr_end = en
            curr_text = txt
            continue

        if should_stitch(curr_text, txt):
            curr_ids.append(sid)
            if en is not None:
                curr_end = en
            curr_text = (curr_text + " " + txt).strip()
        else:
            flush_unit()
            curr_ids = [sid]
            curr_start = st
            curr_end = en
            curr_text = txt

    flush_unit()
    return units

# core processing
def process_video_units(
    video_id: str,
    units: List[Dict[str, Any]],
    surface_to_cat: Dict[str, str],
    parser: Parser,
    device: str
) -> List[Dict[str, Any]]:
    rows = []
    for unit in units:
        seg_group_id = unit["seg_group_id"]
        seg_ids = unit["seg_ids"]
        start = unit["start"]
        end = unit["end"]
        for sent_ix, sentence in enumerate(unit["sentences"]):
            pred = parser.predict([sentence], lang="en", prob=True, verbose=False, device=device)[0]
            words = list(pred.words)
            arcs  = list(pred.arcs)   # 1-based heads, 0=ROOT
            rels  = list(pred.rels)
            for idx, w in enumerate(words):
                lw = w.lower()
                if lw in surface_to_cat:
                    aspect_cat = surface_to_cat[lw]
                    dep_rel = rels[idx] if idx < len(rels) else "dep"
                    head_idx = arcs[idx] if idx < len(arcs) else 0
                    head_word = words[head_idx - 1] if head_idx and 0 <= head_idx - 1 < len(words) else "ROOT"
                    rows.append({
                        "video_id": video_id,
                        "seg_group_id": seg_group_id,
                        "seg_ids": "|".join(str(i) for i in seg_ids),
                        "start": start,
                        "end": end,
                        "sent_ix": sent_ix,
                        "sentence": sentence,
                        "aspect": aspect_cat,
                        "aspect_surface": w,
                        "dep_rel": dep_rel,
                        "head_form": head_word,
                        "dep_form": w
                    })
    return rows

def run_pipeline(input_dir: str, entity_xlsx: str, output_csv: str,
                 model_name: str, device: str):
    surface_to_cat = load_entity_xlsx(entity_xlsx)
    print(f"Loaded {len(surface_to_cat)} entity surface forms from {os.path.basename(entity_xlsx)}.")
    videos = gather_segments(input_dir)
    print(f"Found {len(videos)} videos with caption segments in {input_dir}.")

    parser = Parser.load(model_name, weights_only=True)
    parser.model = parser.model.to(device)

    all_rows = []
    for vid, segs in tqdm(videos.items(), desc="Stitching + parsing"):
        units = stitch_segments(segs)
        rows = process_video_units(vid, units, surface_to_cat, parser, device)
        all_rows.extend(rows)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "video_id","seg_group_id","seg_ids","start","end","sent_ix","sentence",
            "aspect","aspect_surface","dep_rel","head_form","dep_form"
        ])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"WROTE {output_csv} rows: {len(all_rows)}")

# CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with Whisper .json caption files")
    ap.add_argument("--entity_xlsx", required=True, help="Excel entity dict (columns=categories, cells=surface forms)")
    ap.add_argument("--output_csv", required=True, help="Destination CSV")
    ap.add_argument("--model_name", default="biaffine-dep-en", help="supar checkpoint name")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_pipeline(args.input_dir, args.entity_xlsx, args.output_csv, args.model_name, device)

if __name__ == "__main__":
    main()
