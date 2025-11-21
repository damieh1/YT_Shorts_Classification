# Parsing Module (Biaffine-SuPar)

We use the SuPar biaffine dependency parser to all Whisper transcripts.  
It converts each caption segment into dependency arcs, POS tags, and aspect-aligned spans used for ABSA and event extraction.

**Provides:**  
- Sentence segmentation + dependency parsing  
- Token–head–relation structure (`dep_rel`, `head_form`)  
- Aspect/entity alignment from the Excel dictionary  
- Time-coded parsed sentences (`start`, `end`, `seg_ids`, `sentence_index`)  
- Output CSV consumed by ABSA + event extraction

## Script

- `parse_whisper_captions_biaffine.py`  
  Takes Whisper JSON transcripts and outputs dependency-parsed CSV files.

## Example Command

```bash
source bin/activate
source biaffine-env/bin/activate

python parse_whisper_captions_biaffine.py \
  --input_dir <captions_dir> \
  --entity_xlsx configs/entity_dict_18_04_2025.xlsx \
  --output_csv data/<OUTLET>_absa_from_parses.csv \
  --model_name "biaffine-dep-en"
```
--- 

## Output

`CSV` file containing:
- `sentence index`
- `original sentence`
- `dependency relations`
- `aspect term alignments`
- `timestamps`
