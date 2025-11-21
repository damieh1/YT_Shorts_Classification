# Parsing Module (Biaffine-SuPar)

This directory contains scripts used to perform dependency parsing on Whisper transcripts using SuPar’s biaffine dependency parser.

## Script

- `parse_whisper_captions_biaffine.py`  
  Takes Whisper JSON transcripts and outputs dependency-parsed CSV files.

## Requirements
Parsing was executed inside a dedicated environment (`biaffine-env`) containing:
- supar
- torch (GPU-enabled)
- transformers
- pandas, numpy

## Example Command

```bash
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
