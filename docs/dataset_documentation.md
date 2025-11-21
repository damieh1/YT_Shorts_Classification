# Dataset Documentation (Summary)

## Overview
This project analyzes YouTube Shorts from four international news outlets. All data were collected through the official YouTube Data API and supplemented with Whisper-based re-transcriptions. Because of YouTube API terms and institutional rules, raw data cannot be shared.

To support reproducibility, we release:
- the full schema of the dataset,
- code used for preprocessing,
- and a synthetic example dataset mirroring real structure.

## Collection
- Source: Public YouTube Shorts
- Method: YouTube Data API v3
- Channels: Al Jazeera (English), TRT World (English), BBC News (English), Deutsche Welle (English)
- Date Range: 10/07/2023–10/07/2024
- Data acquisition took place after each short had existed for two weeks
- Extracted Metadata:
  - video_id
  - description
  - title
  - publish_date
  - views
  - likes
  

## Preprocessing Steps
1. Whisper transcription for all videos    
2. Dependency parsing using Biaffine-SUPAR  
4. Filtering to content-bearing sentences  
5. ABSA predictions  
6. Event tuple extraction  
7. Aggregation and outlet-level comparisons  

## Privacy & Compliance
- No private or non-public data used
- No user-level data or comments included
- No redistribution of video content
- Only aggregated results reported
- Synthetic dataset provided for illustration

## Data Schema (Real Dataset/Event-Level)

## Video Metadata
- video_title: `string`  
- seg_group_id: `string`  
- seg_ids: `string`  
- start_time: `float`  
- end_time: `float`  
- sentence_index: `int`  

## Text Fields
- sentence: `string`  

## ABSA Fields
- aspect_category: `string`  
- aspect_term: `string`  

## Dependency Parse Fields
- dep_role: `string`  
- head_verb: `string`  
- event_agent: `string`  


