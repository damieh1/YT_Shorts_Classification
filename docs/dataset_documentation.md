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
- Channels: [anonymous for review]
- Date Range: 10/07/2023--10/072024
- Data acquisition took place after each short had existed for two weeks
- Extracted Metadata:
  - video_id
  - title
  - publish_date
  - views
  - likes
  - channel metadata

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

## Data Schema (Real Dataset)

