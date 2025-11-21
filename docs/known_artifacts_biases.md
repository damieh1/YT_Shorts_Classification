# Known Artifacts and Biases

## Platform Bias
We observed that YouTube Shorts content is algorithmically curated towards short clips that generate high engagement. We therefore collected the data by manually selecting every video, simply by watching all the Shorts before including them in our playlists, which we later scraped.

## ASR Artifacts (Whisper)
Whisper errors occur especially in:
- rapid speech
- overlapping (music/speech or engine noise) and weak audio signals
- complex multilingual utterances
- background noise
- broadcast accents
These propagate into parsing and event extraction.

## Spoken-Language Structure
Short-form videos contain:
- disfluencies
- incomplete clauses
- rhetorical fragments
- narrator/editor switching
Such structures reduce dependency parsing accuracy.

## Parser Limitations
Syntactic parsers trained on formal text often fail on:
- colloquial grammar
- missing subjects
- dramatic transitions
- Shorts-style fragmented narration

## Sentiment/ABSA Domain Shift
ABSA models trained on review-like data struggle with:
- journalistic neutrality
- implied sentiment
- indirect agency framing

## Channel-Level Narrative Disparities
Different outlets emphasize different viewpoints, causing:
- asymmetric distributions of aspects
- topic clustering
- diverging linguistic patterns

