# Tier-1 Audio Event Annotation Cheat Sheet (Span-level, Multilabel)

**Core labels (we use for Label Studio span annotations):**
- **Speech** – normal talking, interviews, reporter VO. *Exclude* rhythmic chanting.
- **Shout/Scream** – brief, high-intensity vocal bursts (merge “Shout” and “Screaming” at evaluation time). You may keep **Speech** if talking continues.
- **Siren** – emergency sirens (police/ambulance/fire); tonal rise/fall or multi-tone.
- **Chant** – rhythmic/unison slogans (“From the river…”, call-and-response). Distinct from normal speech.
- **Music** – any non-diegetic or diegetic music. **Set role attribute**:
  - **Music_FG** = foreground (no voice-over; “no-comment” B-roll)
  - **Music_BG** = background (under speech)
- **Crowd_noise** – non-chant crowd ambience: murmur, cheering, applause, booing.

**General rules**
- Spans may **overlap** (e.g., Speech + Music_BG).
- Long continuous sections can be a **single span** (no micro-slicing).
- **Chant vs Speech:** rhythmic/unison --> *Chant*; otherwise *Speech*.
- **Shout/Scream vs Speech:** short high-intensity --> *Shout/Scream*; keep *Speech* if talking continues.
- **Music role:** set **FG** only when music is the foreground (no speech). Otherwise **BG**. If unclear, leave unset—converter defaults to BG.

**Positive / Negative examples**
- *Speech (✓):* anchor reads; field reporter; interview answer.  *(✗):* chant; singing.
- *Shout/Scream (✓):* “Hey!”, panic scream spike.  *(✗):* normal loud speech; chant.
- *Siren (✓):* police/ambulance siren, multi-tone cycles.  *(✗):* microwave/beep alarms.
- *Chant (✓):* rhythmic slogans in unison.  *(✗):* crowd murmurs, individual speech.
- *Music_FG (✓):* montage/no-comment scene with soundtrack only.  *(✗):* music under talking.
- *Music_BG (✓):* music under voice-over or interview.  *(✗):* no music present.
- *Crowd_noise (✓):* cheering, applause, booing, indistinct crowd.  *(✗):* chant, solo speaker.

**Quality Checks**
- Prefer **precise span boundaries**, but ±0.2s tolerance is fine.
- For ambiguous cases, choose the **broader** label (e.g., Crowd_noise over applause vs cheer).
- If unsure about Music role, set **BG** (converter treats unspecified as BG).

**Export & conversion**
- Export **Label Studio JSON**.
- Convert to windows (1.0s, hop 1.0s) with **union/majority** as agreed.
- Default output labels: `Speech, Shout_Scream, Siren, Chant, Music_FG, Music_BG, Crowd_noise`.
- “Shout” and “Screaming” are **collapsed** to `Shout_Scream` by default in the converter.

**Common confusions**
- Chant vs Speech --> rhythm/unison; if mixed, both can be present.
- Siren under crowd noise --> temporal smoothing (≈3s) helps.
- Applause may be brief; label if clearly audible.

**Operating points**
- Screening (human-in-the-loop): choose **high recall** thresholds (R≥0.85).
- Auto-tagging: choose **high precision** thresholds (P≥0.80).

**What we report in the paper**
- We use **PR-AUC/AP** per label; F1 at selected operating points.
- We include **cross-outlet** evaluation and an ablation for “no Music label” to show schema value.
