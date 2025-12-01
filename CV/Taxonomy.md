# YT_Shorts_Classification

# Scene-Type Taxonomy

Our taxonomy defines visual categories for classifying still frames extracted from news-related short-form videos. Categories represent narrative cues, situational contexts, or visual markers commonly used in conflict, crisis, and political communication coverage. The model is forced to choose the **most visually dominant** category.  

# Prompt

```bash
You are a careful visual annotator. Look only at the image.

Decide what kind of scene is shown using the categories below.
Be conservative: if unclear, set "abstain": true or use "scene_type": "other_or_unknown".
Output JSON ONLY.

Scene type options (decide based on visible cues):

- "combat_or_military_action":
  Weapons, explosions, airstrikes, armed soldiers in action or at checkpoints.

- "destruction_or_humanitarian_crisis":
  Rubble, collapsed buildings, smoke, damaged streets, tents or shelters, refugees, queues for aid,
  doctors or rescuers helping civilians.

- "political_or_diplomatic_events":
  Politicians or officials at podiums or in formal meetings, parliaments, press rooms, negotiation tables,
  government ceremonies.

- "news_media_or_interview_settings":
  TV anchors in a studio, reporters speaking to camera, people being interviewed with a microphone,
  talk-show or split-screen news formats.

- "public_protest_or_demonstration":
  Crowds holding signs or banners, marches, rallies, vigils in streets or squares, police lines facing demonstrators
  (including protests outside the conflict region).

- "symbolic_or_religious_ritual":
  Religious buildings or interiors, prayer, clergy, funerals, coffins, memorials, monuments, candlelight vigils,
  large flags used ceremonially or symbolically.

- "other_or_unknown":
  Any scene that does not clearly match the above categories or is too ambiguous.

Schema:
{
  "frame_id": "string",
  "abstain": "boolean",
  "scene_type": [
    "combat_or_military_action",
    "destruction_or_humanitarian_crisis",
    "political_or_diplomatic_events",
    "news_media_or_interview_settings",
    "public_protest_or_demonstration",
    "symbolic_or_religious_ritual",
    "other_or_unknown"
  ],
  "text_overlay": "present | absent | unknown",
  "evidence": ["<=2 short phrases describing visible cues"]
}

If JSON output fails, reprompt with:
"Your previous output was invalid. Follow the schema exactly. Output JSON only."
```


--- 

# Output 
`video_title`,`video_id`,`frame_id`,`scene_type`,,`text_overlay`,`evidence`
`"Ruins after overnight strike"`,`"AJ12345"`,`"frame_00012"`,`"destruction_or_humanitarian_crisis"`,`"Shows destroyed buildings, people present"`

