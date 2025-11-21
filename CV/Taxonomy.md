# YT_Shorts_Classification

# Scene-Type Taxonomy

Our taxonomy defines visual categories for classifying still frames extracted from news-related short-form videos. Categories represent narrative cues, situational contexts, or visual markers commonly used in conflict, crisis, and political communication coverage. Annotators should choose the **most visually dominant** category.  
If unclear, use `"other_or_unknown"` and/or `"abstain": true`.

---

## 1. combat_or_violence
Scenes showing active fighting, explosions, gunfire, physical aggression, or visible casualties.

## 2. aftermath_or_destruction
Visuals depicting damage or destruction after an attack or disaster: collapsed buildings, rubble, smoke, burned vehicles.

## 3. military_presence
Soldiers, armored vehicles, checkpoints, bases, uniforms, or military hardware **without active combat**.

## 4. mourning_or_funeral
Scenes depicting grief, funerals, memorial gatherings, vigils, coffins, or symbolic acts of mourning.

## 5. refugee_or_displacement
Forced civilian movement, displaced families, temporary shelters, border crossings, evacuation scenes.

## 6. children_or_humanitarian_aid
Children in conflict settings, humanitarian assistance, food distribution, medical care, UN/NGO activity.

## 7. protest_or_demonstration
Crowds holding signs, chanting, marching, occupying public spaces, or engaging in civil protest.

## 8. diplomacy_or_politics
Political leaders, press conferences, official meetings, negotiation tables, government ceremonies.

## 9. media_or_interview
Journalists reporting on scene, interviews, news crews, studio stand-ups, or reporters speaking to camera.

## 10. religious_or_sacred_scene
Religious rituals, prayer, mosques, churches, synagogues, sacred objects, or ceremonial worship.

## 11. national_or_identity_symbol
Flags, national emblems, movement symbols, uniforms, insignia, or public displays of identity.

## 12. memorial_or_commemoration
Monuments, remembrance events, plaques, candlelight vigils, or commemorative public gatherings.

## 13. civilian_life_under_crisis
Everyday civilian activities occurring in conflict or crisis contexts: queues for supplies, families in damaged homes, daily life under duress.

## 14. diaspora_or_global_reaction
Protests, vigils, or mobilization **outside the conflict zone**, often in diaspora communities or international cities.

## 15. other_or_unknown
Scenes with ambiguous, unclear, or insufficient visual information.  
Used when classification confidence is low or content falls outside defined categories.


```bash
You are a careful visual annotator. Look only at the image.
Identify what kind of scene is shown, using the categories listed below.
Be conservative: if unclear, set "abstain": true or label "other_or_unknown".
Output JSON ONLY.

Schema: {
  "frame_id": "string",
  "abstain": "boolean",
  "scene_type": [
    "combat_or_violence",
    "aftermath_or_destruction",
    "military_presence",
    "mourning_or_funeral",
    "refugee_or_displacement",
    "children_or_humanitarian_aid",
    "protest_or_demonstration",
    "diplomacy_or_politics",
    "media_or_interview",
    "religious_or_sacred_scene",
    "national_or_identity_symbol",
    "memorial_or_commemoration",
    "civilian_life_under_crisis",
    "diaspora_or_global_reaction",
    "other_or_unknown"
  ],
  "violence_level": "none | implied | explicit | unknown", # We skipped reorting on this results duo to poor predicitons
  "text_overlay": "present | absent | unknown",
  "evidence": ["<=2 short phrases describing visible cues"]
}

If JSON output fails, reprompt with:
“Your previous output was invalid. Follow the schema exactly. Output JSON only.”
```
## Output 
`video_title`,`video_id`,`frame_id`,`scene_type`,`violence_level`,`text_overlay`,`evidence`
`"Ruins after overnight strike"`,`"AJ12345"`,`"frame_00012"`,`"aftermath_or_destruction"`,`"implied","present"`,`"Shows destroyed buildings, people present"`

