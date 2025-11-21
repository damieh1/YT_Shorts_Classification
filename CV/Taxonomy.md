# YT_Shorts_Classification

# Scene Classification Taxonomy (MPED – Shorts)

The proposed taxonomy defines **scene types** for classifying segmented video shots (10–15s) from YouTube Shorts and similar short-form content.  
The goal is to achieve a **clear visual distinction** and **feasible detection** for affect mobilization in computer vision tasks.

*Work in progress...*

---

## 1. Conflict & Security
Scenes showing **violence, destruction, or organized force**.

- **Active combat:** explosions, firefights, airstrikes, rubble falling, gunfire  
- **Military activity:** soldiers with weapons, armored vehicles, tanks, checkpoints  
- **Aftermath of attacks:** destroyed buildings, burned cars, debris, smoke plumes  
- **Policing / security enforcement:** riot police lines, arrests, tear gas, batons  
- **Detentions / searches:** handcuffs, police restraining or pushing civilians  

---

## 2. Protest & Mobilization
Scenes of **collective civic or political action**.

- **Street protests / rallies:** marches, chanting crowds, placards, megaphones  
- **Protest encampments / sit-ins:** tents on campus lawns, occupations, blockades  
- **Solidarity actions:** candlelight vigils, symbolic flag displays, human chains  
- **Demonstrations with banners:** protest slogans, “Free Palestine,” “Ceasefire Now”  
- **Counter-demonstrations:** opposing groups, police separating crowds  

---

## 3. Humanitarian & Civilian Life
Scenes showing **non-military human conditions**.

- **Refugee / aid camps:** rows of tents, UN/NGO logos, food/aid distribution lines  
- **Medical settings:** ambulances, doctors treating injured, hospital interiors  
- **Civilian daily life:** markets, homes, children playing, family gatherings  
- **Displacement:** families carrying belongings, queues at aid points, shelters  
- **Humanitarian relief:** Red Cross/Red Crescent, volunteers, donation handouts  

---

## 4. Institutional & Media Settings
Scenes of **formal communication or controlled environments**.

- **Political speeches / press conferences:** podiums, leaders, national flags  
- **Studio interviews / talk shows:** panel discussions, guests at a table, microphones  
- **Educational contexts:** classrooms, academic lectures, university debates  
- **Official ceremonies:** government signings, assemblies, formal events  
- **Media coverage:** reporters on location, live news desks, anchors in studios  

---

## 5. Cross-Cutting Attributes
These are **not standalone scene types**, but important **features that can occur in any category**.  
They should be treated as **secondary tags** in annotation or model output.

### Flags & Emblems
Visual symbols of identity, belonging, or political alignment.  
   - National flags: Israeli, Palestinian, American, Turkish, etc.  
   - Political movement flags: Hamas, Hezbollah, Fatah, Islamic Jihad, student groups  
   - Organizational emblems: UN, Red Cross/Red Crescent, Amnesty International  
   - Flag displays: waved in crowds, draped over bodies, hanging from balconies, painted on walls  
   - Symbolic clothing: keffiyehs, IDF uniforms, armbands with insignia  

### Posters / Flyers / Inscriptions
Printed or written text meant to convey a message.  
   - Protest placards: “Free Palestine,” “Ceasefire Now,” “Stop the Occupation”  
   - Flyers/leaflets: calls for demonstrations, boycott/divestment campaigns (BDS)  
   - Wall inscriptions / graffiti: slogans, political messages, murals  
   - Banners: stretched across streets, hung from buildings, behind podiums  
   - Digital posters / screenshots: social media announcements or infographics shown in videos  

### Memorial / Mourning Symbols
Objects and rituals connected to grief, tribute, or remembrance.  
   - Candlelight vigils: groups holding candles, night-time gatherings  
   - Flowers and wreaths: laid at sites of attacks or memorial walls  
   - Portraits/photos of victims: displayed in rallies, held up by family members  
   - Black clothing, armbands, ribbons: visual markers of mourning  
   - Shrines/altars: improvised displays with photos, candles, teddy bears  
   - Moments of silence: crowds standing still, heads bowed  


---

### Notes
- **Every segment** should be classified into **exactly one of the four core categories** (1–4).  
- **Attributes (5)** can be assigned in addition to the main category, providing extra semantic detail (only if feasible).


## Annotation Protocol

1. **Segmentation:**  
   - Videos are split into 10–15 second **segments**.  
   - Each segment must receive **exactly one primary scene label** (1–4).  
   - Cross-cutting attributes (5) may be assigned in addition, if present.  

2. **Segment-Level Example:**  
   - *Scene:* Street protest with Palestinian flags -->> `Protest & Mobilization`  
   - *Attributes:* Flags -->> `Flags & emblems`  

3. **Video-Level Aggregation:**  
   - A video may contain **multiple scene types**, depending on its segments.  
   - The video’s overall label is the **union of all scene labels** across segments.  
   - Attributes are also aggregated at the video level.  

   **Example:**  
   - Segments: `[Conflict & Security]`, `[Protest & Mobilization + Flags]`, `[Institutional & Media Settings + Posters]`  
   - Video-level summary:  
     - **Scenes:** `Conflict & Security`, `Protest & Mobilization`, `Institutional & Media Settings`  
     - **Attributes:** `Flags & emblems`, `Posters / inscriptions` (only if feasible).



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

