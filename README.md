# YT_Shorts_Classification

# Scene Classification Taxonomy (MPED – Shorts)

The proposed taxonomy defines **scene types** for classifying segmented video shots (10–15s) from YouTube Shorts and similar short-form content.  
It balances **clear visual distinctions** and **feasible detection** for affect mobilization in Computer Vision tasks.  

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

- **Flags & emblems:** Israeli, Palestinian, organizational, or political symbols  
- **Posters / flyers / inscriptions:** protest signs, BDS material, OCR-relevant text  
- **Memorial / mourning symbols:** candles, flowers, portraits, tributes  

---

### Notes
- **Every segment** should be classified into **exactly one of the four core categories** (1–4).  
- **Attributes (5)** can be assigned in addition to the main category, providing extra semantic detail.    
