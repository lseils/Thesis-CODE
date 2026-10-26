# Apartment pipeline — playbook

How to turn programmatic apartment boxes in Rhino into walled apartments with a fin facade, front
doors, interior doors and windows. Written so that either `apartment_pipeline.py` can run it, or
Claude (via rhinomcp) can follow it step by step.

---

## 1. Input the model must have

| Thing | Convention |
|---|---|
| Room boxes | One sublayer per room type: `APARTMENTS::Living`, `Bedroom`, `Kitchen`, `Bathroom`, `Shared` |
| Apartments | Each apartment is a **Rhino group** of its room boxes. Shared boxes are **ungrouped**. |
| Grid | Boxes sit on a regular plan grid (default **6 ft**) laid along an arc (straight bars are fine — treated as a huge radius). Depths and lengths are multiples of 6 ft. |
| Floors | Every **10 ft**. A bar may slope gently; floors follow one sloped plane per bar. |
| Facade module | Layer `Layer 02`; the fin pattern is in `CONFIG["fin_pattern"]` (11 fins, 2" deep, 60" repeat). |

Everything numeric lives in `CONFIG` at the top of the script.

---

## 2. The rules (decisions already made)

**Clean-up**
- A group that is not a full apartment (missing Living, Kitchen or Bathroom) and any room box in no
  group becomes **ungrouped Shared**.
- Boxes are rebuilt as **exact wedges** on the arc grid: ends on radial lines, faces on the arcs,
  floors on the bar's sloped plane, 10 ft apart. Layer, group, name and colour are kept.

**Walls** (`WALLS::Exterior 12in / Party 6in / Interior 4in`)
- **Shared space counts as outside.** Shared boxes get no walls.
- Room ↔ outside or Shared → **Exterior 12"**, placed **inside the room** (outer face on the grid line).
- Rooms of **different apartments** → **Party 6"**, centred on the line.
- Rooms of the **same apartment** → **Interior 4"**, centred on the line.
- Walls run floor plane → ceiling plane.
- Junctions: along-bar (arc) walls run through; radial walls stop at their face.
  - An arc wall that ends at a junction extends only as far as the radial wall **on each half of its
    thickness** needs (trim the rest — otherwise stubs poke into rooms and outside).
  - Where an arc wall changes thickness at a junction, fill the pocket next to the radial wall
    with a small filler on the radial wall's layer.

**Facade** (`Layer 02`)
- Fins on every exterior face of every room (faces onto open air **and** onto Shared), 2" outside the
  wall face, floor plane → ceiling plane.
- Each continuous face is its own run of the pattern, starting at one end; partial fins at the end are cut.
- **2"×2" post** at each outside corner; at inside corners the radial face's fins are pulled back 2".

**Interior doors** (`Frames`, `Doors`, in the apartment group)
- **One door per room**, leading back toward the apartment's Living room (BFS tree from Living).
- 3'×7' clear, 2" jambs + head as deep as the wall, 1¾" leaf centred.
- Frame edge **6" from the perpendicular wall face** at one end; if that leaves less than 6" at the
  far end, **centre** it.

**Front doors**
- One per **Living room**, on its longest exterior face, **preferring a face onto Shared**.
- Same size; frame runs through 12" wall + 2" fins (14" deep); leaf centred in the wall; fins cut
  back to the head. 6" rule as above.

**Windows** (`Frames`, `Glass`)
- **Living rooms and Bedrooms**, on the longest exterior face, **preferring open air**.
- 3'×5' clear, sill 3' above floor, 2" jambs/head/sill through wall + fins, ¼" glass centred.
- One window 6" from the perpendicular wall; a face **2 or more grid cells long** (12 ft+) gets a
  second one 6" from the other end. **Count cells, not feet** — a 6 ft cell on the outside of the
  curve measures ~6.04 ft.
- Keep ≥ 6" clear of the front door; if a Living room has no space for a window, try its other faces;
  if still none, put **one window in that apartment's Kitchen**.

---

## 3. Running it

### In Rhino
`RunPythonScript` → `apartment_pipeline.py` → choose a stage → choose dry run or not.

### From Claude (rhinomcp)
Rhino cannot read Claude's session folder — copy the script to a path Rhino can see first (e.g.
`%TEMP%\apartment_pipeline.py`), then:

```python
__name__ = "__main__"
PIPELINE_ARGS = dict(stage="report")                       # always start here
exec(open(r"C:\Users\<you>\AppData\Local\Temp\apartment_pipeline.py").read())
```

`PIPELINE_ARGS` keys: `stage`, `bars` (names from `report`, e.g. `["B1"]`, or `None` for all),
`levels` (e.g. `[0, 1]` or `None`), `dry_run`.

### Order

| # | Stage | Writes? | Notes |
|---|---|---|---|
| 1 | `report` | no | Bars, arc centre/radius, grid snap error, layer counts. Snap error should be ~0 after `wedges`. |
| 2 | `normalize` | yes | Odd groups / stray rooms → Shared. |
| 3 | `wedges` | yes | Prints how far edges and floors move. Expect ≤ 0.2 ft plan, ≤ 0.5 ft height. |
| 4 | `walls` | yes | Run per bar. |
| 5 | `facade` | yes | Run per bar **and per pair of levels** (≈2,500 fins per call). |
| 6 | `doors` | yes | Per bar. |
| 7 | `frontdoors` | yes | Per bar. |
| 8 | `windows` | yes | Per bar and per pair of levels. |
| 9 | `verify` | no | Layer counts, all walls closed solids. |

Run every write stage with `dry_run=True` first and compare its numbers with what you expect.

### Reference numbers (the model this was built on: 825 boxes, 2 bars, 108 apartments)

| Stage | Result |
|---|---|
| walls (before junction fixes) | B2: 396 EXT / 112 PARTY / 275 INT · B1: 396 / 148 / 302 |
| walls (after fixes) | 284 overhang trims, 82 fillers |
| facade | 20,530 fins incl. 326 corner posts |
| interior doors | 427 (535 rooms − 108 Living rooms) |
| front doors | 108 (29 onto Shared, 79 onto open air) |
| windows | 409 (390 Living/Bedroom + 19 Kitchen); 10 Bedrooms have no exterior wall |

---

## 4. Pitfalls (each one happened)

1. **Empty boolean result ≠ "nothing left".** `Brep.CreateBooleanDifference` returns an **empty list**
   when the breps don't touch. Treating that as the result deleted 40 walls and 50 fins. Use
   `safe_difference()` — and after any cutting stage check that wall counts did **not go down**.
2. **Cut only the floor you're on.** A door cutter reaches 1 ft below the floor; without a level
   filter it notched the wall on the floor below instead of cutting the right one.
3. **Bounding boxes are not geometry.** Overlap tests on bounding boxes produce false "clashes";
   confirm with an actual boolean intersection volume.
4. **Two windows by cell count, not length** (see above).
5. **Junction extensions per half-thickness** (see Walls). The first version left 268 stubs.
6. **Tool time limits.** Large stages time out over rhinomcp, but Rhino keeps running and usually
   finishes. After a timeout, **wait, then re-read layer counts** before doing anything else — never
   re-run blindly.
7. **Script errors don't always roll back.** Build everything in memory, check it, and only then add
   to the document (the script does this). If something partial lands, `sc.doc.Undo()` — one call per
   undo, and re-check counts after each.
8. **Closures in loops** (Python): pass loop values in as default arguments or helper parameters, or
   later iterations overwrite them (this once placed the north bar's doors with the south bar's grid).

---

## 5. Verification checklist (after each write stage)

- [ ] Layer counts: walls never decrease during door/window stages; fins only increase (split around
      openings).
- [ ] All walls are closed solids (`verify`).
- [ ] No wall overlaps another (sample: vertex-inside test between neighbours, confirm with boolean volume).
- [ ] Fins: none inside walls, rooms or other fins; every fin as tall as its floor.
- [ ] Doors/windows: every frame/leaf/pane clash-free; door leaves 3'×7', panes 3'×5';
      jamb-to-perpendicular-wall gap 6" (or centred).
- [ ] No two panes closer than one window width.
- [ ] Look at it: top view of one floor + a perspective close-up.

---

## 6. Doing it by hand with Claude (no script)

Ask Claude to follow sections 2–5 in order, one stage per turn, per bar, reporting counts against
section 3 and running the checklist in section 5 before moving on. The script's functions are the
reference implementation for each rule; Claude can paste the relevant function into
`execute_rhinoscript_python_code` instead of rewriting it.
