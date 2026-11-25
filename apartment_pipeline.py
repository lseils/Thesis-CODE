# -*- coding: utf-8 -*-
"""
apartment_pipeline.py -- turn programmatic apartment boxes into walled, fenestrated apartments.

Runs inside Rhino (Rhino 7/8, IronPython 2.7 or CPython 3 via rhinoscriptsyntax).

Input model
-----------
* One sublayer per type: Apartments::Living / Bedroom / Kitchen / Bathroom / Shared / Public / Vertcirc.
* Each apartment is a Rhino group holding its room boxes; each public space is a group of Public boxes.
* Shared counts as outside.  A Shared space grouped with an apartment becomes Living (still grouped)
  if apartment rooms close it in on every side; if any side touches the outside, another Shared
  space or a core it is ungrouped and stays Shared (see resolve_shared).  Vertcirc = multi-storey stair / elevator core (left as it is).
* Public groups are one open space each: walls round the edge, none between their own boxes.
* Boxes sit on a regular grid (default 8 ft) laid out along a gently curving arc (a straight bar
  also works - it is treated as an arc with a huge radius), floors every FLOOR_HEIGHT.

Stages (run in this order; each one can be run on its own, per bar and per pair of levels)
------
  report      read-only: bars, grid fit, counts of everything each stage would make
  normalize   odd groups / stray rooms -> Shared, ungrouped
  wedges      snap every box onto the arc grid as an exact wedge, floors on one sloped plane per bar
  walls       exterior / party / interior walls + junction fixes (fillers, overhang trims)
  facade      fin facade on every exterior face, corner posts, inside-corner clipping
  panels      folded panels on the long outer sides of the apartments (after OTHER::Facade)
  balconies   walkways for apartments whose Living / Kitchen cannot reach a core through Shared space
  doors       one interior door per room (tree back to the Living room)
  frontdoors  one front door per Living room (prefers a face onto Shared space)
  windows     Living + Bedroom windows; Kitchen window where the Living room has no room for one
  verify      read-only checks (solids, counts, nothing deleted)

Usage
-----
* Rhino: RunPythonScript this file -> pick a stage from the list.
* Scripted / Claude via rhinomcp:
      PIPELINE_ARGS = dict(stage="walls", bars=["N"], levels=None, dry_run=True)
      exec(open(r"<path>/apartment_pipeline.py").read())
  Big stages (facade, windows) should be run per bar and per pair of levels to stay under
  tool time limits. Every write stage prints the layer counts before/after.
"""
from __future__ import print_function
import heapq
import math
import random
from collections import defaultdict, Counter, deque

import Rhino
import System.Drawing
import rhinoscriptsyntax as rs
import scriptcontext as sc
from Rhino.Geometry import (Point3d, Vector3d, Plane, Brep, BoundingBox, Interval, Arc, Circle,
                            ArcCurve, LineCurve, PolyCurve, Surface, Extrusion, BrepSolidOrientation)

# --------------------------------------------------------------------------------------------
# CONFIG - every number the process depends on (feet)
# --------------------------------------------------------------------------------------------
IN = 1.0 / 12.0
CONFIG = dict(
    room_layers=dict((k, "Apartments::" + k) for k in
                     ("Living", "Bedroom", "Kitchen", "Bathroom", "Shared", "Public", "Vertcirc")),
    only_bars_with="Public",  # work only on bars holding this type (= the newest generation);
                              # None = every bar in the model
    grid=8.0,                 # plan grid (ft) along and across the bar
    floor_height=10.0,
    bar_gap=0.3,              # boxes whose sides are closer than this belong to the same bar
    wall_t=dict(EXT=1.0, PARTY=0.5, INT=4 * IN),
    wall_layers=dict(EXT="WALLS::Exterior 12in", PARTY="WALLS::Party 6in", INT="WALLS::Interior 4in"),
    wall_colors=dict(EXT=(40, 40, 40), PARTY=(200, 70, 60), INT=(230, 200, 120)),
    shared_is_outside=True,   # walls facing Shared are exterior walls with facade
    fin_layer="FACADE::Fins",
    fin_depth=2 * IN,
    fin_direction="horizontal",   # or "vertical" (the original, with corner posts)
    fin_module=60 * IN,       # pattern repeat
    fin_pattern=[(0, 2), (4, 8), (10, 16), (18, 20), (22, 26), (28, 32), (34, 36), (38, 44),
                 (46, 48), (50, 52), (54, 58)],   # inches inside one module
    panels_on=True,           # folded panels on the long outer sides; fins skip those walls
    panel_layer="FACADE::Panels",
    panel_seed=1,             # change for a different arrangement
    panel_slits=True,         # cut horizontal slits (fin_pattern on its side); False = plain panels
    panel_t=0.5,              # panel thickness
    panel_offsets=(0.0, 0.5, 1.0),   # how far the bottom, seam and top edges may stand out
    panel_seam=(2, 8),        # seam height range above the floor, whole feet
    panel_seam_step=(1, 4),   # how much the seam may rise or fall along one panel
    panel_lengths=((2, 22), (4, 17), (5, 6), (3, 2), (1, 1)),   # (grid cells, weight)
    panel_gap=0.02,           # hairline behind the panels, so they never touch the wall
    balconies_on=True,        # balconies link apartments to egress where Shared space cannot
    balcony_entry=("Living", "Kitchen"),  # rooms that may be entered; add "Bedroom" for fewer balconies
    slab_layer="BALCONIES::Slabs", rail_layer="BALCONIES::Panels",
    balcony_depth=4.0, balcony_t=1.0,     # slab; its top is the floor level
    rail_below=2.5, rail_above=3.5,       # outer panel: below the slab's underside / above its top
    rail_offsets=(0.0, 0.5), rail_seam=(2, 5), rail_seam_step=(1, 2),
    balcony_ends=True,        # slatted guard across both ends (not in OTHER)
    frame=2 * IN,
    clearance=6 * IN,         # door/window frame edge to the perpendicular wall face
    door_w=3.0, door_h=7.0, leaf_t=1.75 * IN,
    win_w=3.0, win_h=5.0, sill=3.0, glass_t=0.25 * IN,
    frames_layer="OPENINGS::Frames", doors_layer="OPENINGS::Doors", glass_layer="OPENINGS::Glass",
    window_rooms=("Living", "Bedroom"),
    two_windows_min_cells=2,  # a face this many grid cells long gets two windows
    kitchen_fallback=True,    # Living room with no window -> one window in its Kitchen
)
E = 0.1          # cutter overreach past faces (ft)
TOL = sc.doc.ModelAbsoluteTolerance


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------
def brep_of(geo):
    return geo.ToBrep() if isinstance(geo, Extrusion) else geo


def layer_objs(path):
    idx = sc.doc.Layers.FindByFullPath(path, -1)
    if idx < 0:
        return []
    return list(sc.doc.Objects.FindByLayer(sc.doc.Layers[idx]) or [])


def ensure_layer(path, color=(0, 0, 0)):
    """Make sure this exact full path exists.  (rs.IsLayer also matches a short name elsewhere in the
    tree: "FACADE" once matched BUILDING::Facade and the fins landed on the wrong layer.)"""
    layers = sc.doc.Layers
    parts = path.split("::")
    parent = None
    for k in range(len(parts)):
        full = "::".join(parts[:k + 1])
        idx = layers.FindByFullPath(full, -1)
        if idx < 0:
            new = Rhino.DocObjects.Layer()
            new.Name = parts[k]
            new.Color = System.Drawing.Color.FromArgb(*color)
            if parent is not None:
                new.ParentLayerId = layers[parent].Id
            idx = layers.Add(new)
            if idx < 0:
                raise RuntimeError("could not make layer " + full)
        parent = idx


def new_group(name):
    """Name of a fresh, empty group."""
    while sc.doc.Groups.Find(name) >= 0:
        name += "'"
    sc.doc.Groups.Add(name)
    return name


def add_brep(brep, layer, group=None, attr=None):
    a = attr.Duplicate() if attr else Rhino.DocObjects.ObjectAttributes()
    if layer:
        a.LayerIndex = sc.doc.Layers.FindByFullPath(layer, -1)
        if a.LayerIndex < 0:
            raise RuntimeError("no such layer: " + layer)
    if group:
        gi = sc.doc.Groups.Find(group)
        if gi >= 0:
            a.AddToGroup(gi)
    return sc.doc.Objects.AddBrep(brep, a)


def closed_prism(B, Tp):
    """Solid from 4 bottom + 4 top corner points (planar faces)."""
    faces = [Brep.CreateFromCornerPoints(B[0], B[1], B[2], B[3], TOL),
             Brep.CreateFromCornerPoints(Tp[0], Tp[1], Tp[2], Tp[3], TOL)]
    for n in range(4):
        m = (n + 1) % 4
        faces.append(Brep.CreateFromCornerPoints(B[n], B[m], Tp[m], Tp[n], TOL))
    if any(f is None for f in faces):
        return None
    j = Brep.JoinBreps(faces, TOL)
    if not j or len(j) != 1:
        return None
    br = j[0]
    if br.SolidOrientation == BrepSolidOrientation.Inward:
        br.Flip()
    return br


def counts():
    c = dict((k, len(layer_objs(v))) for k, v in CONFIG["wall_layers"].items())
    for k in ("fin_layer", "panel_layer", "slab_layer", "rail_layer", "frames_layer", "doors_layer", "glass_layer"):
        c[k.split("_")[0]] = len(layer_objs(CONFIG[k]))
    return c


def safe_difference(target, cutter):
    """Boolean difference that never deletes: returns (pieces, changed).
    Rhino returns an EMPTY list when the two breps don't actually touch - that means 'no cut',
    not 'nothing left'. Treating it as a result deleted 40 walls and 50 fins in the original run."""
    res = Brep.CreateBooleanDifference(target, cutter, TOL)
    if res is None:
        raise RuntimeError("boolean difference failed")
    if len(res) == 0:
        return [target], False
    if abs(target.GetVolume() - sum(r.GetVolume() for r in res)) <= 1e-4:
        return [target], False
    if not all(r.IsSolid for r in res):
        raise RuntimeError("boolean difference produced an open brep")
    if any(r.SolidOrientation == BrepSolidOrientation.Inward for r in res):
        raise RuntimeError("boolean difference produced an inside-out brep")   # the caller retries
    vol = target.GetVolume()
    if sum(r.GetVolume() for r in res) > vol + 1e-4:
        raise RuntimeError("boolean difference added volume")
    return list(res), True


# --------------------------------------------------------------------------------------------
# reading boxes, bars and the arc grid
# --------------------------------------------------------------------------------------------
class Box(object):
    def __init__(self, ro, kind):
        self.id = ro.Id
        self.kind = kind
        self.brep = brep_of(ro.Geometry)
        vs = sorted([v.Location for v in self.brep.Vertices], key=lambda p: p.Z)
        self.bot, self.top = vs[:4], vs[4:8] if len(vs) >= 8 else vs[4:]
        self.zmin = vs[0].Z
        self.cx = sum(p.X for p in self.bot) / len(self.bot)
        self.cy = sum(p.Y for p in self.bot) / len(self.bot)
        g = rs.ObjectGroups(ro.Id)
        self.group = g[0] if g else None
        self.bb = self.brep.GetBoundingBox(True)
        self.zb = sum(p.Z for p in self.bot) / len(self.bot)
        self.zt = sum(p.Z for p in self.top) / len(self.top)
        self.corners = [(p.X, p.Y) for p in
                        sorted(self.bot, key=lambda p: math.atan2(p.Y - self.cy, p.X - self.cx))]


def read_boxes():
    out = []
    for kind, path in CONFIG["room_layers"].items():
        for ro in layer_objs(path):
            out.append(Box(ro, kind))
    return out


def plan_gap(a, b):
    """Plan distance between two boxes (separating axis; negative = overlapping)."""
    best = -1e9
    for c in (a.corners, b.corners):
        for k in range(4):
            dx, dy = c[(k + 1) % 4][0] - c[k][0], c[(k + 1) % 4][1] - c[k][1]
            L = math.hypot(dx, dy)
            if L < 1e-9:
                continue
            nx, ny = dy / L, -dx / L
            pa = [q[0] * nx + q[1] * ny for q in a.corners]; pb = [q[0] * nx + q[1] * ny for q in b.corners]
            best = max(best, min(pb) - max(pa), min(pa) - max(pb))
    return best


def split_bars(boxes):
    """Connected groups of boxes in plan (union-find on real side-to-side contact)."""
    parent = list(range(len(boxes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    G = 20.0
    pad = CONFIG["bar_gap"]
    buckets = defaultdict(list)
    for n, b in enumerate(boxes):
        for x in range(int((b.bb.Min.X - pad) // G), int((b.bb.Max.X + pad) // G) + 1):
            for y in range(int((b.bb.Min.Y - pad) // G), int((b.bb.Max.Y + pad) // G) + 1):
                buckets[(x, y)].append(n)
    for lst in buckets.values():
        for a in range(len(lst)):
            ba = boxes[lst[a]].bb
            for c in range(a + 1, len(lst)):
                bc = boxes[lst[c]].bb
                if (ba.Min.X - pad <= bc.Max.X and bc.Min.X - pad <= ba.Max.X and
                        ba.Min.Y - pad <= bc.Max.Y and bc.Min.Y - pad <= ba.Max.Y and
                        min(ba.Max.Z, bc.Max.Z) - max(ba.Min.Z, bc.Min.Z) > -1.0 and
                        find(lst[a]) != find(lst[c]) and plan_gap(boxes[lst[a]], boxes[lst[c]]) < pad):
                    ra, rc = find(lst[a]), find(lst[c])
                    if ra != rc:
                        parent[ra] = rc
    groups = defaultdict(list)
    for n in range(len(boxes)):
        groups[find(n)].append(boxes[n])
    need = CONFIG["only_bars_with"]
    bars = [bs for bs in groups.values() if need is None or any(b.kind == need for b in bs)]
    # name bars by position: west-most first -> B1, B2, ...
    bars.sort(key=lambda bs: (min(b.cx for b in bs), min(b.cy for b in bs)))
    return dict(("B%d" % (k + 1), bs) for k, bs in enumerate(bars))


ROOMS = ("Living", "Bedroom", "Kitchen", "Bathroom")


def resolve_shared(boxes):
    """Decide what Shared boxes grouped with an apartment are.  Touching Shared boxes of one apartment
    are judged together as one space:
      every side an apartment room (own or a neighbour's)  -> Living, stays grouped
      any side outside, another Shared space or a core     -> Shared, ungrouped
      otherwise (closed in, but partly by Public)          -> left alone, reported
    Changes the Box objects only; stage_normalize writes it to the model.  Returns a list of
    (box, group, verdict)."""
    homes = set(b.group for b in boxes if b.kind == "Living" and b.group)
    todo = [b for b in boxes if b.kind == "Shared" and b.group in homes]
    if not todo:
        return []
    g = Grid(boxes)
    everything = {}
    for d in (g.cell, g.shared, g.core):
        everything.update(d)

    def neighbours(b):
        for i in range(b.i0, b.i1):
            for j in range(b.j0, b.j1):
                for k in ((b.L, i + 1, j), (b.L, i - 1, j), (b.L, i, j + 1), (b.L, i, j - 1)):
                    o = everything.get(k)
                    if o is not b:
                        yield o
    done = set(); out = []
    for b0 in todo:
        if b0 in done:
            continue
        space = [b0]; done.add(b0); sides = []
        for b in space:                                  # grows as touching boxes are found
            for o in neighbours(b):
                if o is not None and o.kind == "Shared" and o.group == b0.group:
                    if o not in done:
                        done.add(o); space.append(o)
                else:
                    sides.append(o)
        if any(o is None or o.kind in ("Shared", "Vertcirc") for o in sides):
            verdict = "shared"
        elif all(o.kind in ROOMS for o in sides):
            verdict = "living"
        else:
            verdict = "unclear"
        out += [(b, b0.group, verdict) for b in space]
    for b, grp, verdict in out:                          # all judged before anything changes
        if verdict == "living":
            b.kind = "Living"
        elif verdict == "shared":
            b.group = None
    return out


def main_living(boxes):
    """Each apartment's main Living room: its largest (a former Shared space may be a second one)."""
    best = {}
    for b in boxes:
        if b.kind == "Living" and b.group is not None:
            c = (b.i1 - b.i0) * (b.j1 - b.j0)
            if b.group not in best or c > best[b.group][0]:
                best[b.group] = (c, b)
    return set(b for c, b in best.values())


class Grid(object):
    """Polar grid of one bar: centre, ring radii R(j), radial-line angles A(i), floor planes."""

    def __init__(self, boxes):
        G = CONFIG["grid"]
        self.boxes = boxes
        # bar axis from box centres (PCA)
        mx = sum(b.cx for b in boxes) / len(boxes)
        my = sum(b.cy for b in boxes) / len(boxes)
        sxx = sum((b.cx - mx) ** 2 for b in boxes); syy = sum((b.cy - my) ** 2 for b in boxes)
        sxy = sum((b.cx - mx) * (b.cy - my) for b in boxes)
        ang = 0.5 * math.atan2(2 * sxy, sxx - syy)
        axis = (math.cos(ang), math.sin(ang))
        # centre = least-squares meeting point of all "across the bar" bottom edges
        A = [[0.0, 0.0], [0.0, 0.0]]; bv = [0.0, 0.0]; n_lines = 0
        for b in boxes:
            for e in b.brep.Edges:
                if not e.IsLinear(TOL):
                    continue
                p, q = e.PointAtStart, e.PointAtEnd
                if abs(p.Z - b.zmin) > 0.5 or abs(q.Z - b.zmin) > 0.5:
                    continue
                d = (q.X - p.X, q.Y - p.Y); L = math.hypot(*d)
                if L < 1.0:
                    continue
                d = (d[0] / L, d[1] / L)
                if abs(d[0] * axis[0] + d[1] * axis[1]) > 0.5:
                    continue          # runs along the bar - not a radial line
                mxp, myp = (p.X + q.X) / 2, (p.Y + q.Y) / 2
                M = [[1 - d[0] * d[0], -d[0] * d[1]], [-d[0] * d[1], 1 - d[1] * d[1]]]
                for i in range(2):
                    for j in range(2):
                        A[i][j] += M[i][j]
                    bv[i] += M[i][0] * mxp + M[i][1] * myp
                n_lines += 1
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if n_lines < 3 or abs(det) < 1e-9 * max(1.0, n_lines) ** 2:
            # straight bar: put the centre very far away on the side of the axis normal
            nrm = (-axis[1], axis[0])
            self.C = (mx - nrm[0] * 1e6, my - nrm[1] * 1e6)
        else:
            self.C = ((bv[0] * A[1][1] - A[0][1] * bv[1]) / det, (A[0][0] * bv[1] - A[1][0] * bv[0]) / det)
        C = self.C
        self.th = math.atan2(my - C[1], mx - C[0])          # reference direction (avoids +-pi wrap)

        def polar(p):
            vx, vy = p.X - C[0], p.Y - C[1]
            c, s = math.cos(self.th), math.sin(self.th)
            return math.hypot(vx, vy), math.atan2(-s * vx + c * vy, c * vx + s * vy)
        self.polar = polar
        for b in boxes:
            pr = [polar(p) for p in b.bot]
            rr = sorted(r for r, a in pr); aa = sorted(a for r, a in pr)
            b.rin, b.rout = (rr[0] + rr[1]) / 2, (rr[2] + rr[3]) / 2
            b.a1, b.a2 = (aa[0] + aa[1]) / 2, (aa[2] + aa[3]) / 2
        rmin = min(b.rin for b in boxes)
        redges = [v for b in boxes for v in (b.rin, b.rout)]
        self.R0 = rmin + sum((v - rmin) - G * round((v - rmin) / G) for v in redges) / len(redges)
        Rm = sum((b.rin + b.rout) / 2 for b in boxes) / len(boxes)
        ws = [(b.a2 - b.a1, max(1, int(round((b.a2 - b.a1) * Rm / G)))) for b in boxes]
        self.da = sum(w for w, k in ws) / sum(k for w, k in ws)
        amin = min(b.a1 for b in boxes)
        aedges = [v for b in boxes for v in (b.a1, b.a2)]
        self.A0 = amin + sum((v - amin) - self.da * round((v - amin) / self.da) for v in aedges) / len(aedges)
        self.Rm = Rm
        FH = CONFIG["floor_height"]
        for b in boxes:
            b.i0 = int(round((b.a1 - self.A0) / self.da)); b.i1 = int(round((b.a2 - self.A0) / self.da))
            b.j0 = int(round((b.rin - self.R0) / G)); b.j1 = int(round((b.rout - self.R0) / G))
            b.snap_err = max(abs((b.a1 - self.A0) / self.da - b.i0), abs((b.a2 - self.A0) / self.da - b.i1),
                             abs((b.rin - self.R0) / G - b.j0), abs((b.rout - self.R0) / G - b.j1))
        # levels and floor planes together: one sloped ground plane for the bar, floors exactly FH
        # apart.  Vertcirc spans several floors, so it never drives the fit.
        single = [b for b in boxes if b.kind != "Vertcirc"]
        zbase = min(b.zb for b in single)
        for b in single:
            b.L = int(round((b.zb - zbase) / FH))
        for _ in range(4):
            pts = [(p, FH * b.L) for b in single for p in b.bot] + \
                  [(p, FH * (b.L + 1)) for b in single for p in b.top]
            x0 = sum(p.X for p, k in pts) / len(pts); y0 = sum(p.Y for p, k in pts) / len(pts)
            M = [[0.0] * 3 for _k in range(3)]; v = [0.0] * 3
            for p, k in pts:
                r = [1.0, p.X - x0, p.Y - y0]
                for i in range(3):
                    v[i] += r[i] * (p.Z - k)
                    for j in range(3):
                        M[i][j] += r[i] * r[j]
            a0, bx, by = _solve3(M, v)
            new = [int(round((b.zb - a0 - bx * (b.cx - x0) - by * (b.cy - y0)) / FH)) for b in single]
            if new == [b.L for b in single]:
                break
            for b, L in zip(single, new):
                b.L = L
        for b in boxes:
            if b.kind == "Vertcirc":
                b.L = int(round((b.zb - a0 - bx * (b.cx - x0) - by * (b.cy - y0)) / FH))
            b.L1 = b.L + max(1, int(round((b.zt - b.zb) / FH)))
        self.levels = sorted(set(b.L for b in single))
        self.ground = (x0, y0, a0, bx, by)
        self.max_z_move = max(abs(p.Z - k - (a0 + bx * (p.X - x0) + by * (p.Y - y0))) for p, k in pts)
        # cells: rooms / public; circulation (Shared) and cores (Vertcirc) kept apart
        self.cell, self.shared, self.core = {}, {}, {}
        self.clashes = []
        for b in boxes:
            target = self.shared if b.kind == "Shared" else self.core if b.kind == "Vertcirc" else self.cell
            for L in range(b.L, b.L1):
                for i in range(b.i0, b.i1):
                    for j in range(b.j0, b.j1):
                        k = (L, i, j)
                        if k in self.cell or k in self.shared or k in self.core:
                            self.clashes.append((k, b.kind))
                        target[k] = b
        if not CONFIG["shared_is_outside"]:
            raise NotImplementedError("only shared_is_outside=True is implemented")

    def at(self, k):
        """What walls see at a cell: a room, a core, or None (outside / circulation)."""
        return self.cell.get(k) or self.core.get(k)

    # geometry along the grid
    def R(self, j):
        return self.R0 + CONFIG["grid"] * j

    def A(self, i):
        return self.th + self.A0 + self.da * i            # absolute angle

    def zat(self, k, x, y):
        x0, y0, a0, bx, by = self.ground
        return a0 + bx * (x - x0) + by * (y - y0) + CONFIG["floor_height"] * k

    def floor_plane(self, k):
        x0, y0, a0, bx, by = self.ground
        n = Vector3d(-bx, -by, 1.0); n.Unitize()
        return Plane(Point3d(x0, y0, a0 + CONFIG["floor_height"] * k), n)

    def u(self, a):
        return Vector3d(math.cos(a), math.sin(a), 0)

    def t(self, a):
        return Vector3d(-math.sin(a), math.cos(a), 0)

    def offline(self, a, rho, d):
        """Point on the line parallel to radial line `a`, offset d sideways, at distance rho from C."""
        C = self.C
        h = math.sqrt(rho * rho - d * d)
        return (C[0] + math.cos(a) * h - math.sin(a) * d, C[1] + math.sin(a) * h + math.cos(a) * d)

    def polarxy(self, rho, a):
        return (self.C[0] + rho * math.cos(a), self.C[1] + rho * math.sin(a))

    def prism(self, xy, L, ztop=None, zbot=None):
        """Vertical prism over 4 plan points, floor plane L to ceiling plane L+1 (or given z)."""
        B = [Point3d(x, y, zbot if zbot is not None else self.zat(L, x, y)) for x, y in xy]
        Tp = [Point3d(x, y, ztop if ztop is not None else self.zat(L + 1, x, y)) for x, y in xy]
        return closed_prism(B, Tp)

    def trimmed(self, curve, L):
        """Extrude a closed plan curve and trim it between floor L and floor L+1."""
        tall = Surface.CreateExtrusion(curve, Vector3d(0, 0, 1000)).ToBrep().CapPlanarHoles(TOL)
        if tall is None:
            return None
        if tall.SolidOrientation == BrepSolidOrientation.Inward:
            tall.Flip()
        top = self.floor_plane(L + 1)
        bot = self.floor_plane(L); bot.Flip()
        a = tall.Trim(top, TOL)
        if not a:
            return None
        b = a[0].Trim(bot, TOL)
        if not b:
            return None
        pc = b[0].CapPlanarHoles(TOL)
        if pc is None:
            return None
        pc.Faces.SplitKinkyFaces()
        return pc

    def sector_curve(self, r1, r2, a1, a2, z=-500.0, ext=None):
        """Closed plan curve of an annular sector; ext=(d_start, d_end) pushes the radial ends
        sideways by a fixed distance (so they line up with a wall face)."""
        base = Plane(Point3d(self.C[0], self.C[1], z), Vector3d.ZAxis)
        ds, de = ext if ext else (0.0, 0.0)
        ai = Arc(Circle(base, r1), Interval(a1 + math.asin(ds / r1), a2 + math.asin(de / r1)))
        ao = Arc(Circle(base, r2), Interval(a1 + math.asin(ds / r2), a2 + math.asin(de / r2)))
        crv = PolyCurve(); crv.Append(ArcCurve(ai)); crv.Append(LineCurve(ai.EndPoint, ao.EndPoint))
        rev = ArcCurve(ao); rev.Reverse(); crv.Append(rev); crv.Append(LineCurve(ao.StartPoint, ai.StartPoint))
        return crv


def _solve3(M, v):
    A = [row[:] + [v[i]] for i, row in enumerate(M)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(A[r][c])); A[c], A[p] = A[p], A[c]
        for r in range(3):
            if r != c:
                f = A[r][c] / A[c][c]
                for k in range(4):
                    A[r][k] -= f * A[c][k]
    return [A[i][3] / A[i][i] for i in range(3)]


# --------------------------------------------------------------------------------------------
# wall classification on the grid
# --------------------------------------------------------------------------------------------
def wall_kind(a, b):
    """a, b = room boxes on either side of a grid edge (None = outside / Shared)."""
    if a is None and b is None:
        return None
    core_a, core_b = a is not None and a.kind == "Vertcirc", b is not None and b.kind == "Vertcirc"
    if core_a or core_b:         # rooms get a party wall against a core; a core gets no walls itself
        other = b if core_a else a
        return None if other is None or (core_a and core_b) else ("PARTY", 0)
    if a is None:
        return ("EXT", +1)       # room on the + side
    if b is None:
        return ("EXT", -1)
    if a.id == b.id:
        return None
    if a.group is not None and a.group == b.group and a.kind == b.kind and a.kind == "Public":
        return None              # one open space
    return ("INT", 0) if a.group is not None and a.group == b.group else ("PARTY", 0)


def wall_band(k):
    t = CONFIG["wall_t"][k[0]]
    if k[0] == "EXT":
        return (0.0, t) if k[1] > 0 else (-t, 0.0)    # exterior walls sit inside the room
    return (-t / 2.0, t / 2.0)                          # shared walls straddle the line


def classify(g):
    """AE[(L, j, i)]: edge on ring j under column i.  RE[(L, i, j)]: edge on radial line i beside ring cell j.
    Values: (type, (lo, hi)) offsets of the wall band across the line."""
    AE, RE = {}, {}
    for (L, i, j) in list(g.cell):
        for (ii, jj) in ((i, j), (i, j + 1)):
            k = wall_kind(g.at((L, ii, jj - 1)), g.at((L, ii, jj)))
            if k:
                AE[(L, jj, ii)] = (k[0], wall_band(k))
        for (ii, jj) in ((i, j), (i + 1, j)):
            k = wall_kind(g.at((L, ii - 1, jj)), g.at((L, ii, jj)))
            if k:
                RE[(L, ii, jj)] = (k[0], wall_band(k))
    return AE, RE


def runs_of(keys_by_line, val, split=lambda line, n: False):
    out = []
    for line, idx in keys_by_line.items():
        idx.sort(); cur = None
        for n in idx:
            v = val(line, n)
            if cur and n == cur[2] and v == cur[3] and not split(line, n):
                cur[2] = n + 1
            else:
                if cur:
                    out.append(tuple(cur))
                cur = [line, n, n + 1, v]
        if cur:
            out.append(tuple(cur))
    return out


# --------------------------------------------------------------------------------------------
# STAGE normalize
# --------------------------------------------------------------------------------------------
def stage_normalize(bars, dry_run):
    """Broken apartments (a group with rooms but no Living, Kitchen or Bathroom) and loose rooms
    become ungrouped Shared space.  Shared spaces grouped with an apartment follow resolve_shared.
    Public groups, cores and other bars are never touched."""
    shared = CONFIG["room_layers"]["Shared"]
    moved = []
    verdicts = [v for boxes in bars.values() for v in boxes[0].verdicts]
    print("grouped Shared boxes: %s" % dict(Counter(v[2] for v in verdicts)))
    for boxes in bars.values():
        grp = defaultdict(list)
        for b in boxes:
            if b.group is None:
                if b.kind in ROOMS:
                    moved.append((b.id, None))
            else:
                grp[b.group].append(b)
        for gname, members in grp.items():
            kinds = set(b.kind for b in members)
            if kinds & set(ROOMS) and not set(("Living", "Kitchen", "Bathroom")) <= kinds:
                moved += [(b.id, gname) for b in members]
    print("normalize: %d boxes become ungrouped Shared" % len(moved))
    if dry_run:
        return
    for b, gname, verdict in verdicts:
        if verdict == "living":
            rs.ObjectLayer(b.id, CONFIG["room_layers"]["Living"])
        elif verdict == "shared":
            rs.RemoveObjectFromGroup(b.id, gname)
    for o, gname in moved:
        if gname:
            rs.RemoveObjectFromGroup(o, gname)
        rs.ObjectLayer(o, shared)
    for gname in set(gname for o, gname in moved if gname):
        if not rs.ObjectsByGroup(gname):
            rs.DeleteGroup(gname)


# --------------------------------------------------------------------------------------------
# STAGE wedges
# --------------------------------------------------------------------------------------------
def check_grid(name, g):
    bad = [b for b in g.boxes if b.snap_err > 0.2]
    if bad or g.clashes:
        raise RuntimeError("%s does not sit on the %.1f ft grid: %d boxes off by > 0.2 cell, %d doubled cells"
                           % (name, CONFIG["grid"], len(bad), len(g.clashes)))


def stage_wedges(bars, dry_run):
    for name, boxes in bars.items():
        g = Grid(boxes)
        check_grid(name, g)
        solo = [b for b in boxes if b.kind != "Vertcirc"]       # cores are left as modelled
        moves = []
        for b in solo:
            moves.append(max(abs(g.R(b.j0) - b.rin), abs(g.R(b.j1) - b.rout),
                             abs(g.A0 + g.da * b.i0 - b.a1) * g.Rm, abs(g.A0 + g.da * b.i1 - b.a2) * g.Rm))
        moves.sort()
        print("%s: %d boxes | arc radius ~%.0f ft | step %.4f ft | edge moves median %.2f max %.2f ft | z moves max %.2f ft"
              % (name, len(boxes), g.Rm, g.da * g.Rm, moves[len(moves) // 2], moves[-1], g.max_z_move))
        if dry_run:
            continue
        built = []
        for b in solo:
            w = g.trimmed(g.sector_curve(g.R(b.j0), g.R(b.j1), g.A(b.i0), g.A(b.i1)), b.L)
            if w is None or not w.IsSolid:
                raise RuntimeError("wedge failed for %s" % b.id)
            built.append((b, w))
        for b, w in built:
            ro = sc.doc.Objects.FindId(b.id)
            if ro is None or not sc.doc.Objects.Replace(b.id, w):
                raise RuntimeError("could not replace %s" % b.id)
    sc.doc.Views.Redraw()


# --------------------------------------------------------------------------------------------
# STAGE walls
# --------------------------------------------------------------------------------------------
def plan_walls(g):
    AE, RE = classify(g)
    arc_lines, rad_lines = defaultdict(list), defaultdict(list)
    for (L, j, i) in AE:
        arc_lines[(L, j)].append(i)
    for (L, i, j) in RE:
        rad_lines[(L, i)].append(j)
    arc_runs = runs_of(arc_lines, lambda line, n: AE[(line[0], line[1], n)])
    # radial walls stop at every node an arc wall passes through
    rad_runs = runs_of(rad_lines, lambda line, n: RE[(line[0], line[1], n)],
                       split=lambda line, n: (line[0], n, line[1] - 1) in AE or (line[0], n, line[1]) in AE)
    return AE, RE, arc_runs, rad_runs


def build_walls(g):
    AE, RE, arc_runs, rad_runs = plan_walls(g)
    walls = []            # [brep, type]
    arc_index = {}        # (L, j, i) -> index into walls, for overhang trimming

    for (L, j), i_s, i_e, (typ, (lo, hi)) in arc_runs:
        def ends(n, side):
            if (L, j, n - 1 if side < 0 else n) in AE:
                return 0.0
            bs = [RE[(L, n, jj)][1] for jj in (j - 1, j) if (L, n, jj) in RE]
            if not bs:
                return 0.0
            return min(0.0, min(b[0] for b in bs)) if side < 0 else max(0.0, max(b[1] for b in bs))
        crv = g.sector_curve(g.R(j) + lo, g.R(j) + hi, g.A(i_s), g.A(i_e), ext=(ends(i_s, -1), ends(i_e, +1)))
        w = g.trimmed(crv, L)
        if w is None or not w.IsSolid:
            raise RuntimeError("arc wall failed L%d ring %d %d-%d" % (L, j, i_s, i_e))
        for i in range(i_s, i_e):
            arc_index[(L, j, i)] = len(walls)
        walls.append([w, typ])

    for (L, i), j_s, j_e, (typ, (lb, rb)) in rad_runs:
        his = [AE[(L, j_s, ii)][1][1] for ii in (i - 1, i) if (L, j_s, ii) in AE]
        los = [AE[(L, j_e, ii)][1][0] for ii in (i - 1, i) if (L, j_e, ii) in AE]
        rs_ = g.R(j_s) + (max(his) if his else 0.0); re_ = g.R(j_e) + (min(los) if los else 0.0)
        if re_ - rs_ < 1e-3:
            continue
        a = g.A(i)
        w = g.prism([g.offline(a, rs_, lb), g.offline(a, re_, lb), g.offline(a, re_, rb), g.offline(a, rs_, rb)], L)
        if w is None or not w.IsSolid:
            raise RuntimeError("radial wall failed")
        walls.append([w, typ])

    # (a) overhang trims: an arc wall that ends at a junction only extends as far as the radial
    #     wall on EACH half of its thickness needs (it used to take the widest of both halves)
    trims = 0
    for (L, j, i), (typ, (lo, hi)) in AE.items():
        for n, side in ((i, -1), (i + 1, +1)):
            if (L, j, n - 1 if side < 0 else n) in AE:
                continue
            need = {}
            for half, jj in (("in", j - 1), ("out", j)):
                b = RE.get((L, n, jj))
                need[half] = 0.0 if b is None else (min(0.0, b[1][0]) if side < 0 else max(0.0, b[1][1]))
            went = min(need.values()) if side < 0 else max(need.values())
            for half in ("in", "out"):
                if abs(went - need[half]) < 1e-6:
                    continue
                r0, r1 = (g.R(j) + lo, g.R(j)) if half == "in" else (g.R(j), g.R(j) + hi)
                if r1 - r0 < 1e-6:
                    continue
                d0, d1 = (need[half], went + E) if side > 0 else (went - E, need[half])
                if half == "in":
                    r0 -= E
                else:
                    r1 += E
                a = g.A(n)
                cutter = g.prism([g.offline(a, r0, d0), g.offline(a, r1, d0), g.offline(a, r1, d1), g.offline(a, r0, d1)],
                                 L, ztop=g.zat(L + 1, *g.offline(a, r0, d0)) + 1, zbot=g.zat(L, *g.offline(a, r0, d0)) - 1)
                k = arc_index[(L, j, i)]
                pieces, changed = safe_difference(walls[k][0], cutter)
                if changed:
                    if len(pieces) != 1:
                        raise RuntimeError("overhang trim split a wall")
                    walls[k][0] = pieces[0]; trims += 1

    # (b) fillers: where a radial wall meets arc walls of different thickness, fill the pocket
    fillers = 0
    for (L, i, j), (typ, (lb, rb)) in RE.items():
        a = g.A(i)

        def piece(d0, d1, rho0, rho1):
            if d1 - d0 < 1e-6 or rho1 - rho0 < 1e-6:
                return 0
            walls.append([g.prism([g.offline(a, rho0, d0), g.offline(a, rho1, d0), g.offline(a, rho1, d1),
                                   g.offline(a, rho0, d1)], L), typ])
            return 1
        w_, e_ = AE.get((L, j, i - 1)), AE.get((L, j, i))
        if w_ and e_:
            hmax = max(w_[1][1], e_[1][1])
            fillers += piece(max(0.0, lb), rb, g.R(j) + e_[1][1], g.R(j) + hmax)
            fillers += piece(lb, min(0.0, rb), g.R(j) + w_[1][1], g.R(j) + hmax)
        w_, e_ = AE.get((L, j + 1, i - 1)), AE.get((L, j + 1, i))
        if w_ and e_:
            lmin = min(w_[1][0], e_[1][0])
            fillers += piece(max(0.0, lb), rb, g.R(j + 1) + lmin, g.R(j + 1) + e_[1][0])
            fillers += piece(lb, min(0.0, rb), g.R(j + 1) + lmin, g.R(j + 1) + w_[1][0])
    return walls, trims, fillers, arc_runs, rad_runs


def stage_walls(bars, dry_run):
    if not dry_run:
        for k, col in CONFIG["wall_colors"].items():
            ensure_layer(CONFIG["wall_layers"][k], col)
    for name, boxes in bars.items():
        g = Grid(boxes)
        check_grid(name, g)
        if dry_run:
            AE, RE, arc_runs, rad_runs = plan_walls(g)
            c = Counter(r[3][0] for r in arc_runs) + Counter(r[3][0] for r in rad_runs)
            print("%s walls (before junction fixes): %s" % (name, dict(c)))
            continue
        walls, trims, fillers, a_r, r_r = build_walls(g)
        bad = [w for w in walls if w[0] is None or not w[0].IsSolid]
        if bad:
            raise RuntimeError("%d walls are not solids - nothing added" % len(bad))
        for w, typ in walls:
            add_brep(w, CONFIG["wall_layers"][typ])
        print("%s: %d walls added (%d overhang trims, %d fillers) %s"
              % (name, len(walls), trims, fillers, dict(Counter(t for w, t in walls))))
    sc.doc.Views.Redraw()


# --------------------------------------------------------------------------------------------
# exterior faces (shared by facade, front doors, windows)
# --------------------------------------------------------------------------------------------
def exterior_faces(g, b):
    """Runs of a room's exterior edges: (outside kind 'open'/'shared', n_cells, key, n_s, n_e);
    key = (kind 'arc'/'rad', line index, room side +1/-1, level)."""
    faces = defaultdict(list)
    for (L, i, j), c in g.cell.items():
        if c is not b:
            continue
        for (oi, oj, kd, line, n, rsign) in ((i, j - 1, "arc", j, i, +1), (i, j + 1, "arc", j + 1, i, -1),
                                              (i - 1, j, "rad", i, j, +1), (i + 1, j, "rad", i + 1, j, -1)):
            if (L, oi, oj) in g.cell or (L, oi, oj) in g.core:
                continue
            out_box = g.shared.get((L, oi, oj))
            faces[(kd, line, rsign, L)].append((n, "open" if out_box is None else "shared"))
    out = []
    for key, items in faces.items():
        items.sort(); run = [items[0]]
        for it in items[1:] + [None]:
            if it and it[0] == run[-1][0] + 1 and it[1] == run[-1][1]:
                run.append(it); continue
            out.append((run[0][1], len(run), key, run[0][0], run[-1][0] + 1))
            if it:
                run = [it]
    return out


def face_frame(g, AE, RE, key, n_s, n_e):
    """Returns (length, near, far, frame(s)->(origin, along, across), offset-along-line).
    near/far = position of the perpendicular wall faces at each end of the run."""
    kd, line, rsign, L = key
    if kd == "arc":
        j = line; R = g.R(j); a_s = g.A(n_s)
        length = g.da * (n_e - n_s) * R; off = g.da * n_s * R
        fs = [RE[(L, n_s, jj)][1][1] for jj in (j - 1, j) if (L, n_s, jj) in RE]
        fe = [RE[(L, n_e, jj)][1][0] for jj in (j - 1, j) if (L, n_e, jj) in RE]

        def frame(s, R=R, a_s=a_s):
            a = a_s + s / R
            return Point3d(g.C[0], g.C[1], 0) + g.u(a) * R, g.t(a), g.u(a)
    else:
        i = line; length = CONFIG["grid"] * (n_e - n_s); off = CONFIG["grid"] * n_s
        fs = [AE[(L, n_s, ii)][1][1] for ii in (i - 1, i) if (L, n_s, ii) in AE]
        fe = [AE[(L, n_e, ii)][1][0] for ii in (i - 1, i) if (L, n_e, ii) in AE]
        a = g.A(i); r_s = g.R(n_s)

        def frame(s, a=a, r_s=r_s):
            return Point3d(g.C[0], g.C[1], 0) + g.u(a) * (r_s + s), g.u(a), g.t(a)
    near = max(fs) if fs else 0.0
    far = length + (min(fe) if fe else 0.0)
    return length, near, far, frame, off


def place(near, far, width):
    """Frame start: clearance from the near wall face; centred if the far side would be tighter."""
    clr = CONFIG["clearance"]
    s0 = near + clr
    if far - (s0 + width) < clr:
        s0 = (near + far - width) / 2.0
    return s0


def local(frame, s0, s1):
    o, tt, nn = frame((s0 + s1) / 2.0)
    return lambda s, n: o + tt * (s - (s0 + s1) / 2.0) + nn * n


def ext_box(g, P, L, a0, a1, n0, n1, ztop, zbot=None):
    xy = [P(a0, n0), P(a1, n0), P(a1, n1), P(a0, n1)]
    return g.prism([(p.X, p.Y) for p in xy], L, ztop=ztop, zbot=zbot)


# --------------------------------------------------------------------------------------------
# egress and balconies: planning
# --------------------------------------------------------------------------------------------
def egress(g):
    """What is linked to a core.  Returns dict(linked, faces, runs, homes, stuck):
    linked = set of core / Shared cells (L, i, j) and balcony faces (L, "F", j, sg, i);
    faces  = set of (L, j, sg, i) that carry a balcony;  runs = [(L, j, sg, i0, i1)];
    homes  = [(group, level, room kind it is entered by)] served by a balcony;  stuck = no route.
    An apartment is linked when a balcony_entry room touches linked Shared space.  Otherwise the
    shortest balcony along a long outer side is added (nearest apartments first; balconies and the
    Shared spaces they reach are shared by the next ones)."""
    if hasattr(g, "egress_plan"):
        return g.egress_plan
    ext = {}
    for d in (g.cell, g.shared, g.core):
        for (L, i, j) in d:
            lo, hi = ext.get((L, i), (j, j))
            ext[(L, i)] = (min(lo, j), max(hi, j))
    built = lambda k: k in g.cell or k in g.shared or k in g.core
    F = {}; back = defaultdict(list)
    for (L, i) in sorted(ext):
        lo, hi = ext[(L, i)]
        for jl, sg, c, o in ((lo, -1, lo, lo - 1), (hi + 1, 1, hi, hi + 1)):
            if not built((L - 1, i, o)):                  # the slab would be inside the floor below
                F[(L, "F", jl, sg, i)] = (L, i, c); back[(L, i, c)].append((L, "F", jl, sg, i))

    def nb(k):
        return [(k[0], k[1] + 1, k[2]), (k[0], k[1] - 1, k[2]), (k[0], k[1], k[2] + 1), (k[0], k[1], k[2] - 1)]
    linked = set(g.core)

    def grow():
        q = deque(linked)
        while q:
            n = q.popleft()
            for m in ([F[n]] if len(n) == 5 else nb(n)):
                if m in g.shared and m not in linked:
                    linked.add(m); q.append(m)
    grow()
    need = {}; spare = defaultdict(list)                  # spare: the rooms one would rather not enter by
    if CONFIG["balconies_on"]:
        mains = dict((b.group, b) for b in main_living(g.boxes))
        for x in mains:
            need[x] = []
        for k in sorted(g.cell):
            v = g.cell[k]
            if v.group in need and v.kind == "Bedroom" and "Bedroom" not in CONFIG["balcony_entry"]:
                spare[v.group] += [(0.4, m, v.kind) for m in nb(k) if m in g.shared]
                spare[v.group] += [(1.4, f, v.kind) for f in back[k]]
            if v.group in need and v.kind in CONFIG["balcony_entry"]:
                w = 0 if v is mains[v.group] else 0.1 if v.kind == "Living" else 0.2 if v.kind == "Kitchen" else 0.4
                need[v.group] += [(w, m, v.kind) for m in nb(k) if m in g.shared]
                need[v.group] += [(1 + w, f, v.kind) for f in back[k]]
        for x in list(need):
            if any(m in linked for w, m, kd in need[x]):
                del need[x]

    def route(src):
        h = []; seen = set(); c = 0
        for w, m, kd in src:
            heapq.heappush(h, (w, c, m, (m,), kd)); c += 1
        while h:
            cost, _, m, path, kd = heapq.heappop(h)
            if m in seen:
                continue
            seen.add(m)
            if m in linked:
                return cost, path, kd
            if len(m) == 5:
                ns = [(f, 1) for f in ((m[0], "F", m[2], m[3], m[4] + 1), (m[0], "F", m[2], m[3], m[4] - 1)) if f in F]
                if F[m] in g.shared:
                    ns.append((F[m], 0))
            else:
                ns = [(q, 0) for q in nb(m) if q in g.shared] + [(f, 1) for f in back[m]]
            for q, w in ns:
                heapq.heappush(h, (cost + w, c, q, path + (q,), kd)); c += 1
        return None
    homes = []; level = dict((v.group, k[0]) for k, v in g.cell.items())
    while need:
        found = []
        for x in sorted(need):
            rt = route(need[x])
            if rt:
                found.append((rt[0], x, rt))
        if not found:
            if not any(spare[x] for x in need):
                break
            for x in need:                                # nothing else left: enter by a bedroom
                need[x] = need[x] + spare.pop(x, [])
            continue
        cost, x, rt = min(found, key=lambda t: (t[0], t[1]))
        linked.update(rt[1]); grow()
        for y in sorted(need):
            if y == x or any(m in linked for w, m, kd in need[y]):
                kd = rt[2] if y == x else [kd for w, m, kd in sorted(need[y], key=lambda t: t[0]) if m in linked][0]
                homes.append((y, level[y], kd)); del need[y]
    faces = set((n[0], n[2], n[3], n[4]) for n in linked if len(n) == 5)
    lines = defaultdict(list); runs = []
    for L, j, sg, i in faces:
        lines[(L, j, sg)].append(i)
    for key in sorted(lines):
        idx = sorted(lines[key]); a = prev = idx[0]
        for n in idx[1:] + [None]:
            if n is not None and n == prev + 1:
                prev = n; continue
            runs.append(key + (a, prev + 1))
            if n is not None:
                a = prev = n
    g.egress_plan = dict(linked=linked, faces=faces, runs=runs, homes=homes, stuck=sorted(need))
    return g.egress_plan


def under_slab(g, L, j, sg, i):
    """Is there a balcony slab at the top of this wall (one floor up)?"""
    return (L + 1, j, sg, i) in egress(g)["faces"]


def door_faces(g, b):
    """Runs of a room's outside wall that lead to egress: (kind 'shared'/'balcony', n_cells, key, n_s, n_e)."""
    e = egress(g); faces = defaultdict(list)
    for (L, i, j), c in g.cell.items():
        if c is not b:
            continue
        for (oi, oj, kd, line, n, rsign) in ((i, j - 1, "arc", j, i, +1), (i, j + 1, "arc", j + 1, i, -1),
                                              (i - 1, j, "rad", i, j, +1), (i + 1, j, "rad", i + 1, j, -1)):
            if (L, oi, oj) in g.shared and (L, oi, oj) in e["linked"]:
                faces[(kd, line, rsign, L)].append((n, "shared"))
            elif kd == "arc" and (L, line, -rsign, i) in e["faces"]:
                faces[(kd, line, rsign, L)].append((n, "balcony"))
    out = []
    for key in sorted(faces):
        items = sorted(faces[key]); run = [items[0]]
        for it in items[1:] + [None]:
            if it and it[0] == run[-1][0] + 1 and it[1] == run[-1][1]:
                run.append(it); continue
            out.append((run[0][1], len(run), key, run[0][0], run[-1][0] + 1))
            run = [it]
    return out


# --------------------------------------------------------------------------------------------
# STAGE panels
# --------------------------------------------------------------------------------------------
def long_faces(g):
    """{(L, j, i): +1/-1}: apartment walls on the bar's long outer sides - arc faces with nothing
    (program, Shared or core) further out on that floor.  Walls behind a balcony are left out (they keep their fins)."""
    if not CONFIG["panels_on"]:
        return {}
    extent = {}
    for d in (g.cell, g.shared, g.core):
        for (L, i, j) in d:
            lo, hi = extent.get((L, i), (j, j))
            extent[(L, i)] = (min(lo, j), max(hi, j))
    out = {}
    for (L, i, j), b in g.cell.items():
        if b.kind in ROOMS:
            lo, hi = extent[(L, i)]
            if j == lo and (L, j, -1, i) not in egress(g)["faces"]:
                out[(L, j, i)] = -1
            if j == hi and (L, j + 1, 1, i) not in egress(g)["faces"]:
                out[(L, j + 1, i)] = +1
    return out


def plan_panels(g, name):
    """Panels: dict(L, j, sg, i0, i1, h0, h1, off=(bottom, seam, top)).  Seams and offsets carry on
    between neighbours along a run, and the floor-line offset carries on between storeys."""
    seed = CONFIG["panel_seed"]                          # a number: text seeds change when Rhino restarts
    for ch in name:
        seed = (seed * 31 + ord(ch)) % 1000003
    rnd = random.Random(seed)
    LF = long_faces(g)
    lines = defaultdict(list)
    for (L, j, i), sg in LF.items():
        lines[(L, j, sg)].append(i)
    runs = []
    for key in sorted(lines):
        idx = sorted(lines[key]); a = idx[0]; prev = a
        for n in idx[1:] + [None]:
            if n is not None and n == prev + 1:
                prev = n; continue
            runs.append(key + (a, prev + 1))
            if n is not None:
                a = prev = n
    # runs stacked over each other share the offset at the floor between them
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x
    for k, (L, j, sg, a, b) in enumerate(runs):
        for m, (L2, j2, sg2, a2, b2) in enumerate(runs):
            if L2 == L + 1 and (j2, sg2) == (j, sg) and min(b, b2) > max(a, a2):
                parent[find((m, "bot"))] = find((k, "top"))
    offs = CONFIG["panel_offsets"]; edge = {}

    def edge_off(x):
        x = find(x)
        if x not in edge:
            edge[x] = rnd.choice(offs)
        return edge[x]
    lo_h, hi_h = CONFIG["panel_seam"]; lo_s, hi_s = CONFIG["panel_seam_step"]
    out = []
    for k, (L, j, sg, a, b) in enumerate(runs):
        bot, top = edge_off((k, "bot")), edge_off((k, "top"))
        mid = rnd.choice([o for o in offs if not (o == bot == top)])
        jj = j if sg < 0 else j - 1
        stops = [i for i in range(a, b + 1) if i in (a, b) or g.cell[(L, i, jj)] is not g.cell[(L, i - 1, jj)]]
        h = rnd.randint(lo_h, hi_h); i = a
        while i < b:
            choice = [(n, w) for n, w in CONFIG["panel_lengths"] if i + n in stops]
            if not choice:
                choice = [(min(t for t in stops if t > i) - i, 1)]
            pick = rnd.random() * sum(w for n, w in choice)
            for n, w in choice:
                pick -= w
                if pick <= 0:
                    break
            nxt = [v for v in range(lo_h, hi_h + 1) if lo_s <= abs(v - h) <= hi_s]
            h2 = rnd.choice(nxt)
            out.append(dict(L=L, j=j, sg=sg, i0=i, i1=i + n, h0=h, h1=h2, off=(bot, mid, top)))
            i += n; h = h2
    return out


def slat_heights():
    """(bottom, top) of each slat above its floor: the fin pattern turned on its side."""
    FH = CONFIG["floor_height"]; m = CONFIG["fin_module"]; out = []; k = 0
    while k * m < FH - 0.01:
        for a, b in CONFIG["fin_pattern"]:
            z0, z1 = k * m + a * IN, min(k * m + b * IN, FH)
            if z1 - z0 > 0.01:
                out.append((z0, z1))
        k += 1
    return out


def strip_solid(sections):
    """Closed solid through equal point rings; each ring is (outer points bottom up, inner points bottom up)."""
    m = len(sections[0][0]); faces = []
    for o, w in (sections[0], sections[-1]):
        for c in range(m - 1):
            faces.append(Brep.CreateFromCornerPoints(w[c], o[c], o[c + 1], w[c + 1], TOL))
    for A, B in zip(sections, sections[1:]):
        ra, rb = A[0] + A[1][::-1], B[0] + B[1][::-1]
        for c in range(2 * m):
            d = (c + 1) % (2 * m)
            faces.append(Brep.CreateFromCornerPoints(ra[c], ra[d], rb[d], rb[c], TOL))
    jn = Brep.JoinBreps(faces, TOL) if all(f is not None for f in faces) else None
    br = jn[0] if jn and len(jn) == 1 and jn[0].IsSolid else None
    if br is not None and br.SolidOrientation == BrepSolidOrientation.Inward:
        br.Flip()
    return br


def panel_strip(g, q, z0, z1, ka=0, kb=None):
    """The part of a panel between heights z0 and z1 above its bottom edge (and cells ka..kb of its
    length), as one closed solid that follows the fold.  None if it fails.
    q may carry H (panel height), base (its bottom, from the floor) and out (its distance from the wall)."""
    R = g.R(q["j"]); n = q["i1"] - q["i0"]; t = CONFIG["panel_t"]
    FH = q.get("H", CONFIG["floor_height"]); base = q.get("base", 0.0); out = q.get("out", 0.0)
    kb = n if kb is None else kb
    bot, mid, top = [o + CONFIG["panel_gap"] + out for o in q["off"]]
    lo, hi = z0 + 0.15 * (z1 - z0), z1 - 0.15 * (z1 - z0)

    def seam(k):
        return q["h0"] + (q["h1"] - q["h0"]) * k / float(n)

    def off(h, k):
        zs = seam(k)
        return bot + (mid - bot) * h / zs if h <= zs else mid + (top - mid) * (h - zs) / (FH - zs)
    crossed = max(seam(ka), seam(kb)) > z0 and min(seam(ka), seam(kb)) < z1
    ks = [float(k) for k in range(ka, kb + 1)]
    if crossed:                                     # extra stations where the seam enters and leaves
        for h in ((z0, lo, hi, z1) if q["h0"] != q["h1"] else ()):
            k = n * (h - q["h0"]) / float(q["h1"] - q["h0"])
            if all(abs(k - o) > 0.01 for o in ks) and ka < k < kb:
                ks.append(k)
        ks.sort()

    def section(k):
        a = g.A(q["i0"]) + (g.A(q["i1"]) - g.A(q["i0"])) * k / n
        zf = g.zat(q["L"], *g.polarxy(R, a)) + base
        hs = [z0, min(max(seam(k), lo), hi), z1] if crossed else [z0, z1]

        def pt(o, h):
            x, y = g.polarxy(R + q["sg"] * o, a)
            return Point3d(x, y, zf + h)
        return [pt(off(h, k) + t, h) for h in hs], [pt(off(h, k), h) for h in hs]
    return strip_solid([section(k) for k in ks])


def slat_heights(H):
    """(bottom, top) of each slat over a height H: the fin pattern turned on its side."""
    if not CONFIG["panel_slits"]:
        return [(0.0, H)]
    m = CONFIG["fin_module"]; out = []; k = 0
    while k * m < H - 0.01:
        for a, b in CONFIG["fin_pattern"]:
            z0, z1 = k * m + a * IN, min(k * m + b * IN, H)
            if z1 - z0 > 0.01:
                out.append((z0, z1))
        k += 1
    return out


def panel_breps(g, q):
    """A panel with horizontal slits cut right through it: a stack of slats following the fold
    (or the whole panel).  Slats stop under a balcony slab.  None in the list = that part failed."""
    n = q["i1"] - q["i0"]; FH = CONFIG["floor_height"]; cap = FH - CONFIG["balcony_t"]
    under = [under_slab(g, q["L"], q["j"], q["sg"], q["i0"] + k) for k in range(n)]
    out = []
    for z0, z1 in slat_heights(FH):
        if z1 <= cap + 0.001 or not any(under):
            out.append(panel_strip(g, q, z0, z1)); continue
        a = 0
        while a < n:
            b = a
            while b < n and under[b] == under[a]:
                b += 1
            zt = min(z1, cap) if under[a] else z1
            if zt - z0 > 0.01:
                out.append(panel_strip(g, q, z0, zt, a, b))
            a = b
    return out


def stage_panels(bars, levels, dry_run):
    if not dry_run:
        ensure_layer(CONFIG["panel_layer"])
    for name, boxes in bars.items():
        g = Grid(boxes)
        plan = [q for q in plan_panels(g, name) if levels is None or q["L"] in levels]
        sets = [(q, panel_breps(g, q)) for q in plan]
        breps = [b for q, bs in sets for b in bs]
        bad = sum(1 for b in breps if b is None or not b.IsSolid)
        print("%s: %d panels, %d solids, %.0f ft of wall, lengths (cells) %s | parts that failed: %d"
              % (name, len(plan), len(breps), sum(q["i1"] - q["i0"] for q in plan) * CONFIG["grid"],
                 dict(Counter(q["i1"] - q["i0"] for q in plan)), bad))
        if dry_run:
            continue
        if bad:
            raise RuntimeError("panel failed - nothing added")
        for q, bs in sets:
            grp = new_group("Panel %s L%d %d %d-%d" % (name, q["L"], q["j"], q["i0"], q["i1"]))
            for b in bs:
                add_brep(b, CONFIG["panel_layer"], grp)
    sc.doc.Views.Redraw()


# --------------------------------------------------------------------------------------------
# STAGE balconies
# --------------------------------------------------------------------------------------------
def rail_panels(g, name, run):
    """The folded outer panels of one balcony, as panel dicts (2-5 cells long)."""
    L, j, sg, a, b = run
    seed = CONFIG["panel_seed"]
    for ch in "%s/%s/%s/%s/%s" % (name, L, j, sg, a):
        seed = (seed * 31 + ord(ch)) % 1000003
    rnd = random.Random(seed)
    offs = CONFIG["rail_offsets"]; bot, top = rnd.choice(offs), rnd.choice(offs)
    mid = rnd.choice([o for o in offs if not (o == bot == top)])
    lo, hi = CONFIG["rail_seam"]; s0, s1 = CONFIG["rail_seam_step"]
    below = CONFIG["balcony_t"] + CONFIG["rail_below"]
    h = rnd.randint(lo, hi); i = a; out = []
    while i < b:
        left = b - i
        fit = [(n, w) for n, w in CONFIG["panel_lengths"] if n > 1 and (n == left or left - n > 1)] or [(left, 1)]
        pick = rnd.random() * sum(w for n, w in fit)
        for n, w in fit:
            pick -= w
            if pick <= 0:
                break
        h2 = rnd.choice([v for v in range(lo, hi + 1) if s0 <= abs(v - h) <= s1])
        out.append(dict(L=L, j=j, sg=sg, i0=i, i1=i + n, h0=h, h1=h2, off=(bot, mid, top),
                        base=-below, H=below + CONFIG["rail_above"], out=CONFIG["balcony_depth"]))
        i += n; h = h2
    return out


def balcony_parts(g, name, run):
    """(slab, [outer panel slats], [end guard slats])."""
    L, j, sg, a, b = run
    R = g.R(j); gap = CONFIG["panel_gap"]; dep = CONFIG["balcony_depth"]; T = CONFIG["balcony_t"]

    def box(angles, o0, o1, h0, h1):
        secs = []
        for ang in angles:
            zf = g.zat(L, *g.polarxy(R, ang))

            def pt(o, h, ang=ang, zf=zf):
                x, y = g.polarxy(R + sg * o, ang)
                return Point3d(x, y, zf + h)
            secs.append(([pt(o1, h0), pt(o1, h1)], [pt(o0, h0), pt(o0, h1)]))
        return strip_solid(secs)
    slab = box([g.A(i) for i in range(a, b + 1)], gap, dep, -T, 0.0)
    rail = []
    for q in rail_panels(g, name, run):
        rail += [panel_strip(g, q, z0, z1) for z0, z1 in slat_heights(q["H"])]
    ends = []
    if CONFIG["balcony_ends"]:
        w = CONFIG["panel_t"] / R; o0 = CONFIG["fin_depth"] + gap
        for a0, a1 in ((g.A(a), g.A(a) + w), (g.A(b) - w, g.A(b))):
            ends += [box([a0, a1], o0, dep, z0, z1) for z0, z1 in slat_heights(CONFIG["rail_above"])]
    return slab, rail, ends


def stage_balconies(bars, levels, dry_run):
    for name, boxes in bars.items():
        g = Grid(boxes); e = egress(g)
        runs = [q for q in e["runs"] if levels is None or q[0] in levels]
        slabs, rails, names = [], [], []
        for run in runs:
            slab, rail, ends = balcony_parts(g, name, run)
            slabs.append(slab); rails += rail + ends
            names += ["Balcony %s L%d %d %d-%d" % (name, run[0], run[1], run[3], run[4])] * len(rail + ends)
        bad = sum(1 for b in slabs + rails if b is None or not b.IsSolid)
        print("%s: %d apartments reach egress only by balcony %s | %d balconies, %d ft | %d panel slats | no route: %s | parts that failed: %d"
              % (name, len(e["homes"]), ["%s L%d %s" % h for h in e["homes"]], len(runs),
                 sum(q[4] - q[3] for q in runs) * CONFIG["grid"], len(rails), e["stuck"] or "none", bad))
        for q in runs:
            print("   level %d, line %d, cells %d-%d: %d ft" % (q[0], q[1], q[3], q[4], (q[4] - q[3]) * CONFIG["grid"]))
        if dry_run or not runs:
            continue
        if bad:
            raise RuntimeError("balcony failed - nothing added")
        ensure_layer(CONFIG["slab_layer"]); ensure_layer(CONFIG["rail_layer"])
        for b in slabs:
            add_brep(b, CONFIG["slab_layer"])
        grps = dict((k, new_group(k)) for k in sorted(set(names)))
        for b, k in zip(rails, names):
            add_brep(b, CONFIG["rail_layer"], grps[k])
    sc.doc.Views.Redraw()


# --------------------------------------------------------------------------------------------
# STAGE facade
# --------------------------------------------------------------------------------------------
def horizontal_fins(g, arc_runs, rad_runs, AF, LF, away, empty):
    """[((L, label), [solids])]: one list per continuous wall stretch.  Fins sit at the slat heights of the
    panels, wrap outside corners, and stop under balcony slabs and hanging balcony panels."""
    D = CONFIG["fin_depth"]; FH = CONFIG["floor_height"]; hair = 0.005      # off the curved wall face
    T = CONFIG["balcony_t"]; dep = CONFIG["balcony_depth"]; low = T + CONFIG["rail_below"]
    far = dep + max(CONFIG["rail_offsets"]) + CONFIG["panel_t"] + 0.05
    heights = slat_heights(FH); out = []

    def solid(L, rings, z0, z1):
        """rings: [(outer xy, inner xy)] along the stretch."""
        def pt(q, h):
            return Point3d(q[0], q[1], g.zat(L, q[0], q[1]) + h)
        return strip_solid([([pt(o, z0), pt(o, z1)], [pt(w, z0), pt(w, z1)]) for o, w in rings])

    for (L, j, sg), i_s, i_e in sorted(arc_runs):
        R = g.R(j); bs = []
        ring = lambda i: (g.polarxy(R + sg * D, g.A(i)), g.polarxy(R + sg * hair, g.A(i)))
        under = [under_slab(g, L, j, sg, i) for i in range(i_s, i_e)]
        for z0, z1 in heights:
            a = i_s
            while a < i_e:
                b = a; low_wall = z1 > FH - T + 0.001
                while b < i_e and (not low_wall or under[b - i_s] == under[a - i_s]):
                    b += 1
                zt = min(z1, FH - T) if low_wall and under[a - i_s] else z1
                if zt - z0 > 0.01:
                    bs.append(solid(L, [ring(i) for i in range(a, b + 1)], z0, zt))
                a = b
        out.append(((L, "long %d %d-%d" % (j, i_s, i_e)), bs))
    for (L, i, sg), j_s, j_e in sorted(rad_runs):
        a = g.A(i); col = i - 1 if sg < 0 else i; room = i if sg < 0 else i - 1
        r0, r1 = g.R(j_s), g.R(j_e)
        if AF.get((L, j_s, col)) == +1:                   # inside corner: stop at the other wall's fins
            r0 += away if (L, j_s, col) in LF else D
        elif empty((L, room, j_s - 1)) and empty((L, col, j_s - 1)):
            r0 -= D                                       # outside corner: wrap round
        if AF.get((L, j_e, col)) == -1:
            r1 -= away if (L, j_e, col) in LF else D
        elif empty((L, room, j_e)) and empty((L, col, j_e)):
            r1 += D
        bs = []
        for z0, z1 in heights:
            parts = [(r0, r1)]
            for (LL, jj, ss, ia, ib) in egress(g)["runs"]:
                if LL != L + 1 or not ia <= col < ib:     # is this fin under a balcony?
                    continue
                Rl = g.R(jj)
                for top, o0, o1 in ((T, 0.001, dep), (low, dep, far)):
                    if z1 > FH - top + 0.001:
                        x0, x1 = sorted((Rl + ss * o0, Rl + ss * o1))
                        parts = [q for u, v in parts for q in ((u, min(v, x0)), (max(u, x1), v)) if q[1] - q[0] > 0.01]
            for u, v in parts:
                ring = lambda q: (g.offline(a, q, sg * D), g.offline(a, q, sg * hair))
                bs.append(solid(L, [ring(u), ring(v)], z0, z1))
        out.append(((L, "end %d %d-%d" % (i, j_s, j_e)), bs))
    return out


def stage_facade(bars, levels, dry_run):
    D = CONFIG["fin_depth"]; MODL = CONFIG["fin_module"]
    pattern = [(a * IN, b * IN) for a, b in CONFIG["fin_pattern"]]

    def segs(length, c0, c1):
        out = []; k = 0
        while k * MODL < length:
            for a, b in pattern:
                p0, p1 = max(k * MODL + a, c0), min(k * MODL + b, length - c1)
                if p1 - p0 > 0.01:
                    out.append((p0, p1))
            k += 1
        return out
    if not dry_run:
        ensure_layer(CONFIG["fin_layer"])
    for name, boxes in bars.items():
        g = Grid(boxes)
        lv = levels if levels is not None else g.levels
        empty = lambda k: k not in g.cell and k not in g.core
        AF, RF = {}, {}
        for (L, i, j) in g.cell:
            if L not in lv:
                continue
            if empty((L, i, j - 1)): AF[(L, j, i)] = -1
            if empty((L, i, j + 1)): AF[(L, j + 1, i)] = +1
            if empty((L, i - 1, j)): RF[(L, i, j)] = -1
            if empty((L, i + 1, j)): RF[(L, i + 1, j)] = +1

        LF = long_faces(g)
        away = CONFIG["panel_t"] + max(CONFIG["panel_offsets"]) + CONFIG["panel_gap"]
        for k in LF:
            AF[k] = AF.get(k, LF[k])

        def runs(d):
            lines = defaultdict(list)
            for (L, ln, n), sg in d.items():
                lines[(L, ln, sg)].append(n)
            out = []
            for key, idx in lines.items():
                idx.sort(); s = prev = idx[0]
                for n in idx[1:] + [None]:
                    if n is not None and n == prev + 1:
                        prev = n; continue
                    out.append((key, s, prev + 1))
                    if n is not None:
                        s = prev = n
            return out
        if CONFIG["fin_direction"] == "horizontal":
            sets = horizontal_fins(g, runs(dict((k, v) for k, v in AF.items() if k not in LF)), runs(RF), AF, LF, away, empty)
            solids = [b for key, bs in sets for b in bs]
            bad = sum(1 for b in solids if b is None or not b.IsSolid)
            print("%s levels %s: %d horizontal fins on %d wall stretches | parts that failed: %d"
                  % (name, lv, len(solids), len(sets), bad))
            if dry_run:
                continue
            if bad:
                raise RuntimeError("fin failed - nothing added")
            for key, bs in sets:
                grp = new_group("Fins %s L%d %s" % (name, key[0], key[1]))
                for b in bs:
                    add_brep(b, CONFIG["fin_layer"], grp)
            continue
        pieces = []
        for (L, j, sg), i_s, i_e in runs(dict((k, v) for k, v in AF.items() if k not in LF)):
            R = g.R(j); length = g.da * (i_e - i_s) * R
            r_lo, r_hi = (R - D, R) if sg < 0 else (R, R + D)
            for p0, p1 in segs(length, 0, 0):
                a0, a1 = g.A(i_s) + p0 / R, g.A(i_s) + p1 / R
                pieces.append(([g.polarxy(r_lo, a0), g.polarxy(r_lo, a1), g.polarxy(r_hi, a1), g.polarxy(r_hi, a0)], L))
        for (L, i, sg), j_s, j_e in runs(RF):
            a = g.A(i); d0, d1 = (-D, 0.0) if sg < 0 else (0.0, D)
            col = i - 1 if sg < 0 else i
            c0 = (away if (L, j_s, col) in LF else D) if AF.get((L, j_s, col)) == +1 else 0.0       # inside corner: pull back
            c1 = (away if (L, j_e, col) in LF else D) if AF.get((L, j_e, col)) == -1 else 0.0
            rs_ = g.R(j_s)
            for p0, p1 in segs(g.R(j_e) - rs_, c0, c1):
                pieces.append(([g.offline(a, rs_ + p0, d0), g.offline(a, rs_ + p1, d0),
                                g.offline(a, rs_ + p1, d1), g.offline(a, rs_ + p0, d1)], L))
        posts = 0
        for (L, i, j) in g.cell:
            if L not in lv:
                continue
            for di, dj in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                if empty((L, i + di, j)) and empty((L, i, j + dj)) and empty((L, i + di, j + dj)):
                    a = g.A(i if di < 0 else i + 1); R = g.R(j if dj < 0 else j + 1)
                    d0, d1 = (-D, 0.0) if di < 0 else (0.0, D)
                    r0, r1 = (R - D, R) if dj < 0 else (R, R + D)
                    pieces.append(([g.offline(a, r0, d0), g.offline(a, r1, d0), g.offline(a, r1, d1), g.offline(a, r0, d1)], L))
                    posts += 1
        print("%s levels %s: %d fins (%d corner posts)" % (name, lv, len(pieces), posts))
        if dry_run:
            continue
        slabs = defaultdict(list)                    # fins stop under a balcony slab
        for (L, j, sg, a, b) in egress(g)["runs"]:
            slabs[L - 1].append((g.R(j), sg, g.A(a), g.A(b)))

        def fin(xy, L):
            m = (sum(p[0] for p in xy) / 4.0, sum(p[1] for p in xy) / 4.0)
            hit = 0.0                                # how far below the floor above the fin must stop
            dep = CONFIG["balcony_depth"]; low = CONFIG["balcony_t"] + CONFIG["rail_below"]
            far = dep + max(CONFIG["rail_offsets"]) + CONFIG["panel_t"] + 0.05
            for R, sg, a0, a1 in slabs[L]:
                for x, y in xy + [m]:
                    dx, dy = x - g.C[0], y - g.C[1]
                    o = sg * (math.hypot(dx, dy) - R)
                    turn = math.atan2(dx * g.t(a0).X + dy * g.t(a0).Y, dx * g.u(a0).X + dy * g.u(a0).Y)
                    if 0.01 < o < far and 0.001 / R < turn < a1 - a0 - 0.001 / R:
                        hit = max(hit, low + 0.01 if o > dep - 0.01 else CONFIG["balcony_t"])
            if not hit:
                return g.prism(xy, L)
            return closed_prism([Point3d(x, y, g.zat(L, x, y)) for x, y in xy],
                                [Point3d(x, y, g.zat(L + 1, x, y) - hit) for x, y in xy])
        built = [fin(list(xy), L) for xy, L in pieces]
        if any(b is None or not b.IsSolid for b in built):
            raise RuntimeError("fin failed - nothing added")
        for b in built:
            add_brep(b, CONFIG["fin_layer"])
    sc.doc.Views.Redraw()


# --------------------------------------------------------------------------------------------
# openings: cut walls/fins on ONE floor and add frame + leaf/glass
# --------------------------------------------------------------------------------------------
def apply_openings(g, openings, cut_layers, wall_layers, dry_run, label):
    """openings: dict(L, P, s0, s1, n0, n1, z0, z1, parts=[(brep, layer)], group).
    Only objects whose centre is on the opening's floor are cut (a cutter that reaches below the
    floor once notched the wall underneath)."""
    FH = CONFIG["floor_height"]
    recut = CONFIG.get("recut_fins_only")            # set by stage 'recutfins'
    if recut:
        cut_layers = [CONFIG["fin_layer"]]
    if not openings:
        print("%s: nothing to do" % label)
        return
    if dry_run:
        bad = sum(1 for op in openings for b, l in op["parts"] if b is None or not b.IsSolid)
        print("%s: %d planned (dry run, nothing cut) | parts that failed to build: %d" % (label, len(openings), bad))
        return
    for k in ("frames_layer", "doors_layer", "glass_layer"):
        ensure_layer(CONFIG[k])
    # only objects of this bar, on the floors being worked on
    area = BoundingBox.Empty
    for b in g.boxes:
        area.Union(b.bb)
    area.Inflate(3.0)
    mids = [(op["L"], op["P"]((op["s0"] + op["s1"]) / 2.0, 0)) for op in openings]
    zs = [g.zat(L, m.X, m.Y) for L, m in mids]
    zlo, zhi = min(zs) - 1.0, max(zs) + FH + 1.0
    cand = []
    for path in cut_layers:
        for ro in layer_objs(path):
            bb = ro.Geometry.GetBoundingBox(False)
            if not area.Contains(bb.Center) or not (zlo < bb.Center.Z < zhi):
                continue
            cand.append((ro.Id, bb, path))
    newgeo = {}; fails = []
    for op in openings:
        P, L = op["P"], op["L"]
        mid = P((op["s0"] + op["s1"]) / 2.0, 0); zb = g.zat(L, mid.X, mid.Y)
        c0, c1 = op.get("cut", (op["n0"], op["n1"]))
        cutter = ext_box(g, P, L, op["s0"], op["s1"], c0 - E, c1 + E, op["z1"], zbot=op["z0"])
        cbb = cutter.GetBoundingBox(True)
        hit = 0
        for oid, bb, path in cand:
            zc = g.zat(L, bb.Center.X, bb.Center.Y)      # the floor where the object is: bars slope
            if not (zc - 0.5 < bb.Center.Z < zc + FH):
                continue
            if not BoundingBox.Intersection(bb, cbb).IsValid:
                continue
            srcs = newgeo.get(oid) or [brep_of(sc.doc.Objects.FindId(oid).Geometry)]
            out = []; changed = False
            for gb in srcs:
                if not BoundingBox.Intersection(gb.GetBoundingBox(True), cbb).IsValid:
                    out.append(gb); continue
                for d in (0.0, 0.003, -0.003, 0.007, -0.007, 0.01):  # a hair's change rescues odd cuts
                    try:
                        c = cutter if d == 0 else ext_box(g, P, L, op["s0"] - d, op["s1"] + d, c0 - E - d,
                                                          c1 + E + d, op["z1"] + d, zbot=op["z0"] - d)
                        pcs, ch = safe_difference(gb, c)
                        break
                    except RuntimeError:
                        if d == 0.01:
                            raise
                out += pcs; changed = changed or ch
            if changed:
                newgeo[oid] = out
                if path in wall_layers:
                    hit += 1
        if hit < 1 and not recut:
            fails.append((L, round(mid.X, 1), round(mid.Y, 1)))
    print("%s: %d openings | objects cut %d | openings that found no wall: %s"
          % (label, len(openings), len(newgeo), fails[:5] or "none"))
    if dry_run or fails:
        if fails:
            print("  nothing applied")
        return
    for oid, res in newgeo.items():
        ro = sc.doc.Objects.FindId(oid); attr = ro.Attributes.Duplicate()
        sc.doc.Objects.Delete(ro, True)
        for r in res:
            sc.doc.Objects.AddBrep(r, attr)
    for op in openings if not recut else ():
        for b, layer in op["parts"]:
            add_brep(b, layer, op.get("group"))
    sc.doc.Views.Redraw()


def opening_parts(g, P, L, s0, s1, n0, n1, kind, mid_n):
    F = CONFIG["frame"]
    mid = P((s0 + s1) / 2.0, 0); zb = g.zat(L, mid.X, mid.Y)
    if kind == "door":
        zh = zb + CONFIG["door_h"]; t = CONFIG["leaf_t"]
        parts = [(ext_box(g, P, L, s0, s0 + F, n0, n1, zh + F), CONFIG["frames_layer"]),
                 (ext_box(g, P, L, s1 - F, s1, n0, n1, zh + F), CONFIG["frames_layer"]),
                 (ext_box(g, P, L, s0 + F, s1 - F, n0, n1, zh + F, zbot=zh), CONFIG["frames_layer"]),
                 (ext_box(g, P, L, s0 + F, s1 - F, mid_n - t / 2, mid_n + t / 2, zh), CONFIG["doors_layer"])]
        return parts, zb - 1.0, zh + F
    zs = zb + CONFIG["sill"]; zh = zs + CONFIG["win_h"]; t = CONFIG["glass_t"]
    parts = [(ext_box(g, P, L, s0, s0 + F, n0, n1, zh + F, zbot=zs - F), CONFIG["frames_layer"]),
             (ext_box(g, P, L, s1 - F, s1, n0, n1, zh + F, zbot=zs - F), CONFIG["frames_layer"]),
             (ext_box(g, P, L, s0 + F, s1 - F, n0, n1, zh + F, zbot=zh), CONFIG["frames_layer"]),
             (ext_box(g, P, L, s0 + F, s1 - F, n0, n1, zs, zbot=zs - F), CONFIG["frames_layer"]),
             (ext_box(g, P, L, s0 + F, s1 - F, mid_n - t / 2, mid_n + t / 2, zh, zbot=zs), CONFIG["glass_layer"])]
    return parts, zs - F, zh + F


# --------------------------------------------------------------------------------------------
# STAGE doors (interior)
# --------------------------------------------------------------------------------------------
def door_tree(g):
    """Room pairs that get a door: BFS from each Living room, one door per room."""
    pairs = defaultdict(list)
    for (L, i, j), a in g.cell.items():
        for (ii, jj, kd) in ((i, j + 1, "arc"), (i + 1, j, "rad")):
            b = g.cell.get((L, ii, jj))
            if b and b.id != a.id and b.group is not None and b.group == a.group and a.kind != "Public":
                pairs[tuple(sorted((str(a.id), str(b.id))))].append((kd, L, i, j))
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b); adj[b].add(a)
    by_g = defaultdict(list)
    for b in g.boxes:
        if b.kind not in ("Shared", "Public", "Vertcirc") and b.group is not None:
            by_g[b.group].append(b)
    tree = []; cut_off = 0
    mains = main_living(g.boxes)
    for grp, rooms in by_g.items():
        liv = [r for r in rooms if r in mains]
        if not liv:
            continue
        start = str(liv[0].id); seen = set([start]); q = deque([start])
        while q:
            x = q.popleft()
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y); q.append(y); tree.append(tuple(sorted((x, y))))
        cut_off += len(rooms) - len(seen)
    if cut_off:
        print("  WARNING: %d rooms cannot be reached from their Living room" % cut_off)
    return pairs, tree


def stage_doors(bars, levels, dry_run):
    W = CONFIG["door_w"] + 2 * CONFIG["frame"]; t = CONFIG["wall_t"]["INT"]
    for name, boxes in bars.items():
        g = Grid(boxes); AE, RE = classify(g)
        pairs, tree = door_tree(g)
        info = dict((str(b.id), b) for b in boxes)
        ops = []
        for key in tree:
            edges = pairs[key]; kd, L = edges[0][0], edges[0][1]
            if levels is not None and L not in levels:
                continue
            if kd == "arc":
                j = edges[0][3] + 1; idx = sorted(e[2] for e in edges)
                fkey, n_s, n_e = ("arc", j, 0, L), idx[0], idx[-1] + 1
            else:
                i = edges[0][2] + 1; idx = sorted(e[3] for e in edges)
                fkey, n_s, n_e = ("rad", i, 0, L), idx[0], idx[-1] + 1
            length, near, far, frame, off = face_frame(g, AE, RE, fkey, n_s, n_e)
            s0 = place(near, far, W); s1 = s0 + W
            P = local(frame, s0, s1)
            parts, z0, z1 = opening_parts(g, P, L, s0, s1, -t / 2, t / 2, "door", 0.0)
            ops.append(dict(L=L, P=P, s0=s0, s1=s1, n0=-t / 2, n1=t / 2, z0=z0, z1=z1, parts=parts, group=info[key[0]].group))
        apply_openings(g, ops, [CONFIG["wall_layers"]["INT"]], [CONFIG["wall_layers"]["INT"]], dry_run,
                       "%s interior doors" % name)


# --------------------------------------------------------------------------------------------
# STAGE frontdoors / windows
# --------------------------------------------------------------------------------------------
def entry_room(g, living):
    """Where the front door goes: the first of main Living, other Living, Kitchen, then any room with a
    wall onto Shared space or a balcony that leads to egress; failing that, with any outside wall."""
    home = [b for b in g.boxes if b.group is not None and b.group == living.group
            and b.kind not in ("Shared", "Public")]
    rank = dict(Living=1, Kitchen=2, Bedroom=3)
    home.sort(key=lambda b: 0 if b is living else rank.get(b.kind, 4))
    for faces in (door_faces, exterior_faces):
        for b in home:
            if faces(g, b):
                return b
    return None


def front_door_of(g, AE, RE, b):
    """(key, n_s, n_e, s0, s1, frame, off) for the Living room's front door."""
    runs = door_faces(g, b) or exterior_faces(g, b)
    if not runs:
        return None
    best = max(runs, key=lambda r: (r[0] == "shared", r[1]))
    W = CONFIG["door_w"] + 2 * CONFIG["frame"]
    length, near, far, frame, off = face_frame(g, AE, RE, best[2], best[3], best[4])
    s0 = place(near, far, W)
    return best[2], best[3], best[4], s0, s0 + W, frame, off


def ext_opening(g, key, frame, s0, s1, kind, group):
    D = CONFIG["fin_depth"]; T = CONFIG["wall_t"]["EXT"]
    rsign, L = key[2], key[3]
    n0, n1 = (-D, T) if rsign > 0 else (-T, D)
    mid_n = T / 2 if rsign > 0 else -T / 2
    P = local(frame, s0, s1)
    parts, z0, z1 = opening_parts(g, P, L, s0, s1, n0, n1, kind, mid_n)
    out = CONFIG["panel_t"] + max(CONFIG["panel_offsets"]) + CONFIG["panel_gap"] if CONFIG["panels_on"] else D
    cut = (-out, T) if rsign > 0 else (-T, out)
    return dict(L=L, P=P, s0=s0, s1=s1, n0=n0, n1=n1, z0=z0, z1=z1, parts=parts, group=group, cut=cut)


def stage_frontdoors(bars, levels, dry_run):
    for name, boxes in bars.items():
        g = Grid(boxes); AE, RE = classify(g)
        ops = []; facing = Counter(); mains = main_living(boxes)
        for b in boxes:
            if b not in mains or (levels is not None and b.L not in levels):
                continue
            home = entry_room(g, b)
            if home is None:
                facing["NO outside wall"] += 1; continue
            key, n_s, n_e, s0, s1, frame, off = front_door_of(g, AE, RE, home)
            ways = door_faces(g, home)
            face = [r for r in ways or exterior_faces(g, home) if r[2] == key and r[3] == n_s][0][0]
            face = ("" if ways else "NO EGRESS, ") + face
            facing["%s from %s" % (face, home.kind)] += 1
            ops.append(ext_opening(g, key, frame, s0, s1, "door", b.group))
        print("%s front doors facing: %s" % (name, dict(facing)))
        apply_openings(g, ops, [CONFIG["wall_layers"]["EXT"], CONFIG["fin_layer"], CONFIG["panel_layer"]], [CONFIG["wall_layers"]["EXT"]],
                       dry_run, "%s front doors" % name)


def stage_windows(bars, levels, dry_run):
    W = CONFIG["win_w"] + 2 * CONFIG["frame"]; clr = CONFIG["clearance"]
    for name, boxes in bars.items():
        g = Grid(boxes); AE, RE = classify(g)
        ops = []; notes = Counter()
        by_group = defaultdict(list)
        for b in boxes:
            by_group[b.group].append(b)
        taken = {}                                   # room id -> its front door (line, s0, s1)
        mains = main_living(boxes)
        for b in mains:
            home = entry_room(g, b)
            if home is not None:
                fd = front_door_of(g, AE, RE, home)
                taken[str(home.id)] = (fd[0], fd[6] + fd[3], fd[6] + fd[4])

        def plan_room(b, door, n_max):
            runs = exterior_faces(g, b)
            for wr in sorted(runs, key=lambda r: (r[0] == "open", r[1]), reverse=True):
                key, n_s, n_e = wr[2], wr[3], wr[4]
                length, near, far, frame, off = face_frame(g, AE, RE, key, n_s, n_e)
                if (n_max > 1 and n_e - n_s >= CONFIG["two_windows_min_cells"] and
                        far - near >= 2 * W + 3 * clr):
                    slots = [near + clr, far - clr - W]
                else:
                    slots = [place(near, far, W)]
                ok = [s0 for s0 in slots if not (door and door[0] == key and
                                                 not (off + s0 + W + clr <= door[1] or off + s0 >= door[2] + clr))]
                if ok:
                    return [(key, frame, s0) for s0 in ok], wr[0]
            return [], None
        for b in boxes:
            if b.kind not in CONFIG["window_rooms"] or (levels is not None and b.L not in levels):
                continue
            if b not in mains and not exterior_faces(g, b):
                notes[b.kind + ": no exterior wall"] += 1; continue
            door = taken.get(str(b.id))
            wins, facing = plan_room(b, door, 2)
            if not wins and b in mains and CONFIG["kitchen_fallback"]:
                kit = [k for k in by_group[b.group] if k.kind == "Kitchen"]
                if kit:
                    wins, facing = plan_room(kit[0], taken.get(str(kit[0].id)), 1)
                    if wins:
                        notes["Living -> Kitchen window"] += 1
            if not wins:
                notes[b.kind + ": no room for a window"] += 1; continue
            notes["%s: %d on %s" % (b.kind, len(wins), facing)] += 1
            for key, frame, s0 in wins:
                ops.append(ext_opening(g, key, frame, s0, s0 + W, "window", b.group))
        print("%s windows: %s" % (name, dict(notes)))
        apply_openings(g, ops, [CONFIG["wall_layers"]["EXT"], CONFIG["fin_layer"], CONFIG["panel_layer"]], [CONFIG["wall_layers"]["EXT"]],
                       dry_run, "%s windows" % name)


# --------------------------------------------------------------------------------------------
# STAGE report / verify
# --------------------------------------------------------------------------------------------
def stage_report(bars):
    for name, boxes in bars.items():
        g = Grid(boxes)
        kinds = Counter(b.kind for b in boxes)
        homes = len(set(b.group for b in boxes if b.kind == "Living"))
        pubs = len(set(b.group for b in boxes if b.kind == "Public"))
        print("%s: %d boxes %s | %d apartments, %d public spaces | levels %s | arc radius ~%.0f ft, centre (%.1f, %.1f) | "
              "grid snap error max %.3f cells | doubled cells %d | floors off plane max %.2f ft"
              % (name, len(boxes), dict(kinds), homes, pubs, g.levels, g.Rm, g.C[0], g.C[1], max(b.snap_err for b in boxes),
                 len(g.clashes), g.max_z_move))
    print("layer counts:", counts())


def stage_verify():
    c = counts()
    bad = 0
    for path in CONFIG["wall_layers"].values():
        for ro in layer_objs(path):
            if not brep_of(ro.Geometry).IsSolid:
                bad += 1
    print("layer counts:", c)
    print("walls that are not closed solids:", bad)


# --------------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------------
STAGES = ["report", "normalize", "wedges", "walls", "facade", "panels", "balconies", "doors", "frontdoors", "windows", "recutfins", "verify"]


def run(stage, bars=None, levels=None, dry_run=False):
    """stage: one of STAGES.  bars: list of bar names (from 'report') or None for all.
    levels: list of level indexes or None for all.  dry_run: plan and report, change nothing."""
    before = counts()
    if stage == "verify":
        stage_verify(); return
    all_bars = split_bars(read_boxes())
    sel = dict((k, v) for k, v in all_bars.items() if bars is None or k in bars)
    for boxes in sel.values():
        verdicts = resolve_shared(boxes)
        for b in boxes:
            b.verdicts = verdicts
    if not sel:
        raise ValueError("no bars selected (found: %s)" % sorted(all_bars))
    if stage == "report":
        stage_report(sel)
    elif stage == "normalize":
        stage_normalize(sel, dry_run)
    elif stage == "wedges":
        stage_wedges(sel, dry_run)
    elif stage == "walls":
        stage_walls(sel, dry_run)
    elif stage == "facade":
        stage_facade(sel, levels, dry_run)
    elif stage == "panels":
        stage_panels(sel, levels, dry_run)
    elif stage == "recutfins":                       # after rebuilding only the fins of a finished bar
        CONFIG["recut_fins_only"] = True
        stage_frontdoors(sel, levels, dry_run)
        stage_windows(sel, levels, dry_run)
        CONFIG["recut_fins_only"] = False
    elif stage == "balconies":
        stage_balconies(sel, levels, dry_run)
    elif stage == "doors":
        stage_doors(sel, levels, dry_run)
    elif stage == "frontdoors":
        stage_frontdoors(sel, levels, dry_run)
    elif stage == "windows":
        stage_windows(sel, levels, dry_run)
    else:
        raise ValueError("unknown stage %r" % stage)
    if not dry_run and stage != "report":
        after = counts()
        print("counts before:", before)
        print("counts after: ", after)
        # nothing on a wall layer may ever go DOWN in a stage that only cuts openings
        for k in CONFIG["wall_layers"]:
            if after[k] < before[k] and stage in ("doors", "frontdoors", "windows"):
                print("WARNING: %s walls decreased (%d -> %d) - undo this stage" % (k, before[k], after[k]))


def run_interactive():
    stage = rs.ListBox(STAGES, "Stage to run", "Apartment pipeline")
    if not stage:
        return
    dry = rs.GetBoolean("Dry run?", ("DryRun", "No", "Yes"), (True,))
    run(stage, dry_run=bool(dry and dry[0]))


if __name__ == "__main__":
    _args = globals().get("PIPELINE_ARGS")
    if _args:
        run(**_args)
    else:
        run_interactive()
