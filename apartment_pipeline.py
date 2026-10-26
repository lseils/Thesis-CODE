# -*- coding: utf-8 -*-
"""
apartment_pipeline.py -- turn programmatic apartment boxes into walled, fenestrated apartments.

Runs inside Rhino (Rhino 7/8, IronPython 2.7 or CPython 3 via rhinoscriptsyntax).

Input model
-----------
* One sublayer per room type (default APARTMENTS::Living / Bedroom / Kitchen / Bathroom / Shared).
* Each apartment is a Rhino group holding its room boxes. Shared boxes are ungrouped.
* Boxes sit on a regular grid (default 6 ft) laid out along a gently curving arc (a straight bar
  also works - it is treated as an arc with a huge radius), floors every FLOOR_HEIGHT.

Stages (run in this order; each one can be run on its own, per bar and per pair of levels)
------
  report      read-only: bars, grid fit, counts of everything each stage would make
  normalize   odd groups / stray rooms -> Shared, ungrouped
  wedges      snap every box onto the arc grid as an exact wedge, floors on one sloped plane per bar
  walls       exterior / party / interior walls + junction fixes (fillers, overhang trims)
  facade      fin facade on every exterior face, corner posts, inside-corner clipping
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
import math
from collections import defaultdict, Counter, deque

import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
from Rhino.Geometry import (Point3d, Vector3d, Plane, Brep, BoundingBox, Interval, Arc, Circle,
                            ArcCurve, LineCurve, PolyCurve, Surface, Extrusion, BrepSolidOrientation)

# --------------------------------------------------------------------------------------------
# CONFIG - every number the process depends on (feet)
# --------------------------------------------------------------------------------------------
IN = 1.0 / 12.0
CONFIG = dict(
    room_layers=dict(Living="APARTMENTS::Living", Bedroom="APARTMENTS::Bedroom",
                     Kitchen="APARTMENTS::Kitchen", Bathroom="APARTMENTS::Bathroom",
                     Shared="APARTMENTS::Shared"),
    grid=6.0,                 # plan grid (ft) along and across the bar
    floor_height=10.0,
    bar_gap=2.0,              # boxes closer than this belong to the same bar
    wall_t=dict(EXT=1.0, PARTY=0.5, INT=4 * IN),
    wall_layers=dict(EXT="WALLS::Exterior 12in", PARTY="WALLS::Party 6in", INT="WALLS::Interior 4in"),
    wall_colors=dict(EXT=(40, 40, 40), PARTY=(200, 70, 60), INT=(230, 200, 120)),
    shared_is_outside=True,   # walls facing Shared are exterior walls with facade
    fin_layer="Layer 02",
    fin_depth=2 * IN,
    fin_module=60 * IN,       # pattern repeat
    fin_pattern=[(0, 2), (4, 8), (10, 16), (18, 20), (22, 26), (28, 32), (34, 36), (38, 44),
                 (46, 48), (50, 52), (54, 58)],   # inches inside one module
    frame=2 * IN,
    clearance=6 * IN,         # door/window frame edge to the perpendicular wall face
    door_w=3.0, door_h=7.0, leaf_t=1.75 * IN,
    win_w=3.0, win_h=5.0, sill=3.0, glass_t=0.25 * IN,
    frames_layer="Frames", doors_layer="Doors", glass_layer="Glass",
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
    if rs.IsLayer(path):
        return
    parts = path.split("::")
    parent = None
    for k in range(len(parts)):
        full = "::".join(parts[:k + 1])
        if not rs.IsLayer(full):
            rs.AddLayer(parts[k], color, parent=parent)
        parent = full


def add_brep(brep, layer, group=None, attr=None):
    a = attr.Duplicate() if attr else Rhino.DocObjects.ObjectAttributes()
    if layer:
        a.LayerIndex = sc.doc.Layers.FindByFullPath(layer, -1)
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
    for k in ("fin_layer", "frames_layer", "doors_layer", "glass_layer"):
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


def read_boxes():
    out = []
    for kind, path in CONFIG["room_layers"].items():
        for ro in layer_objs(path):
            out.append(Box(ro, kind))
    return out


def split_bars(boxes):
    """Connected groups of boxes in plan (union-find on padded bounding boxes)."""
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
                        ba.Min.Y - pad <= bc.Max.Y and bc.Min.Y - pad <= ba.Max.Y):
                    ra, rc = find(lst[a]), find(lst[c])
                    if ra != rc:
                        parent[ra] = rc
    groups = defaultdict(list)
    for n in range(len(boxes)):
        groups[find(n)].append(boxes[n])
    bars = sorted(groups.values(), key=lambda bs: -len(bs))
    # name bars by position: west-most first -> B1, B2, ...
    bars.sort(key=lambda bs: (min(b.cy for b in bs), min(b.cx for b in bs)))
    return dict(("B%d" % (k + 1), bs) for k, bs in enumerate(bars))


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
        zbase = min(b.zmin for b in boxes)
        FH = CONFIG["floor_height"]
        for b in boxes:
            b.L = int(round((b.zmin - zbase) / FH))
            b.i0 = int(round((b.a1 - self.A0) / self.da)); b.i1 = int(round((b.a2 - self.A0) / self.da))
            b.j0 = int(round((b.rin - self.R0) / G)); b.j1 = int(round((b.rout - self.R0) / G))
            b.snap_err = max(abs((b.a1 - self.A0) / self.da - b.i0), abs((b.a2 - self.A0) / self.da - b.i1),
                             abs((b.rin - self.R0) / G - b.j0), abs((b.rout - self.R0) / G - b.j1))
        self.levels = sorted(set(b.L for b in boxes))
        # floor planes: one sloped ground plane for the bar, floors exactly FH apart
        pts = [(p, FH * b.L) for b in boxes for p in b.bot] + [(p, FH * (b.L + 1)) for b in boxes for p in b.top]
        x0 = sum(p.X for p, k in pts) / len(pts); y0 = sum(p.Y for p, k in pts) / len(pts)
        M = [[0.0] * 3 for _ in range(3)]; v = [0.0] * 3
        for p, k in pts:
            r = [1.0, p.X - x0, p.Y - y0]
            for i in range(3):
                v[i] += r[i] * (p.Z - k)
                for j in range(3):
                    M[i][j] += r[i] * r[j]
        a0, bx, by = _solve3(M, v)
        self.ground = (x0, y0, a0, bx, by)
        self.max_z_move = max(abs(p.Z - k - (a0 + bx * (p.X - x0) + by * (p.Y - y0))) for p, k in pts)
        # cells: rooms only when shared counts as outside
        self.cell, self.shared = {}, {}
        for b in boxes:
            target = self.shared if b.kind == "Shared" else self.cell
            for i in range(b.i0, b.i1):
                for j in range(b.j0, b.j1):
                    target[(b.L, i, j)] = b
        if not CONFIG["shared_is_outside"]:
            raise NotImplementedError("only shared_is_outside=True is implemented")

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
    if a is None:
        return ("EXT", +1)       # room on the + side
    if b is None:
        return ("EXT", -1)
    if a.id == b.id:
        return None
    return ("INT", 0) if a.group == b.group else ("PARTY", 0)


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
            k = wall_kind(g.cell.get((L, ii, jj - 1)), g.cell.get((L, ii, jj)))
            if k:
                AE[(L, jj, ii)] = (k[0], wall_band(k))
        for (ii, jj) in ((i, j), (i + 1, j)):
            k = wall_kind(g.cell.get((L, ii - 1, jj)), g.cell.get((L, ii, jj)))
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
def stage_normalize(dry_run):
    """Groups that are not a full apartment (no Living, Kitchen or Bathroom) and rooms that are
    in no group become ungrouped Shared space."""
    shared = CONFIG["room_layers"]["Shared"]
    moved = []
    grp = defaultdict(list)
    for kind, path in CONFIG["room_layers"].items():
        for ro in layer_objs(path):
            gs = rs.ObjectGroups(ro.Id)
            if not gs:
                if kind != "Shared":
                    moved.append((ro.Id, None))
                continue
            for gname in gs:
                grp[gname].append((ro.Id, kind))
    for gname, members in grp.items():
        kinds = set(k for o, k in members)
        if not set(("Living", "Kitchen", "Bathroom")) <= kinds:
            moved += [(o, gname) for o, k in members]
    print("normalize: %d boxes become ungrouped Shared" % len(moved))
    if dry_run:
        return
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
def stage_wedges(bars, dry_run):
    for name, boxes in bars.items():
        g = Grid(boxes)
        moves = []
        for b in boxes:
            moves.append(max(abs(g.R(b.j0) - b.rin), abs(g.R(b.j1) - b.rout),
                             abs(g.A0 + g.da * b.i0 - b.a1) * g.Rm, abs(g.A0 + g.da * b.i1 - b.a2) * g.Rm))
        moves.sort()
        print("%s: %d boxes | arc radius ~%.0f ft | step %.4f ft | edge moves median %.2f max %.2f ft | z moves max %.2f ft"
              % (name, len(boxes), g.Rm, g.da * g.Rm, moves[len(moves) // 2], moves[-1], g.max_z_move))
        if dry_run:
            continue
        built = []
        for b in boxes:
            w = g.trimmed(g.sector_curve(g.R(b.j0), g.R(b.j1), g.A(b.i0), g.A(b.i1)), b.L)
            if w is None or not w.IsSolid:
                raise RuntimeError("wedge failed for %s" % b.id)
            built.append((b, w))
        for b, w in built:
            ro = sc.doc.Objects.FindId(b.id)
            sc.doc.Objects.Replace(b.id, w) if ro else None
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
    for k, col in CONFIG["wall_colors"].items():
        ensure_layer(CONFIG["wall_layers"][k], col)
    for name, boxes in bars.items():
        g = Grid(boxes)
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
            if (L, oi, oj) in g.cell:
                continue
            faces[(kd, line, rsign, L)].append((n, "shared" if (L, oi, oj) in g.shared else "open"))
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
# STAGE facade
# --------------------------------------------------------------------------------------------
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
    ensure_layer(CONFIG["fin_layer"])
    for name, boxes in bars.items():
        g = Grid(boxes)
        lv = levels if levels is not None else g.levels
        empty = lambda k: k not in g.cell
        AF, RF = {}, {}
        for (L, i, j) in g.cell:
            if L not in lv:
                continue
            if empty((L, i, j - 1)): AF[(L, j, i)] = -1
            if empty((L, i, j + 1)): AF[(L, j + 1, i)] = +1
            if empty((L, i - 1, j)): RF[(L, i, j)] = -1
            if empty((L, i + 1, j)): RF[(L, i + 1, j)] = +1

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
        pieces = []
        for (L, j, sg), i_s, i_e in runs(AF):
            R = g.R(j); length = g.da * (i_e - i_s) * R
            r_lo, r_hi = (R - D, R) if sg < 0 else (R, R + D)
            for p0, p1 in segs(length, 0, 0):
                a0, a1 = g.A(i_s) + p0 / R, g.A(i_s) + p1 / R
                pieces.append(([g.polarxy(r_lo, a0), g.polarxy(r_lo, a1), g.polarxy(r_hi, a1), g.polarxy(r_hi, a0)], L))
        for (L, i, sg), j_s, j_e in runs(RF):
            a = g.A(i); d0, d1 = (-D, 0.0) if sg < 0 else (0.0, D)
            col = i - 1 if sg < 0 else i
            c0 = D if AF.get((L, j_s, col)) == +1 else 0.0       # inside corner: pull back
            c1 = D if AF.get((L, j_e, col)) == -1 else 0.0
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
        built = [g.prism(xy, L) for xy, L in pieces]
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
    if not openings:
        print("%s: nothing to do" % label)
        return
    if dry_run:
        bad = sum(1 for op in openings for b, l in op["parts"] if b is None or not b.IsSolid)
        print("%s: %d planned (dry run, nothing cut) | parts that failed to build: %d" % (label, len(openings), bad))
        return
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
        cutter = ext_box(g, P, L, op["s0"], op["s1"], op["n0"] - E, op["n1"] + E, op["z1"], zbot=op["z0"])
        cbb = cutter.GetBoundingBox(True)
        hit = 0
        for oid, bb, path in cand:
            if not (zb - 0.5 < bb.Center.Z < zb + FH):
                continue
            if not BoundingBox.Intersection(bb, cbb).IsValid:
                continue
            srcs = newgeo.get(oid) or [brep_of(sc.doc.Objects.FindId(oid).Geometry)]
            out = []; changed = False
            for gb in srcs:
                if not BoundingBox.Intersection(gb.GetBoundingBox(True), cbb).IsValid:
                    out.append(gb); continue
                pcs, ch = safe_difference(gb, cutter)
                out += pcs; changed = changed or ch
            if changed:
                newgeo[oid] = out
                if path in wall_layers:
                    hit += 1
        if hit < 1:
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
    for op in openings:
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
            if b and b.id != a.id and b.group == a.group:
                pairs[tuple(sorted((str(a.id), str(b.id))))].append((kd, L, i, j))
    adj = defaultdict(set)
    for a, b in pairs:
        adj[a].add(b); adj[b].add(a)
    by_g = defaultdict(list)
    for b in g.boxes:
        if b.kind != "Shared":
            by_g[b.group].append(b)
    tree = []
    for grp, rooms in by_g.items():
        liv = [r for r in rooms if r.kind == "Living"]
        if not liv:
            continue
        start = str(liv[0].id); seen = set([start]); q = deque([start])
        while q:
            x = q.popleft()
            for y in sorted(adj[x]):
                if y not in seen:
                    seen.add(y); q.append(y); tree.append(tuple(sorted((x, y))))
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
def front_door_of(g, AE, RE, b):
    """(key, n_s, n_e, s0, s1, frame, off) for the Living room's front door."""
    runs = exterior_faces(g, b)
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
    return dict(L=L, P=P, s0=s0, s1=s1, n0=n0, n1=n1, z0=z0, z1=z1, parts=parts, group=group)


def stage_frontdoors(bars, levels, dry_run):
    for name, boxes in bars.items():
        g = Grid(boxes); AE, RE = classify(g)
        ops = []; facing = Counter()
        for b in boxes:
            if b.kind != "Living" or (levels is not None and b.L not in levels):
                continue
            fd = front_door_of(g, AE, RE, b)
            if fd is None:
                facing["no exterior wall"] += 1; continue
            key, n_s, n_e, s0, s1, frame, off = fd
            facing[[r for r in exterior_faces(g, b) if r[2] == key and r[3] == n_s][0][0]] += 1
            ops.append(ext_opening(g, key, frame, s0, s1, "door", b.group))
        print("%s front doors facing: %s" % (name, dict(facing)))
        apply_openings(g, ops, [CONFIG["wall_layers"]["EXT"], CONFIG["fin_layer"]], [CONFIG["wall_layers"]["EXT"]],
                       dry_run, "%s front doors" % name)


def stage_windows(bars, levels, dry_run):
    W = CONFIG["win_w"] + 2 * CONFIG["frame"]; clr = CONFIG["clearance"]
    for name, boxes in bars.items():
        g = Grid(boxes); AE, RE = classify(g)
        ops = []; notes = Counter()
        by_group = defaultdict(list)
        for b in boxes:
            by_group[b.group].append(b)

        def plan_room(b, door, n_max):
            runs = exterior_faces(g, b)
            for wr in sorted(runs, key=lambda r: (r[0] == "open", r[1]), reverse=True):
                key, n_s, n_e = wr[2], wr[3], wr[4]
                length, near, far, frame, off = face_frame(g, AE, RE, key, n_s, n_e)
                if n_max > 1 and n_e - n_s >= CONFIG["two_windows_min_cells"]:
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
            if not exterior_faces(g, b):
                notes[b.kind + ": no exterior wall"] += 1; continue
            door = None
            if b.kind == "Living":
                fd = front_door_of(g, AE, RE, b)
                if fd:
                    door = (fd[0], fd[6] + fd[3], fd[6] + fd[4])
            wins, facing = plan_room(b, door, 2)
            if not wins and b.kind == "Living" and CONFIG["kitchen_fallback"]:
                kit = [k for k in by_group[b.group] if k.kind == "Kitchen"]
                if kit:
                    wins, facing = plan_room(kit[0], None, 1)
                    if wins:
                        notes["Living -> Kitchen window"] += 1
            if not wins:
                notes[b.kind + ": no room for a window"] += 1; continue
            notes["%s: %d on %s" % (b.kind, len(wins), facing)] += 1
            for key, frame, s0 in wins:
                ops.append(ext_opening(g, key, frame, s0, s0 + W, "window", b.group))
        print("%s windows: %s" % (name, dict(notes)))
        apply_openings(g, ops, [CONFIG["wall_layers"]["EXT"], CONFIG["fin_layer"]], [CONFIG["wall_layers"]["EXT"]],
                       dry_run, "%s windows" % name)


# --------------------------------------------------------------------------------------------
# STAGE report / verify
# --------------------------------------------------------------------------------------------
def stage_report(bars):
    for name, boxes in bars.items():
        g = Grid(boxes)
        kinds = Counter(b.kind for b in boxes)
        groups = len(set(b.group for b in boxes if b.group))
        print("%s: %d boxes %s | %d apartments | levels %s | arc radius ~%.0f ft, centre (%.1f, %.1f) | "
              "grid snap error max %.3f cells"
              % (name, len(boxes), dict(kinds), groups, g.levels, g.Rm, g.C[0], g.C[1], max(b.snap_err for b in boxes)))
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
STAGES = ["report", "normalize", "wedges", "walls", "facade", "doors", "frontdoors", "windows", "verify"]


def run(stage, bars=None, levels=None, dry_run=False):
    """stage: one of STAGES.  bars: list of bar names (from 'report') or None for all.
    levels: list of level indexes or None for all.  dry_run: plan and report, change nothing."""
    before = counts()
    if stage == "normalize":
        stage_normalize(dry_run); return
    if stage == "verify":
        stage_verify(); return
    all_bars = split_bars(read_boxes())
    sel = dict((k, v) for k, v in all_bars.items() if bars is None or k in bars)
    if stage == "report":
        stage_report(sel)
    elif stage == "wedges":
        stage_wedges(sel, dry_run)
    elif stage == "walls":
        stage_walls(sel, dry_run)
    elif stage == "facade":
        stage_facade(sel, levels, dry_run)
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
