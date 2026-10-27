# -*- coding: utf-8 -*-
"""
cladding.py -- clad every exposed vertical wall face with a board pattern you can adjust.

Runs in Rhino 7/8 (RunPythonScript, or the Script Editor).  Units: model units (feet here).

What it does
------------
1. Reads the wall volumes on WALL_LAYER (vertical prisms: walls straight up and down, flat tops).
2. Finds every vertical face and walks along it to keep only the stretches that are really outside:
   internal faces and the buried parts of partly-internal faces (left behind by booleans) are skipped.
3. Lays the board pattern on each exposed stretch, in rows (horizontal boards) or columns (vertical
   boards), and cuts every board to the wall's ends, bottom and top.
4. Where walls meet (inside corners, T-junctions), the board on the shorter wall gives way so no two
   boards overlap.
5. Puts the boards on OUTPUT_LAYER (cleared first if CLEAR_OUTPUT).

Adjust the PATTERN block, then run.  Set DRY_RUN = True to only print what would be made.
"""
from __future__ import print_function
import math
from collections import defaultdict, Counter

import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
from Rhino.Geometry import (Point3d, Vector3d, Brep, BoundingBox, PointFaceRelation, BrepSolidOrientation)

IN = 1.0 / 12.0     # one inch in feet

# ============================================================================================
# SETTINGS
# ============================================================================================
WALL_LAYER = "SEE::Planes-Booled"          # the wall volumes
OUTPUT_LAYER = "SEE::Western Red Cedar"     # where the boards go
CLEAR_OUTPUT = True                         # delete what is already ON OUTPUT_LAYER (not its sublayers)
DRY_RUN = False

# Read the pattern from existing sample boards instead of the numbers below
# (horizontal boards on a wall running along x or y). None = use PATTERN as written.
PATTERN_FROM_SAMPLE = None                  # e.g. "SEE::Western Red Cedar::Original sample"

PATTERN = dict(
    orientation="horizontal",      # "horizontal" boards in rows, or "vertical" boards in columns
    board_length=12.0,             # along the board
    board_width=7.125 * IN,        # face width of one board (row height for horizontal boards)
    board_thickness=0.057,         # depth off the wall
    standoff=0.0,                  # air gap between wall face and back of board (sample is flush)
    course_gap=1.0 * IN,           # gap between rows (horizontal) / columns (vertical)
    joint_gap=1.0 * IN,            # gap where two boards meet end to end
    course_offsets=[0.0, 6.0],     # joint shift of each row, repeating: [0, 6] = running bond,
                                   # [0, 4, 8] = thirds, [0] = stacked joints
    board_lengths=None,            # optional repeating list of lengths within a row, e.g. [12, 8, 10]
                                   # (None = every board is board_length)
    start="bottom",                # first row at the "bottom" of each wall, or "top"
    partial_last_course=False,     # True: cut a final row to fit; False: only full rows (strip left over)
    min_piece=1.0 * IN,            # drop cut pieces shorter / narrower than this
    joint_anchor=dict(x=3444.917, y=-102.8613),
                                   # where joint lines line up, measured along each wall from the origin:
                                   # x = walls running along x, y = walls running along y, other = any
                                   # other angle (a single number works for all). Change to slide the
                                   # pattern sideways.
    course_anchor=None,            # None = rows start at each wall's own bottom; a z value = rows line up
                                   # across the whole building from that height
)

ONLY_LEVELS_BETWEEN = None                  # (zmin, zmax) to clad only walls in that height band
SAMPLE_STEP = 0.25                          # how finely to look for exposed stretches along a wall
MIN_WALL_HEIGHT = 1.0                       # ignore vertical faces shorter than this

# ============================================================================================
TOL = sc.doc.ModelAbsoluteTolerance


def layer_objs(path):
    idx = sc.doc.Layers.FindByFullPath(path, -1)
    if idx < 0:
        raise ValueError("layer not found: %s" % path)
    return list(sc.doc.Objects.FindByLayer(sc.doc.Layers[idx]) or [])


def as_brep(g):
    return g.ToBrep() if hasattr(g, "ToBrep") and not isinstance(g, Brep) else g


def ensure_layer(path):
    if rs.IsLayer(path):
        return
    parts = path.split("::"); parent = None
    for k in range(len(parts)):
        full = "::".join(parts[:k + 1])
        if not rs.IsLayer(full):
            rs.AddLayer(parts[k], parent=parent)
        parent = full


# --------------------------------------------------------------------------------------------
# pattern from sample boards
# --------------------------------------------------------------------------------------------
def pattern_from_sample(path):
    boxes = [as_brep(o.Geometry).GetBoundingBox(True) for o in layer_objs(path)]
    if not boxes:
        raise ValueError("no sample boards on %s" % path)
    along_x = sum(1 for b in boxes if b.Max.X - b.Min.X > b.Max.Y - b.Min.Y) > len(boxes) / 2
    L = Counter(round(max(b.Max.X - b.Min.X, b.Max.Y - b.Min.Y), 4) for b in boxes).most_common(1)[0][0]
    T = Counter(round(min(b.Max.X - b.Min.X, b.Max.Y - b.Min.Y), 4) for b in boxes).most_common(1)[0][0]
    H = Counter(round(b.Max.Z - b.Min.Z, 4) for b in boxes).most_common(1)[0][0]
    zs = sorted(set(round(b.Min.Z, 4) for b in boxes))
    pitch = Counter(round(zs[k + 1] - zs[k], 4) for k in range(len(zs) - 1)).most_common(1)[0][0]
    rows = defaultdict(list)
    for b in boxes:
        rows[round(b.Min.Z, 4)].append(b.Min.X if along_x else b.Min.Y)
    starts = sorted(set(round(s, 4) for s in rows[zs[0]]))
    module = Counter(round(starts[k + 1] - starts[k], 4) for k in range(len(starts) - 1)).most_common(1)[0][0] if len(starts) > 1 else L
    nxt = sorted(set(round(s, 4) for s in rows[zs[1]])) if len(zs) > 1 else starts
    shift = round((nxt[0] - starts[0]) % module, 4)
    # standoff: from each board's two broad faces, step straight back (perpendicular to the wall)
    # until a wall volume is hit; the nearer hit is the gap behind the board. Median over boards.
    walls = [as_brep(o.Geometry) for o in layer_objs(WALL_LAYER)]
    gaps = []
    for b in boxes[::max(1, len(boxes) // 25)]:
        c = b.Center
        for sgn in (-1, 1):
            if along_x:
                start = Point3d(c.X, b.Min.Y if sgn < 0 else b.Max.Y, c.Z); d = Vector3d(0, sgn, 0)
            else:
                start = Point3d(b.Min.X if sgn < 0 else b.Max.X, c.Y, c.Z); d = Vector3d(sgn, 0, 0)
            ray = Rhino.Geometry.LineCurve(start, start + d * 2.0)
            for w in walls:
                ok, crvs, pts = Rhino.Geometry.Intersect.Intersection.CurveBrep(ray, w, TOL)
                if ok and pts:
                    gaps.append(min(start.DistanceTo(q) for q in pts))
    gaps.sort()
    standoff = gaps[len(gaps) // 2] if gaps else PATTERN["standoff"]
    p = dict(PATTERN)
    ja = dict(p["joint_anchor"]) if isinstance(p["joint_anchor"], dict) else {}
    ja["x" if along_x else "y"] = starts[0]
    p.update(orientation="horizontal", board_length=L, board_width=H, board_thickness=T,
             course_gap=round(pitch - H, 5), joint_gap=round(module - L, 5),
             course_offsets=[0.0, shift] if shift > 1e-3 else [0.0],
             standoff=round(standoff, 4), joint_anchor=ja or starts[0])
    print("pattern read from %s: %s" % (path, dict((k, p[k]) for k in
          ("board_length", "board_width", "board_thickness", "standoff", "course_gap", "joint_gap", "course_offsets", "joint_anchor"))))
    return p


# --------------------------------------------------------------------------------------------
# which side of a wall is outside
# --------------------------------------------------------------------------------------------
class Volume(object):
    """A wall volume. Inside test uses its top faces as the plan footprint (robust against the
    internal faces and non-manifold edges booleans leave behind); falls back to the solid test."""

    def __init__(self, brep):
        self.brep = brep
        self.bb = brep.GetBoundingBox(True)
        self.tops = []
        for f in brep.Faces:
            fb = f.GetBoundingBox(True)
            if abs(fb.Min.Z - self.bb.Max.Z) < 0.01 and abs(fb.Max.Z - self.bb.Max.Z) < 0.01:
                self.tops.append((f, fb))
        area = sum(Rhino.Geometry.AreaMassProperties.Compute(f.DuplicateFace(False)).Area for f, fb in self.tops)
        vol = brep.GetVolume() if brep.IsSolid else 0
        h = self.bb.Max.Z - self.bb.Min.Z
        self.prism = bool(self.tops) and (not brep.IsSolid or abs(area * h - vol) < 0.01 * max(vol, 1))

    def covers_z(self, z):
        return self.bb.Min.Z - 0.01 <= z <= self.bb.Max.Z + 0.01

    def inside(self, x, y, z):
        if not self.covers_z(z):
            return False
        if self.prism:
            for f, fb in self.tops:
                if fb.Min.X - 0.01 <= x <= fb.Max.X + 0.01 and fb.Min.Y - 0.01 <= y <= fb.Max.Y + 0.01:
                    ok, u, v = f.ClosestPoint(Point3d(x, y, self.bb.Max.Z))
                    if ok and f.IsPointOnFace(u, v) == PointFaceRelation.Interior:
                        return True
            return False
        return self.brep.IsPointInside(Point3d(x, y, z), 0.001, True)


def inside_any(vols, x, y, z):
    return any(v.inside(x, y, z) for v in vols)


def exposed_runs(vols, origin, t, n, lo, hi, z):
    """Along the wall line origin + t*s (s in [lo, hi]) at height z: stretches where exactly one side
    is inside a volume. Returns [(s0, s1, side)] with side = +1 if the outside is along +n."""
    def state(s):
        p = origin + t * s
        a = inside_any(vols, p.X + n.X * 0.05, p.Y + n.Y * 0.05, z)
        b = inside_any(vols, p.X - n.X * 0.05, p.Y - n.Y * 0.05, z)
        return 0 if a == b else (-1 if a else +1)

    def edge(a, b, want):
        for _ in range(12):
            m = (a + b) / 2.0
            if state(m) == want:
                a = m
            else:
                b = m
        return (a + b) / 2.0
    cnt = max(2, int((hi - lo) / SAMPLE_STEP) + 1)
    ss = [lo + (hi - lo) * k / (cnt - 1.0) for k in range(cnt)]
    ss[0] += 0.01; ss[-1] -= 0.01
    st = [state(s) for s in ss]
    runs = []; k = 0
    while k < cnt:
        if st[k] == 0:
            k += 1; continue
        j = k
        while j + 1 < cnt and st[j + 1] == st[k]:
            j += 1
        s0 = lo if k == 0 else edge(ss[k], ss[k - 1], st[k])
        s1 = hi if j == cnt - 1 else edge(ss[j], ss[j + 1], st[k])
        runs.append((s0, s1, st[k]))
        k = j + 1
    return runs


def wall_stretches(vols):
    """Every exposed stretch of vertical wall: dict(origin, t, n(outward), s0, s1, z0, z1, length)."""
    lines = {}
    for v in vols:
        b = v.brep.DuplicateBrep(); b.MergeCoplanarFaces(TOL)
        for f in b.Faces:
            ok, pl = f.TryGetPlane(0.01)
            if not ok or abs(pl.Normal.Z) > 0.01:
                continue
            fb = f.GetBoundingBox(True)
            if fb.Max.Z - fb.Min.Z < MIN_WALL_HEIGHT:
                continue
            if ONLY_LEVELS_BETWEEN and not (ONLY_LEVELS_BETWEEN[0] <= fb.Min.Z <= ONLY_LEVELS_BETWEEN[1]):
                continue
            n = Vector3d(pl.Normal.X, pl.Normal.Y, 0); n.Unitize()
            t = Vector3d(-n.Y, n.X, 0)
            if t.X < -1e-9 or (abs(t.X) < 1e-9 and t.Y < 0):
                t = -t                                   # walls always run toward +x / +y
            n = Vector3d(-t.Y, t.X, 0)                   # one normal per wall line (outside picked later)
            c = fb.Center
            off = round(c.X * n.X + c.Y * n.Y, 4)        # position of the wall line across
            ends = []
            for vi in f.AdjacentEdges():
                for p in (b.Edges[vi].PointAtStart, b.Edges[vi].PointAtEnd):
                    ends.append(p.X * t.X + p.Y * t.Y)
            key = (round(n.X, 4), round(n.Y, 4), off, round(fb.Min.Z, 3), round(fb.Max.Z, 3))
            lines.setdefault(key, []).append((min(ends), max(ends)))
    out = []
    for (nx, ny, off, z0, z1), spans in lines.items():
        n = Vector3d(nx, ny, 0); t = Vector3d(ny, -nx, 0)
        origin = Point3d(n.X * off, n.Y * off, 0)
        spans.sort(); merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        zmid = (z0 + z1) / 2.0
        for a, b in merged:
            for s0, s1, side in exposed_runs(vols, origin, t, n, a, b, zmid):
                out.append(dict(origin=origin, t=t, n=n * side, s0=s0, s1=s1, z0=z0, z1=z1, length=s1 - s0))
    return out


# --------------------------------------------------------------------------------------------
# laying out the pattern on one stretch (u = along the wall, v = up)
# --------------------------------------------------------------------------------------------
def layout(st, P):
    """Board rectangles on one wall stretch: list of (u0, u1, v0, v1)."""
    W, G = P["board_width"], P["course_gap"]
    pitch = W + G
    lengths = P["board_lengths"] or [P["board_length"]]
    offs = P["course_offsets"] or [0.0]
    u_lo, u_hi, v_lo, v_hi = st["s0"], st["s1"], st["z0"], st["z1"]
    horizontal = P["orientation"] == "horizontal"
    # course axis: v for horizontal boards, u for vertical boards
    c_lo, c_hi = (v_lo, v_hi) if horizontal else (u_lo, u_hi)
    a_lo, a_hi = (u_lo, u_hi) if horizontal else (v_lo, v_hi)
    if horizontal and P["course_anchor"] is not None:
        first = P["course_anchor"] + math.ceil((c_lo - P["course_anchor"]) / pitch - 1e-9) * pitch
    else:
        first = c_lo
    courses = []
    if P["start"] == "top":
        c = c_hi - W; k = 0
        while c >= c_lo - 1e-6:
            courses.append((k, c)); c -= pitch; k += 1
    else:
        c = first; k = 0
        while c + W <= c_hi + 1e-6:
            courses.append((k, c)); c += pitch; k += 1
    if P["partial_last_course"]:
        if P["start"] == "top":
            c = courses[-1][1] - pitch if courses else c_hi - W
            if c + W - c_lo >= P["min_piece"]:
                courses.append((len(courses), c))
        else:
            c = courses[-1][1] + pitch if courses else first
            if c_hi - c >= P["min_piece"]:
                courses.append((len(courses), c))
    if horizontal:
        ja = P["joint_anchor"]
        if isinstance(ja, dict):
            t = st["t"]
            key = "x" if abs(t.X) > 0.999 else "y" if abs(t.Y) > 0.999 else "other"
            anchor = ja.get(key, ja.get("other", 0.0))
        else:
            anchor = ja
    else:
        anchor = P["course_anchor"] or 0.0
    cycle = sum(L + P["joint_gap"] for L in lengths)
    out = []
    for k, c in courses:
        c0, c1 = max(c, c_lo), min(c + W, c_hi)
        if c1 - c0 < P["min_piece"]:
            continue
        start = anchor + offs[k % len(offs)]
        start -= math.ceil((start - a_lo) / cycle + 1) * cycle          # step back past the wall start
        idx = 0; b0 = start
        while b0 < a_hi:
            L = lengths[idx % len(lengths)]
            b1 = b0 + L
            x0, x1 = max(b0, a_lo), min(b1, a_hi)
            if x1 - x0 >= P["min_piece"]:
                out.append((x0, x1, c0, c1) if horizontal else (c0, c1, x0, x1))
            b0 = b1 + P["joint_gap"]; idx += 1
    return out


class Board(object):
    __slots__ = ("o", "t", "n", "u0", "u1", "v0", "v1", "w0", "w1", "rank", "alive")

    def __init__(self, st, u0, u1, v0, v1, P):
        self.o, self.t, self.n = st["origin"], st["t"], st["n"]
        self.u0, self.u1, self.v0, self.v1 = u0, u1, v0, v1
        self.w0 = P["standoff"]; self.w1 = P["standoff"] + P["board_thickness"]
        self.rank = st["length"]
        self.alive = True

    def corners(self, u0=None, u1=None):
        u0 = self.u0 if u0 is None else u0; u1 = self.u1 if u1 is None else u1
        return [self.o + self.t * u + self.n * w for u, w in ((u0, self.w0), (u1, self.w0), (u1, self.w1), (u0, self.w1))]

    def bbox(self):
        bb = BoundingBox([Point3d(p.X, p.Y, z) for p in self.corners() for z in (self.v0, self.v1)])
        return bb


def rect_overlap(A, B):
    """Plan overlap of two boards' rectangles (separating axis test) plus height overlap."""
    if min(A.v1, B.v1) - max(A.v0, B.v0) <= 1e-4:
        return False
    pa, pb = A.corners(), B.corners()
    for ax in (A.t, A.n, B.t, B.n):
        a = [p.X * ax.X + p.Y * ax.Y for p in pa]; b = [p.X * ax.X + p.Y * ax.Y for p in pb]
        if min(max(a), max(b)) - max(min(a), min(b)) <= 1e-4:
            return False
    return True


def covered_interval(A, B):
    """Part of A's length (u) taken by B's rectangle within A's depth band."""
    pts = []
    for p in B.corners():
        v = p - A.o
        pts.append((v.X * A.t.X + v.Y * A.t.Y, v.X * A.n.X + v.Y * A.n.Y))
    for lim, keep_above in ((A.w0, True), (A.w1, False)):           # clip polygon to w0 <= w <= w1
        res = []
        for i in range(len(pts)):
            p, q = pts[i], pts[(i + 1) % len(pts)]
            pin = p[1] >= lim if keep_above else p[1] <= lim
            qin = q[1] >= lim if keep_above else q[1] <= lim
            if pin:
                res.append(p)
            if pin != qin:
                f = (lim - p[1]) / (q[1] - p[1])
                res.append((p[0] + f * (q[0] - p[0]), lim))
        pts = res
        if not pts:
            return None
    us = [p[0] for p in pts]
    return min(us), max(us)


def resolve_overlaps(boards, P):
    """Where boards from different walls collide, the one on the shorter wall is cut back."""
    passes = 0; trims = 0
    while passes < 12:
        passes += 1
        grid = defaultdict(list)
        for i, b in enumerate(boards):
            if not b.alive:
                continue
            bb = b.bbox()
            for x in range(int(bb.Min.X // 4), int(bb.Max.X // 4) + 1):
                for y in range(int(bb.Min.Y // 4), int(bb.Max.Y // 4) + 1):
                    grid[(x, y, int(bb.Min.Z // 4))].append(i)
        new = []; changed = False; seen = set()
        for lst in grid.values():
            for a in range(len(lst)):
                for c in range(a + 1, len(lst)):
                    i, j = lst[a], lst[c]
                    if (i, j) in seen:
                        continue
                    seen.add((i, j))
                    A, B = boards[i], boards[j]
                    if not (A.alive and B.alive) or not rect_overlap(A, B):
                        continue
                    lose, keep = (A, B) if (A.rank, -i) < (B.rank, -j) else (B, A)
                    cov = covered_interval(lose, keep)
                    lose.alive = False; changed = True; trims += 1
                    if cov is None:
                        continue
                    for u0, u1 in ((lose.u0, cov[0]), (cov[1], lose.u1)):
                        if u1 - u0 >= P["min_piece"]:
                            nb = Board.__new__(Board)
                            for s in Board.__slots__:
                                setattr(nb, s, getattr(lose, s))
                            nb.u0, nb.u1, nb.alive = u0, u1, True
                            new.append(nb)
        boards = [b for b in boards if b.alive] + new
        if not changed:
            break
    return boards, trims, passes


def make_brep(b):
    pts = b.corners()
    B = [Point3d(p.X, p.Y, b.v0) for p in pts]; T = [Point3d(p.X, p.Y, b.v1) for p in pts]
    faces = [Brep.CreateFromCornerPoints(B[0], B[1], B[2], B[3], TOL), Brep.CreateFromCornerPoints(T[0], T[1], T[2], T[3], TOL)]
    for k in range(4):
        m = (k + 1) % 4
        faces.append(Brep.CreateFromCornerPoints(B[k], B[m], T[m], T[k], TOL))
    j = Brep.JoinBreps(faces, TOL)
    if not j or len(j) != 1:
        return None
    br = j[0]
    if br.SolidOrientation == BrepSolidOrientation.Inward:
        br.Flip()
    return br


# --------------------------------------------------------------------------------------------
def main():
    P = pattern_from_sample(PATTERN_FROM_SAMPLE) if PATTERN_FROM_SAMPLE else dict(PATTERN)
    vols = [Volume(as_brep(o.Geometry)) for o in layer_objs(WALL_LAYER)]
    print("wall volumes: %d (%d read as vertical prisms)" % (len(vols), sum(v.prism for v in vols)))
    stretches = wall_stretches(vols)
    boards = []
    for st in stretches:
        for u0, u1, v0, v1 in layout(st, P):
            boards.append(Board(st, u0, u1, v0, v1, P))
    n_laid = len(boards)
    boards, trims, passes = resolve_overlaps(boards, P)
    lengths = Counter("full" if abs((b.u1 - b.u0 if P["orientation"] == "horizontal" else b.v1 - b.v0) - P["board_length"]) < 1e-4
                      else "cut" for b in boards)
    print("exposed wall stretches: %d | boards laid: %d | corner/T clean-ups: %d (%d passes) | final boards: %d %s"
          % (len(stretches), n_laid, trims, passes, len(boards), dict(lengths)))
    if DRY_RUN:
        print("dry run - nothing changed")
        return
    breps = [make_brep(b) for b in boards]
    if any(br is None or not br.IsSolid for br in breps):
        raise RuntimeError("some boards failed to build - nothing changed")
    ensure_layer(OUTPUT_LAYER)
    if CLEAR_OUTPUT:
        old = layer_objs(OUTPUT_LAYER)
        for o in old:
            sc.doc.Objects.Delete(o, True)
        print("cleared %d objects from %s" % (len(old), OUTPUT_LAYER))
    li = sc.doc.Layers.FindByFullPath(OUTPUT_LAYER, -1)
    for br in breps:
        a = Rhino.DocObjects.ObjectAttributes(); a.LayerIndex = li
        sc.doc.Objects.AddBrep(br, a)
    sc.doc.Views.Redraw()
    print("added %d boards to %s" % (len(breps), OUTPUT_LAYER))


if __name__ == "__main__":
    main()
