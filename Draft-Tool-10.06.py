
# r: numpy
from __future__ import annotations
import Rhino
import Rhino.Geometry as rg
from Rhino.Geometry.Intersect import Intersection
import scriptcontext as sc
import rhinoscriptsyntax as rs #slower than rg
import math
import random
import System.Drawing
from System.Collections.Generic import List
from enum import IntEnum
from typing import NamedTuple, Iterator, Sequence
import numpy as np
OT = Rhino.DocObjects.ObjectType
import copy
#import Rhino.Geometry.RTree
#import Rhino.display - temporary graphics

#=================================================
#=================================================
#=================================================
#-------------------------------------- PARAMETERS 

node_distance = 10.0   # default node spacing across and along the road (asked at run time)
floor_height = None    # default floor height; None = same as node_distance (asked at run time)
density = 0

# How likely each apartment type is to be picked on each placement round.
# Weights are relative: 3 is picked three times as often as 1, 0 never.
# Big apartments fail to fit more often, so the final mix leans smaller than the weights.
apartment_weights = {
    "onebedA":   1,
    "onebedB":   1,
    "onebedC":   1,
    "twobedA":   1,
    "twobedB":   1,
    "threebedA": 1,
    "threebedB": 1,
}

#also ask user to select bridge and topo layer

#=================================================
#------------------------------------- PREPARATION

# Ask user for Road and Topography
def get_one(prompt, geometry_filter): 
    go = Rhino.Input.Custom.GetObject()
    go.SetCommandPrompt(prompt)
    go.GeometryFilter = geometry_filter
    go.SubObjectSelect = False
    go.EnablePreSelect(False, True)

    go.Get()
    if go.CommandResult() != Rhino.Commands.Result.Success:
        return None
    return go.Object(0)

def pick_road():
    ref = get_one("Select the road surface", OT.Surface | OT.PolysrfFilter)
    return ref.Brep() if ref else None

def pick_topo():
    ref = get_one("Select the Topo (mesh or surface)", OT.Mesh | OT.Surface | OT.PolysrfFilter)
    if ref is None:
            return None
    mesh = ref.Mesh()
    if mesh is not None:
        return mesh
    parts = rg.Mesh.CreateFromBrep(ref.Brep(), rg.MeshingParameters.Default)
    if not parts:
        return None
    joined = rg.Mesh()
    for m in parts:
        joined.Append(m)
    return joined

#=================================================
#------------------------------------------ DEFINE
#______________________________ 
class ProgramState(IntEnum):
    EMPTY = 0
    USED = 1
    SHARED = 2
    LIVING = 3
    BEDROOM = 4
    KITCHEN = 5
    BATHROOM = 6
    OUTSIDE = 7   # past the road edge or below the topo, never buildable

FACE_OFFSETS = np.array([
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
])

#______________________________
class Node(NamedTuple):
    #a node's position in the grid's *index* space
    #neighbors are always index +/= 1, no matter how the curve bends
    #Use Grid.position(node) to get the node's real (X, Y, Z) in Rhino
    #i = along the road, j = across the road, k = floors down (k=0 is right under the road)
    i: int
    j: int
    k: int

#______________________________
class Grid:
    def __init__(self, positions: np.ndarray, corners: np.ndarray | None = None):
        #positions: float array of shape (n_div, n_off, n_z, 3) holding world X, Y, Z
        #corners: optional (n_div+1, n_off+1, n_z+1, 3) lattice of cell corners.
        #   Node (i, j, k) is the cell between corners [i:i+2, j:j+2, k:k+2].
        self.positions = positions
        self.corners = corners
        self.shape = positions.shape[:3]
        self.states = np.full(self.shape, ProgramState.EMPTY, dtype=np.uint8)

    @classmethod
    def from_points(cls, points: Sequence[tuple[float, float, float]],
                    n_div: int, n_off: int, n_z: int) -> Grid:
        arr = np.asarray(points, dtype=float)
        expected = n_div * n_off * n_z
        if arr.shape != (expected, 3):
            raise ValueError(f"expected {expected} points of (x, y, z), got shape {arr.shape}")
        return cls(arr.reshape(n_div, n_off, n_z, 3))


    # -- world-space lookups

    def position(self, node: Node) -> tuple[float, float, float]:
        x, y, z = self.positions[node]
        return float(x), float(y), float(z)

    def nearest_node(self, x: float, y: float, z: float) -> Node:
        #node closest to a world-space point
        flat = self.positions.reshape(-1, 3)
        dist_sq = np.sum((flat - (x, y, z)) ** 2, axis=1)
        return Node(*map(int, np.unravel_index(np.argmin(dist_sq), self.shape)))

    def box_corners(self, i: int, j: int, k: int, di: int, dj: int, dk: int) -> np.ndarray:
        #world position of the 8 cornder nodes of an index-space box
        i2, j2, k2 = i + di - 1, j + dj - 1, k + dk - 1
        return np.array([self.positions[a, b, c]
                        for a in (i, i2) for b in (j, j2) for c in (k, k2)])

    def cell_corners(self, i: int, j: int, k: int, di: int, dj: int, dk: int) -> np.ndarray:
        #8 outer corners of an index-space box of cells, in Rhino box order:
        #bottom face (0-3) then top face (4-7). k grows downward, so k + dk is the bottom.
        c = self.corners
        i2, j2, k2 = i + di, j + dj, k + dk
        return np.array([c[i, j, k2], c[i2, j, k2], c[i2, j2, k2], c[i, j2, k2],
                         c[i, j, k],  c[i2, j, k],  c[i2, j2, k],  c[i, j2, k]])


    # -- single nodes
    
    def in_bounds(self, i: int, j: int, k: int) -> bool:
        a, b, c = self.shape
        return 0 <= i < a and 0 <= j < b and 0 <= k < c

    def get_state(self, i: int, j: int, k: int) -> ProgramState:
        return ProgramState(self.states[i, j, k])

    def is_empty(self, i: int, j: int, k: int) -> bool:
        return self.in_bounds(i, j, k) and self.states[i, j, k] == ProgramState.EMPTY

    def occupy(self, i: int, j: int, k: int, state: ProgramState = ProgramState.USED) -> None:
        self.states[i, j, k] = state

    def release(self,  i: int, j: int, k: int) -> None:
        self.states[i, j, k] = ProgramState.EMPTY


    # -- neighbors

    def neighbors(self, node: Node) -> Iterator[Node]:
        #yield the in-bounds face neighbors of a node
        coords = np.asarray(node) + FACE_OFFSETS
        mask = np.all((coords >= 0) & (coords < self.shape), axis=1)
        for c in coords[mask]:
            yield Node(*map(int, c))

    def empty_neighbors(self, node: Node) -> Iterator[Node]:
        for n in self.neighbors(node):
            if self.states[n] == ProgramState.EMPTY:
                yield n

    # -- boxes (sizes are in nodes, not world units) --

    def box_fits(self, i: int, j: int, k: int, di: int, dj: int, dk: int) -> bool:
        #True if a di*dj*dk box at (i, j, k) is fully in bounds and empty.
        if i < 0 or j < 0 or k < 0:
            return False
        region = self.states[i:i + di, j:j + dj, k:k + dk]
        return region.shape == (di, dj, dk) and not region.any()

    def place_box(self, i: int, j: int, k: int, di: int, dj: int, dk: int, state: ProgramState = ProgramState.USED) -> bool:
        #Fill a box if it fits. Return False and changes nothing if no fit
        if not self.box_fits(i, j, k, di, dj, dk):
            return False
        self.states[i:i + di, j:j + dj, k:k + dk] = state
        return True

    def clear_box(self, i: int, j: int, k: int, di: int, dj: int, dk: int) -> None:
        self.states[i:i + di, j:j + dj, k:k + dk] = ProgramState.EMPTY


    # -- free nodes --

    def count_free(self) -> int:
        return int(np.count_nonzero(self.states == ProgramState.EMPTY))

    def first_free(self) -> Node | None:
        # the first empty node in i, j, k order
        flat = np.flatnonzero(self.states == ProgramState.EMPTY)
        if flat.size == 0:
            return None
        return Node(*map(int, np.unravel_index(flat[0], self.shape)))

    def free_nodes(self) -> list[Node]:
        return [Node(*map(int, c)) for c in np.argwhere(self.states == ProgramState.EMPTY)]

        
#______________________________
class ProgramLayout(IntEnum):
    OXW = 0   # One x Two
    WXW = 1   # tWo x Two
    HXW = 2   # tHree x Two

LAYOUT_WIDTH = {ProgramLayout.OXW: 1, ProgramLayout.WXW: 2, ProgramLayout.HXW: 3}
PROGRAM_DEPTH = 2    # every layout is 2 nodes deep
PROGRAM_HEIGHT = 1   # every program sits on one floor

# code -> (state, layout). The number in the code is the width in nodes.
PROGRAMS = {
    "L1":  (ProgramState.LIVING,   ProgramLayout.OXW),
    "L2":  (ProgramState.LIVING,   ProgramLayout.WXW),
    "BA":  (ProgramState.BATHROOM, ProgramLayout.OXW),
    "BR1": (ProgramState.BEDROOM,  ProgramLayout.WXW),
    "BR2": (ProgramState.BEDROOM,  ProgramLayout.HXW),
    "K1":  (ProgramState.KITCHEN,  ProgramLayout.WXW),
    "K2":  (ProgramState.KITCHEN,  ProgramLayout.HXW),
    "S1":  (ProgramState.SHARED,   ProgramLayout.OXW),
    "S2":  (ProgramState.SHARED,   ProgramLayout.WXW),
    "S3":  (ProgramState.SHARED,   ProgramLayout.HXW),
}

#______________________________
class Program:
    #One room of an apartment, as a box of nodes: width x 2 deep x 1 floor.
    #rotated=True swaps width and depth (the box turns 90 degrees in plan).
    #A program only exists in the grid after try_place_at / place_first / place_adjacent succeeds.

    def __init__(self, code: str):
        if code not in PROGRAMS:
            raise ValueError(f"unknown program code {code!r}")
        self.code = code
        self.type, self.layout = PROGRAMS[code]
        self.origin: Node | None = None
        self.rotated = False
        self.nodes: list[Node] = []

    @classmethod
    def from_code(cls, code: str) -> Program:
        return cls(code)

    def __repr__(self) -> str:
        if self.origin is None:
            return f"Program({self.code}, unplaced)"
        turn = ", rotated" if self.rotated else ""
        return f"Program({self.code}, at {tuple(self.origin)}{turn})"

    @property
    def is_placed(self) -> bool:
        return self.origin is not None


    # -- size and conflicts

    def footprint(self, rotated: bool = False) -> tuple[int, int, int]:
        #(di, dj, dk) size of the box in nodes
        w = LAYOUT_WIDTH[self.layout]
        if rotated:
            return PROGRAM_DEPTH, w, PROGRAM_HEIGHT
        return w, PROGRAM_DEPTH, PROGRAM_HEIGHT

    def check_conflict(self, grid: Grid, i: int, j: int, k: int, rotated: bool = False) -> bool:
        #True if the box at (i, j, k) runs out of bounds or hits a used node
        return not grid.box_fits(i, j, k, *self.footprint(rotated))

    def candidate_boxes(self, anchor: Node) -> list[tuple[int, int, int, bool]]:
        #Every box origin (i, j, k, rotated) whose box still covers the anchor node.
        #Order: as-is first, then moved over (smallest shift first), then rotated.
        options = [False]
        if self.footprint(False) != self.footprint(True):
            options.append(True)   # a square layout looks the same rotated
        out = []
        for rotated in options:
            di, dj, _ = self.footprint(rotated)
            shifts = sorted(((si, sj) for si in range(di) for sj in range(dj)),
                            key=lambda s: s[0] + s[1])
            for si, sj in shifts:
                out.append((anchor.i - si, anchor.j - sj, anchor.k, rotated))
        return out


    # -- placing

    def try_place_at(self, grid: Grid, anchor: Node) -> bool:
        #Place the box so it covers the anchor, moving over or rotating on conflict
        for i, j, k, rotated in self.candidate_boxes(anchor):
            if self.check_conflict(grid, i, j, k, rotated):
                continue
            di, dj, dk = self.footprint(rotated)
            grid.place_box(i, j, k, di, dj, dk, state=self.type)
            self.origin = Node(i, j, k)
            self.rotated = rotated
            self.nodes = [Node(a, b, c)
                          for a in range(i, i + di)
                          for b in range(j, j + dj)
                          for c in range(k, k + dk)]
            return True
        return False

    def first_anchors(self, grid: Grid, k: int) -> list[Node]:
        #Free nodes on floor k: smallest-i column first, random order inside a column
        free = [n for n in grid.free_nodes() if n.k == k]
        anchors = []
        for col in sorted({n.i for n in free}):
            column = [n for n in free if n.i == col]
            random.shuffle(column)
            anchors.extend(column)
        return anchors

    def place_first(self, grid: Grid, k: int) -> bool:
        #Living: random free node in the smallest-i column of floor k.
        #If nothing fits there, try the next column over.
        return any(self.try_place_at(grid, n) for n in self.first_anchors(grid, k))

    def place_adjacent(self, grid: Grid, placed: Sequence[Program]) -> bool:
        #Other programs: start from an empty node touching an already placed program
        frontier = sorted({n for p in placed for node in p.nodes
                           for n in grid.empty_neighbors(node) if n.k == node.k})
        random.shuffle(frontier)
        for node in frontier:
            if self.try_place_at(grid, node):
                return True
        return False

    def remove(self, grid: Grid) -> None:
        if self.origin is None:
            return
        grid.clear_box(*self.origin, *self.footprint(self.rotated))
        self.origin = None
        self.rotated = False
        self.nodes = []

    def world_corners(self, grid: Grid) -> np.ndarray:
        #8 world-space corner points of the placed box (for drawing in Rhino)
        return grid.cell_corners(*self.origin, *self.footprint(self.rotated))

#______________________________
class Apartment:
    def __init__(self, type: str, programs: Sequence[str], floor: int = 0, number: int | None = None):
        self.type = type
        self.programs = [Program(code) for code in programs]
        self.floor = floor
        self.number = number

    def __repr__(self) -> str:
        return f"Apartment({self.type} #{self.number}, floor {self.floor}, {self.programs})"

    # -- presets (edit these to change what each apartment holds)
    @classmethod
    def onebedA(cls, floor: int = 0):
        return cls("onebedA", ["L1", "BA", "K1"], floor)
    @classmethod
    def onebedB(cls, floor: int = 0):
        return cls("onebedB", ["L1", "BA", "BR1", "K1"], floor)
    @classmethod
    def onebedC(cls, floor: int = 0):
        return cls("onebedC", ["L2", "BA", "BR1", "K1"], floor)
    @classmethod
    def twobedA(cls, floor: int = 0):
        return cls("twobedA", ["L2", "BA", "BR1", "BR1", "K1"], floor)
    @classmethod
    def twobedB(cls, floor: int = 0):
        return cls("twobedB", ["L2", "BA", "BA", "BR2", "BR1", "K2"], floor)
    @classmethod
    def threebedA(cls, floor: int = 0):
        return cls("threebedA", ["L2", "BA", "BA", "BR2", "BR1", "BR1", "K2"], floor)
    @classmethod
    def threebedB(cls, floor: int = 0):
        return cls("threebedB", ["L2", "BA", "BA", "BR2", "BR2", "BR1", "K2"], floor)

    def place(self, grid: Grid) -> bool:
        #Living first, then every other program next to what is already placed.
        #If the rest doesn't fit around that Living, undo and try the next Living spot.
        #If no Living spot works, the grid is left unchanged and this returns False.
        living = [p for p in self.programs if p.type == ProgramState.LIVING]
        if not living:
            raise ValueError(f"{self.type} has no living program")
        first = living[0]
        rest = [p for p in self.programs if p is not first]

        tried = set()
        for anchor in first.first_anchors(grid, self.floor):
            if not first.try_place_at(grid, anchor):
                continue
            key = (first.origin, first.rotated)
            if key in tried:          # same Living box as an earlier anchor
                first.remove(grid)
                continue
            tried.add(key)

            placed = [first]
            for p in rest:
                if not p.place_adjacent(grid, placed):
                    break
                placed.append(p)
            else:
                return True
            for q in placed:
                q.remove(grid)
        return False

APARTMENT_TYPES = [
    Apartment.onebedA, Apartment.onebedB, Apartment.onebedC,
    Apartment.twobedA, Apartment.twobedB,
    Apartment.threebedA, Apartment.threebedB,
]

#______________________________
def fill_shared_space(grid: Grid, floor: int = 0) -> list[Program]:
    #Leftover gaps become shared space, biggest piece first (S3 -> S2 -> S1).
    #A single node with no empty neighbor stays EMPTY.
    shared = []
    for node in grid.free_nodes():
        if node.k != floor or not grid.is_empty(*node):
            continue
        for code in ("S3", "S2", "S1"):
            s = Program(code)
            if s.try_place_at(grid, node):
                shared.append(s)
                break
    return shared

def type_weights(apartment_types, weights: dict[str, float] | None) -> list[float]:
    #One weight per apartment type, looked up by preset name (missing names count as 1)
    if weights is None:
        return [1.0] * len(apartment_types)
    names = {t.__name__ for t in apartment_types}
    unknown = set(weights) - names
    if unknown:
        raise ValueError(f"unknown apartment types in weights: {sorted(unknown)}")
    out = [float(weights.get(t.__name__, 1)) for t in apartment_types]
    if any(w < 0 for w in out) or sum(out) == 0:
        raise ValueError("apartment weights must be >= 0 and not all 0")
    return out

def populate(grid: Grid, floor: int = 0, apartment_types=None, weights=None,
             max_failures: int = 20, fill_shared: bool = True):
    #Each round pick an apartment type (by weight) and try to place it on the floor.
    #Stop after max_failures misses in a row or when the floor is full.
    #Returns (apartments, shared_programs).
    if apartment_types is None:
        apartment_types = APARTMENT_TYPES
    if weights is None:
        weights = apartment_weights
    w = type_weights(apartment_types, weights)
    apartments = []
    failures = 0
    while failures < max_failures and any(n.k == floor for n in grid.free_nodes()):
        apt = random.choices(apartment_types, weights=w)[0](floor)
        if apt.place(grid):
            apt.number = len(apartments) + 1
            apartments.append(apt)
            failures = 0
        else:
            failures += 1
    shared = fill_shared_space(grid, floor) if fill_shared else []
    return apartments, shared

def summary(grid: Grid, apartments: Sequence[Apartment], shared: Sequence[Program]) -> str:
    #Text report: total apartments, count per type, count per floor
    total = len(apartments)
    lines = [f"Generated {total} apartments on {grid.shape[2]} floors",
             "By type:"]
    width = max(len(t.__name__) for t in APARTMENT_TYPES)
    for t in APARTMENT_TYPES:
        n = sum(1 for a in apartments if a.type == t.__name__)
        pct = f"{100 * n / total:5.1f}%" if total else "    -"
        lines.append(f"  {t.__name__:<{width}}  {n:4d}  {pct}")
    lines.append("By floor (0 = right under the road):")
    for k in range(grid.shape[2]):
        n = sum(1 for a in apartments if a.floor == k)
        lines.append(f"  floor {k}  {n:4d}")
    lines.append(f"Shared spaces: {len(shared)}, empty nodes left: {grid.count_free()}")
    return "\n".join(lines)



#=================================================
#--------------------------------- AVAILABLE NODES
# The grid is a lattice of cells that follows the road:
#   i = along the centerline, j = across toward the road edges, k = floors down.
# Each cell's top corners sit on the road. A cell is only buildable when
#   - all 4 of its top corners are inside the road edges (contained under the road)
#   - its bottom is still above the topo at all 4 corners (standing on the topo)
# Everything else is marked OUTSIDE, so programs can never be placed there.

def closest_point_on(crv, pt):
    ok, t = crv.ClosestPoint(pt)
    return crv.PointAt(t)

def road_guides(road, iso_param=0.5):
    #(centerline, edge_a, edge_b) of the road, or None
    # -------------------------------------------- two longest road edges
    # These are the road's side boundaries; they define where the road ends
    # across its width ("out of bounds").
    naked = [e for e in road.Edges if e.Valence == rg.EdgeAdjacency.Naked]
    if len(naked) < 2:
        print("Road needs at least 2 naked edges")
        return None
    naked.sort(key=lambda e: e.GetLength())
    edge_a = naked[-1].DuplicateCurve()
    edge_b = naked[-2].DuplicateCurve()

    # -------------------------------------------- centerline (Isocurve)
    face = road.Faces[0]
    ok, u_dim, v_dim = face.GetSurfaceSize()
    if u_dim >= v_dim:
        # road runs along U: hold V fixed at iso_param
        centerline = face.IsoCurve(0, face.Domain(1).ParameterAt(iso_param))
    else:
        # road runs along V: hold U fixed at iso_param
        centerline = face.IsoCurve(1, face.Domain(0).ParameterAt(iso_param))
    return centerline, edge_a, edge_b

def road_frame(centerline, edges, s):
    #Point s along the centerline, plus (unit direction, distance) to each road edge
    s = min(max(s, 0.0), centerline.GetLength())
    ok, t = centerline.LengthParameter(s)
    p = centerline.PointAt(t)
    sides = []
    for edge in edges:
        v = closest_point_on(edge, p) - p
        width = v.Length
        v.Unitize()
        sides.append((v, width))
    return p, sides

def grid_from_road_samples(top: np.ndarray, corner_ok: np.ndarray, depth: np.ndarray,
                           floor_height: float, tol: float = 0.0) -> Grid | None:
    #top:       (n_div+1, n_off+1, 3) corner points on the road surface
    #corner_ok: (n_div+1, n_off+1) True where the corner is inside the road edges
    #depth:     (n_div+1, n_off+1) distance straight down to the topo (nan = no topo below)
    d = np.where(corner_ok & np.isfinite(depth), depth, -np.inf)

    # a column of cells is only as deep as its shallowest corner
    col = np.minimum.reduce([d[:-1, :-1], d[1:, :-1], d[1:, 1:], d[:-1, 1:]])
    levels = np.zeros(col.shape, dtype=int)
    finite = np.isfinite(col)
    levels[finite] = np.floor((col[finite] + tol) / floor_height).astype(int)
    levels = np.maximum(levels, 0)
    n_z = int(levels.max()) if levels.size else 0
    if n_z == 0:
        return None

    # stack the road-level corners down one floor at a time
    corners = np.repeat(top[:, :, None, :], n_z + 1, axis=2).astype(float)
    corners[..., 2] -= np.arange(n_z + 1) * floor_height

    # node position = center of its cell
    positions = sum(corners[a:a + corners.shape[0] - 1,
                            b:b + corners.shape[1] - 1,
                            c:c + corners.shape[2] - 1]
                    for a in (0, 1) for b in (0, 1) for c in (0, 1)) / 8.0

    grid = Grid(positions, corners)
    below_topo = np.arange(n_z)[None, None, :] >= levels[:, :, None]
    grid.states[below_topo] = ProgramState.OUTSIDE
    return grid

def build_grid(road, topo, node_distance: float, floor_height: float) -> Grid | None:
    tol = sc.doc.ModelAbsoluteTolerance
    guides = road_guides(road)
    if guides is None:
        return None
    centerline, edge_a, edge_b = guides

    # ----- 1. cell boundaries along the centerline, centered on the road length
    length = centerline.GetLength()
    n_div = int((length + tol) / node_distance)
    if n_div < 1:
        print("Road is shorter than one node distance")
        return None
    start = (length - n_div * node_distance) / 2
    frames = [road_frame(centerline, (edge_a, edge_b), start + b * node_distance)
              for b in range(n_div + 1)]

    # ----- 2. cell boundaries across the road, every node_distance from the centerline
    m = max(int((w + tol) / node_distance) for _, sides in frames for _, w in sides)
    if m == 0:
        print("Road is narrower than one node distance")
        return None
    n_off = 2 * m

    # ----- 3. road-level corners: inside the road edges? how far down to the topo?
    down = rg.Vector3d(0, 0, -1)
    top = np.zeros((n_div + 1, n_off + 1, 3))
    corner_ok = np.zeros((n_div + 1, n_off + 1), dtype=bool)
    depth = np.full((n_div + 1, n_off + 1), np.nan)
    for b, (p, sides) in enumerate(frames):
        for c in range(n_off + 1):
            x = (c - m) * node_distance          # > 0 toward edge_a, < 0 toward edge_b
            v, width = sides[0] if x > 0 else sides[1]
            pt = p + v * abs(x)
            top[b, c] = (pt.X, pt.Y, pt.Z)
            corner_ok[b, c] = abs(x) <= width + tol
            if corner_ok[b, c]:
                t = Intersection.MeshRay(topo, rg.Ray3d(pt, down))
                if t >= 0:
                    depth[b, c] = t

    # ----- 4. stack floors down to the topo
    grid = grid_from_road_samples(top, corner_ok, depth, floor_height, tol)
    if grid is None:
        print("No room for a single floor between the road and the topo")
    return grid

#=================================================
#--------------------------------------------- DRAW
# Nothing is added to the Rhino document until draw() runs at the very end.

LAYER_ROOT = "Apartments"
PROGRAM_COLORS = {
    ProgramState.LIVING:   (129, 113, 68), # Olive Wood
    ProgramState.BEDROOM:  (69, 87, 34), # Olive Leaf
    ProgramState.KITCHEN:  (161, 144, 96), # Camel
    ProgramState.BATHROOM: (81, 86, 71), # Ebony
    ProgramState.SHARED:   (41, 76, 96), # Charcoal Blue
}

def program_layer(state: ProgramState) -> int:
    name = state.name.capitalize()
    full = f"{LAYER_ROOT}::{name}"
    if not rs.IsLayer(full):
        if not rs.IsLayer(LAYER_ROOT):
            rs.AddLayer(LAYER_ROOT)
        rs.AddLayer(name, System.Drawing.Color.FromArgb(*PROGRAM_COLORS[state]), parent=LAYER_ROOT)
    return sc.doc.Layers.FindByFullPath(full, -1)

def program_brep(grid: Grid, program: Program):
    pts = List[rg.Point3d]([rg.Point3d(*map(float, c)) for c in program.world_corners(grid)])
    brep = rg.Brep.CreateFromBox(pts)
    if brep is not None and brep.SolidOrientation == rg.BrepSolidOrientation.Inward:
        brep.Flip()
    return brep

def add_program(grid: Grid, program: Program, name: str):
    brep = program_brep(grid, program)
    if brep is None:
        return None
    attr = Rhino.DocObjects.ObjectAttributes()
    attr.LayerIndex = program_layer(program.type)
    attr.Name = name
    gid = sc.doc.Objects.AddBrep(brep, attr)
    return None if gid == System.Guid.Empty else gid

def draw(grid: Grid, apartments: Sequence[Apartment], shared: Sequence[Program]) -> None:
    sc.doc.Views.RedrawEnabled = False
    try:
        for apt in apartments:
            ids = [add_program(grid, p, f"{apt.type} #{apt.number} floor {apt.floor} - {p.code}")
                   for p in apt.programs]
            ids = [i for i in ids if i is not None]
            if ids:
                sc.doc.Groups.Add(List[System.Guid](ids))   # one group per apartment
        for s in shared:
            add_program(grid, s, f"shared floor {s.origin.k} - {s.code}")
    finally:
        sc.doc.Views.RedrawEnabled = True
        sc.doc.Views.Redraw()

#=================================================
#--------------------------------------------- MAIN

def main():
    road = pick_road()
    if road is None:
        return
    topo = pick_topo()
    if topo is None:
        return

    dist = rs.GetReal("Node distance", node_distance, 0.001)
    if dist is None:
        return
    fh = rs.GetReal("Floor height", floor_height or dist, 0.001)
    if fh is None:
        return

    # ----- generate (internal only, nothing drawn yet)
    grid = build_grid(road, topo, dist, fh)
    if grid is None:
        return
    apartments, shared = [], []
    for k in range(grid.shape[2]):
        a, s = populate(grid, floor=k)
        apartments += a
        shared += s
    for n, apt in enumerate(apartments, 1):
        apt.number = n

    # ----- draw everything at the end
    draw(grid, apartments, shared)
    print(summary(grid, apartments, shared))

if __name__ == "__main__":
    main()
