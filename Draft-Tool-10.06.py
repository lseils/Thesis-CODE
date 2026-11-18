
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
import Rhino.UI
import Eto.Forms as forms
import Eto.Drawing as drawing
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

# Density: how likely a round places a shared space instead of an apartment,
# on the same scale as the apartment weights. With the 7 weights above at 1,
# shared_space_weight = 1 means about 1 round in 8. 0 = shared space only fills leftover gaps.
shared_space_weight = 1   # the only weight asked at run time; apartment weights are set here only

# Vertical circulation: one 1x3 core every this many nodes along the road (asked at run time).
# Cores are placed before everything else (shared space, conditions, apartments) and run from
# the top floor down to the topo. A core that doesn't fit slides along the road to the nearest spot.
vertcirc_spacing = 20
# On every floor, at least this share of the buildable cells touching a core become shared space.
core_shared_share = 0.5

# Obstacles (other roads, structures...) under the main road: nothing is built within this
# clearance above them, but the space between the clearance and the main road is still buildable.
obstacle_clearance_ft = 16.0   # in feet, converted to model units (asked at run time)
obstacle_samples = 5           # rays per cell side when looking for obstacles (5 = 25 rays per column)

# Cell checks. Rows of cells point straight across the road at each step along it, so on a curve
# the cells fan out and the building bends with the road. A column of cells is only used if:
road_coverage = 0.75   # at least this share of it sits on the road surface (asked at run time);
                       # an end cut at an angle becomes a staircase of whole cells
coverage_samples = 4   # sample points per cell side for the coverage check (4 = 16 per column)
max_cell_skew = 45.0   # no corner more than this many degrees off a right angle
min_cell_area = 0.6    # and at least this share of a full node_distance x node_distance cell

# ---- Conditions (only used when "Conditions" is picked at run time)
# Ground engagement: Public pods at an interval along the road. Each pod stands on the topo,
# spills past the road edges in steps (widest at the ground and next to the building, shorter
# the further out it reaches) and rises a few floors, each floor smaller than the one below.
# Everywhere else, the floor touching the topo is left empty (open air).
ge_spacing = 23      # nodes between pod centers along the road (asked at run time)
ge_length = 10       # pod length along the road on the ground floor, in nodes (asked at run time)
ge_spill = 4         # nodes the pod reaches past each road edge on the ground floor (asked at run time)
ge_floors = 2        # floors of public, counted up from the topo (asked at run time)
ge_taper = 1         # each step further out past the road edge, the pod is this many nodes shorter at each end
ge_floor_step = 2    # each floor up, the pod is this many nodes shorter at each end and spills this many fewer
ge_void = 0.08       # share of each pod's nodes randomly left as voids (open air), 0 = solid (asked at run time)

# Pathway engagement: wherever an obstacle (a road or path picked as an obstacle) crosses under
# the road, a ground engagement pod is centered on the crossing. The pod is the crossing plus
# `pe_width` nodes on each side of it; wherever the crossing (its clearance) actually is, the pod's
# nodes are voided, so the growth stands on the topo on both sides of the crossing road.
# The steps taper the same way as ground engagement (ge_taper, ge_floor_step).
pe_width = 5         # pod growth along the road on each side of the crossing, ground floor, in nodes (asked)
pe_spill = 4         # nodes the pod reaches past each road edge on the ground floor (asked at run time)
pe_floors = 2        # floors of public, counted up from the topo (asked at run time)
pe_void = 0.08       # share of each pod's nodes randomly left as voids (asked at run time)

# Ends / ramp down: each short end is treated as if another road crossed right past it, so it
# gets half of a pathway engagement pod: the pod is centered on the end, and only the half inside
# the generation is built. The floors above the pod are left as they are (nothing is voided).
# The end is the first / last node along the road whose column reaches the topo
# (a crossing road right at the end is skipped).
er_width = 5         # pod growth inward from the end on the ground floor, in nodes, like pe_width (asked)
er_spill = 4         # nodes the end pod reaches past each road edge on the ground floor (asked)
er_floors = 2        # floors of public in the end pod, counted up from the topo (asked)
er_void = 0.08       # share of each end pod's nodes randomly left as voids (asked)

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

def ref_to_mesh(ref):
    #Selected mesh as-is, or a surface / polysurface meshed into one mesh (None if it fails)
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

def pick_topo():
    ref = get_one("Select the Topo (mesh or surface)", OT.Mesh | OT.Surface | OT.PolysrfFilter)
    if ref is None:
        return None
    return ref_to_mesh(ref)

def pick_obstacles():
    #Optional obstacles under the road, joined into one mesh.
    #Returns an empty mesh if Enter is pressed (no obstacles), None if cancelled.
    go = Rhino.Input.Custom.GetObject()
    go.SetCommandPrompt("Select obstacles under the road (Enter for none)")
    go.GeometryFilter = OT.Mesh | OT.Surface | OT.PolysrfFilter
    go.SubObjectSelect = False
    go.EnablePreSelect(False, True)
    go.AcceptNothing(True)

    res = go.GetMultiple(1, 0)
    joined = rg.Mesh()
    if res == Rhino.Input.GetResult.Nothing:
        return joined
    if res != Rhino.Input.GetResult.Object:
        return None
    for n in range(go.ObjectCount):
        mesh = ref_to_mesh(go.Object(n))
        if mesh is not None:
            joined.Append(mesh)
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
    OUTSIDE = 7   # past the road edge / off the topo: open air, never buildable
    GROUND = 8    # below the topo: solid earth, never buildable
    VERTCIRC = 9  # vertical circulation core, spans every floor
    PUBLIC = 10   # ground / pathway engagement: public space

# what a living room may face on its open side
OPEN_STATES = (ProgramState.OUTSIDE, ProgramState.SHARED, ProgramState.PUBLIC)

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
        # per column (i, j), filled in by grid_from_road_samples:
        #   levels      = floors that fit between the road and the topo
        #   open_levels = floors above any obstacle's clearance (<= levels)
        #   in_road     = True if the column is under the road
        self.levels: np.ndarray | None = None
        self.open_levels: np.ndarray | None = None
        self.in_road: np.ndarray | None = None
        # filled in by build_grid: distance down from the road to the top of an obstacle,
        # per column (inf = no obstacle below), and the floor height
        self.obstacle_hit: np.ndarray | None = None
        self.floor_height: float | None = None
        # filled in by build_grid: centerline length and node distance, in model units
        self.road_length: float | None = None
        self.node_distance: float | None = None

    def ground_k(self, i: int, j: int) -> int | None:
        #k of the cell standing on the topo in column (i, j), or None
        #(no topo below, or the column is cut short by an obstacle's clearance)
        lv, op = int(self.levels[i, j]), int(self.open_levels[i, j])
        if lv < 1 or op < lv or lv > self.shape[2]:
            return None
        return lv - 1

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

    def is_open(self, i: int, j: int, k: int) -> bool:
        #Off the grid, open air past the road edge, or shared space (not earth, not a room)
        return not self.in_bounds(i, j, k) or self.states[i, j, k] in OPEN_STATES

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
    OXO = 3   # One x One
    OXH = 4   # One x tHree

# (width, depth) in nodes for each layout
LAYOUT_SIZE = {
    ProgramLayout.OXO: (1, 1),
    ProgramLayout.OXH: (1, 3),
    ProgramLayout.OXW: (1, 2),
    ProgramLayout.WXW: (2, 2),
    ProgramLayout.HXW: (3, 2),
}
PROGRAM_HEIGHT = 1   # every program sits on one floor (except vertical circulation)

# code -> (state, layout). The number in the code is the width in nodes (S0 = 1x1).
PROGRAMS = {
    "S0":  (ProgramState.SHARED,   ProgramLayout.OXO),
    "L1":  (ProgramState.LIVING,   ProgramLayout.WXW),
    "L2":  (ProgramState.LIVING,   ProgramLayout.HXW),
    "BA":  (ProgramState.BATHROOM, ProgramLayout.OXW),
    "BR1": (ProgramState.BEDROOM,  ProgramLayout.WXW),
    "BR2": (ProgramState.BEDROOM,  ProgramLayout.HXW),
    "K1":  (ProgramState.KITCHEN,  ProgramLayout.OXW),
    "K2":  (ProgramState.KITCHEN,  ProgramLayout.WXW),
    "S1":  (ProgramState.SHARED,   ProgramLayout.OXW),
    "S2":  (ProgramState.SHARED,   ProgramLayout.WXW),
    "S3":  (ProgramState.SHARED,   ProgramLayout.HXW),
    "VC":  (ProgramState.VERTCIRC, ProgramLayout.OXH),
}

#______________________________
class Program:
    #One room of an apartment, as a box of nodes: width x 2 deep x 1 floor.
    #rotated=True swaps width and depth (the box turns 90 degrees in plan).
    #A program only exists in the grid after try_place_at / place_first / place_adjacent succeeds.

    def __init__(self, code: str, height: int = PROGRAM_HEIGHT):
        #height = floors the box spans downward from its origin (only vertical circulation uses > 1)
        if code not in PROGRAMS:
            raise ValueError(f"unknown program code {code!r}")
        self.code = code
        self.type, self.layout = PROGRAMS[code]
        self.height = height
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
        w, d = LAYOUT_SIZE[self.layout]
        if rotated:
            return d, w, self.height
        return w, d, self.height

    def sides(self) -> dict[str, list[Node]]:
        #The 4 rows of nodes just outside the placed box, in plan (may be out of bounds)
        i, j, k = self.origin
        di, dj, _ = self.footprint(self.rotated)
        return {
            "-i": [Node(i - 1, j + b, k) for b in range(dj)],
            "+i": [Node(i + di, j + b, k) for b in range(dj)],
            "-j": [Node(i + a, j - 1, k) for a in range(di)],
            "+j": [Node(i + a, j + dj, k) for a in range(di)],
        }

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
        #Other programs: start from an empty node touching one of the placed programs,
        #so the box always shares at least one face with them (same floor)
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
        self.shared: list[Program] = []   # 1x1 shared cells that open up the living room

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
        return cls("twobedA", ["L2", "BA", "BR1", "BR1", "K2"], floor)
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
        #Living first. Then:
        #  - one full side of the living room must be open (road edge / off the grid / shared space)
        #  - every other room must touch the living room (at least one face)
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

            if self.place_around_living(grid, first, rest):
                return True
            first.remove(grid)
        return False

    def place_around_living(self, grid: Grid, living: Program, rest: Sequence[Program]) -> bool:
        #Make sure the living room has an open side, then put every room against it.
        #If no side is open yet, a side of empty nodes is turned into 1x1 shared space (S0).
        sides = list(living.sides().values())
        random.shuffle(sides)
        if any(all(grid.is_open(*n) for n in side) for side in sides):
            openings = [[]]                    # already open, nothing to add
        else:
            openings = [[n for n in side if not grid.is_open(*n)]
                        for side in sides
                        if all(grid.is_open(*n) or grid.is_empty(*n) for n in side)]

        for to_open in openings:
            entry = []
            for n in to_open:
                s = Program("S0")
                s.try_place_at(grid, n)        # n is empty, so a 1x1 always fits
                entry.append(s)

            placed = []
            for p in rest:
                if not p.place_adjacent(grid, [living]):
                    break
                placed.append(p)
            else:
                self.shared = entry
                return True
            for q in placed + entry:
                q.remove(grid)
        return False

APARTMENT_TYPES = [
    Apartment.onebedA, Apartment.onebedB, Apartment.onebedC,
    Apartment.twobedA, Apartment.twobedB,
    Apartment.threebedA, Apartment.threebedB,
]

#______________________________
def core_depth(grid: Grid, i: int, j: int, w: int, d: int) -> int:
    #How many floors a w x d core at (i, j) can run down from the top floor (0 = doesn't fit)
    dk = 0
    while dk < grid.shape[2] and grid.box_fits(i, j, 0, w, d, dk + 1):
        dk += 1
    return dk

def core_grounded(grid: Grid, i: int, j: int, w: int, d: int, dk: int) -> bool:
    #True if a core dk floors deep at (i, j) reaches the topo (not cut short by an obstacle)
    if dk == 0 or grid.levels is None:
        return dk > 0
    lowest = min(int(grid.levels[a, b]) for a in range(i, i + w) for b in range(j, j + d))
    return dk >= lowest

def place_vertcirc(grid: Grid, spacing: int = vertcirc_spacing) -> list[Program]:
    #Runs first, before shared space, the engagement conditions and apartments: circulation
    #comes before everything else. One 1x3 core every `spacing` nodes along the road,
    #its 3-node length across the road, centered on the centerline.
    #The series is centered along the road so both ends have the same leftover gap.
    #Each core runs from the top floor (k=0) down as far as all 3 of its columns are buildable.
    #If a core's spot doesn't work (a crossing road's clearance, a stepped end...), it slides
    #along the road to the nearest spot within half the spacing where it fits, preferring one
    #that reaches the topo. It is only dropped if nothing in that range fits.
    n_div, n_off, n_z = grid.shape
    w, d = LAYOUT_SIZE[ProgramLayout.OXH]
    if n_off < d:
        print("Road is too narrow for vertical circulation")
        return []
    j = (n_off - d) // 2
    start = ((n_div - 1) % spacing) // 2
    reach = spacing // 2
    offsets = [0] + [o for s in range(1, reach + 1) for o in (s, -s)]   # nearest first
    cores = []
    for target in range(start, n_div, spacing):
        best = None                       # (i, dk, grounded)
        for off in offsets:
            i = target + off
            if not 0 <= i <= n_div - w:
                continue
            dk = core_depth(grid, i, j, w, d)
            if dk == 0:
                continue
            grounded = core_grounded(grid, i, j, w, d, dk)
            if grounded:
                best = (i, dk, True)
                break
            if best is None or dk > best[1]:
                best = (i, dk, False)
        if best is None:
            print(f"Vertical circulation: no room within {reach} nodes of node {target}, core dropped")
            continue
        i, dk, grounded = best
        if i != target:
            print(f"Vertical circulation: core moved from node {target} to node {i} to fit")
        if not grounded:
            print(f"Vertical circulation: core at node {i} can't reach the topo ({dk} floors)")
        core = Program("VC", height=dk)
        core.try_place_at(grid, Node(i, j, 0))   # the box at the anchor fits, checked above
        cores.append(core)
    return cores

def place_core_shared(grid: Grid, cores: Sequence[Program], share: float = core_shared_share) -> list[Program]:
    #Runs right after place_vertcirc, before any apartment. On every floor, at least `share`
    #of the buildable cells touching a core's sides become 1x1 shared space (S0).
    #Order: one long side in full, then the ends, then the other long side.
    #The same side comes first on every floor, so the shared space stacks.
    shared = []
    for core in cores:
        sides = list(core.sides().values())
        random.shuffle(sides)
        sides.sort(key=len, reverse=True)      # long sides first, random which one leads
        order = sides[0] + sides[2] + sides[3] + sides[1]
        k0, dk = core.origin.k, core.height
        for k in range(k0, k0 + dk):
            cells = [Node(n.i, n.j, k) for n in order]
            cells = [c for c in cells if grid.in_bounds(*c)
                     and grid.get_state(*c) not in (ProgramState.OUTSIDE, ProgramState.GROUND)]
            needed = math.ceil(share * len(cells))
            needed -= sum(1 for c in cells if grid.get_state(*c) == ProgramState.SHARED)
            for c in cells:
                if needed <= 0:
                    break
                if grid.is_empty(*c):
                    s = Program("S0")
                    s.try_place_at(grid, c)
                    shared.append(s)
                    needed -= 1
    return shared

#______________________________
class Block:
    #A free-size box of cells with one state. Used for public space, whose shape
    #comes from the engagement steps instead of a fixed layout.
    #condition = which condition made it ("Ground engagement"...), group = its pod / wing name
    def __init__(self, state: ProgramState, i: int, j: int, k: int, di: int, dj: int,
                 condition: str = "", group: str = ""):
        self.type = state
        self.code = "P"
        self.origin = Node(i, j, k)
        self.size = (di, dj, 1)
        self.condition = condition
        self.group = group
        self.nodes = [Node(a, b, k) for a in range(i, i + di) for b in range(j, j + dj)]

    def world_corners(self, grid: Grid) -> np.ndarray:
        return grid.cell_corners(*self.origin, *self.size)

def merge_cells(cells: set[Node]) -> list[tuple[int, int, int, int, int]]:
    #Merge cells into as few boxes as possible: runs along i, then equal runs side by side in j.
    #Returns (i, j, k, di, dj) per box.
    runs = {}                                   # (k, i0, di) -> sorted j list
    for (k, j) in sorted({(c.k, c.j) for c in cells}):
        row = sorted(c.i for c in cells if c.k == k and c.j == j)
        start = prev = row[0]
        for i in row[1:] + [None]:
            if i is not None and i == prev + 1:
                prev = i
                continue
            runs.setdefault((k, start, prev - start + 1), []).append(j)
            if i is not None:
                start = prev = i
    boxes = []
    for (k, i0, di), js in runs.items():
        j0 = prev = js[0]
        for j in js[1:] + [None]:
            if j is not None and j == prev + 1:
                prev = j
                continue
            boxes.append((i0, j0, k, di, prev - j0 + 1))
            if j is not None:
                j0 = prev = j
    return boxes

def pod_centers(n_div: int, spacing: int) -> list[float]:
    #Pod centers along the road (index i), `spacing` apart, the series centered on the road length
    count = max(1, n_div // spacing)
    mid = (n_div - 1) / 2
    return [mid + (p - (count - 1) / 2) * spacing for p in range(count)]

def build_pod(grid: Grid, center: float, length: int, spill: int, floors: int,
              taper: int = ge_taper, floor_step: int = ge_floor_step) -> set[Node]:
    #One stepped public pod centered on `center` (index i along the road), standing on the topo.
    #A cell only becomes public if it is still empty or open air, so cores, other public space and
    #anything in an obstacle's clearance (columns cut by an obstacle have no ground cell) stay as they are.
    n_div, n_off, _ = grid.shape
    pod = set()
    for f in range(floors):
        reach = spill - floor_step * f
        for s in range(max(reach, 0) + 1):
            run = length - 2 * floor_step * f - 2 * taper * s
            if run <= 0:
                break
            i0 = int(math.floor(center - run / 2 + 0.5))
            for i in range(max(i0, 0), min(i0 + run, n_div)):
                road_js = np.flatnonzero(grid.in_road[i])
                if road_js.size == 0:
                    continue
                lo, hi = int(road_js.min()), int(road_js.max())
                js = range(lo, hi + 1) if s == 0 else (lo - s, hi + s)
                for j in js:
                    if not 0 <= j < n_off:
                        continue
                    kg = grid.ground_k(i, j)
                    if kg is None or kg - f < 0:
                        continue
                    k = kg - f
                    if grid.states[i, j, k] in (ProgramState.EMPTY, ProgramState.OUTSIDE):
                        grid.states[i, j, k] = ProgramState.PUBLIC
                        pod.add(Node(i, j, k))
    return pod

def place_ground_engagement(grid: Grid, spacing: int = ge_spacing, length: int = ge_length,
                            spill: int = ge_spill, floors: int = ge_floors,
                            taper: int = ge_taper, floor_step: int = ge_floor_step,
                            void: float = ge_void) -> tuple[list[Block], int]:
    #Runs after the cores and their shared landings, before anything else. Builds the public pods, then leaves
    #the floor touching the topo empty everywhere else.
    #Returns (public blocks, number of void nodes).
    #A pod on floor f (0 = the cell on the topo) is made of steps s:
    #   s = 0         the road's own columns, `length - 2 * floor_step * f` nodes long
    #   s = 1, 2, ... one column further out past each road edge, `2 * taper` nodes shorter per step,
    #                 up to `spill - floor_step * f` columns out
    #Every step and floor is centered on the pod center, so each one sits on the one below.
    n_div = grid.shape[0]
    cells_by_pod = [build_pod(grid, center, length, spill, floors, taper, floor_step)
                    for center in pod_centers(n_div, spacing)]

    voids = carve_voids(grid, cells_by_pod, void)

    # the floor touching the topo stays empty under the rest of the road
    for i in range(n_div):
        for j in np.flatnonzero(grid.in_road[i]):
            kg = grid.ground_k(i, int(j))
            if kg is not None and grid.states[i, j, kg] == ProgramState.EMPTY:
                grid.states[i, j, kg] = ProgramState.OUTSIDE

    blocks = []
    for n, pod in enumerate(cells_by_pod, 1):
        if pod:
            blocks += [Block(ProgramState.PUBLIC, *box, condition="Ground engagement",
                             group=f"ground engagement pod #{n}") for box in merge_cells(pod)]
    return blocks, voids

def carve_voids(grid: Grid, groups: Sequence[set[Node]], void: float) -> int:
    #Random voids: about `void` of each group's public nodes are left as open air, so the steps
    #aren't perfect. Only nodes with no public directly above them, so nothing upstairs floats.
    #Removes the voids from the groups and returns how many there are.
    voids = 0
    for cells in groups:
        tops = sorted(n for n in cells if Node(n.i, n.j, n.k - 1) not in cells)
        for n in random.sample(tops, min(len(tops), round(void * len(cells)))):
            grid.states[n] = ProgramState.OUTSIDE
            cells.discard(n)
            voids += 1
    return voids

def find_crossings(grid: Grid) -> list[tuple[int, int, float]]:
    #Runs of nodes along the road (i) where an obstacle cuts in under the road.
    #Returns (first i, last i, pathway depth) per crossing; pathway depth = distance down
    #from the road to the highest obstacle surface in that run.
    if grid.obstacle_hit is None:
        return []
    cut = grid.in_road & np.isfinite(grid.obstacle_hit) & (grid.open_levels < grid.levels)
    rows = cut.any(axis=1)
    crossings, i, n_div = [], 0, grid.shape[0]
    while i < n_div:
        if not rows[i]:
            i += 1
            continue
        start = i
        while i + 1 < n_div and rows[i + 1]:
            i += 1
        depth = float(grid.obstacle_hit[start:i + 1][cut[start:i + 1]].min())
        crossings.append((start, i, depth))
        i += 1
    return crossings

def place_pathway_engagement(grid: Grid, width: int = pe_width, spill: int = pe_spill,
                             floors: int = pe_floors, void: float = pe_void,
                             taper: int = ge_taper, floor_step: int = ge_floor_step) -> tuple[list[Block], int, int]:
    #Runs after place_ground_engagement (if any), before shared space and apartments.
    #For each crossing, a ground engagement pod centered on it, as long as the crossing plus
    #`width` nodes on each side. The crossing's own columns (the obstacle and its clearance) have
    #no ground cell, so they are voided automatically and the pod grows on both sides of it.
    #Returns (public blocks, number of crossings, number of void nodes).
    crossings = find_crossings(grid)
    pods = []
    for start, end, _ in crossings:
        center = (start + end) / 2
        length = (end - start + 1) + 2 * width
        pods.append(build_pod(grid, center, length, spill, floors, taper, floor_step))

    voids = carve_voids(grid, pods, void)
    blocks = []
    for n, pod in enumerate(pods, 1):
        if pod:
            blocks += [Block(ProgramState.PUBLIC, *box, condition="Pathway engagement",
                             group=f"pathway engagement pod #{n}") for box in merge_cells(pod)]
    return blocks, len(crossings), voids

def generation_ends(grid: Grid) -> tuple[int, int] | None:
    #(first i, last i) along the road whose column under the road reaches the topo, or None
    has_ground = grid.in_road & (grid.levels > 0) & (grid.open_levels >= grid.levels)
    rows = np.flatnonzero(has_ground.any(axis=1))
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1])

def place_ends_ramp_down(grid: Grid, width: int = er_width, spill: int = er_spill,
                         floors: int = er_floors, void: float = er_void,
                         taper: int = ge_taper, floor_step: int = ge_floor_step) -> tuple[list[Block], int]:
    #Runs after the other engagement conditions, before shared space and apartments.
    #At each end: half of a pathway engagement pod, as if a road crossed right past the end
    #(a pod 2 * width long centered on the end; the outer half falls off the grid).
    #The floors above the pod are not touched.
    #Returns (public blocks, number of void nodes).
    ends = generation_ends(grid)
    if ends is None:
        return [], 0
    first, last = ends
    pods = [build_pod(grid, first - 0.5, 2 * width, spill, floors, taper, floor_step),
            build_pod(grid, last + 0.5, 2 * width, spill, floors, taper, floor_step)]
    voids = carve_voids(grid, pods, void)

    blocks = []
    for name, pod in zip(("start", "end"), pods):
        if pod:
            blocks += [Block(ProgramState.PUBLIC, *box, condition="Ends / ramp down",
                             group=f"ends ramp down pod ({name})") for box in merge_cells(pod)]
    return blocks, voids

def fill_shared_space(grid: Grid, floor: int = 0) -> list[Program]:
    #Runs after all apartments on the floor are placed.
    #Leftover gaps become shared space, biggest piece first (S3 -> S2 -> S1 -> S0).
    shared = []
    for node in grid.free_nodes():
        if node.k != floor or not grid.is_empty(*node):
            continue
        for code in ("S3", "S2", "S1", "S0"):
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

def place_shared_piece(grid: Grid, floor: int) -> Program | None:
    #One shared piece of random size (S1-S3, smaller if it doesn't fit), placed at the
    #same sweep position a living room would take: smallest free column on the floor.
    sizes = ["S3", "S2", "S1", "S0"]
    start = random.randrange(3)            # S3, S2 or S1 first, then smaller
    for code in sizes[start:]:
        s = Program(code)
        if s.place_first(grid, floor):
            return s
    return None

def populate(grid: Grid, floor: int = 0, apartment_types=None, weights=None,
             shared_weight: float | None = None,
             max_failures: int = 20, fill_shared: bool = True):
    #Each round pick an apartment type or a shared space (by weight) and place it on the floor.
    #Stop after max_failures apartment misses in a row or when the floor is full.
    #Returns (apartments, scattered, filler):
    #   scattered = shared pieces picked by shared_weight, between apartments
    #   filler    = shared space filling the leftover gaps at the end
    #The 1x1 shared cells that open each living room live on apartment.shared.
    if apartment_types is None:
        apartment_types = APARTMENT_TYPES
    if weights is None:
        weights = apartment_weights
    if shared_weight is None:
        shared_weight = shared_space_weight
    if shared_weight < 0:
        raise ValueError("shared space weight must be >= 0")
    choices = list(apartment_types) + [None]          # None = shared space
    w = type_weights(apartment_types, weights) + [float(shared_weight)]

    apartments, scattered = [], []
    failures = 0
    while failures < max_failures and any(n.k == floor for n in grid.free_nodes()):
        pick = random.choices(choices, weights=w)[0]
        if pick is None:
            piece = place_shared_piece(grid, floor)
            if piece is not None:
                scattered.append(piece)
            continue
        apt = pick(floor)
        if apt.place(grid):
            apt.number = len(apartments) + 1
            apartments.append(apt)
            failures = 0
        else:
            failures += 1
    filler = fill_shared_space(grid, floor) if fill_shared else []
    return apartments, scattered, filler

def summary(grid: Grid, cores: Sequence[Program], core_shared: Sequence[Program],
            apartments: Sequence[Apartment], scattered: Sequence[Program],
            filler: Sequence[Program], public: Sequence[Block] = (),
            voids: dict[str, int] | None = None, crossings: int = 0) -> str:
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
    entry = sum(len(a.shared) for a in apartments)
    lines.append("Shared spaces:")
    lines.append(f"  {len(scattered):4d}  placed between apartments (shared space weight)")
    lines.append(f"  {entry:4d}  1x1 cells opening living rooms")
    lines.append(f"  {len(filler):4d}  filling leftover gaps")
    lines.append(f"  {len(core_shared):4d}  1x1 cells around vertical circulation")
    lines.append(f"Vertical circulation: {len(cores)} cores")
    voids = voids or {}
    for condition in ("Ground engagement", "Pathway engagement", "Ends / ramp down"):
        blocks = [b for b in public if b.condition == condition]
        if not blocks and condition not in voids:
            continue
        groups = len({b.group for b in blocks})
        cells = sum(len(b.nodes) for b in blocks)
        extra = f"{crossings} crossings, " if condition == "Pathway engagement" else ""
        lines.append(f"{condition}: {extra}{groups} public pods, {cells} public nodes"
                     f" in {len(blocks)} boxes, {voids.get(condition, 0)} void nodes")

    # density = share of buildable nodes that ended up as apartments
    apt_nodes = sum(len(p.nodes) for a in apartments for p in a.programs)
    buildable = int(np.count_nonzero(~np.isin(grid.states, (ProgramState.OUTSIDE, ProgramState.GROUND))))
    if buildable:
        lines.append(f"Density: {100 * apt_nodes / buildable:.1f}% of buildable nodes are apartments")
    lines.append(f"Empty nodes left: {grid.count_free()}")
    return "\n".join(lines)



#=================================================
#--------------------------------- AVAILABLE NODES
# The grid is a lattice of cells that follows the road:
#   i = along the centerline, j = across toward the road edges, k = floors down.
# Each cell's top corners sit on the road. A cell is only buildable when
#   - all 4 of its top corners are inside the road edges (contained under the road)
#   - its bottom is still above the topo at all 4 corners (standing on the topo)
#   - its bottom is still above the clearance line of any obstacle under it
# Columns past the road edge and cells in an obstacle's clearance are marked OUTSIDE (open air),
# cells under the topo GROUND (earth).
# Programs can never be placed on either; only OUTSIDE counts as open for a living room.
# With ground engagement on, the grid is `margin` columns wider on each side of the road.
# Those extra columns are OUTSIDE too, but the topo is sampled under them so public pods
# can step out past the road edge and stand on the topo.

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

def road_frame(centerline, edges, s, flat_edges=None):
    #Point s along the centerline, plus (unit direction, distance) to each road edge.
    #The directions point straight across the road (90 degrees to the centerline in plan), so rows
    #of cells turn with the road on curves but never swing toward an edge's end at an angled cut.
    #flat_edges: the edges projected flat onto the world XY plane (computed here if not given).
    s = min(max(s, 0.0), centerline.GetLength())
    ok, t = centerline.LengthParameter(s)
    p = centerline.PointAt(t)
    tangent = centerline.TangentAt(t)
    across = rg.Vector3d.CrossProduct(tangent, rg.Vector3d.ZAxis)
    across.Unitize()
    if flat_edges is None:
        flat_edges = [rg.Curve.ProjectToPlane(e, rg.Plane.WorldXY) for e in edges]
    p_flat = rg.Point3d(p.X, p.Y, 0.0)
    tol = sc.doc.ModelAbsoluteTolerance
    sides = []
    for edge, flat in zip(edges, flat_edges):
        to_edge = closest_point_on(edge, p) - p
        v = across if rg.Vector3d.Multiply(to_edge, across) >= 0 else rg.Vector3d(-across.X, -across.Y, -across.Z)
        # distance straight across (in plan) to where the edge is
        reach = to_edge.Length + flat.GetBoundingBox(True).Diagonal.Length
        line = rg.LineCurve(p_flat, p_flat + v * reach)
        hits = Intersection.CurveCurve(line, flat, tol, tol)
        dists = [h.PointA.DistanceTo(p_flat) for h in hits] if hits else []
        if dists:
            width = min(dists)
        else:
            width = abs(rg.Vector3d.Multiply(to_edge, v))   # past the edge's end (angled cut)
        sides.append((v, width))
    return p, sides

def cell_quality_ok(top: np.ndarray, node_distance: float, max_skew: float = max_cell_skew,
                    min_area: float = min_cell_area) -> np.ndarray:
    #(n_div, n_off) True where a column's top cell isn't skewed past `max_skew` degrees at any
    #corner and has at least `min_area` of a full node_distance x node_distance cell
    a, b, c, d = top[:-1, :-1], top[1:, :-1], top[1:, 1:], top[:-1, 1:]
    quad = (a, b, c, d)
    ok = np.ones(a.shape[:2], dtype=bool)
    for n in range(4):
        p, prev, nxt = quad[n], quad[n - 1], quad[(n + 1) % 4]
        u, v = prev - p, nxt - p
        norms = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        cos = np.sum(u * v, axis=-1) / np.where(norms > 0, norms, 1.0)
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        ok &= (norms > 0) & (np.abs(angle - 90.0) <= max_skew)
    area = 0.5 * (np.linalg.norm(np.cross(b - a, d - a), axis=-1)
                  + np.linalg.norm(np.cross(b - c, d - c), axis=-1))
    ok &= area >= min_area * node_distance ** 2
    return ok

def road_coverage_ok(top: np.ndarray, candidates: np.ndarray, road_mesh, share: float,
                     samples: int = coverage_samples) -> np.ndarray:
    #(n_div, n_off) True where at least `share` of the column's top cell is over the road.
    #Only columns in `candidates` are tested (the rest are False). Each cell is sampled on a
    #samples x samples grid; a sample counts if a ray straight down from above hits the road.
    n_div, n_off = top.shape[0] - 1, top.shape[1] - 1
    out = np.zeros((n_div, n_off), dtype=bool)
    lift = road_mesh.GetBoundingBox(True).Diagonal.Length + 1.0
    down = rg.Vector3d(0, 0, -1)
    s = (np.arange(samples) + 0.5) / samples          # sample at sub-cell centers
    for b in range(n_div):
        for c in range(n_off):
            if not candidates[b, c]:
                continue
            p00, p10, p11, p01 = top[b, c], top[b + 1, c], top[b + 1, c + 1], top[b, c + 1]
            on = 0
            for u in s:
                for v in s:
                    x, y, z = (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + u * v * p11 + (1 - u) * v * p01
                    ray = rg.Ray3d(rg.Point3d(float(x), float(y), float(z) + lift), down)
                    if Intersection.MeshRay(road_mesh, ray) >= 0:
                        on += 1
            out[b, c] = on >= share * samples * samples - 1e-9
    return out

def grid_from_road_samples(top: np.ndarray, corner_ok: np.ndarray, depth: np.ndarray,
                           floor_height: float, tol: float = 0.0,
                           obstacle_depth: np.ndarray | None = None,
                           column_ok: np.ndarray | None = None,
                           on_road: np.ndarray | None = None) -> Grid | None:
    #top:       (n_div+1, n_off+1, 3) corner points on the road surface
    #corner_ok: (n_div+1, n_off+1) True where the corner is inside the road edges
    #depth:     (n_div+1, n_off+1) distance straight down to the topo (nan = no topo below)
    #obstacle_depth: optional (n_div, n_off) distance down to the clearance line above an
    #           obstacle, per column of cells (inf = no obstacle below)
    #column_ok: optional (n_div, n_off) False for columns too skewed / small to use at all
    #on_road:   optional (n_div, n_off) False for columns not enough on the road surface
    d = np.where(np.isfinite(depth), depth, -np.inf)

    # a column of cells is only as deep as its shallowest corner
    col = np.minimum.reduce([d[:-1, :-1], d[1:, :-1], d[1:, 1:], d[:-1, 1:]])
    if column_ok is not None:
        col[~column_ok] = -np.inf      # too skewed / small: no floors, under the road or past it
    levels = np.zeros(col.shape, dtype=int)
    finite = np.isfinite(col)
    levels[finite] = np.floor((col[finite] + tol) / floor_height).astype(int)
    levels = np.maximum(levels, 0)

    # under the road = all 4 top corners inside the road edges, enough of it on the road
    # surface, with topo below
    ok = corner_ok.astype(bool)
    in_road = ok[:-1, :-1] & ok[1:, :-1] & ok[1:, 1:] & ok[:-1, 1:] & finite
    if on_road is not None:
        in_road &= on_road

    # floors above an obstacle's clearance line (never deeper than the topo)
    open_levels = levels.copy()
    if obstacle_depth is not None:
        hit = np.isfinite(obstacle_depth)
        above = np.floor((obstacle_depth[hit] + tol) / floor_height).astype(int)
        open_levels[hit] = np.minimum(levels[hit], np.maximum(above, 0))

    n_z = int(open_levels.max()) if open_levels.size else 0
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
    grid.levels, grid.open_levels, grid.in_road = levels, open_levels, in_road
    ks = np.arange(n_z)[None, None, :]
    in_clearance = ks >= open_levels[:, :, None]
    below_topo = ks >= levels[:, :, None]
    off_site = np.broadcast_to(~in_road[:, :, None], below_topo.shape)
    grid.states[in_clearance] = ProgramState.OUTSIDE   # open air above an obstacle
    grid.states[below_topo] = ProgramState.GROUND
    grid.states[off_site] = ProgramState.OUTSIDE   # past the road edge or no topo below
    return grid

def obstacle_clearance_depth(top: np.ndarray, sampled: np.ndarray, obstacles,
                             clearance: float, samples: int = obstacle_samples) -> np.ndarray:
    #sampled: (n_div+1, n_off+1) True where the corner has topo below (other columns are skipped)
    #(n_div, n_off) distance down from the road each column of cells may reach before it gets
    #within `clearance` of an obstacle (inf = no obstacle below). Each column is sampled on a
    #samples x samples grid of rays, so an obstacle narrower than a cell is still found.
    down = rg.Vector3d(0, 0, -1)
    n_div, n_off = top.shape[0] - 1, top.shape[1] - 1
    out = np.full((n_div, n_off), np.inf)
    s = np.linspace(0.0, 1.0, samples)
    for b in range(n_div):
        for c in range(n_off):
            if not sampled[b:b + 2, c:c + 2].all():
                continue                       # off site anyway
            p00, p10, p11, p01 = top[b, c], top[b + 1, c], top[b + 1, c + 1], top[b, c + 1]
            for u in s:
                for v in s:
                    x, y, z = (1 - u) * (1 - v) * p00 + u * (1 - v) * p10 + u * v * p11 + (1 - u) * v * p01
                    t = Intersection.MeshRay(obstacles, rg.Ray3d(rg.Point3d(float(x), float(y), float(z)), down))
                    if t >= 0:
                        out[b, c] = min(out[b, c], t - clearance)
    return out

def build_grid(road, topo, node_distance: float, floor_height: float,
               obstacles=None, clearance: float = 0.0, margin: int = 0,
               coverage: float = road_coverage) -> Grid | None:
    #obstacles: optional mesh of things under the road to keep `clearance` above
    #margin: extra columns past each road edge (ground engagement), topo sampled under them
    #coverage: share of a column's top cell that must be on the road surface to build under it
    tol = sc.doc.ModelAbsoluteTolerance
    guides = road_guides(road)
    if guides is None:
        return None
    centerline, edge_a, edge_b = guides
    flat_edges = [rg.Curve.ProjectToPlane(e, rg.Plane.WorldXY) for e in (edge_a, edge_b)]

    # ----- 1. cell boundaries along the centerline, centered on the road length
    length = centerline.GetLength()
    n_div = int((length + tol) / node_distance)
    if n_div < 1:
        print("Road is shorter than one node distance")
        return None
    start = (length - n_div * node_distance) / 2
    frames = [road_frame(centerline, (edge_a, edge_b), start + b * node_distance, flat_edges)
              for b in range(n_div + 1)]

    # ----- 2. cell boundaries across the road, every node_distance from the centerline
    m = max(int((w + tol) / node_distance) for _, sides in frames for _, w in sides)
    if m == 0:
        print("Road is narrower than one node distance")
        return None
    half = m + margin
    n_off = 2 * half

    # ----- 3. road-level corners: inside the road edges? how far down to the topo?
    down = rg.Vector3d(0, 0, -1)
    top = np.zeros((n_div + 1, n_off + 1, 3))
    corner_ok = np.zeros((n_div + 1, n_off + 1), dtype=bool)
    depth = np.full((n_div + 1, n_off + 1), np.nan)
    for b, (p, sides) in enumerate(frames):
        for c in range(n_off + 1):
            x = (c - half) * node_distance       # > 0 toward edge_a, < 0 toward edge_b
            v, width = sides[0] if x > 0 else sides[1]
            pt = p + v * abs(x)
            top[b, c] = (pt.X, pt.Y, pt.Z)
            corner_ok[b, c] = abs(x) <= width + tol
            if corner_ok[b, c] or margin > 0:
                t = Intersection.MeshRay(topo, rg.Ray3d(pt, down))
                if t >= 0:
                    depth[b, c] = t

    # ----- 4. cell checks: skewed / squished cells are never used; under the road, a column
    #          also needs `coverage` of its cell on the road surface (steps angled ends)
    column_ok = cell_quality_ok(top, node_distance)
    ok = corner_ok
    candidates = ok[:-1, :-1] & ok[1:, :-1] & ok[1:, 1:] & ok[:-1, 1:] & column_ok
    road_mesh = rg.Mesh()
    for part in rg.Mesh.CreateFromBrep(road, rg.MeshingParameters.Default) or []:
        road_mesh.Append(part)
    on_road = road_coverage_ok(top, candidates, road_mesh, coverage)

    # ----- 5. obstacles: stop each column short of the clearance above them
    obstacle_depth = None
    if obstacles is not None and obstacles.Faces.Count > 0:
        obstacle_depth = obstacle_clearance_depth(top, np.isfinite(depth), obstacles, clearance)

    # ----- 6. stack floors down to the topo / obstacle clearance
    grid = grid_from_road_samples(top, corner_ok, depth, floor_height, tol, obstacle_depth,
                                  column_ok, on_road)
    if grid is None:
        print("No room for a single floor between the road and the topo")
        return None
    grid.floor_height = floor_height
    grid.road_length, grid.node_distance = length, node_distance
    grid.obstacle_hit = (obstacle_depth + clearance if obstacle_depth is not None
                         else np.full(grid.shape[:2], np.inf))
    return grid

#=================================================
#----------------------------------------- EVALUATE
# Scores a finished generation, measured on the node grid. A "wall" is one side face of an
# apartment's room cell (one node wide, one floor tall). Results print after the summary and
# are saved to two CSV files next to the Rhino file (one row per run, one row per apartment).

view_distance_ft = 20.0   # "can see" a shared space up to this far, across shared space or open air
egress_limit_ft = 250.0   # walking distance to a core that reaches the topo; longer is flagged

ROOM_STATES = (ProgramState.LIVING, ProgramState.BEDROOM, ProgramState.KITCHEN, ProgramState.BATHROOM)
WALL_SIDES = ((1, 0), (-1, 0), (0, 1), (0, -1))
LIGHT_KINDS = ("direct_outside", "direct_shared", "perp_outside", "perp_shared")

# Scoring weights (adjustable in the "Scoring weights" window, remembered for the session)
DEFAULT_WEIGHTS = {
    # overall score: how much each aspect counts
    "w_light": 1.0, "w_community": 1.0, "w_privacy": 1.0, "w_density": 1.0, "w_walking": 1.0,
    # light / air: how much one wall of each kind counts
    "direct_outside": 1.0, "direct_shared": 0.75, "perp_outside": 0.5, "perp_shared": 0.25,
    # privacy gradient: how much a room facing public space counts
    "bedroom": 2.0, "bathroom": 2.0, "living": 1.0, "kitchen": 1.0,
}

# Pareto front: the scores compared across runs (chosen in the "Scoring weights" window).
# A run is on the front if no other run is at least as good on every one of these scores and
# better on at least one. Pick 2-3: with all five, almost every run is best at something.
SCORE_NAMES = ("light", "community", "privacy", "density", "walking")
DEFAULT_PARETO_SCORES = ("density", "light", "walking")

def pareto_flags(points: list[tuple | None]) -> list[bool]:
    #True for each point that no other point dominates (higher is better on every axis).
    #None = missing scores, never on the front.
    flags = []
    for a in points:
        if a is None:
            flags.append(False)
            continue
        dominated = any(b is not None and all(y >= x for x, y in zip(a, b)) and any(y > x for x, y in zip(a, b))
                        for b in points)
        flags.append(not dominated)
    return flags

def pareto_scores_chosen() -> tuple:
    chosen = sc.sticky.get("pareto_scores")
    if not chosen or not all(c in SCORE_NAMES for c in chosen):
        return DEFAULT_PARETO_SCORES
    return tuple(chosen)

def open_air(grid: Grid, i: int, j: int, k: int) -> bool:
    #Off the grid or open air (past the road edge, a void, an obstacle's clearance)
    return not grid.in_bounds(i, j, k) or grid.states[i, j, k] == ProgramState.OUTSIDE

def wall_length(grid: Grid, i: int, j: int, di: int, dj: int) -> float:
    #Plan length of the wall of cell (i, j) on side (di, dj), in model units
    c = grid.corners[:, :, 0]
    if di == 1:
        p, q = c[i + 1, j], c[i + 1, j + 1]
    elif di == -1:
        p, q = c[i, j], c[i, j + 1]
    elif dj == 1:
        p, q = c[i, j + 1], c[i + 1, j + 1]
    else:
        p, q = c[i, j], c[i + 1, j]
    return float(np.linalg.norm(q[:2] - p[:2]))

def cell_areas(grid: Grid) -> np.ndarray:
    #(n_div, n_off) plan area of each column's cell, in model units squared
    c = grid.corners[:, :, 0, :2]
    a, b, cc, d = c[:-1, :-1], c[1:, :-1], c[1:, 1:], c[:-1, 1:]
    def tri(p, q, r):
        return 0.5 * np.abs((q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1])
                            - (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0]))
    return tri(a, b, cc) + tri(a, cc, d)

def wall_light(grid: Grid, i: int, j: int, k: int, di: int, dj: int) -> str | None:
    #What one wall gets: "direct_outside", "direct_shared", "perp_outside", "perp_shared" or None.
    #outside / shared = what is right against the wall. direct = looking straight out, the view
    #stays open (open air or shared space) all the way past the building; perpendicular = it is
    #closed off further out (another room, a core...), so the light comes in from the side.
    a, b = i + di, j + dj
    if open_air(grid, a, b, k):
        kind = "outside"
    elif grid.states[a, b, k] == ProgramState.SHARED:
        kind = "shared"
    else:
        return None
    while grid.in_bounds(a, b, k):
        if grid.states[a, b, k] not in (ProgramState.OUTSIDE, ProgramState.SHARED):
            return "perp_" + kind
        a, b = a + di, b + dj
    return "direct_" + kind

def wall_view(grid: Grid, i: int, j: int, k: int, di: int, dj: int, steps: int) -> list[Node]:
    #Shared cells seen straight out from a wall, across shared space and open air, within `steps` cells
    seen = []
    a, b = i + di, j + dj
    for _ in range(steps):
        if not grid.in_bounds(a, b, k):
            break
        s = grid.states[a, b, k]
        if s == ProgramState.SHARED:
            seen.append(Node(a, b, k))
        elif s != ProgramState.OUTSIDE:
            break
        a, b = a + di, b + dj
    return seen

def wall_public(grid: Grid, i: int, j: int, k: int, di: int, dj: int, steps: int) -> str | None:
    #"direct" if public space is right against the wall, "buffered" if shared space sits between
    #the wall and public space (within `steps` cells), otherwise None
    a, b = i + di, j + dj
    for step in range(steps):
        if not grid.in_bounds(a, b, k):
            return None
        s = grid.states[a, b, k]
        if s == ProgramState.PUBLIC:
            return "direct" if step == 0 else "buffered"
        if s != ProgramState.SHARED:
            return None
        a, b = a + di, b + dj
    return None

def shared_regions(grid: Grid) -> dict[Node, int]:
    #Each shared cell -> id of its shared space (shared cells touching side by side on one floor)
    region, next_id = {}, 0
    for start in map(lambda c: Node(*map(int, c)), np.argwhere(grid.states == ProgramState.SHARED)):
        if start in region:
            continue
        stack = [start]
        region[start] = next_id
        while stack:
            n = stack.pop()
            for di, dj in WALL_SIDES:
                m = Node(n.i + di, n.j + dj, n.k)
                if grid.in_bounds(*m) and m not in region and grid.states[m] == ProgramState.SHARED:
                    region[m] = next_id
                    stack.append(m)
        next_id += 1
    return region

def egress_distances(grid: Grid, cores: Sequence[Program]) -> tuple[dict[Node, float], int]:
    #Walking distance (model units) from every shared / public cell to the nearest core that
    #reaches the topo, walking cell to cell on one floor. Returns ({cell: distance}, grounded cores).
    import heapq
    pos = grid.positions
    dist, heap, grounded = {}, [], 0
    for core in cores:
        w, d, dk = core.footprint(core.rotated)
        if not core_grounded(grid, core.origin.i, core.origin.j, w, d, dk):
            continue
        grounded += 1
        for n in core.nodes:
            dist[n] = 0.0
            heapq.heappush(heap, (0.0, n))
    walkable = (ProgramState.SHARED, ProgramState.PUBLIC)
    while heap:
        dn, n = heapq.heappop(heap)
        if dn > dist.get(n, math.inf):
            continue
        for di, dj in WALL_SIDES:
            m = Node(n.i + di, n.j + dj, n.k)
            if not grid.in_bounds(*m) or grid.states[m] not in walkable:
                continue
            dm = dn + float(np.linalg.norm(pos[m] - pos[n]))
            if dm < dist.get(m, math.inf):
                dist[m] = dm
                heapq.heappush(heap, (dm, m))
    return dist, grounded

def densification_potential(grid: Grid, scattered: Sequence[Program], filler: Sequence[Program]) -> int:
    #How many more apartments fit into the current shared and leftover space without moving
    #anything: a trial copy of the grid frees the shared space placed between apartments and as
    #leftover fill (core landings and living room openings stay) and fills it with apartments.
    trial = copy.deepcopy(grid)
    for p in list(scattered) + list(filler):
        for n in p.nodes:
            trial.states[n] = ProgramState.EMPTY
    added = 0
    for k in range(trial.shape[2]):
        placed, _, _ = populate(trial, floor=k, shared_weight=0.0, fill_shared=False)
        added += len(placed)
    return added

def evaluate(grid: Grid, cores: Sequence[Program], apartments: Sequence[Apartment],
             scattered: Sequence[Program], filler: Sequence[Program], ft: float,
             weights: dict | None = None) -> dict:
    #Scores one generation. ft = model units per foot. Returns {"summary": {...}, "apartments": [...]}
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    nd = grid.node_distance or float(np.linalg.norm(grid.positions[1, 0, 0] - grid.positions[0, 0, 0]))
    fh = grid.floor_height or nd
    sqft = ft * ft
    steps = 1 + int(math.floor(view_distance_ft * ft / nd + 1e-9))
    room_weight = {ProgramState.BEDROOM: w["bedroom"], ProgramState.BATHROOM: w["bathroom"],
                   ProgramState.LIVING: w["living"], ProgramState.KITCHEN: w["kitchen"]}
    owner = {n: a for a, apt in enumerate(apartments) for p in apt.programs for n in p.nodes}
    regions = shared_regions(grid)
    egress, grounded_cores = egress_distances(grid, cores)
    pos = grid.positions
    viewers = {}                              # shared space id -> apartments that see it
    rows = []
    for a, apt in enumerate(apartments):
        light = {kind: 0 for kind in LIGHT_KINDS}
        light_sqft = {kind: 0.0 for kind in LIGHT_KINDS}
        walls = 0
        connected = False
        seen = set()
        public_direct = public_buffered = 0
        privacy_direct_w = privacy_buffered_w = 0.0
        walk = math.inf
        for p in apt.programs:
            for n in p.nodes:
                for di, dj in WALL_SIDES:
                    m = Node(n.i + di, n.j + dj, n.k)
                    if owner.get(m) == a:
                        continue                  # inside the apartment
                    walls += 1
                    kind = wall_light(grid, n.i, n.j, n.k, di, dj)
                    if kind:
                        light[kind] += 1
                        light_sqft[kind] += wall_length(grid, n.i, n.j, di, dj) * fh / sqft
                    if grid.in_bounds(*m) and grid.states[m] == ProgramState.SHARED:
                        connected = True
                    seen.update(regions[c] for c in wall_view(grid, n.i, n.j, n.k, di, dj, steps))
                    exposure = wall_public(grid, n.i, n.j, n.k, di, dj, steps)
                    if exposure == "direct":
                        public_direct += 1
                        privacy_direct_w += room_weight[p.type]
                    elif exposure == "buffered":
                        public_buffered += 1
                        privacy_buffered_w += room_weight[p.type]
                    if grid.in_bounds(*m) and m in egress:
                        walk = min(walk, egress[m] + float(np.linalg.norm(pos[m] - pos[n])))
        for r in seen:
            viewers.setdefault(r, set()).add(a)
        lit = sum(w[kind] * light[kind] for kind in LIGHT_KINDS)
        rows.append({
            "apartment": apt.number, "type": apt.type, "floor": apt.floor,
            **{f"walls_{kind}": light[kind] for kind in LIGHT_KINDS},
            **{f"sqft_{kind}": round(light_sqft[kind], 1) for kind in LIGHT_KINDS},
            "light_score": round(lit / walls, 3) if walls else 0.0,
            "connected_to_shared": connected, "sees_shared": bool(seen),
            "public_walls_direct": public_direct, "public_walls_buffered": public_buffered,
            "privacy_direct_weighted": privacy_direct_w, "privacy_buffered_weighted": privacy_buffered_w,
            "walk_to_egress_ft": round(walk / ft, 1) if math.isfinite(walk) else None,
            "over_egress_limit": (not math.isfinite(walk)) or walk / ft > egress_limit_ft,
        })

    n_apts = len(rows)
    areas = cell_areas(grid)
    def area_of(states):
        mask = np.isin(grid.states, states)
        return float((areas[:, :, None] * mask).sum()) / sqft
    buildable = int(np.count_nonzero(~np.isin(grid.states, (ProgramState.OUTSIDE, ProgramState.GROUND))))
    apt_nodes = sum(len(p.nodes) for apt in apartments for p in apt.programs)
    road_ft = (grid.road_length or grid.shape[0] * nd) / ft
    per_floor = [sum(1 for r in rows if r["floor"] == k) for k in range(grid.shape[2])]
    region_views = [len(viewers.get(r, ())) for r in set(regions.values())]
    walks = [r["walk_to_egress_ft"] for r in rows if r["walk_to_egress_ft"] is not None]
    direct_w = sum(r["privacy_direct_weighted"] for r in rows)
    buffered_w = sum(r["privacy_buffered_weighted"] for r in rows)

    scores = {
        "light": float(np.mean([r["light_score"] for r in rows])) if rows else 0.0,
        "community": (0.5 * sum(r["connected_to_shared"] for r in rows) / n_apts
                      + 0.5 * sum(r["sees_shared"] for r in rows) / n_apts) if n_apts else 0.0,
        "privacy": buffered_w / (buffered_w + direct_w) if (buffered_w + direct_w) else 1.0,
        "density": apt_nodes / buildable if buildable else 0.0,
        "walking": sum(1 for r in rows if not r["over_egress_limit"]) / n_apts if n_apts else 0.0,
    }
    total_w = sum(w["w_" + k] for k in scores)
    overall = sum(w["w_" + k] * v for k, v in scores.items()) / total_w if total_w else 0.0

    summary = {
        "apartments": n_apts,
        "apartments_per_floor": per_floor,
        "apartments_per_100ft": round(n_apts / (road_ft / 100.0), 2) if road_ft else 0.0,
        "road_length_ft": round(road_ft, 1),
        "sqft_apartments": round(area_of(ROOM_STATES), 0),
        "sqft_shared": round(area_of((ProgramState.SHARED,)), 0),
        "sqft_public": round(area_of((ProgramState.PUBLIC,)), 0),
        "sqft_cores": round(area_of((ProgramState.VERTCIRC,)), 0),
        "densification_potential": densification_potential(grid, scattered, filler),
        **{f"walls_{kind}": sum(r[f"walls_{kind}"] for r in rows) for kind in LIGHT_KINDS},
        **{f"sqft_{kind}": round(sum(r[f"sqft_{kind}"] for r in rows), 0) for kind in LIGHT_KINDS},
        "apartments_connected_to_shared": sum(r["connected_to_shared"] for r in rows),
        "apartments_seeing_shared": sum(r["sees_shared"] for r in rows),
        "shared_spaces": len(region_views),
        "shared_space_viewers_avg": round(float(np.mean(region_views)), 2) if region_views else 0.0,
        "shared_space_viewers_max": max(region_views) if region_views else 0,
        "shared_spaces_unseen": sum(1 for v in region_views if v == 0),
        "public_walls_direct": sum(r["public_walls_direct"] for r in rows),
        "public_walls_buffered": sum(r["public_walls_buffered"] for r in rows),
        "grounded_cores": grounded_cores,
        "walk_avg_ft": round(float(np.mean(walks)), 1) if walks else None,
        "walk_max_ft": round(max(walks), 1) if walks else None,
        "apartments_over_egress_limit": sum(r["over_egress_limit"] for r in rows),
        "apartments_without_egress": sum(1 for r in rows if r["walk_to_egress_ft"] is None),
        **{f"score_{k}": round(v, 3) for k, v in scores.items()},
        "score_overall": round(overall, 3),
    }
    return {"summary": summary, "apartments": rows}

def format_evaluation(result: dict) -> str:
    s = result["summary"]
    lines = ["Evaluation",
             f"  Density: {s['apartments']} apartments, {s['apartments_per_100ft']} per 100 ft of road "
             f"({s['road_length_ft']} ft); per floor {s['apartments_per_floor']}",
             f"    area (sq ft): apartments {s['sqft_apartments']:.0f}, shared {s['sqft_shared']:.0f}, "
             f"public {s['sqft_public']:.0f}, cores {s['sqft_cores']:.0f}",
             f"    densification potential: {s['densification_potential']} more apartments fit "
             f"in today's shared / leftover space",
             "  Light / air (walls, sq ft):"]
    names = {"direct_outside": "direct to outside", "direct_shared": "direct to shared",
             "perp_outside": "perpendicular to outside", "perp_shared": "perpendicular to shared"}
    for kind in LIGHT_KINDS:
        lines.append(f"    {names[kind]:<26} {s['walls_' + kind]:5d}  {s['sqft_' + kind]:8.0f}")
    lines += [
        f"  Community: {s['apartments_connected_to_shared']} apartments connected to shared space, "
        f"{s['apartments_seeing_shared']} can see one (within {view_distance_ft:.0f} ft)",
        f"    {s['shared_spaces']} shared spaces, seen by {s['shared_space_viewers_avg']} apartments "
        f"on average (most {s['shared_space_viewers_max']}, {s['shared_spaces_unseen']} unseen)",
        f"    privacy gradient: {s['public_walls_direct']} walls face public directly, "
        f"{s['public_walls_buffered']} with shared space in between",
        f"  Walking to egress ({s['grounded_cores']} cores reach the topo): "
        f"average {s['walk_avg_ft']} ft, longest {s['walk_max_ft']} ft",
        f"    {s['apartments_over_egress_limit']} apartments over {egress_limit_ft:.0f} ft "
        f"({s['apartments_without_egress']} with no route to a core)",
        f"  Scores (0-1): light {s['score_light']}, community {s['score_community']}, "
        f"privacy {s['score_privacy']}, density {s['score_density']}, walking {s['score_walking']}"
        f"  ->  overall {s['score_overall']}",
    ]
    return "\n".join(lines)

def save_evaluation(result: dict, settings: dict) -> dict | None:
    #Adds this run to <file>_evaluation_runs.csv and its apartments to <file>_evaluation_apartments.csv,
    #next to the Rhino file (Documents if it isn't saved). The runs file is rewritten every time with
    #a "pareto_front" column (yes / no) worked out over all runs in it, using the chosen scores.
    #Returns {"folder", "runs", "front": [run rows on the front], "on_front": bool, "scores"},
    #or None if the files couldn't be written (e.g. open in Excel).
    import csv, os, datetime
    folder = os.path.dirname(sc.doc.Path) if sc.doc.Path else os.path.join(os.path.expanduser("~"), "Documents")
    name = os.path.splitext(sc.doc.Name)[0] if sc.doc.Name else "untitled"
    run = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    runs_path = os.path.join(folder, f"{name}_evaluation_runs.csv")
    apts_path = os.path.join(folder, f"{name}_evaluation_apartments.csv")
    chosen = pareto_scores_chosen()

    row = {"run": run, "seed": settings.get("seed", "")}
    row.update({f"set_{k}": v for k, v in settings.items() if k != "seed"})
    row.update({k: (" ".join(map(str, v)) if isinstance(v, list) else v) for k, v in result["summary"].items()})
    row = {k: ("" if v is None else str(v)) for k, v in row.items()}

    try:
        rows, fields = [], []
        if os.path.exists(runs_path):
            with open(runs_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fields = list(reader.fieldnames or [])
                rows = list(reader)
        rows.append(row)
        for key in list(row.keys()) + ["pareto_front", "pareto_scores"]:
            if key not in fields:
                fields.append(key)

        def point(r):
            try:
                return tuple(float(r[f"score_{c}"]) for c in chosen)
            except (KeyError, ValueError, TypeError):
                return None

        flags = pareto_flags([point(r) for r in rows])
        for r, on in zip(rows, flags):
            r["pareto_front"] = "yes" if on else "no"
            r["pareto_scores"] = "+".join(chosen)
        with open(runs_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)

        apt_rows = [{"run": run, **r} for r in result["apartments"]]
        if apt_rows:
            new = not os.path.exists(apts_path)
            with open(apts_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(apt_rows[0].keys()), extrasaction="ignore")
                if new:
                    writer.writeheader()
                writer.writerows(apt_rows)
    except OSError as e:
        print(f"Evaluation: couldn't write the CSV files ({e}). Close them if they're open in Excel.")
        return None
    return {"folder": folder, "runs": len(rows), "scores": chosen, "on_front": flags[-1],
            "front": [r for r, on in zip(rows, flags) if on]}

def format_front(saved: dict, this_run: bool = True) -> str:
    #Printout of the Pareto front over every run in the runs file.
    #this_run: also say whether the latest run is on it (single runs).
    chosen = saved["scores"]
    lines = [f"Pareto front ({' / '.join(chosen)}): {len(saved['front'])} of {saved['runs']} runs in the file"
             + (f" - this run is {'ON' if saved['on_front'] else 'not on'} the front" if this_run else "")]
    for r in saved["front"][-12:]:
        scores = ", ".join(f"{c} {r.get('score_' + c, '')}" for c in chosen)
        lines.append(f"    {r.get('run', '')}  seed {r.get('seed') or '-':>6}  {scores}")
    if len(saved["front"]) > 12:
        lines.append(f"    ... {len(saved['front']) - 12} more in the CSV (pareto_front = yes)")
    return "\n".join(lines)

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
    ProgramState.VERTCIRC: (0, 27, 46),   # Prussian Blue
    ProgramState.PUBLIC:   (214, 204, 178), # Bone
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

def add_program(grid: Grid, program: Program, name: str, offset: rg.Vector3d | None = None):
    brep = program_brep(grid, program)
    if brep is None:
        return None
    if offset is not None:
        brep.Transform(rg.Transform.Translation(offset))
    attr = Rhino.DocObjects.ObjectAttributes()
    attr.LayerIndex = program_layer(program.type)
    attr.Name = name
    gid = sc.doc.Objects.AddBrep(brep, attr)
    return None if gid == System.Guid.Empty else gid

def draw(grid: Grid, cores: Sequence[Program], core_shared: Sequence[Program],
         apartments: Sequence[Apartment], scattered: Sequence[Program],
         filler: Sequence[Program], public: Sequence[Block] = (),
         offset: rg.Vector3d | None = None, prefix: str = "") -> None:
    #offset: shift everything (drawing batch runs side by side); prefix: added to every name
    def add(program, name):
        return add_program(grid, program, prefix + name, offset)
    sc.doc.Views.RedrawEnabled = False
    try:
        for group in sorted({b.group for b in public}):
            ids = [add(b, f"public floor {b.origin.k} - {group}")
                   for b in public if b.group == group]
            ids = [i for i in ids if i is not None]
            if ids:
                sc.doc.Groups.Add(List[System.Guid](ids))   # one group per pod / wing
        for n, c in enumerate(cores, 1):
            add(c, f"vertical circulation #{n} - {c.height} floors")
        for s in core_shared:
            add(s, f"shared floor {s.origin.k} - {s.code} (around vertical circulation)")
        for apt in apartments:
            ids = [add(p, f"{apt.type} #{apt.number} floor {apt.floor} - {p.code}")
                   for p in apt.programs]
            ids += [add(s, f"{apt.type} #{apt.number} floor {apt.floor} - living room opening")
                    for s in apt.shared]
            ids = [i for i in ids if i is not None]
            if ids:
                sc.doc.Groups.Add(List[System.Guid](ids))   # one group per apartment
        for s in scattered:
            add(s, f"shared floor {s.origin.k} - {s.code} (between apartments)")
        for s in filler:
            add(s, f"shared floor {s.origin.k} - {s.code} (leftover fill)")
    finally:
        sc.doc.Views.RedrawEnabled = True
        sc.doc.Views.Redraw()

#=================================================
#--------------------------------------------- BATCH
# Generates and evaluates many runs in one go without drawing them: every run gets its own seed,
# and the settings ticked in the batch window are picked at random within their range. Every run
# is saved to the runs CSV (with its seed and settings, so any run can be rebuilt as a single run).

# (section, key) -> (label, kind, lowest allowed, highest allowed, default from, default to, ticked)
BATCH_PARAMS = [
    ("grid", "shared_weight", "Shared space weight", "float", 0.0, 10.0, 0.0, 4.0, True),
    ("grid", "spacing", "Core spacing", "int", 1, 60, 12, 28, True),
    ("ground_engagement", "spacing", "Ground: pod spacing", "int", 1, 60, 15, 40, True),
    ("ground_engagement", "length", "Ground: pod length", "int", 1, 40, 6, 14, True),
    ("ground_engagement", "spill", "Ground: spill past edge", "int", 0, 10, 2, 6, False),
    ("ground_engagement", "floors", "Ground: public floors", "int", 1, 6, 1, 3, False),
    ("ground_engagement", "void", "Ground: voids", "float", 0.0, 1.0, 0.0, 0.15, False),
    ("pathway_engagement", "width", "Pathway: growth each side", "int", 1, 15, 3, 8, False),
    ("pathway_engagement", "spill", "Pathway: spill past edge", "int", 0, 10, 2, 6, False),
    ("pathway_engagement", "floors", "Pathway: public floors", "int", 1, 6, 1, 3, False),
    ("pathway_engagement", "void", "Pathway: voids", "float", 0.0, 1.0, 0.0, 0.15, False),
    ("ends_ramp_down", "width", "Ends: growth from end", "int", 1, 15, 3, 8, False),
    ("ends_ramp_down", "spill", "Ends: spill past edge", "int", 0, 10, 2, 6, False),
    ("ends_ramp_down", "floors", "Ends: public floors", "int", 1, 6, 1, 3, False),
    ("ends_ramp_down", "void", "Ends: voids", "float", 0.0, 1.0, 0.0, 0.15, False),
]
BATCH_DRAW = ("Nothing (just the CSV)", "The best run (overall score)", "Every Pareto front run, side by side")

def generate(base_grid: Grid, spacing: int, shared_weight: float, ge, pe, er, seed: int) -> dict:
    #One complete generation on a copy of base_grid. ge / pe / er are the condition settings
    #(None = off). Every random choice follows from `seed`, so the same inputs give the same layout.
    random.seed(seed)
    grid = copy.deepcopy(base_grid)
    # circulation first: cores, then the shared landings around them on every floor, so the
    # engagement conditions below work around them and never take their place
    cores = place_vertcirc(grid, spacing)
    core_shared = place_core_shared(grid, cores)
    public, voids, crossings = [], {}, 0
    if ge:
        g_spacing, g_length, g_spill, g_floors, g_void = ge
        blocks, voids["Ground engagement"] = place_ground_engagement(
            grid, g_spacing, g_length, g_spill, g_floors, void=g_void)
        public += blocks
    if pe:
        p_width, p_spill, p_floors, p_void = pe
        blocks, crossings, voids["Pathway engagement"] = place_pathway_engagement(
            grid, p_width, p_spill, p_floors, void=p_void)
        public += blocks
        if crossings == 0:
            print("Pathway engagement: no obstacle crosses under the road, nothing added")
    if er:
        e_width, e_spill, e_floors, e_void = er
        blocks, voids["Ends / ramp down"] = place_ends_ramp_down(
            grid, e_width, e_spill, e_floors, void=e_void)
        public += blocks
    apartments, scattered, filler = [], [], []
    for k in range(grid.shape[2]):
        a, s, f = populate(grid, floor=k, shared_weight=shared_weight)
        apartments += a
        scattered += s
        filler += f
    for n, apt in enumerate(apartments, 1):
        apt.number = n
    return dict(grid=grid, cores=cores, core_shared=core_shared, apartments=apartments,
                scattered=scattered, filler=filler, public=public, voids=voids, crossings=crossings)

def condition_tuples(conditions: dict, sections: dict):
    #(ge, pe, er) settings tuples for generate(), None for a condition that is off
    g, p, e = sections["ground_engagement"], sections["pathway_engagement"], sections["ends_ramp_down"]
    ge = (g["spacing"], g["length"], g["spill"], g["floors"], g["void"]) if conditions["ground_engagement"] else None
    pe = (p["width"], p["spill"], p["floors"], p["void"]) if conditions["pathway_engagement"] else None
    er = (e["width"], e["spill"], e["floors"], e["void"]) if conditions["ends_ramp_down"] else None
    return ge, pe, er

def grid_margin(ge, pe, er) -> int:
    #Extra columns past each road edge the conditions need (their largest spill)
    return max(ge[2] if ge else 0, pe[1] if pe else 0, er[1] if er else 0)

def run_settings_row(grid_set: dict, seed: int, conditions: dict, sections: dict) -> dict:
    #The settings saved with a run in the runs CSV
    row = dict(grid_set)
    row["seed"] = seed
    row.update(conditions)
    for key in ("ground_engagement", "pathway_engagement", "ends_ramp_down"):
        if conditions[key]:
            row.update({f"{key}_{k}": v for k, v in sections[key].items()})
    return row

def ask_batch(owner, has_obstacles: bool) -> dict | None:
    #The batch window. Returns {"runs", "draw", "toggle_conditions", "vary": {(section, key): (lo, hi, kind)}}
    #or None if cancelled.
    last = sc.sticky.get("batch_settings") or {}
    run_fields = [dict(key="runs", label="Runs", value=last.get("runs", 20), kind="int", low=1, high=300,
                       hint="How many layouts to generate and evaluate")]
    run_box, run_boxes, _ = settings_section("Batch", "Each run gets its own seed and is saved to the runs CSV. "
                                             "Press Esc in Rhino to stop early.", run_fields)

    draw_list = forms.RadioButtonList()
    draw_list.Orientation = forms.Orientation.Vertical
    for text in BATCH_DRAW:
        draw_list.Items.Add(text)
    draw_list.SelectedIndex = int(last.get("draw", 1))
    toggle = forms.CheckBox()
    toggle.Text = "Turn each condition on or off at random"
    toggle.Checked = bool(last.get("toggle_conditions", False))
    toggle.ToolTip = "Off: every run uses the conditions ticked in the settings window"
    run_layout = forms.StackLayout()          # stacked from the top, no stretching
    run_layout.Spacing = 8
    run_layout.HorizontalContentAlignment = forms.HorizontalAlignment.Stretch
    for item in (run_box, hint_label("Draw afterwards"), draw_list, toggle):
        run_layout.Items.Add(forms.StackLayoutItem(item))

    # vary: one row per setting, a tick box and a from / to range
    vary_layout = forms.TableLayout()
    vary_layout.Spacing = drawing.Size(8, 5)
    hdr = [hint_label("Pick at random"), hint_label("from"), hint_label("to")]
    vary_layout.Rows.Add(forms.TableRow(forms.TableCell(hdr[0], True), forms.TableCell(hdr[1]), forms.TableCell(hdr[2])))
    rows = {}
    saved_vary = last.get("vary", {})
    for section, key, text, kind, low, high, lo_default, hi_default, ticked in BATCH_PARAMS:
        name = f"{section}.{key}"
        check = forms.CheckBox()
        check.Text = text
        prev = saved_vary.get(name)
        check.Checked = bool(prev[2]) if prev else ticked
        if section == "pathway_engagement" and not has_obstacles:
            check.Checked, check.Enabled = False, False
        boxes = []
        for value in ((prev[0] if prev else lo_default), (prev[1] if prev else hi_default)):
            box = forms.NumericStepper()
            box.DecimalPlaces = 0 if kind == "int" else 2
            box.Increment = 1.0 if kind == "int" else 0.05
            box.MinValue, box.MaxValue = float(low), float(high)
            box.Value = float(value)
            box.Width = 70
            boxes.append(box)
        vary_layout.Rows.Add(forms.TableRow(forms.TableCell(check, True), forms.TableCell(boxes[0]),
                                            forms.TableCell(boxes[1])))
        rows[(section, key)] = (check, boxes[0], boxes[1], kind)
    vary_box = forms.GroupBox()
    vary_box.Text = "Vary between runs"
    vary_box.Padding = drawing.Padding(10, 6, 10, 10)
    vary_box.Content = vary_layout

    dialog = forms.Dialog()
    dialog.Title = "Batch runs"
    dialog.Padding = drawing.Padding(14)
    dialog.Resizable = False
    result = {"ok": False}
    start, cancel = forms.Button(), forms.Button()
    start.Text, cancel.Text = "Run batch", "Cancel"

    def on_start(sender, e):
        result["ok"] = True
        dialog.Close()

    def on_cancel(sender, e):
        dialog.Close()

    start.Click += on_start
    cancel.Click += on_cancel
    dialog.DefaultButton, dialog.AbortButton = start, cancel
    buttons = forms.TableLayout()
    buttons.Spacing = drawing.Size(8, 0)
    buttons.Rows.Add(forms.TableRow(forms.TableCell(None, True), forms.TableCell(cancel), forms.TableCell(start)))
    page = forms.TableLayout()
    page.Spacing = drawing.Size(12, 12)
    page.Rows.Add(forms.TableRow(forms.TableCell(run_layout), forms.TableCell(vary_box, True)))
    page.Rows.Add(forms.TableRow(forms.TableCell(None), forms.TableCell(buttons)))
    dialog.Content = page
    dialog.ShowModal(owner or Rhino.UI.RhinoEtoApp.MainWindow)
    if not result["ok"]:
        return None

    vary, remember = {}, {}
    for (section, key), (check, lo_box, hi_box, kind) in rows.items():
        lo, hi = sorted((lo_box.Value, hi_box.Value))
        if kind == "int":
            lo, hi = int(round(lo)), int(round(hi))
        remember[f"{section}.{key}"] = (lo, hi, bool(check.Checked))
        if check.Checked and check.Enabled:
            vary[(section, key)] = (lo, hi, kind)
    config = {"runs": int(round(run_boxes["runs"].Value)), "draw": int(draw_list.SelectedIndex),
              "toggle_conditions": bool(toggle.Checked), "vary": vary}
    sc.sticky["batch_settings"] = {"runs": config["runs"], "draw": config["draw"],
                                   "toggle_conditions": config["toggle_conditions"], "vary": remember}
    return config

def run_batch(road, topo, obstacles, has_obstacles: bool, settings: dict, batch: dict) -> list[dict]:
    #Generates, evaluates and saves batch["runs"] runs, then draws what the batch window asked for.
    #Returns the runs (each {gen, result, row, seed}).
    import io, contextlib
    ft = Rhino.RhinoMath.UnitScale(Rhino.UnitSystem.Feet, sc.doc.ModelUnitSystem)
    weights = remembered("evaluation_weights", DEFAULT_WEIGHTS)
    picker = random.Random(random.SystemRandom().randrange(1, 2 ** 31))
    grids = {}                                    # margin -> grid built for it (built once each)
    done, saved = [], None
    n = batch["runs"]
    print(f"Batch: {n} runs. Press Esc to stop early.")
    Rhino.UI.StatusBar.ShowProgressMeter(0, n, "Batch runs", True, True)
    try:
        for i in range(n):
            if sc.escape_test(False):
                print(f"Batch stopped after {len(done)} runs")
                break
            grid_set = dict(settings["grid"])
            sections = {k: dict(settings[k]) for k in ("ground_engagement", "pathway_engagement", "ends_ramp_down")}
            conditions = dict(settings["conditions"])
            if batch["toggle_conditions"]:
                conditions = {k: picker.random() < 0.5 for k in conditions}
            if not has_obstacles:
                conditions["pathway_engagement"] = False
            for (section, key), (lo, hi, kind) in batch["vary"].items():
                value = picker.randint(lo, hi) if kind == "int" else round(picker.uniform(lo, hi), 2)
                (grid_set if section == "grid" else sections[section])[key] = value
            ge, pe, er = condition_tuples(conditions, sections)
            margin = grid_margin(ge, pe, er)
            if margin not in grids:
                clearance = grid_set["clearance"] if has_obstacles else 0.0
                grids[margin] = build_grid(road, topo, grid_set["dist"], grid_set["fh"], obstacles,
                                           clearance, margin, grid_set["coverage"])
            if grids[margin] is None:
                print("Batch: no room for a single floor between the road and the topo")
                return done
            seed = picker.randrange(1, 1000000)
            with contextlib.redirect_stdout(io.StringIO()):       # keep the command line readable
                gen = generate(grids[margin], grid_set["spacing"], grid_set["shared_weight"], ge, pe, er, seed)
                result = evaluate(gen["grid"], gen["cores"], gen["apartments"], gen["scattered"],
                                  gen["filler"], ft, weights)
            row = run_settings_row(grid_set, seed, conditions, sections)
            saved = save_evaluation(result, row) or saved
            s = result["summary"]
            on = "".join(l for k, l in (("ground_engagement", "G"), ("pathway_engagement", "P"),
                                        ("ends_ramp_down", "E")) if conditions[k]) or "-"
            print(f"  run {i + 1}/{n}  seed {seed:>6}  [{on}]  {s['apartments']} apartments  "
                  f"light {s['score_light']}  density {s['score_density']}  walking {s['score_walking']}  "
                  f"overall {s['score_overall']}")
            done.append(dict(gen=gen, result=result, row=row, seed=seed))
            Rhino.UI.StatusBar.UpdateProgressMeter(i + 1, True)
            Rhino.RhinoApp.Wait()                 # keep Rhino responsive (and Esc readable)
    finally:
        Rhino.UI.StatusBar.HideProgressMeter()
    if not done:
        return done

    # the front among this batch's runs, on the chosen scores
    chosen = pareto_scores_chosen()
    flags = pareto_flags([tuple(r["result"]["summary"][f"score_{c}"] for c in chosen) for r in done])
    front = [r for r, on in zip(done, flags) if on]
    best = max(done, key=lambda r: r["result"]["summary"]["score_overall"])
    print(f"Batch done: {len(done)} runs, {len(front)} on the Pareto front ({' / '.join(chosen)}) "
          f"among these runs. Best overall: seed {best['seed']} ({best['result']['summary']['score_overall']}).")
    if saved:                                     # the front over every run in the CSV file
        print(format_front(saved, this_run=False))
        print(f"Evaluation saved to {saved['folder']}")
    if batch["draw"] == 1:
        draw_run(best)
        print(summary(best["gen"]["grid"], best["gen"]["cores"], best["gen"]["core_shared"],
                      best["gen"]["apartments"], best["gen"]["scattered"], best["gen"]["filler"],
                      best["gen"]["public"], best["gen"]["voids"], best["gen"]["crossings"]))
        print(format_evaluation(best["result"]))
    elif batch["draw"] == 2:
        step = side_by_side_step([r["gen"]["grid"] for r in front])
        for n_run, r in enumerate(sorted(front, key=lambda r: -r["result"]["summary"]["score_overall"])):
            draw_run(r, step * n_run)
        print(f"Drew {len(front)} front runs side by side, best overall first, each labelled with its seed.")
    return done

def side_by_side_step(grids: Sequence[Grid]) -> rg.Vector3d:
    #How far to shift each drawn run so they sit side by side across the road's short direction,
    #sized by the widest of the runs (their pods can spill different amounts)
    c = np.concatenate([g.corners.reshape(-1, 3) for g in grids])
    dx, dy = float(np.ptp(c[:, 0])), float(np.ptp(c[:, 1]))
    gap = 2 * (grids[0].node_distance or 1.0)
    if dx >= dy:
        return rg.Vector3d(0, -(dy * 1.4 + gap), 0)
    return rg.Vector3d(dx * 1.4 + gap, 0, 0)

def draw_run(r: dict, offset: rg.Vector3d | None = None) -> None:
    #Draw one batch run, shifted by `offset`, with a text dot naming its seed and scores
    gen, s = r["gen"], r["result"]["summary"]
    draw(gen["grid"], gen["cores"], gen["core_shared"], gen["apartments"], gen["scattered"],
         gen["filler"], gen["public"], offset=offset, prefix=f"seed {r['seed']} - ")
    c = gen["grid"].corners[0, 0, 0]
    at = rg.Point3d(float(c[0]), float(c[1]), float(c[2]))
    if offset is not None:
        at += offset
    if not rs.IsLayer(LAYER_ROOT):
        rs.AddLayer(LAYER_ROOT)
    attr = Rhino.DocObjects.ObjectAttributes()
    attr.LayerIndex = sc.doc.Layers.FindByFullPath(LAYER_ROOT, -1)
    sc.doc.Objects.AddTextDot(f"seed {r['seed']} | overall {s['score_overall']}", at, attr)
    sc.doc.Views.Redraw()

#=================================================
#--------------------------------------------- MAIN

# ----- settings window: every setting in one window, grouped into sections
# Each section starts from the values used last in this Rhino session (sc.sticky), falling back
# to the parameters at the top of this script. Hover a setting for a longer explanation.

def remembered(name: str, defaults: dict) -> dict:
    #Last values used this Rhino session for a section, filled in from the defaults
    last = sc.sticky.get(name)
    out = dict(defaults)
    if isinstance(last, dict):
        out.update({k: v for k, v in last.items() if k in defaults})
    return out

def hint_label(text: str):
    #Small grey explanation line
    label = forms.Label()
    label.Text = text
    label.TextColor = drawing.Colors.Gray
    label.Wrap = forms.WrapMode.Word
    return label

def number_row(field: dict):
    #(label, slider or None, number box) for one numeric setting, the slider and box kept in sync.
    #Fractions slide in hundredths; whole numbers slide in steps of 1.
    is_int = field["kind"] == "int"
    low, high = float(field.get("low", 0)), float(field.get("high", 1e6))
    box = forms.NumericStepper()
    box.DecimalPlaces = 0 if is_int else field.get("decimals", 2)
    box.Increment = field.get("step", 1.0 if is_int else 0.05)
    box.MinValue, box.MaxValue = low, high
    box.Value = min(max(float(field["value"]), low), high)
    box.Width = 75
    label = forms.Label()
    label.Text = field["label"]
    label.VerticalAlignment = forms.VerticalAlignment.Center
    label.Width = 170
    if field.get("hint"):
        label.ToolTip = box.ToolTip = field["hint"]
    slider = None
    if "high" in field and field.get("slider", True):
        scale = 1 if is_int else 100
        slider = forms.Slider()
        slider.MinValue, slider.MaxValue = int(round(low * scale)), int(round(high * scale))
        slider.Value = int(round(box.Value * scale))
        slider.TickFrequency = max(1, (slider.MaxValue - slider.MinValue) // 10)   # about 10 ticks
        slider.Width = 160
        if field.get("hint"):
            slider.ToolTip = field["hint"]
        syncing = [False]

        def from_slider(sender, e):
            if not syncing[0]:
                syncing[0] = True
                box.Value = slider.Value / scale
                syncing[0] = False

        def from_box(sender, e):
            if not syncing[0]:
                syncing[0] = True
                slider.Value = int(round(box.Value * scale))
                syncing[0] = False

        slider.ValueChanged += from_slider
        box.ValueChanged += from_box
    return label, slider, box

def settings_section(title: str, subtitle: str, fields: list[dict],
                     toggle: dict | None = None, available: bool = True):
    #A boxed section. toggle = {"label", "value"} adds an on/off checkbox at the top that greys
    #out the rest of the section when off. available=False keeps the whole section off.
    #Returns (group box, {key: number box}, checkbox or None).
    group = forms.GroupBox()
    group.Text = title
    group.Padding = drawing.Padding(10, 6, 10, 10)
    table = forms.TableLayout()
    table.Spacing = drawing.Size(10, 6)
    check = None
    if toggle is not None:
        check = forms.CheckBox()
        check.Text = toggle["label"]
        check.Checked = bool(toggle["value"]) and available
        check.Enabled = available
        table.Rows.Add(forms.TableRow(forms.TableCell(check)))
    if subtitle:
        table.Rows.Add(forms.TableRow(forms.TableCell(hint_label(subtitle))))
    rows = forms.TableLayout()
    rows.Spacing = drawing.Size(10, 6)
    boxes, dependents = {}, []
    for f in fields:
        label, slider, box = number_row(f)
        rows.Rows.Add(forms.TableRow(forms.TableCell(label, True), forms.TableCell(slider),
                                     forms.TableCell(box)))
        boxes[f["key"]] = box
        dependents += [label, box] + ([slider] if slider is not None else [])
    table.Rows.Add(forms.TableRow(forms.TableCell(rows)))
    if check is not None:
        def refresh(sender=None, e=None):
            on = bool(check.Checked)
            for c in dependents:
                c.Enabled = on
        check.CheckedChanged += refresh
        refresh()
    group.Content = table
    return group, boxes, check

def read_numbers(fields: list[dict], boxes: dict) -> dict:
    return {f["key"]: (int(round(boxes[f["key"]].Value)) if f["kind"] == "int"
                       else float(boxes[f["key"]].Value)) for f in fields}

def pod_section_fields(last: dict, first: list[dict]) -> list[dict]:
    #The settings every engagement pod shares, after the section's own first settings
    return first + [
        dict(key="spill", label="Spill past edge", value=last["spill"], kind="int", low=0, high=10,
             hint="Nodes the pod reaches past each road edge on the ground floor"),
        dict(key="floors", label="Public floors", value=last["floors"], kind="int", low=1, high=6,
             hint="Floors of public space, counted up from the topo"),
        dict(key="void", label="Voids", value=last["void"], kind="float", low=0.0, high=1.0,
             hint="Share of the pod's nodes randomly left open (0 = solid)"),
    ]

def ask_weights(owner=None) -> dict | None:
    #The "Scoring weights" window: how much each aspect counts in the overall score, how much each
    #kind of lit wall counts, and how much each room type counts for privacy.
    #Saves to the session (sc.sticky) and returns the weights, or None if cancelled.
    last = remembered("evaluation_weights", DEFAULT_WEIGHTS)
    overall_fields = [
        dict(key="w_light", label="Light / air", value=last["w_light"], kind="float", low=0, high=5,
             step=0.25, hint="How much light / air counts in the overall score"),
        dict(key="w_community", label="Community", value=last["w_community"], kind="float", low=0, high=5,
             step=0.25, hint="Apartments connected to and able to see shared space"),
        dict(key="w_privacy", label="Privacy gradient", value=last["w_privacy"], kind="float", low=0, high=5,
             step=0.25, hint="Public space reached through shared space rather than directly"),
        dict(key="w_density", label="Density", value=last["w_density"], kind="float", low=0, high=5,
             step=0.25, hint="Share of the buildable nodes that are apartments"),
        dict(key="w_walking", label="Walking to egress", value=last["w_walking"], kind="float", low=0, high=5,
             step=0.25, hint=f"Share of apartments within {egress_limit_ft:.0f} ft of a core that reaches the topo"),
    ]
    light_fields = [
        dict(key="direct_outside", label="Direct to outside", value=last["direct_outside"], kind="float",
             low=0, high=1, hint="A wall facing straight out to open air"),
        dict(key="direct_shared", label="Direct to shared", value=last["direct_shared"], kind="float",
             low=0, high=1, hint="A wall facing shared space, with the view open beyond it"),
        dict(key="perp_outside", label="Perpendicular to outside", value=last["perp_outside"], kind="float",
             low=0, high=1, hint="A wall facing open air that is closed off further out (light from the side)"),
        dict(key="perp_shared", label="Perpendicular to shared", value=last["perp_shared"], kind="float",
             low=0, high=1, hint="A wall facing shared space that is closed off further out"),
    ]
    room_fields = [
        dict(key="bedroom", label="Bedroom", value=last["bedroom"], kind="float", low=0, high=5, step=0.25,
             hint="How much a bedroom facing public space counts"),
        dict(key="bathroom", label="Bathroom", value=last["bathroom"], kind="float", low=0, high=5, step=0.25,
             hint="How much a bathroom facing public space counts"),
        dict(key="living", label="Living", value=last["living"], kind="float", low=0, high=5, step=0.25,
             hint="How much a living room facing public space counts"),
        dict(key="kitchen", label="Kitchen", value=last["kitchen"], kind="float", low=0, high=5, step=0.25,
             hint="How much a kitchen facing public space counts"),
    ]
    overall_box, overall_boxes, _ = settings_section(
        "Overall score", "How much each aspect counts (0 = ignore it).", overall_fields)
    light_box, light_boxes, _ = settings_section(
        "Light / air", "How much one wall of each kind counts.", light_fields)
    room_box, room_boxes, _ = settings_section(
        "Privacy gradient", "How much each room facing public space counts.", room_fields)

    # Pareto front: which scores to compare across runs
    chosen = pareto_scores_chosen()
    pareto_box = forms.GroupBox()
    pareto_box.Text = "Pareto front"
    pareto_box.Padding = drawing.Padding(10, 6, 10, 10)
    pareto_layout = forms.TableLayout()
    pareto_layout.Spacing = drawing.Size(10, 6)
    pareto_layout.Rows.Add(forms.TableRow(forms.TableCell(hint_label(
        "Scores compared across runs. Pick 2-3: with all five, almost every run is on the front."))))
    pareto_checks = {}
    labels = {"light": "Light / air", "community": "Community", "privacy": "Privacy gradient",
              "density": "Density", "walking": "Walking to egress"}
    for name in SCORE_NAMES:
        check = forms.CheckBox()
        check.Text = labels[name]
        check.Checked = name in chosen
        pareto_checks[name] = check
        pareto_layout.Rows.Add(forms.TableRow(forms.TableCell(check)))
    pareto_box.Content = pareto_layout

    dialog = forms.Dialog()
    dialog.Title = "Scoring weights"
    dialog.Padding = drawing.Padding(14)
    dialog.Resizable = False
    result = {"ok": False}
    ok, cancel, reset = forms.Button(), forms.Button(), forms.Button()
    ok.Text, cancel.Text, reset.Text = "OK", "Cancel", "Reset to defaults"

    def on_ok(sender, e):
        result["ok"] = True
        dialog.Close()

    def on_cancel(sender, e):
        dialog.Close()

    def on_reset(sender, e):
        for boxes in (overall_boxes, light_boxes, room_boxes):
            for key, box in boxes.items():
                box.Value = float(DEFAULT_WEIGHTS[key])
        for name, check in pareto_checks.items():
            check.Checked = name in DEFAULT_PARETO_SCORES

    ok.Click += on_ok
    cancel.Click += on_cancel
    reset.Click += on_reset
    dialog.DefaultButton, dialog.AbortButton = ok, cancel

    left, right = forms.StackLayout(), forms.StackLayout()
    for column, boxes in ((left, (overall_box, pareto_box)), (right, (light_box, room_box))):
        column.Spacing = 10
        column.HorizontalContentAlignment = forms.HorizontalAlignment.Stretch
        for b in boxes:
            column.Items.Add(forms.StackLayoutItem(b))
    buttons = forms.TableLayout()
    buttons.Spacing = drawing.Size(8, 0)
    buttons.Rows.Add(forms.TableRow(forms.TableCell(None, True), forms.TableCell(cancel), forms.TableCell(ok)))
    reset_row = forms.TableLayout()
    reset_row.Rows.Add(forms.TableRow(forms.TableCell(reset), forms.TableCell(None, True)))
    page = forms.TableLayout()
    page.Spacing = drawing.Size(12, 12)
    page.Rows.Add(forms.TableRow(forms.TableCell(left, True), forms.TableCell(right, True)))
    page.Rows.Add(forms.TableRow(forms.TableCell(reset_row), forms.TableCell(buttons)))
    dialog.Content = page
    dialog.ShowModal(owner or Rhino.UI.RhinoEtoApp.MainWindow)
    if not result["ok"]:
        return None
    values = dict(last)
    for fields, boxes in ((overall_fields, overall_boxes), (light_fields, light_boxes), (room_fields, room_boxes)):
        values.update(read_numbers(fields, boxes))
    sc.sticky["evaluation_weights"] = values
    picked = [name for name in SCORE_NAMES if pareto_checks[name].Checked]
    sc.sticky["pareto_scores"] = picked if len(picked) >= 2 else list(DEFAULT_PARETO_SCORES)
    if len(picked) < 2:
        print("Pareto front needs at least 2 scores; using " + ", ".join(DEFAULT_PARETO_SCORES))
    return values

def ask_settings(has_obstacles: bool) -> dict | None:
    #Every setting in one window. Returns
    #  {"grid": {...}, "conditions": {key: bool}, "ground_engagement": {...},
    #   "pathway_engagement": {...}, "ends_ramp_down": {...}}
    #or None if cancelled.
    ft = Rhino.RhinoMath.UnitScale(Rhino.UnitSystem.Feet, sc.doc.ModelUnitSystem)
    grid_last = remembered("grid_settings", {
        "dist": node_distance, "fh": floor_height or node_distance, "coverage": road_coverage,
        "spacing": vertcirc_spacing, "shared_weight": float(shared_space_weight),
        "clearance": obstacle_clearance_ft * ft, "seed": 0})
    on_last = remembered("conditions_on", {k: False for k in
                                           ("ground_engagement", "pathway_engagement", "ends_ramp_down")})
    ge_last = remembered("ground_engagement_settings", {
        "spacing": ge_spacing, "length": ge_length, "spill": ge_spill, "floors": ge_floors, "void": ge_void})
    pe_last = remembered("pathway_engagement_settings", {
        "width": pe_width, "spill": pe_spill, "floors": pe_floors, "void": pe_void})
    er_last = remembered("ends_ramp_down_settings", {
        "width": er_width, "spill": er_spill, "floors": er_floors, "void": er_void})

    grid_fields = [
        dict(key="dist", label="Node distance", value=grid_last["dist"], kind="float", low=0.01, step=1,
             hint="Size of one node (cell) along and across the road, in model units"),
        dict(key="fh", label="Floor height", value=grid_last["fh"], kind="float", low=0.01, step=1,
             hint="Height of one floor, in model units"),
        dict(key="coverage", label="Cell on road", value=grid_last["coverage"], kind="float",
             low=0.05, high=1.0,
             hint="Share of a cell that must sit on the road surface. Angled ends become a "
                  "staircase of whole cells; lower keeps more cells along the angle"),
        dict(key="spacing", label="Core spacing", value=grid_last["spacing"], kind="int", low=1, high=60,
             hint="Nodes between vertical circulation cores along the road"),
        dict(key="shared_weight", label="Shared space", value=grid_last["shared_weight"], kind="float",
             low=0.0, high=10.0, step=0.25,
             hint="How often a shared space is placed instead of an apartment (0 = only leftover gaps)"),
        dict(key="seed", label="Random seed", value=grid_last["seed"], kind="int", low=0, high=999999,
             slider=False,
             hint="0 = a new random layout every run. Any other number rebuilds the same layout "
                  "(same road, settings and script). Each run's seed is printed and saved to the CSV"),
    ]
    if has_obstacles:
        grid_fields.append(dict(key="clearance", label="Obstacle clearance", value=grid_last["clearance"],
                                kind="float", low=0.0, step=1,
                                hint="Nothing is built within this distance above an obstacle"))
    ge_fields = pod_section_fields(ge_last, [
        dict(key="spacing", label="Pod spacing", value=ge_last["spacing"], kind="int", low=1, high=60,
             hint="Nodes between pod centers along the road"),
        dict(key="length", label="Pod length", value=ge_last["length"], kind="int", low=1, high=40,
             hint="Pod length along the road on the ground floor, in nodes"),
    ])
    pe_fields = pod_section_fields(pe_last, [
        dict(key="width", label="Growth each side", value=pe_last["width"], kind="int", low=1, high=15,
             hint="Nodes the pod grows along the road on each side of the crossing"),
    ])
    er_fields = pod_section_fields(er_last, [
        dict(key="width", label="Growth from end", value=er_last["width"], kind="int", low=1, high=15,
             hint="Nodes the half pod grows inward from each end"),
    ])

    grid_box, grid_boxes, _ = settings_section(
        "Grid and circulation", "Leave every condition off for Normal.", grid_fields)
    ge_box, ge_boxes, ge_check = settings_section(
        "Ground engagement", "Public pods at an interval, standing on the topo.", ge_fields,
        toggle={"label": "Use ground engagement", "value": on_last["ground_engagement"]})
    pe_box, pe_boxes, pe_check = settings_section(
        "Pathway engagement",
        "A pod split by each road that crosses under." if has_obstacles
        else "Pick the crossing roads as obstacles to use this.",
        pe_fields, toggle={"label": "Use pathway engagement", "value": on_last["pathway_engagement"]},
        available=has_obstacles)
    er_box, er_boxes, er_check = settings_section(
        "Ends / ramp down", "Half a pod at each short end of the generation.", er_fields,
        toggle={"label": "Use ends / ramp down", "value": on_last["ends_ramp_down"]})

    dialog = forms.Dialog()
    dialog.Title = "Generator settings"
    dialog.Padding = drawing.Padding(14)
    dialog.Resizable = False
    result = {"ok": False}
    generate, cancel, weights = forms.Button(), forms.Button(), forms.Button()
    generate.Text, cancel.Text, weights.Text = "Generate", "Cancel", "Scoring weights..."
    weights.ToolTip = "How the evaluation scores each aspect"

    def on_weights(sender, e):
        ask_weights(dialog)

    weights.Click += on_weights

    def on_generate(sender, e):
        result["ok"] = True
        dialog.Close()

    batch = forms.Button()
    batch.Text = "Run batch..."
    batch.ToolTip = "Generate and evaluate many runs with these settings, varying the ones you pick"

    def on_batch(sender, e):
        config = ask_batch(dialog, has_obstacles)
        if config is not None:
            result["ok"], result["batch"] = True, config
            dialog.Close()

    batch.Click += on_batch

    def on_cancel(sender, e):
        dialog.Close()

    generate.Click += on_generate
    cancel.Click += on_cancel
    dialog.DefaultButton, dialog.AbortButton = generate, cancel

    left, right = forms.StackLayout(), forms.StackLayout()
    for column, boxes in ((left, (grid_box, ge_box)), (right, (pe_box, er_box))):
        column.Spacing = 10
        column.HorizontalContentAlignment = forms.HorizontalAlignment.Stretch
        for b in boxes:
            column.Items.Add(forms.StackLayoutItem(b))
    buttons = forms.TableLayout()
    buttons.Spacing = drawing.Size(8, 0)
    buttons.Rows.Add(forms.TableRow(forms.TableCell(None, True), forms.TableCell(cancel),
                                    forms.TableCell(batch), forms.TableCell(generate)))
    page = forms.TableLayout()
    page.Spacing = drawing.Size(12, 12)
    page.Rows.Add(forms.TableRow(forms.TableCell(left, True), forms.TableCell(right, True)))
    weights_row = forms.TableLayout()
    weights_row.Rows.Add(forms.TableRow(forms.TableCell(weights), forms.TableCell(None, True)))
    page.Rows.Add(forms.TableRow(forms.TableCell(weights_row), forms.TableCell(buttons)))
    dialog.Content = page
    dialog.ShowModal(Rhino.UI.RhinoEtoApp.MainWindow)
    if not result["ok"]:
        return None

    out = {
        "grid": dict(grid_last, **read_numbers(grid_fields, grid_boxes)),
        "conditions": {"ground_engagement": bool(ge_check.Checked),
                       "pathway_engagement": bool(pe_check.Checked) and has_obstacles,
                       "ends_ramp_down": bool(er_check.Checked)},
        "ground_engagement": read_numbers(ge_fields, ge_boxes),
        "pathway_engagement": read_numbers(pe_fields, pe_boxes),
        "ends_ramp_down": read_numbers(er_fields, er_boxes),
        "batch": result.get("batch"),
    }
    sc.sticky["grid_settings"] = out["grid"]
    sc.sticky["conditions_on"] = {k: (v or (k == "pathway_engagement" and on_last[k] and not has_obstacles))
                                  for k, v in out["conditions"].items()}
    sc.sticky["ground_engagement_settings"] = out["ground_engagement"]
    sc.sticky["pathway_engagement_settings"] = out["pathway_engagement"]
    sc.sticky["ends_ramp_down_settings"] = out["ends_ramp_down"]
    return out

def main():
    road = pick_road()
    if road is None:
        return
    topo = pick_topo()
    if topo is None:
        return
    obstacles = pick_obstacles()
    if obstacles is None:
        return
    has_obstacles = obstacles.Faces.Count > 0

    settings = ask_settings(has_obstacles)
    if settings is None:
        return
    if settings["batch"]:
        run_batch(road, topo, obstacles, has_obstacles, settings, settings["batch"])
        return

    grid_set, conditions = settings["grid"], settings["conditions"]
    sections = {k: settings[k] for k in ("ground_engagement", "pathway_engagement", "ends_ramp_down")}
    clearance = grid_set["clearance"] if has_obstacles else 0.0
    seed = int(grid_set.get("seed") or 0) or random.SystemRandom().randrange(1, 1000000)
    print(f"Random seed: {seed}")
    ge, pe, er = condition_tuples(conditions, sections)

    # ----- generate (internal only, nothing drawn yet)
    grid = build_grid(road, topo, grid_set["dist"], grid_set["fh"], obstacles, clearance,
                      grid_margin(ge, pe, er), grid_set["coverage"])
    if grid is None:
        return
    gen = generate(grid, grid_set["spacing"], grid_set["shared_weight"], ge, pe, er, seed)

    # ----- draw everything at the end
    draw(gen["grid"], gen["cores"], gen["core_shared"], gen["apartments"], gen["scattered"],
         gen["filler"], gen["public"])
    print(summary(gen["grid"], gen["cores"], gen["core_shared"], gen["apartments"], gen["scattered"],
                  gen["filler"], gen["public"], gen["voids"], gen["crossings"]))

    # ----- evaluate, print and save to CSV
    ft = Rhino.RhinoMath.UnitScale(Rhino.UnitSystem.Feet, sc.doc.ModelUnitSystem)
    result = evaluate(gen["grid"], gen["cores"], gen["apartments"], gen["scattered"], gen["filler"], ft,
                      remembered("evaluation_weights", DEFAULT_WEIGHTS))
    print(format_evaluation(result))
    saved = save_evaluation(result, run_settings_row(grid_set, seed, conditions, sections))
    if saved:
        print(format_front(saved))
        print(f"Evaluation saved to {saved['folder']}")

if __name__ == "__main__":
    main()
