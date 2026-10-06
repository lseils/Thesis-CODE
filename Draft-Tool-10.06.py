
# r: numpy
import Rhino
import Rhino.Geometry as rg
from Rhino.Geometry.Intersect import Intersection
import scriptcontext as sc
import rhinoscriptsyntax as rs #slower than rg
import math
import random
import System.Drawing
from enum import IntEnum
from typing import NamedTuple, Iterator, Sequence
import numpy as np
from __future__ import annotations
OT = Rhino.DocObjects.ObjectType
import copy
#import Rhino.Geometry.RTree
#import Rhino.display - temporary graphics

#=================================================
#=================================================
#=================================================
#-------------------------------------- PARAMETERS 

node_distance = 0
floor_height = 0
density = 0

#also ask user to select bridge and topo layer

available_node_list = []

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
    i: int
    j: int
    k: int

#______________________________
class Grid:
    def __init__(self, positions: np.ndarray):
        #positions: float array of shape (n_div, n_off, n_z, 3) holding world X, Y, Z
        self.positions = positions
        self.shape = positions.shape[:3]
        self.states = np.full(self.shape, ProgramState.EMPTY, dtype=np.uint8)

    @classmethod
    def from_points(cls, points: Sequence[tuple[float, float, float]],
                    n_div: int, n_off: int, n_z: int) -> Grid:
        arr = np.asarray(points, dtype=float)
        expected = n_div * n_off * n_z
        if arr.shape != (expected, 3)
            raise ValueError(f" ahh")
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


    # -- single nodes
    
    def in_bounds(self, i: int, j: int, k: int) -> bool:
        a, b, c = self.shape
        return 0 <= i < a and 0 <= j < b and 0 <= k < c

    def get_state(self, i: int, j: int, k: int) -> ProgramState:
        return ProgramState(self.states[i, j, k])

    def is_empty(self, i: int, j: int, k: int) -> bool:
        return self.in_bounds(i, j, k) and self.states[i, j, k] == ProgramState.EMPTY

    def occupy(self, i: int, j: int, kz: int, state: ProgramState = ProgramState.USED) -> None:
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

    #Use
    # pts = []
    # for i in range(6):
    #     angle = i * np.pi / 10
    #     for j in range(3):
    #         r = 10 + j * 2
    #         for k in range(4):
    #             pts.append((r * np.cos(angle), r * np.sin(angle), k * 3.0))

    # g = Grid.from_points(pts, n_div=6, n_off=3, n_z=4)
    # assert g.shape == (6, 3, 4)
    # assert g.place_box(0, 0, 0, 2, 2, 2)
    # assert not g.box_fits(1, 1, 1, 2, 2, 2)      # overlaps
    # assert not g.box_fits(5, 0, 0, 2, 1, 1)      # runs off the end of the curve
    # assert g.count_free() == 72 - 8
    # assert g.first_free() == Node(0, 0, 2)
    # assert len(list(g.neighbors(Node(0, 0, 0)))) == 3
    # assert g.nearest_node(*g.position(Node(4, 2, 3))) == Node(4, 2, 3)
    # assert g.nearest_node(12.1, 0.1, 5.8) == Node(0, 1, 2)
    # assert g.box_corners(0, 0, 0, 2, 2, 2).shape == (8, 3)
    # print("all checks passed")
        
#______________________________
class ProgramLayout(IntEnum):
    OXW = 0
    WXW = 1
    HXW = 2

#______________________________
class Program:
    def __init__(self, type, layout, nodes):
        self.type = type
        self.layout = layout
        self.nodes = nodes


    def is_valid_node(self, coordinates):
        x, y, z = coordinates
        if coordinates in available_node_list:
            return True
        else:
            False

    def check_conflict(self, nodedims):
        checkmatrix = []
        if nodedims == 0: #One x Two layout
            checkmatrix = [1, 2]
        elif nodedims == 1: #Two x Two layout
            checkmatrix = [2, 2]
        elif nodedims == 2: #Three x Two layout
            checkmatrix = [3, 2]
        else:
            raise("No layout assigned")

            

    """
    L1 = 1
    L2 = 2
    BA = 1
    BR1 = 2
    BR2 = 3
    K1 = 2
    K2 = 3
    S1 = 1
    S2 = 2
    S3 = 3
    """

#______________________________
class Apartment:
    def __int__(self, type, number, floor):
        self.type = type
        self.number = number
        self.floor = floor

    """
    @classmethod
    def studio(cls):
        return cls([L1, BA1, BR1, K1])
    @classmethod
    def onebedA(cls):
        return cls([L1, BA1, BR1, K1])
    @classmethod
    def onebedB(cls):
        return cls([L1, BA1, BR1, K1])
    @classmethod
    def twobedA(cls):
        return cls([L1, BA1, BR1, K1])
    @classmethod
    def twobedB(cls):
        return cls([L1, BA1, BR1, K1])
    @classmethod
    def threebedA(cls):
        return cls([L1, BA1, BR1, K1])
    @classmethod
    def threebedB(cls):
        return cls([L1, BA1, BR1, K1])
    """





                

# class Populate:
#     def __init__(self):
#         pass

    


        #Make list of all first floor Nodes
        #Each round we pick an apartment type
            #In that we pick program
            #First we place Living 
                #Randomly choose one of the smallest x node
                #search for the nodes in the desired dim 
                #if avaliable for the entire dim, set nodes' state to Living
                #elif, depending on where there was conflict, we move over
                #else, shared space?
            #For the rest of the programs
                #Search the nodes around the living nodes
                #once find empty next to living, search for available for entire dim
                #if available set nodes to program
                #elif, depending on where there was conflict, we rotate or move over
                #else, shared space?

                
        



#=================================================
#--------------------------------- AVAILABLE NODES

def divide_distance(crv, dist):
    ts = crv.DivideByLength(dist, True)
    if ts is None:
        return [crv.PointAtStart]
    return [crv.PointAt(t) for t in ts]

def closeset_point_on(crv, pt):
    ok, t = crv.ClosestPoint(pt)
    return crv.PointAt(t)

def initial_nodes():
    tol = sc.doc.ModelAbsoluteTolerance

    ROAD = pick_road()
    if ROAD is None:
        return
    TOPO = pick_topo()
    if TOPO is None:
        return

    node_distance = rs.GetReal("Node distance", 10.0, 0.001)
    if node_distance is None:
        return
    iso_param = 0.5


    # -------------------------------------------- 1. two longest road edges
    # These are the road's side boundaries; they define where the road ends
    # across its width ("out of bounds").
    naked = [e for e in ROAD.Edges if e.Valence == rg.EdgeAdjacency.Naked]
    if len(naked) < 2:
        print("Road needs at least 2 naked edges")
        return

    naked.sort(key=lambda e: e.GetLength())
    edge_a = naked[-1].DuplicateCurve()
    edge_b = naked[-2].DuplicateCurve()

    # -------------------------------------------- 2. centerline (Isocurve)
    face = ROAD.Faces[0]
    ok, u_dim, v_dim = face.GetSurfaceSize()
    if u_dim >= v_dim:
    # road runs along U: hold V fixed at iso_param
        centerline = face.IsoCurve(0, face.Domain(1).ParameterAt(iso_param))
    else:
    # road runs along V: hold U fixed at iso_param
        centerline = face.IsoCurve(1, face.Domain(0).ParameterAt(iso_param))

    # -------------------------------------------- 3. divide center line
    stations = divide_distance(centerline, node_distance)

    # ----- 4. offset sideways every node_distance until past the road edge
    road_nodes = []
    for p in stations:
        road_nodes.append(p)
        for edge in (edge_a, edge_b):
            target = closest_point_on(edge, p)
            v = target - p
            width = v.Length
            if not v.Unitize():
                continue
            steps = int((width + tol) / node_distance)
            for k in range(1, steps + 1):
                road_nodes.append(p + v * (k * node_distance))
 
    # ----- 5. offset every node down until it would pass below the topo
    down = rg.Vector3d(0, 0, -1)
    available_node_list = []
    for node in road_nodes:
        available_node_list.append(node)
        t = Intersection.MeshRay(TOPO, rg.Ray3d(node, down))
        if t < 0:
            continue                          # no topo below this node
        steps = int((t + tol) / node_distance)
        for k in range(1, steps + 1):
            available_node_list.append(
                rg.Point3d(node.X, node.Y, node.Z - k * node_distance)
            )

    available_node_list = [(p.X, p.Y, p.Z) for p in available_node_list]
    # ----- 6. add the nodes to the model
    # for pt in available_node_list:
    #     sc.doc.Objects.AddPoint(pt)
    # sc.doc.Views.Redraw()
    # print("Added {} nodes".format(len(available_node_list)))


    

#=================================================
#------------------------------------------ 