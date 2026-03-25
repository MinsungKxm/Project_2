"""CSC111 Winter 2026 Project: Constellation Explorer

similarity.py - Graph Similarity Scoring

This module computes a similarity score between a user's drawn constellation
and the actual constellation shape. It normalizes both graphs geometrically
(accounting for translation, scale, and rotation) and computes a combined
structural and geometric similarity score.


"""

from __future__ import annotations
import math
import doctest
import python_ta


###############################################################################
# Data Types
###############################################################################

class StarNode:
    """A node in a constellation graph representing a single star.

    Instance Attributes:
        - star_id: A unique identifier for this star (e.g. HIP catalog number or name).
        - x: The projected x-coordinate of the star on the 2D sky map (in pixels or degrees).
        - y: The projected y-coordinate of the star on the 2D sky map.
        - neighbours: A set of star_ids that this star is connected to by a constellation line.

    Representation Invariants:
        - self.star_id not in self.neighbours
        - all edges are undirected: if b in self.neighbours then self appears in b's neighbours
    """
    star_id: str
    x: float
    y: float
    neighbours: set[str]

    def __init__(self, star_id: str, x: float, y: float) -> None:
        """Initialize a StarNode with the given id and coordinates, and no neighbours."""
        self.star_id = star_id
        self.x = x
        self.y = y
        self.neighbours = set()


class ConstellationGraph:
    """A graph representing the stars and connecting lines of a constellation.

    Each vertex is a StarNode. Each edge represents a line drawn between two
    stars in the constellation as seen from Earth (a 2D projection).

    Instance Attributes:
        - name: The name of the constellation (e.g. 'Orion').
        - _nodes: A mapping from star_id to StarNode for all stars in this graph.

    Representation Invariants:
        - All star_ids referenced in any node's neighbours exist as keys in _nodes.
        - For any two nodes u, v: v.star_id in u.neighbours iff u.star_id in v.neighbours.
    """
    name: str
    _nodes: dict[str, StarNode]

    def __init__(self, name: str) -> None:
        """Initialize an empty ConstellationGraph with the given constellation name."""
        self.name = name
        self._nodes = {}

    def add_star(self, star_id: str, x: float, y: float) -> None:
        """Add a star with the given id and coordinates to this graph.

        Preconditions:
            - star_id not in self._nodes
        """
        self._nodes[star_id] = StarNode(star_id, x, y)

    def add_edge(self, id1: str, id2: str) -> None:
        """Add an undirected edge between the two stars with the given ids.

        Preconditions:
            - id1 in self._nodes and id2 in self._nodes
            - id1 != id2
            - id2 not in self._nodes[id1].neighbours
        """
        self._nodes[id1].neighbours.add(id2)
        self._nodes[id2].neighbours.add(id1)

    def get_nodes(self) -> dict[str, StarNode]:
        """Return the dictionary of all StarNodes in this graph."""
        return self._nodes

    def get_edges(self) -> list[tuple[str, str]]:
        """Return a list of all edges as (id1, id2) pairs with id1 < id2 (no duplicates).

        >>> g = ConstellationGraph('Test')
        >>> g.add_star('A', 0.0, 0.0)
        >>> g.add_star('B', 1.0, 0.0)
        >>> g.add_edge('A', 'B')
        >>> g.get_edges()
        [('A', 'B')]
        """
        edges = set()
        for node in self._nodes.values():
            for neighbour_id in node.neighbours:
                edge = (min(node.star_id, neighbour_id), max(node.star_id, neighbour_id))
                edges.add(edge)
        return sorted(edges)

    def num_stars(self) -> int:
        """Return the number of stars (nodes) in this constellation graph.

        >>> g = ConstellationGraph('Test')
        >>> g.add_star('A', 0.0, 0.0)
        >>> g.add_star('B', 1.0, 1.0)
        >>> g.num_stars()
        2
        """
        return len(self._nodes)

    def num_edges(self) -> int:
        """Return the number of edges in this constellation graph.

        >>> g = ConstellationGraph('Test')
        >>> g.add_star('A', 0.0, 0.0)
        >>> g.add_star('B', 1.0, 0.0)
        >>> g.add_edge('A', 'B')
        >>> g.num_edges()
        1
        """
        return len(self.get_edges())


###############################################################################
# Geometric Normalization Helpers
###############################################################################

def _compute_centroid(nodes: dict[str, StarNode]) -> tuple[float, float]:
    """Return the (x, y) centroid of all star positions in the given node dictionary.

    The centroid is the mean x and mean y across all nodes.

    Preconditions:
        - len(nodes) > 0

    >>> from similarity import StarNode
    >>> nodes = {'A': StarNode('A', 0.0, 0.0), 'B': StarNode('B', 2.0, 2.0)}
    >>> _compute_centroid(nodes)
    (1.0, 1.0)
    """
    xs = [node.x for node in nodes.values()]
    ys = [node.y for node in nodes.values()]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _compute_scale(nodes: dict[str, StarNode], cx: float, cy: float) -> float:
    """Return the RMS (root mean square) distance from the centroid to all nodes.

    This is used to normalize the graph to unit scale. Returns 1.0 if all
    nodes are at the centroid (degenerate case) to avoid division by zero.

    Preconditions:
        - len(nodes) > 0

    >>> from similarity import StarNode
    >>> nodes = {'A': StarNode('A', 1.0, 0.0), 'B': StarNode('B', -1.0, 0.0)}
    >>> abs(_compute_scale(nodes, 0.0, 0.0) - 1.0) < 1e-9
    True
    """
    distances_sq = [(node.x - cx) ** 2 + (node.y - cy) ** 2 for node in nodes.values()]
    rms = math.sqrt(sum(distances_sq) / len(distances_sq))
    return rms if rms > 1e-9 else 1.0


def _normalize_positions(nodes: dict[str, StarNode]) -> dict[str, tuple[float, float]]:
    """Return a dictionary mapping star_id to its normalized (x, y) position.

    Normalization steps:
        1. Translate so the centroid is at the origin.
        2. Scale so the RMS distance from the origin is 1.0.

    The original StarNode objects are not mutated.

    Preconditions:
        - len(nodes) > 0

    >>> from similarity import StarNode
    >>> nodes = {'A': StarNode('A', 0.0, 0.0), 'B': StarNode('B', 2.0, 0.0)}
    >>> result = _normalize_positions(nodes)
    >>> abs(result['A'][0] - (-1.0)) < 1e-9
    True
    >>> abs(result['B'][0] - 1.0) < 1e-9
    True
    """
    cx, cy = _compute_centroid(nodes)
    scale = _compute_scale(nodes, cx, cy)
    return {
        sid: ((node.x - cx) / scale, (node.y - cy) / scale)
        for sid, node in nodes.items()
    }


def _best_rotation_angle(
    user_pos: dict[str, tuple[float, float]],
    real_pos: dict[str, tuple[float, float]],
    matching: dict[str, str]
) -> float:
    """Return the rotation angle (in radians) that best aligns the user's normalized
    positions to the real constellation's normalized positions, given a star matching.

    Uses the closed-form solution for the optimal 2D rotation via the cross and dot
    products of matched point pairs (Procrustes analysis).

    If the matching is empty or positions are degenerate, returns 0.0.

    Preconditions:
        - All keys in matching are present in user_pos.
        - All values in matching are present in real_pos.

    >>> angle = _best_rotation_angle({'A': (1.0, 0.0)}, {'X': (0.0, 1.0)}, {'A': 'X'})
    >>> abs(angle - math.pi / 2) < 1e-6
    True
    """
    cross_sum = 0.0
    dot_sum = 0.0
    for user_id, real_id in matching.items():
        ux, uy = user_pos[user_id]
        rx, ry = real_pos[real_id]
        cross_sum += ux * ry - uy * rx
        dot_sum += ux * rx + uy * ry
    if abs(cross_sum) < 1e-12 and abs(dot_sum) < 1e-12:
        return 0.0
    return math.atan2(cross_sum, dot_sum)


def _rotate_positions(
    positions: dict[str, tuple[float, float]],
    angle: float
) -> dict[str, tuple[float, float]]:
    """Return a new dictionary with all positions rotated by the given angle (radians)
    around the origin.

    Preconditions:
        - angle is a valid float (not NaN or inf)

    >>> pos = {'A': (1.0, 0.0)}
    >>> rotated = _rotate_positions(pos, math.pi / 2)
    >>> abs(rotated['A'][0] - 0.0) < 1e-6 and abs(rotated['A'][1] - 1.0) < 1e-6
    True
    """
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return {
        sid: (cos_a * x - sin_a * y, sin_a * x + cos_a * y)
        for sid, (x, y) in positions.items()
    }


###############################################################################
# Star Matching
###############################################################################

def _match_user_to_real(
    user_nodes: dict[str, StarNode],
    real_nodes: dict[str, StarNode]
) -> dict[str, str]:
    """Return a matching from user star_ids to real star_ids based on nearest normalized
    position.

    Each user star is matched to the closest real star (by Euclidean distance after
    normalization). This is a greedy nearest-neighbour matching — not globally optimal,
    but fast and sufficient for scoring purposes.

    If the user graph has more stars than the real graph, excess user stars are
    matched to the nearest real star (duplicates allowed). If fewer, some real
    stars will be unmatched.

    Preconditions:
        - len(user_nodes) > 0
        - len(real_nodes) > 0

    >>> from similarity import StarNode, ConstellationGraph
    >>> u = {'U1': StarNode('U1', 0.0, 0.0), 'U2': StarNode('U2', 1.0, 0.0)}
    >>> r = {'R1': StarNode('R1', 0.1, 0.0), 'R2': StarNode('R2', 0.9, 0.0)}
    >>> m = _match_user_to_real(u, r)
    >>> m['U1'] == 'R1' and m['U2'] == 'R2'
    True
    """
    user_norm = _normalize_positions(user_nodes)
    real_norm = _normalize_positions(real_nodes)

    matching = {}
    for u_id, (ux, uy) in user_norm.items():
        best_id = min(real_norm, key=lambda r_id: (ux - real_norm[r_id][0]) ** 2
                                                   + (uy - real_norm[r_id][1]) ** 2)
        matching[u_id] = best_id
    return matching


###############################################################################
# Similarity Score Components
###############################################################################

def _geometric_score(
    user_nodes: dict[str, StarNode],
    real_nodes: dict[str, StarNode],
    matching: dict[str, str]
) -> float:
    """Return a geometric similarity score in [0.0, 1.0] between the user graph
    and the real constellation graph.

    Steps:
        1. Normalize both graphs (translate + scale).
        2. Find the best rotation to align the user graph to the real graph.
        3. Compute the mean Euclidean distance between matched star pairs after alignment.
        4. Convert to a score: score = exp(-k * mean_distance), where k controls sharpness.
           A perfect alignment gives score 1.0; larger distances decay toward 0.0.

    Preconditions:
        - len(matching) > 0
        - All keys in matching are in user_nodes.
        - All values in matching are in real_nodes.
    """
    user_norm = _normalize_positions(user_nodes)
    real_norm = _normalize_positions(real_nodes)

    angle = _best_rotation_angle(user_norm, real_norm, matching)
    user_rotated = _rotate_positions(user_norm, angle)

    distances = []
    for u_id, r_id in matching.items():
        ux, uy = user_rotated[u_id]
        rx, ry = real_norm[r_id]
        distances.append(math.sqrt((ux - rx) ** 2 + (uy - ry) ** 2))

    mean_dist = sum(distances) / len(distances)
    k = 3.0  # sharpness constant: controls how quickly score falls with distance
    return math.exp(-k * mean_dist)


def _structural_score(
    user_graph: ConstellationGraph,
    real_graph: ConstellationGraph,
    matching: dict[str, str]
) -> float:
    """Return a structural similarity score in [0.0, 1.0] comparing the edge sets
    of the user graph and the real constellation graph.

    An edge (u1, u2) in the user graph is considered "correct" if the matched pair
    (matching[u1], matching[u2]) is an edge in the real graph.

    Score formula:
        structural_score = correct_edges / max(user_edges, real_edges)

    A perfect score of 1.0 means every user edge matches a real edge and no real
    edges are missing.

    Preconditions:
        - All keys in matching are valid star_ids in user_graph.
        - All values in matching are valid star_ids in real_graph.

    >>> g_user = ConstellationGraph('U')
    >>> g_user.add_star('A', 0.0, 0.0)
    >>> g_user.add_star('B', 1.0, 0.0)
    >>> g_user.add_edge('A', 'B')
    >>> g_real = ConstellationGraph('R')
    >>> g_real.add_star('X', 0.0, 0.0)
    >>> g_real.add_star('Y', 1.0, 0.0)
    >>> g_real.add_edge('X', 'Y')
    >>> _structural_score(g_user, g_real, {'A': 'X', 'B': 'Y'})
    1.0
    """
    real_nodes = real_graph.get_nodes()
    real_edge_set = set()
    for node in real_nodes.values():
        for nb in node.neighbours:
            real_edge_set.add(
                (min(node.star_id, nb), max(node.star_id, nb))
            )

    user_nodes = user_graph.get_nodes()
    correct = 0
    for node in user_nodes.values():
        for nb in node.neighbours:
            mapped_u = matching.get(node.star_id)
            mapped_nb = matching.get(nb)
            if mapped_u is not None and mapped_nb is not None:
                mapped_edge = (min(mapped_u, mapped_nb), max(mapped_u, mapped_nb))
                if mapped_edge in real_edge_set:
                    correct += 1

    # Each edge is counted once from both endpoints above, so divide by 2
    correct //= 2

    user_edge_count = user_graph.num_edges()
    real_edge_count = real_graph.num_edges()
    denominator = max(user_edge_count, real_edge_count)

    if denominator == 0:
        return 1.0
    return correct / denominator


###############################################################################
# Main Similarity Interface
###############################################################################

def compute_similarity_score(
    user_graph: ConstellationGraph,
    real_graph: ConstellationGraph,
    geo_weight: float = 0.5,
    struct_weight: float = 0.5
) -> float:
    """Return a combined similarity score in [0.0, 1.0] between the user's drawn
    constellation and the real constellation.

    The score is a weighted average of:
        - A geometric score: how well the star positions align after normalization
          (translation, scale, rotation).
        - A structural score: what fraction of constellation edges were drawn correctly.

    A score of 1.0 means a perfect match. A score of 0.0 means no similarity.

    Parameters:
        - user_graph: The graph built from the user's star placements and connections.
        - real_graph: The ground-truth constellation graph.
        - geo_weight: Weight given to the geometric position score (default 0.5).
        - struct_weight: Weight given to the structural edge score (default 0.5).

    Preconditions:
        - user_graph.num_stars() > 0
        - real_graph.num_stars() > 0
        - abs(geo_weight + struct_weight - 1.0) < 1e-9
        - 0.0 <= geo_weight <= 1.0 and 0.0 <= struct_weight <= 1.0

    >>> g_user = ConstellationGraph('UserOrion')
    >>> g_user.add_star('A', 0.0, 0.0)
    >>> g_user.add_star('B', 1.0, 0.0)
    >>> g_user.add_edge('A', 'B')
    >>> g_real = ConstellationGraph('Orion')
    >>> g_real.add_star('X', 0.0, 0.0)
    >>> g_real.add_star('Y', 1.0, 0.0)
    >>> g_real.add_edge('X', 'Y')
    >>> score = compute_similarity_score(g_user, g_real)
    >>> 0.9 < score <= 1.0
    True
    """
    matching = _match_user_to_real(user_graph.get_nodes(), real_graph.get_nodes())

    geo = _geometric_score(user_graph.get_nodes(), real_graph.get_nodes(), matching)
    struct = _structural_score(user_graph, real_graph, matching)

    return geo_weight * geo + struct_weight * struct


def score_to_percentage(score: float) -> int:
    """Return the similarity score as an integer percentage in [0, 100].

    Preconditions:
        - 0.0 <= score <= 1.0

    >>> score_to_percentage(1.0)
    100
    >>> score_to_percentage(0.0)
    0
    >>> score_to_percentage(0.756)
    76
    """
    return round(score * 100)


###############################################################################
# Main Block
###############################################################################

if __name__ == '__main__':
    doctest.testmod()
    python_ta.check_all(config={
        'extra-imports': ['math'],
        'allowed-io': [],
        'max-line-length': 120
    })
