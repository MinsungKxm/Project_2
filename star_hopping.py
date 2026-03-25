"""CSC111 Winter 2026 Project: Constellation Explorer

star_hopping.py - Star Hopping via Shortest Path

This module builds a "hoppable" graph over all visible stars and finds the
easiest sequence of stars to hop between a start star and a target star.
Star hopping is a real technique used by amateur astronomers to navigate the
night sky by using bright, recognizable stars as stepping stones.

A star hop edge exists between two stars if they are within a maximum angular
separation threshold. The edge weight encodes the visual difficulty of the hop:
a brighter, closer pair is easier (lower weight) than a dim, distant pair.

The easiest path is found using Dijkstra's shortest path algorithm, implemented
from scratch using a min-heap (priority queue).


"""

from __future__ import annotations
import math
import heapq
import doctest
import python_ta


###############################################################################
# Constants
###############################################################################

# Maximum angular separation (in degrees) for two stars to be considered hoppable.
# Typical star hopping uses jumps of under ~15 degrees.
MAX_HOP_DEGREES = 15.0

# A very large number used to represent "infinity" in Dijkstra's algorithm.
_INFINITY = float('inf')


###############################################################################
# Data Types
###############################################################################

class Star:
    """A star in the night sky, used as a node in the hoppable star graph.

    Instance Attributes:
        - star_id: A unique identifier for this star (e.g. HIP number or common name).
        - ra: Right Ascension in decimal degrees [0, 360). Analogous to longitude.
        - dec: Declination in decimal degrees [-90, 90]. Analogous to latitude.
        - magnitude: Apparent visual magnitude. Lower values mean brighter stars.
                     Typical naked-eye stars have magnitude < 6.5.
        - name: A human-readable name for this star (e.g. 'Polaris'). May be empty.
        - constellation: The constellation this star belongs to, or '' if none.

    Representation Invariants:
        - 0.0 <= self.ra < 360.0
        - -90.0 <= self.dec <= 90.0
    """
    star_id: str
    ra: float
    dec: float
    magnitude: float
    name: str
    constellation: str

    def __init__(
        self,
        star_id: str,
        ra: float,
        dec: float,
        magnitude: float,
        name: str = '',
        constellation: str = ''
    ) -> None:
        """Initialize a Star with the given attributes."""
        self.star_id = star_id
        self.ra = ra
        self.dec = dec
        self.magnitude = magnitude
        self.name = name
        self.constellation = constellation

    def display_name(self) -> str:
        """Return a display-friendly name for this star.

        Uses the common name if available, otherwise the star_id.

        >>> s = Star('HIP1234', 100.0, 45.0, 2.5, 'Betelgeuse', 'Orion')
        >>> s.display_name()
        'Betelgeuse'
        >>> s2 = Star('HIP9999', 200.0, -10.0, 4.1)
        >>> s2.display_name()
        'HIP9999'
        """
        return self.name if self.name else self.star_id


class HoppableGraph:
    """A weighted undirected graph of stars, where edges represent possible star hops.

    An edge between two stars exists if their angular separation is at most
    MAX_HOP_DEGREES. The edge weight represents the visual difficulty of the hop.

    Instance Attributes:
        - _stars: Maps star_id to the Star object.
        - _adj: Adjacency list. Maps star_id to a dict of {neighbour_id: weight}.

    Representation Invariants:
        - All star_ids in _adj are keys in _stars.
        - The graph is undirected: if b in _adj[a] then a in _adj[b] with same weight.
        - All edge weights are positive.
    """
    _stars: dict[str, Star]
    _adj: dict[str, dict[str, float]]

    def __init__(self) -> None:
        """Initialize an empty HoppableGraph."""
        self._stars = {}
        self._adj = {}

    def add_star(self, star: Star) -> None:
        """Add the given Star to this graph with no edges.

        Preconditions:
            - star.star_id not in self._stars
        """
        self._stars[star.star_id] = star
        self._adj[star.star_id] = {}

    def add_hop_edge(self, id1: str, id2: str, weight: float) -> None:
        """Add an undirected weighted edge between the two stars.

        Preconditions:
            - id1 in self._stars and id2 in self._stars
            - id1 != id2
            - weight > 0
            - id2 not in self._adj[id1]
        """
        self._adj[id1][id2] = weight
        self._adj[id2][id1] = weight

    def get_star(self, star_id: str) -> Star:
        """Return the Star object for the given star_id.

        Preconditions:
            - star_id in self._stars
        """
        return self._stars[star_id]

    def get_neighbours(self, star_id: str) -> dict[str, float]:
        """Return the adjacency dict {neighbour_id: weight} for the given star.

        Preconditions:
            - star_id in self._stars
        """
        return self._adj[star_id]

    def all_star_ids(self) -> list[str]:
        """Return a list of all star_ids in this graph."""
        return list(self._stars.keys())

    def num_stars(self) -> int:
        """Return the number of stars in this graph.

        >>> g = HoppableGraph()
        >>> g.add_star(Star('S1', 0.0, 0.0, 1.0))
        >>> g.add_star(Star('S2', 1.0, 1.0, 2.0))
        >>> g.num_stars()
        2
        """
        return len(self._stars)

    def num_edges(self) -> int:
        """Return the number of hop edges in this graph.

        >>> g = HoppableGraph()
        >>> g.add_star(Star('S1', 0.0, 0.0, 1.0))
        >>> g.add_star(Star('S2', 1.0, 0.0, 2.0))
        >>> g.add_hop_edge('S1', 'S2', 1.0)
        >>> g.num_edges()
        1
        """
        total = sum(len(neighbours) for neighbours in self._adj.values())
        return total // 2


###############################################################################
# Angular Separation
###############################################################################

def angular_separation(star1: Star, star2: Star) -> float:
    """Return the angular separation in degrees between two stars using the
    Haversine formula applied to spherical (RA, Dec) coordinates.

    The Haversine formula correctly handles the spherical geometry of the
    celestial sphere, unlike a simple Euclidean distance on RA/Dec values.

    Preconditions:
        - 0.0 <= star1.ra < 360.0 and -90.0 <= star1.dec <= 90.0
        - 0.0 <= star2.ra < 360.0 and -90.0 <= star2.dec <= 90.0

    >>> s1 = Star('A', 0.0, 0.0, 1.0)
    >>> s2 = Star('B', 0.0, 0.0, 2.0)
    >>> angular_separation(s1, s2)
    0.0
    >>> s3 = Star('C', 90.0, 0.0, 1.0)
    >>> abs(angular_separation(s1, s3) - 90.0) < 1e-6
    True
    """
    ra1 = math.radians(star1.ra)
    dec1 = math.radians(star1.dec)
    ra2 = math.radians(star2.ra)
    dec2 = math.radians(star2.dec)

    delta_ra = ra2 - ra1
    delta_dec = dec2 - dec1

    a = (math.sin(delta_dec / 2) ** 2
         + math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return math.degrees(c)


###############################################################################
# Edge Weight
###############################################################################

def hop_difficulty(star1: Star, star2: Star, separation: float) -> float:
    """Return the visual difficulty of hopping from star1 to star2.

    A lower value means an easier hop (preferred by Dijkstra's algorithm).
    Difficulty increases with:
        - Greater angular separation (harder to trace a long jump)
        - Fainter stars (higher magnitude values are harder to see)

    Formula:
        difficulty = separation_weight * separation + magnitude_weight * mean_magnitude

    Where mean_magnitude is the average apparent magnitude of the two stars,
    and the weights are chosen so that separation and brightness contribute
    roughly equally over their typical ranges.

    Preconditions:
        - 0.0 < separation <= MAX_HOP_DEGREES
        - Both stars have valid magnitude values

    >>> s1 = Star('A', 0.0, 0.0, 1.0)
    >>> s2 = Star('B', 1.0, 0.0, 1.0)
    >>> d = hop_difficulty(s1, s2, 1.0)
    >>> d > 0
    True
    """
    separation_weight = 2.0
    magnitude_weight = 1.0
    mean_magnitude = (star1.magnitude + star2.magnitude) / 2.0
    return separation_weight * separation + magnitude_weight * mean_magnitude


###############################################################################
# Graph Construction
###############################################################################

def build_hoppable_graph(stars: list[Star], max_degrees: float = MAX_HOP_DEGREES) -> HoppableGraph:
    """Build and return a HoppableGraph from the given list of stars.

    For every pair of stars within max_degrees of angular separation, an edge
    is added with weight equal to the hop difficulty between them.

    This runs in O(n^2) time where n is the number of stars. For large catalogs
    (~9000 stars), this is acceptable for a one-time build at program startup.

    Preconditions:
        - len(stars) > 0
        - max_degrees > 0.0
        - All stars have unique star_ids.

    >>> stars = [Star('S1', 0.0, 0.0, 1.0), Star('S2', 1.0, 0.0, 2.0), Star('S3', 50.0, 0.0, 3.0)]
    >>> g = build_hoppable_graph(stars, max_degrees=5.0)
    >>> g.num_stars()
    3
    >>> g.num_edges()
    1
    """
    graph = HoppableGraph()
    for star in stars:
        graph.add_star(star)

    ids = [star.star_id for star in stars]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s1 = graph.get_star(ids[i])
            s2 = graph.get_star(ids[j])
            sep = angular_separation(s1, s2)
            if sep <= max_degrees:
                weight = hop_difficulty(s1, s2, sep)
                graph.add_hop_edge(ids[i], ids[j], weight)

    return graph


###############################################################################
# Dijkstra's Algorithm
###############################################################################

def find_easiest_path(
    graph: HoppableGraph,
    start_id: str,
    target_id: str
) -> tuple[list[str], float] | tuple[None, None]:
    """Return the easiest star-hopping path from start_id to target_id in graph,
    along with its total difficulty score.

    Uses Dijkstra's shortest path algorithm with a min-heap (priority queue).
    The path is a list of star_ids from start_id to target_id (inclusive).
    The total difficulty is the sum of hop_difficulty values along the path.

    Returns (None, None) if no path exists between the two stars (i.e. they
    are not connected in the graph).

    Preconditions:
        - start_id in graph.all_star_ids()
        - target_id in graph.all_star_ids()
        - start_id != target_id

    >>> stars = [Star('A', 0.0, 0.0, 1.0), Star('B', 1.0, 0.0, 1.0), Star('C', 2.0, 0.0, 1.0)]
    >>> g = build_hoppable_graph(stars, max_degrees=5.0)
    >>> path, cost = find_easiest_path(g, 'A', 'C')
    >>> path == ['A', 'B', 'C'] or path == ['A', 'C']
    True
    >>> cost is not None and cost > 0
    True
    """
    # dist[node] = best known total difficulty from start to node
    dist: dict[str, float] = {sid: _INFINITY for sid in graph.all_star_ids()}
    dist[start_id] = 0.0

    # prev[node] = the node preceding it on the best known path
    prev: dict[str, str | None] = {sid: None for sid in graph.all_star_ids()}

    # Min-heap entries: (cumulative_difficulty, star_id)
    heap: list[tuple[float, str]] = [(0.0, start_id)]

    visited: set[str] = set()

    while heap:
        current_dist, current_id = heapq.heappop(heap)

        if current_id in visited:
            continue
        visited.add(current_id)

        if current_id == target_id:
            break

        for neighbour_id, weight in graph.get_neighbours(current_id).items():
            if neighbour_id in visited:
                continue
            new_dist = current_dist + weight
            if new_dist < dist[neighbour_id]:
                dist[neighbour_id] = new_dist
                prev[neighbour_id] = current_id
                heapq.heappush(heap, (new_dist, neighbour_id))

    # Reconstruct the path by backtracking through prev
    if dist[target_id] == _INFINITY:
        return (None, None)

    path = _reconstruct_path(prev, start_id, target_id)
    return (path, dist[target_id])


def _reconstruct_path(
    prev: dict[str, str | None],
    start_id: str,
    target_id: str
) -> list[str]:
    """Return the path from start_id to target_id by backtracking through prev.

    prev maps each node to the node that preceded it on the best path.

    Preconditions:
        - target_id is reachable from start_id (prev[target_id] is not None
          or target_id == start_id).
        - start_id is a key in prev.

    >>> prev = {'A': None, 'B': 'A', 'C': 'B'}
    >>> _reconstruct_path(prev, 'A', 'C')
    ['A', 'B', 'C']
    """
    path = []
    current: str | None = target_id
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path


###############################################################################
# Nearest Constellation Star
###############################################################################

def nearest_constellation_star(
    graph: HoppableGraph,
    from_id: str,
    constellation_name: str
) -> str | None:
    """Return the star_id of the nearest reachable star (by hop path cost) that
    belongs to the given constellation, starting from from_id.

    Uses a modified Dijkstra's search that stops as soon as it reaches any star
    in the target constellation.

    Returns None if no star in the constellation is reachable from from_id.

    Preconditions:
        - from_id in graph.all_star_ids()
        - constellation_name is a non-empty string

    >>> stars = [
    ...     Star('A', 0.0, 0.0, 1.0, constellation='Orion'),
    ...     Star('B', 1.0, 0.0, 1.0, constellation=''),
    ...     Star('C', 2.0, 0.0, 1.0, constellation='Orion'),
    ... ]
    >>> g = build_hoppable_graph(stars, max_degrees=5.0)
    >>> result = nearest_constellation_star(g, 'B', 'Orion')
    >>> result in ('A', 'C')
    True
    """
    dist: dict[str, float] = {sid: _INFINITY for sid in graph.all_star_ids()}
    dist[from_id] = 0.0

    heap: list[tuple[float, str]] = [(0.0, from_id)]
    visited: set[str] = set()

    while heap:
        current_dist, current_id = heapq.heappop(heap)

        if current_id in visited:
            continue
        visited.add(current_id)

        star = graph.get_star(current_id)
        if star.constellation == constellation_name and current_id != from_id:
            return current_id

        for neighbour_id, weight in graph.get_neighbours(current_id).items():
            if neighbour_id in visited:
                continue
            new_dist = current_dist + weight
            if new_dist < dist[neighbour_id]:
                dist[neighbour_id] = new_dist
                heapq.heappush(heap, (new_dist, neighbour_id))

    return None


###############################################################################
# Path Formatting
###############################################################################

def format_hop_path(graph: HoppableGraph, path: list[str]) -> list[str]:
    """Return a human-readable list of strings describing each hop in the path.

    Each string describes one hop: the star name, its constellation, and the
    hop difficulty to the next star.

    If path has fewer than 2 elements, returns an empty list.

    Preconditions:
        - All star_ids in path are in graph.
        - len(path) >= 1

    >>> stars = [Star('A', 0.0, 0.0, 1.0, 'Sirius', 'CMa'), Star('B', 1.0, 0.0, 2.0, 'Adhara', 'CMa')]
    >>> g = build_hoppable_graph(stars, max_degrees=5.0)
    >>> result = format_hop_path(g, ['A', 'B'])
    >>> len(result) == 1
    True
    """
    if len(path) < 2:
        return []

    steps = []
    for i in range(len(path) - 1):
        current = graph.get_star(path[i])
        nxt = graph.get_star(path[i + 1])
        sep = angular_separation(current, nxt)
        difficulty = graph.get_neighbours(path[i])[path[i + 1]]
        step = (
            f"Hop {i + 1}: {current.display_name()} ({current.constellation}) "
            f"→ {nxt.display_name()} ({nxt.constellation}) "
            f"[sep: {sep:.1f}°, difficulty: {difficulty:.2f}]"
        )
        steps.append(step)
    return steps


###############################################################################
# Main Block
###############################################################################

if __name__ == '__main__':
    doctest.testmod()
    python_ta.check_all(config={
        'extra-imports': ['math', 'heapq'],
        'allowed-io': [],
        'max-line-length': 120
    })
