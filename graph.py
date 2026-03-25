
import csv
import math
from __future__ import annotations
from typing import Any
from typing import Optional


class _Vertex:
    """
    A vertex in a constellation, representing a star.
    The connected vertexes make up a constellation.
    Note: Right Ascension and Declination are celestial coordinates, similar to latitude / longitude, specifically for
    stars and objects in the sky.

    Instance Attributes:
        - name: The name of the star.
        - ra: Right Ascension in decimal degrees [0, 360).
        - dec: Declination in decimal degrees [-90, 90].
        - magnitude: Apparent visual magnitude (lower = brighter).
        - neighbours: Mapping from adjacent _Vertex to edge weight (angular distance in degrees between the two stars

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)
        - 0.0 <= self.ra < 360.0
        - -90.0 <= self.dec <= 90.0

    """
    name: str
    ra: float
    dec: float
    magnitude: float
    neighbours: dict[_Vertex, float]

    def __init__(
            self,
            item: Any,
            ra: float,
            dec: float,
            magnitude: float,
            name: str = "",
    ) -> None:
        """
        Initialise a new star vertex with no neighbours.
        """
        self.item = item
        self.ra = ra
        self.dec = dec
        self.magnitude = magnitude
        self.name = name
        self.neighbours = {}

    def degree(self) -> int:
        """
        Return the number of edges connected to this vertex.
        """
        return len(self.neighbours)

    def angular_distance(self, other: _Vertex) -> float:
        """
        Return the angular separation (degrees) between this star and other
        Uses the spherical law of cosines (standard way to calculate the angular separation between two stars)
        >>> v1 = _Vertex('A', ra=0.0, dec=0.0, magnitude=1.0)
        >>> v2 = _Vertex('B', ra=90.0, dec=0.0, magnitude=2.0)
        >>> round(v1.angular_distance(v2), 4)
        90.0
        """
        ra1, dec1 = math.radians(self.ra), math.radians(self.dec)
        ra2, dec2 = math.radians(other.ra), math.radians(other.dec)

        cos_angle = (math.sin(dec1) * math.sin(dec2) + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
        # Clamp to [-1, 1] to guard against floating-point drift
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.degrees(math.acos(cos_angle))

    def sky_position(self) -> tuple[float, float]:
        """
        Return the (RA, Dec) position of this star as a tuple.
        """
        return self.ra, self.dec

    def similarity_score(self, other: _Vertex) -> float:
        """
        Return a similarity score ranging from 0 to 1.

        The score is based on the set of neighbour's items, similar to the book-review similarity score function so that
        similarity.py can call this on constellation subgraphs.

        A score of 1.0 means identical neighbour sets; 0.0 means disjoint.

        Two stars are "similar" if they are connected to the same neighbouring stars.
        """
        self_items = {v.item for v in self.neighbours}
        other_items = {v.item for v in other.neighbours}
        union = self_items | other_items
        if not union:
            return 0.0
        return len(self_items & other_items) / len(union)


# WeightedGraph
class WeightedGraph:
    """
    A weighted undirected graph representing a network of stars.

    Vertices are stars; edges carry a float weight equal to the angular
    distance (degrees) between the two connected stars.

    The "weight" of an edge that connects two stars represents the angular distance between them.

    Private Instance Attributes:
        _vertices:  Maps item → _Vertex for O(1) look-up.
    """

    _vertices: dict[Any, _Vertex]

    def __init__(self) -> None:
        """
        Initialize an empty graph (no vertices or edges).
        """
        self._vertices = {}

    # Mutation
    def add_vertex(
            self,
            item: Any,
            ra: float,
            dec: float,
            magnitude: float,
            name: str = "",
    ) -> None:
        """
        Add a star vertex to this graph.
        Do nothing if the vertex is already present.
        """
        if item not in self._vertices:
            self._vertices[item] = _Vertex(item, ra, dec, magnitude, name)

    def add_edge(
            self,
            item1: Any,
            item2: Any,
            weight: Optional[float] = None,
    ) -> None:
        """
        Add a weighted undirected edge between *item1* and *item2*.

        If the weight is not given, its set as the angular distance between the two stars.

        Raise ValueError if either item is not a vertex in this graph or if item1 == item2.
        """
        if item1 == item2:
            raise ValueError("You can't add an edge to itself.")
        if item1 not in self._vertices or item2 not in self._vertices:
            raise ValueError(f"One or both vertices not found: {item1!r}, {item2!r}")
        v1 = self._vertices[item1]
        v2 = self._vertices[item2]
        if weight is None:
            weight = v1.angular_distance(v2)
        v1.neighbours[v2] = weight
        v2.neighbours[v1] = weight

    def adjacent(self, item1: Any, item2: Any) -> bool:
        """Return whether the two stars share an edge.

        Return False if any one of the items are not in this graph.
        """
        if item1 not in self._vertices or item2 not in self._vertices:
            return False
        v1 = self._vertices[item1]
        v2 = self._vertices[item2]
        return v2 in v1.neighbours

    def get_edge_weight(self, item1: Any, item2: Any) -> float:
        """Return the weight of the edge between *item1* and *item2*.

        Raise ValueError if the edge does not exist.
        """
        if item1 not in self._vertices or item2 not in self._vertices:
            raise ValueError("One or both vertices not found.")
        v1 = self._vertices[item1]
        v2 = self._vertices[item2]
        if v2 not in v1.neighbours:
            raise ValueError(f"No edge between {item1!r} and {item2!r}.")
        return v1.neighbours[v2]

    def get_neighbours(self, item: Any) -> dict[Any, float]:
        """Return a dict mapping each neighbour's *item* to the edge weight.

        Raise ValueError if *item* is not in this graph.
        """
        if item not in self._vertices:
            raise ValueError(f"Vertex {item!r} not found.")
        v = self._vertices[item]
        return {neighbour.item: weight for neighbour, weight in v.neighbours.items()}

    def get_all_vertices(self) -> set[Any]:
        """Return the set of all vertex items in this graph."""
        return set(self._vertices.keys())

    def has_vertex(self, item: Any) -> bool:
        """Return True iff *item* is a vertex in this graph."""
        return item in self._vertices

    def get_vertex_data(self, item: Any) -> dict[str, Any]:
        """Return a dict of attributes for the star vertex *item*.

        Keys: 'item', 'name', 'ra', 'dec', 'magnitude', 'degree'.

        Raise ValueError if *item* is not in this graph.
        """
        if item not in self._vertices:
            raise ValueError(f"Vertex {item!r} not found.")
        v = self._vertices[item]
        return {
            "item": v.item,
            "name": v.name,
            "ra": v.ra,
            "dec": v.dec,
            "magnitude": v.magnitude,
            "degree": v.degree(),
        }

    # ------------------------------------------------------------------
    # Similarity (used by similarity.py)
    # ------------------------------------------------------------------

    def get_similarity_score(self, item1: Any, item2: Any) -> float:
        """Return the structural similarity score between two star vertices.

        This is the Jaccard coefficient over neighbour-item sets.
        Raise ValueError if either item is not in this graph.

        >>> g = WeightedGraph()
        >>> for label, ra, dec in [('A',0,0),('B',1,0),('C',2,0),('D',3,0),('E',4,0)]:
        ...     g.add_vertex(label, ra, dec, 1.0)
        >>> g.add_edge('A', 'B')
        >>> g.add_edge('A', 'C')
        >>> g.add_edge('D', 'B')
        >>> g.add_edge('D', 'C')
        >>> g.get_similarity_score('A', 'D')
        1.0
        """
        if item1 not in self._vertices or item2 not in self._vertices:
            raise ValueError("One or both vertices not found.")
        return self._vertices[item1].similarity_score(self._vertices[item2])

    # ------------------------------------------------------------------
    # Subgraph extraction (used by constellations.py / similarity.py)
    # ------------------------------------------------------------------

    def get_subgraph(self, items: set[Any]) -> WeightedGraph:
        """Return a new WeightedGraph containing only the vertices in *items*
        and the edges that connect them.

        Vertices not present in this graph are silently skipped.

        This is used by constellations.py to extract a single constellation's
        subgraph from the full star catalog graph.
        """
        sub = WeightedGraph()
        # Add vertices
        for item in items:
            if item in self._vertices:
                v = self._vertices[item]
                sub.add_vertex(v.item, v.ra, v.dec, v.magnitude, v.name)
        # Add edges (only those where both endpoints are in the subgraph)
        visited_pairs: set[frozenset] = set()
        for item in items:
            if item not in self._vertices:
                continue
            v = self._vertices[item]
            for neighbour, weight in v.neighbours.items():
                if neighbour.item in items:
                    pair = frozenset({item, neighbour.item})
                    if pair not in visited_pairs:
                        sub.add_edge(item, neighbour.item, weight)
                        visited_pairs.add(pair)
        return sub


# ---------------------------------------------------------------------------
# Module-level doctest runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import doctest

    doctest.testmod()

