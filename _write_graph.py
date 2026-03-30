write_path = '/Users/jennylin/Documents/GitHub/Project_2/graph.py'

code = '''\
"""CSC111 Winter 2026 Project: Constellation Explorer

graph.py - WeightedGraph Implementation

This module defines the WeightedGraph used to store all stars and their
constellation edges. Each vertex is a star (identified by HIP number),
and each edge represents a constellation line between two stars.
"""

from __future__ import annotations


class _Vertex:
    """A single star vertex in the WeightedGraph.

    Instance Attributes:
        - item: The HIP identifier for this star (int).
        - ra: Right Ascension in decimal degrees.
        - dec: Declination in decimal degrees.
        - magnitude: Apparent visual magnitude.
        - name: Common name of the star (may be empty string).
        - neighbours: Set of neighbouring star HIP ids.
    """
    item: int
    ra: float
    dec: float
    magnitude: float
    name: str
    neighbours: set

    def __init__(self, item: int, ra: float, dec: float,
                 magnitude: float, name: str) -> None:
        self.item = item
        self.ra = ra
        self.dec = dec
        self.magnitude = magnitude
        self.name = name
        self.neighbours = set()


class WeightedGraph:
    """A weighted undirected graph where each vertex is a star.

    Instance Attributes:
        - _vertices: Maps a HIP star id (int) to its _Vertex object.
    """
    _vertices: dict

    def __init__(self) -> None:
        """Initialize an empty WeightedGraph."""
        self._vertices = {}

    def add_vertex(self, item: int, ra: float, dec: float,
                   magnitude: float, name: str) -> None:
        """Add a star vertex to this graph.

        Preconditions:
            - item not in self._vertices
        """
        self._vertices[item] = _Vertex(item, ra, dec, magnitude, name)

    def has_vertex(self, item: int) -> bool:
        """Return whether a vertex with the given item exists."""
        return item in self._vertices

    def add_edge(self, item1: int, item2: int) -> None:
        """Add an undirected edge between two vertices.

        Preconditions:
            - item1 in self._vertices and item2 in self._vertices
            - item1 != item2
        """
        self._vertices[item1].neighbours.add(item2)
        self._vertices[item2].neighbours.add(item1)

    def get_vertex_data(self, item: int) -> dict:
        """Return a dict with data about the given star.

        Preconditions:
            - item in self._vertices
        """
        v = self._vertices[item]
        return {
            \'item\': v.item,
            \'ra\': v.ra,
            \'dec\': v.dec,
            \'magnitude\': v.magnitude,
            \'name\': v.name,
        }

    def get_neighbours(self, item: int) -> set:
        """Return the set of neighbour HIP ids for the given star.

        Preconditions:
            - item in self._vertices
        """
        return self._vertices[item].neighbours

    def all_vertices(self) -> dict:
        """Return the full vertex dictionary mapping HIP id to _Vertex."""
        return self._vertices

    def num_vertices(self) -> int:
        """Return the number of vertices in this graph."""
        return len(self._vertices)

    def num_edges(self) -> int:
        """Return the number of edges in this graph."""
        return sum(len(v.neighbours) for v in self._vertices.values()) // 2
'''

with open(write_path, 'w') as f:
    f.write(code)

print('Done -', len(code.splitlines()), 'lines written')
