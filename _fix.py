code = '''\
"""CSC111 Winter 2026 Project: Constellation Explorer

graph.py - WeightedGraph Implementation
"""

from __future__ import annotations


class _Vertex:
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
    _vertices: dict

    def __init__(self) -> None:
        self._vertices = {}

    def add_vertex(self, item: int, ra: float, dec: float,
                   magnitude: float, name: str) -> None:
        self._vertices[item] = _Vertex(item, ra, dec, magnitude, name)

    def has_vertex(self, item: int) -> bool:
        return item in self._vertices

    def add_edge(self, item1: int, item2: int) -> None:
        self._vertices[item1].neighbours.add(item2)
        self._vertices[item2].neighbours.add(item1)

    def get_vertex_data(self, item: int) -> dict:
        v = self._vertices[item]
        return {
            \'item\': v.item,
            \'ra\': v.ra,
            \'dec\': v.dec,
            \'magnitude\': v.magnitude,
            \'name\': v.name,
        }

    def get_neighbours(self, item: int) -> set:
        return self._vertices[item].neighbours

    def all_vertices(self) -> dict:
        return self._vertices

    def num_vertices(self) -> int:
        return len(self._vertices)

    def num_edges(self) -> int:
        return sum(len(v.neighbours) for v in self._vertices.values()) // 2
'''

with open('/Users/jennylin/Documents/GitHub/Project_2/graph.py', 'w') as f:
    f.write(code)
print('graph.py fixed')
