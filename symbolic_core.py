"""Typed expression DAG, independent of Maxwell, solvers and data sources."""

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

import networkx as nx
import numpy as np
import sympy as sp

Shape = Literal["scalar", "vector"]
COORDINATES = ("x", "y", "z", "t")


@dataclass(frozen=True)
class Node:
    """Arguments are ordered node indices; repetitions are meaningful."""

    kind: str
    shape: Shape
    args: tuple[int, ...] = ()
    name: str = ""
    value: float | None = None
    axis: str = ""
    order: int = 1


class ExpressionDAG:
    """Append-only typed DAG with ordered roots and parameter coordinates.

    Scalar/vector compatibility is checked, but SI dimensional analysis is not
    implemented. Units belong to physical model metadata, not this small core.
    Vector field u reserves u_x, u_y, u_z as well as its base name u, so distinct
    DAG symbols cannot collapse into the same scalar function during export.
    """

    def __init__(self):
        self._nodes: list[Node] = []
        self.roots: dict[str, int] = {}
        self.parameter_order: list[str] = []

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    def _append(self, node: Node) -> int:
        self._nodes.append(node)
        return len(self._nodes) - 1

    def symbol(self, name: str, shape: Shape = "scalar", kind: str = "field") -> int:
        if shape not in ("scalar", "vector") or kind not in (
            "field",
            "parameter",
            "coordinate",
        ):
            raise ValueError("Invalid symbol shape or kind")
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Symbol name must be an identifier")
        if kind != "field" and shape != "scalar":
            raise TypeError("Parameters and coordinates must be scalar")
        if kind == "coordinate" and name not in COORDINATES:
            raise ValueError("Coordinate must be x, y, z, or t")
        if kind != "coordinate" and name in COORDINATES:
            raise ValueError("Coordinate names are reserved")
        requested_names = {name}
        if kind == "field" and shape == "vector":
            requested_names.update(f"{name}_{axis}" for axis in "xyz")
        reserved_names = set()
        for existing in self._nodes:
            if existing.name:
                reserved_names.add(existing.name)
            if existing.kind == "field" and existing.shape == "vector":
                reserved_names.update(f"{existing.name}_{axis}" for axis in "xyz")
        collision = requested_names & reserved_names
        if collision:
            raise ValueError(f"Symbol names conflict with reserved names: {sorted(collision)}")
        if kind == "parameter":
            self.parameter_order.append(name)
        return self._append(Node(kind, shape, name=name))

    def constant(self, value: float) -> int:
        if not np.isscalar(value) or not np.isfinite(value) or np.iscomplexobj(value):
            raise ValueError("Constant must be a finite real number")
        return self._append(Node("constant", "scalar", value=float(value)))

    def operation(self, kind: str, *args: int, axis: str = "", order: int = 1) -> int:
        arities = {
            "add": 2,
            "sub": 2,
            "mul": 2,
            "div": 2,
            "scale": 2,
            "dot": 2,
            "cross": 2,
            "neg": 1,
            "curl": 1,
            "divergence": 1,
            "gradient": 1,
            "diff": 1,
        }
        if kind not in arities or len(args) != arities[kind]:
            raise ValueError(f"Unknown operation or wrong arity: {kind}")
        if any(type(i) is not int or i < 0 or i >= len(self._nodes) for i in args):
            raise ValueError("Arguments must reference existing nodes")
        shapes = tuple(self._nodes[i].shape for i in args)
        if kind == "diff":
            if axis not in COORDINATES or type(order) is not int or order < 1:
                raise ValueError("Derivative needs a coordinate and positive integer order")
        elif axis or order != 1:
            raise ValueError("Derivative metadata is only valid on diff nodes")
        if kind in ("add", "sub"):
            valid, result = shapes[0] == shapes[1], shapes[0]
        elif kind in ("mul", "div"):
            valid, result = shapes == ("scalar", "scalar"), "scalar"
        elif kind == "scale":
            valid, result = shapes == ("scalar", "vector"), "vector"
        elif kind in ("cross", "dot"):
            valid, result = (
                shapes == ("vector", "vector"),
                "vector" if kind == "cross" else "scalar",
            )
        elif kind in ("curl", "divergence"):
            valid, result = (
                shapes == ("vector",),
                "vector" if kind == "curl" else "scalar",
            )
        elif kind == "gradient":
            valid, result = shapes == ("scalar",), "vector"
        else:
            valid, result = True, shapes[0]
        if not valid:
            raise TypeError(f"Incompatible arguments for {kind}: {shapes}")
        return self._append(Node(kind, result, tuple(args), axis=axis, order=order))

    def set_root(self, name: str, node: int) -> None:
        if (
            not isinstance(name, str)
            or not name
            or type(node) is not int
            or not 0 <= node < len(self._nodes)
        ):
            raise ValueError("Invalid root name or node")
        self.roots[name] = node

    def to_dict(self) -> dict:
        records = []
        for node in self._nodes:
            record = asdict(node)
            record["args"] = list(node.args)
            records.append(record)
        return {
            "schema": 1,
            "nodes": records,
            "roots": dict(self.roots),
            "parameter_order": list(self.parameter_order),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpressionDAG":
        if set(data) != {"schema", "nodes", "roots", "parameter_order"} or data["schema"] != 1:
            raise ValueError("Unsupported or incomplete DAG schema")
        graph = cls()
        for record in data["nodes"]:
            node = Node(**{**record, "args": tuple(record["args"])})
            if node.kind in ("field", "parameter", "coordinate"):
                idx = graph.symbol(node.name, node.shape, node.kind)
            elif node.kind == "constant":
                idx = graph.constant(node.value)
            else:
                idx = graph.operation(node.kind, *node.args, axis=node.axis, order=node.order)
            if graph.nodes[idx] != node:
                raise ValueError("Node metadata conflicts with its operation")
        for name, node in data["roots"].items():
            graph.set_root(name, node)
        if graph.parameter_order != data["parameter_order"]:
            raise ValueError("Parameter order is inconsistent with declared symbols")
        return graph

    def adjacency(self) -> np.ndarray:
        """Z[operation, argument] is edge multiplicity (including repeated inputs)."""
        matrix = np.zeros((len(self._nodes), len(self._nodes)), dtype=int)
        for i, node in enumerate(self._nodes):
            for j, count in Counter(node.args).items():
                matrix[i, j] = count
        return matrix

    @classmethod
    def from_adjacency(cls, matrix: np.ndarray, metadata: dict) -> "ExpressionDAG":
        graph = cls.from_dict(metadata)
        supplied = np.asarray(matrix)
        if supplied.shape != graph.adjacency().shape or not np.array_equal(
            supplied, graph.adjacency()
        ):
            raise ValueError("Adjacency and ordered argument metadata disagree")
        return graph

    def networkx(self) -> nx.MultiDiGraph:
        graph = nx.MultiDiGraph()
        for i, node in enumerate(self._nodes):
            graph.add_node(i, kind=node.kind, label=node.name or node.kind, shape=node.shape)
            for port, source in enumerate(node.args):
                graph.add_edge(source, i, port=port)
        return graph

    def to_sympy(self, root: int):
        if type(root) is not int or not 0 <= root < len(self._nodes):
            raise ValueError("Invalid root")
        coordinates = sp.symbols("x y z t", real=True)
        cache = []
        for node in self._nodes[: root + 1]:
            args = [cache[i] for i in node.args]
            kind = node.kind
            if kind in ("parameter", "coordinate"):
                result = sp.Symbol(node.name, real=True)
            elif kind == "constant":
                result = sp.Integer(node.value) if node.value.is_integer() else sp.Float(node.value)
            elif kind == "field":
                result = (
                    sp.Function(node.name)(*coordinates)
                    if node.shape == "scalar"
                    else sp.ImmutableMatrix(
                        [sp.Function(f"{node.name}_{j}")(*coordinates) for j in "xyz"]
                    )
                )
            elif kind == "add":
                result = args[0] + args[1]
            elif kind == "sub":
                result = args[0] - args[1]
            elif kind in ("mul", "scale"):
                result = args[0] * args[1]
            elif kind == "div":
                result = args[0] / args[1]
            elif kind == "neg":
                result = -args[0]
            elif kind == "dot":
                result = args[0].dot(args[1])
            elif kind == "cross":
                result = args[0].cross(args[1])
            elif kind == "diff":
                result = sp.diff(args[0], coordinates[COORDINATES.index(node.axis)], node.order)
            elif kind == "gradient":
                result = sp.ImmutableMatrix([sp.diff(args[0], c) for c in coordinates[:3]])
            elif kind == "divergence":
                result = sum(sp.diff(args[0][j], coordinates[j]) for j in range(3))
            elif kind == "curl":
                result = symbolic_curl(args[0], coordinates[:3])
            cache.append(result)
        return cache[root]


def symbolic_curl(vector, coordinates):
    x, y, z = coordinates
    a, b, c = vector
    return sp.ImmutableMatrix(
        [
            sp.diff(c, y) - sp.diff(b, z),
            sp.diff(a, z) - sp.diff(c, x),
            sp.diff(b, x) - sp.diff(a, y),
        ]
    )
