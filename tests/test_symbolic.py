import copy
import json

import networkx as nx
import pytest
import sympy as sp

from maxwell_symbolic import maxwell_model
from symbolic_comparison import compare_libraries
from symbolic_core import ExpressionDAG, symbolic_curl


def test_maxwell_roundtrip_and_derivatives():
    graph = maxwell_model(lossy=True)
    restored = ExpressionDAG.from_adjacency(
        graph.adjacency(), json.loads(json.dumps(graph.to_dict()))
    )
    assert restored.to_dict() == graph.to_dict()
    assert nx.is_directed_acyclic_graph(restored.networkx())
    for name, root in graph.roots.items():
        assert restored.to_sympy(restored.roots[name]) == graph.to_sympy(root)
    x, y, z, t = sp.symbols("x y z t", real=True)
    residual = graph.to_sympy(graph.roots["e_residual"])
    assert residual[0].has(sp.Derivative(sp.Function("e_x")(x, y, z, t), t))
    assert graph.parameter_order == ["epsilon_r", "mu_r", "c", "gamma"]


def test_repeated_operands_order_and_second_derivative():
    g = ExpressionDAG()
    a = g.symbol("a", kind="parameter")
    b = g.symbol("b", kind="parameter")
    field = g.symbol("u")
    repeated = g.operation("mul", a, a)
    division = g.operation("div", a, b)
    derivative = g.operation("diff", field, axis="z", order=2)
    g.set_root("division", division)
    g.set_root("derivative", derivative)
    r = ExpressionDAG.from_adjacency(g.adjacency(), g.to_dict())
    assert r.nodes[repeated].args == (a, a)
    assert r.adjacency()[repeated, a] == 2
    assert r.nodes[division].args == (a, b)
    assert r.nodes[derivative].order == 2
    assert r.networkx().number_of_edges(a, repeated) == 2
    changed = g.adjacency()
    changed[repeated, a] = 1
    with pytest.raises(ValueError):
        ExpressionDAG.from_adjacency(changed, g.to_dict())


@pytest.mark.parametrize("kind,count", [("add", 1), ("curl", 2), ("unknown", 1)])
def test_bad_arity(kind, count):
    g = ExpressionDAG()
    a = g.symbol("u")
    with pytest.raises(ValueError):
        g.operation(kind, *([a] * count))


def test_type_checks_and_corruption():
    g = ExpressionDAG()
    u = g.symbol("u")
    v = g.symbol("v", "vector")
    for op, args in [
        ("curl", (u,)),
        ("dot", (v, u)),
        ("scale", (v, u)),
        ("add", (u, v)),
        ("div", (v, v)),
    ]:
        with pytest.raises(TypeError):
            g.operation(op, *args)
    for kwargs in [dict(axis="q"), dict(axis="x", order=0), dict(axis="x", order=1.5)]:
        with pytest.raises(ValueError):
            g.operation("diff", u, **kwargs)
    with pytest.raises(ValueError):
        g.operation("neg", -1)
    with pytest.raises(ValueError):
        g.operation("add", u, 999)
    bad = copy.deepcopy(maxwell_model().to_dict())
    bad["nodes"][-1]["args"][0] = len(bad["nodes"]) - 1
    with pytest.raises(ValueError):
        ExpressionDAG.from_dict(bad)
    bad = maxwell_model().to_dict()
    bad["parameter_order"].reverse()
    with pytest.raises(ValueError):
        ExpressionDAG.from_dict(bad)


def test_vector_operations_known_answers():
    x, y, z = sp.symbols("x y z")
    f = sp.Matrix([y * z, x * z, x * y])  # gradient of xyz
    assert symbolic_curl(f, (x, y, z)) == sp.zeros(3, 1)
    g = sp.Matrix([-y, x, z**2])
    assert symbolic_curl(g, (x, y, z)) == sp.Matrix([0, 0, 2])
    assert sum(sp.diff(g[i], a) for i, a in enumerate((x, y, z))) == 2 * z
    a, b = sp.Matrix([1, 2, 3]), sp.Matrix([4, 5, 6])
    assert a.dot(b) == 32
    assert a.cross(b) == sp.Matrix([-3, 6, -3])


def test_dag_vector_compilation_matches_explicit_operations():
    g = ExpressionDAG()
    f = g.symbol("u", "vector")
    h = g.symbol("v", "vector")
    roots = {
        op: g.operation(op, *(args))
        for op, args in [
            ("dot", (f, h)),
            ("cross", (f, h)),
            ("curl", (f,)),
            ("divergence", (f,)),
        ]
    }
    x, y, z, t = sp.symbols("x y z t", real=True)
    subs = dict(zip(g.to_sympy(f), (-y, x, z * z))) | dict(zip(g.to_sympy(h), (1, 2, 3)))
    assert g.to_sympy(roots["curl"]).subs(subs).doit() == sp.Matrix([0, 0, 2])
    assert g.to_sympy(roots["divergence"]).subs(subs).doit() == 2 * z
    assert g.to_sympy(roots["dot"]).subs(subs) == -y + 2 * x + 3 * z * z
    assert g.to_sympy(roots["cross"]).subs(subs) == sp.Matrix([-y, x, z * z]).cross(
        sp.Matrix([1, 2, 3])
    )


def test_two_libraries_agree():
    report = compare_libraries(3)
    assert report["equivalent"] and report["finite"]
    assert report["numeric_difference"] < 1e-14


@pytest.mark.parametrize("vector_first", [True, False])
def test_vector_component_names_cannot_alias_scalar_fields(vector_first):
    graph = ExpressionDAG()
    if vector_first:
        graph.symbol("u", "vector")
        with pytest.raises(ValueError, match="reserved names"):
            graph.symbol("u_x")
    else:
        graph.symbol("u_x")
        with pytest.raises(ValueError, match="reserved names"):
            graph.symbol("u", "vector")


@pytest.mark.parametrize("vector_first", [True, False])
def test_deserialization_rejects_component_name_collisions(vector_first):
    graph = ExpressionDAG()
    if vector_first:
        graph.symbol("u", "vector")
        scalar = graph.symbol("v")
    else:
        scalar = graph.symbol("v")
        graph.symbol("u", "vector")
    metadata = graph.to_dict()
    metadata["nodes"][scalar]["name"] = "u_x"
    with pytest.raises(ValueError, match="reserved names"):
        ExpressionDAG.from_dict(metadata)
