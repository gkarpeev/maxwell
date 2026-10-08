"""Maxwell-specific equations built by the generic typed DAG."""

from symbolic_core import ExpressionDAG


def maxwell_model(lossy: bool = False) -> ExpressionDAG:
    """Normalized fields: e_t=(c curl h - sigma/epsilon0 e)/epsilon_r.

    gamma=sigma/epsilon0 [s^-1]; for the lossless model gamma is absent.
    Both relative material constants and c are scalar parameters.
    In variable media the same curl equations apply with spatial material
    fields; interfaces additionally obey the appropriate Maxwell jump laws.
    """
    graph = ExpressionDAG()
    electric = graph.symbol("e", "vector")
    magnetic = graph.symbol("h", "vector")
    eps = graph.symbol("epsilon_r", kind="parameter")
    mu = graph.symbol("mu_r", kind="parameter")
    c = graph.symbol("c", kind="parameter")
    e_rhs = graph.operation(
        "scale", graph.operation("div", c, eps), graph.operation("curl", magnetic)
    )
    if lossy:
        gamma = graph.symbol("gamma", kind="parameter")
        loss = graph.operation("scale", graph.operation("div", gamma, eps), electric)
        e_rhs = graph.operation("sub", e_rhs, loss)
    h_rhs = graph.operation(
        "neg",
        graph.operation("scale", graph.operation("div", c, mu), graph.operation("curl", electric)),
    )
    graph.set_root("e_rhs", e_rhs)
    graph.set_root("h_rhs", h_rhs)
    graph.set_root(
        "e_residual",
        graph.operation("sub", graph.operation("diff", electric, axis="t"), e_rhs),
    )
    graph.set_root(
        "h_residual",
        graph.operation("sub", graph.operation("diff", magnetic, axis="t"), h_rhs),
    )
    return graph
