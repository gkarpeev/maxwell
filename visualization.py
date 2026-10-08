"""Plots used by the Russian notebook; computation stays in separate modules."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

from maxwell_fdtd import ArenaConfig

NODE_COLORS = {"field": "#b9deeb", "parameter": "#f6d299"}
OPERATION_COLOR = "#d2e5ce"
SHARED_COLOR = "#8556a6"


def _node_label(index, node):
    """Use the same node identifiers and operation names in both figures."""
    name = {"epsilon_r": "εᵣ", "mu_r": "μᵣ", "gamma": "γ"}.get(node.name, node.name)
    if node.kind == "diff":
        name = f"∂{node.axis}" if node.order == 1 else f"∂{node.axis}^{node.order}"
    elif node.kind == "constant":
        name = str(node.value)
    return f"{index}: {name or node.kind}"


def plot_graph(graph):
    """Draw circular DAG nodes, highlighting shared inputs and named roots."""
    network = graph.networkx()
    pos = {}
    for level, generation in enumerate(nx.topological_generations(network)):
        nodes = sorted(generation)
        heights = np.linspace(-1, 1, len(nodes)) if len(nodes) > 1 else [0.0]
        pos.update({node: (level, height) for node, height in zip(nodes, heights)})
    shared = [network.out_degree(i) > 1 for i in range(len(graph.nodes))]
    adjacent_edges = [(s, t) for s, t in network.edges() if pos[t][0] - pos[s][0] <= 2]
    long_edges = [(s, t) for s, t in network.edges() if pos[t][0] - pos[s][0] > 2]
    fig, ax = plt.subplots(figsize=(13, 8.5))
    nx.draw_networkx(
        network,
        pos,
        ax=ax,
        edgelist=adjacent_edges,
        labels={i: _node_label(i, node) for i, node in enumerate(graph.nodes)},
        node_shape="o",
        node_size=1800,
        node_color=[NODE_COLORS.get(node.kind, OPERATION_COLOR) for node in graph.nodes],
        edgecolors=[SHARED_COLOR if value else "#65717c" for value in shared],
        linewidths=[2.4 if value else 1 for value in shared],
        font_size=9,
        edge_color="#7b8997",
        width=1.1,
        arrowsize=16,
    )
    nx.draw_networkx_edges(
        network,
        pos,
        ax=ax,
        edgelist=long_edges,
        node_size=1800,
        edge_color="#7b8997",
        width=1.1,
        arrowsize=16,
        connectionstyle="arc3,rad=-0.16",
    )
    for name, index in graph.roots.items():
        ax.annotate(
            name,
            pos[index],
            xytext=(0, -32),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=9,
            color="#46515c",
        )
    legend = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=11,
            markerfacecolor=color,
            markeredgecolor=border,
            markeredgewidth=width,
            label=label,
        )
        for color, border, width, label in [
            (NODE_COLORS["field"], "#65717c", 1, "Поля e, h"),
            (NODE_COLORS["parameter"], "#65717c", 1, "Параметры"),
            (OPERATION_COLOR, "#65717c", 1, "Операции"),
            ("white", SHARED_COLOR, 2.4, "Узел используется несколько раз"),
        ]
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    ax.set_title("Граф модели с потерями\nСтрелка: аргумент → операция", pad=18)
    ax.margins(0.12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_adjacency(graph):
    """Draw integer edge multiplicities with the full node labels on both axes."""
    matrix = graph.adjacency()
    labels = [_node_label(i, node) for i, node in enumerate(graph.nodes)]
    maximum = max(1, int(matrix.max()))
    cmap = plt.get_cmap("Blues", maximum + 1)
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
    for row, column in zip(*np.nonzero(matrix)):
        ax.text(
            column,
            row,
            str(matrix[row, column]),
            ha="center",
            va="center",
            color="white" if matrix[row, column] > maximum / 2 else "#17324d",
            fontsize=9,
        )
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#dbe3e9", linewidth=0.5)
    ax.tick_params(which="both", length=0)
    ax.set(
        xlabel="Аргумент j",
        ylabel="Узел i, использующий аргумент",
        title="Матрица смежности Zᵢⱼ\nЗначение — число использований j в узле i",
    )
    fig.colorbar(
        im,
        ax=ax,
        ticks=range(maximum + 1),
        shrink=0.65,
        label="Кратность ребра (светлая ячейка = 0)",
    )
    fig.tight_layout()
    return fig


def plot_arena_slices(electric, predicted, frame=7, component=0):
    config = ArenaConfig()
    reference = electric[0, frame, :, :, 16, component]
    approximation = predicted[frame, :, :, 16, component]
    error = approximation - reference
    coordinate = config.coordinates()[..., component, :] * 1e6
    half_cell = config.dx * 1e6 / 2
    extent = (
        coordinate[0, 0, 16, 0] - half_cell,
        coordinate[-1, 0, 16, 0] + half_cell,
        coordinate[0, 0, 16, 1] - half_cell,
        coordinate[0, -1, 16, 1] + half_cell,
    )
    maximum = np.max(np.abs(reference))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, field, title in zip(
        axes,
        [reference, approximation, error],
        ["Внешние данные", "Численное решение", "Модель − данные"],
    ):
        limit = np.max(np.abs(error)) if ax is axes[2] else maximum
        im = ax.imshow(
            field.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="equal",
        )
        ax.set(xlabel="x, мкм", ylabel="y, мкм", title=title)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Нормированное eₓ")
    fig.suptitle(
        f"Отложенный кадр {frame}; t_E={config.times[frame] * 1e12:.4f} пс; z={coordinate[0, 0, 16, 2]:.2f} мкм"
    )
    fig.tight_layout()
    return fig


def plot_arena_time(electric, magnetic, predictions):
    config = ArenaConfig()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    point = (16, 16, 16, 0)
    for index, (observed, times, label) in enumerate(
        [(electric, config.times, "eₓ"), (magnetic, config.magnetic_times, "hₓ")]
    ):
        ax = axes[index]
        ax.plot(
            times * 1e12,
            observed[(0, slice(None)) + point],
            "o",
            label="Внешние данные",
        )
        ax.plot(
            times * 1e12,
            predictions["sample0"][index][(slice(None),) + point],
            label="После подбора",
        )
        ax.plot(
            times * 1e12,
            predictions["baseline"][index][(slice(None),) + point],
            "--",
            label="До подбора: εr=8",
        )
        ax.axvspan(
            times[4] * 1e12,
            times[-1] * 1e12,
            color="gray",
            alpha=0.12,
            label="Отложенные кадры",
        )
        ax.set(
            xlabel="t, пс",
            ylabel=f"Нормированное {label}",
            title="Индекс точки (16,16,16)",
        )
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_fit(report):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    profile = np.asarray(report["profile"])
    history = np.asarray(report["fit"]["history"])
    axes[0].semilogy(profile[:, 0], np.maximum(profile[:, 1], 1e-30), "o-", label="Профиль")
    axes[0].scatter(history[:, 0], history[:, 1], label="Поиск", color="tab:red", s=15)
    axes[0].set(
        xlabel="εr",
        ylabel="Средняя относительная L2²",
        title="Функция потерь на кадрах для подбора",
    )
    axes[0].legend()
    for sample in (0, 1):
        axes[1].semilogy(
            np.arange(8),
            report[f"sample{sample}"]["e_per_frame"],
            "o-",
            label=f"e, траектория {sample}",
        )
        axes[1].semilogy(
            np.arange(8),
            report[f"sample{sample}"]["h_per_frame"],
            ".--",
            label=f"h, траектория {sample}",
        )
    axes[1].axvline(3.5, color="gray", linestyle=":")
    axes[1].set(
        xlabel="Номер кадра",
        ylabel="Относительная L2",
        title="Ошибки полей; траектория 1 целиком отложена",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_convergence(report):
    rows = report["convergence"]
    n = np.array([r["n"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4))
    for field, label in (("electric", "Электрическое поле e"), ("magnetic", "Магнитное поле h")):
        errors = [r[f"{field}_relative_l2"] for r in rows]
        ax.loglog(n, errors, "o-", label=label)
    ax.loglog(n, rows[0]["electric_relative_l2"] * (n[0] / n) ** 2, ":", label="O(Δx²)")
    ax.set(
        xlabel="Число ячеек по оси",
        ylabel="Относительная L2",
        title="Независимое аналитическое решение",
    )
    ax.set_xticks(n, [str(value) for value in n])
    ax.minorticks_off()
    ax.grid(which="major", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_emval(cloud):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(121, projection="3d")
    indices = np.arange(0, len(cloud.points), 10)
    p = cloud.points[indices] * 1e3
    colors = np.log10(np.maximum(np.linalg.norm(cloud.electric[indices], axis=1), 1e-20))
    scat = ax.scatter(*p.T, c=colors, s=6, cmap="viridis")
    ax.set(
        xlabel="x, мм",
        ylabel="y, мм",
        zlabel="z, мм",
        title="Расположение точек EMVal в 3D",
    )
    fig.colorbar(scat, ax=ax, shrink=0.6, label=r"$\log_{10}(\|\hat{E}\|\, /\, (1\,\mathrm{V/m}))$")
    ax = fig.add_subplot(122)
    point = 0
    times = np.linspace(0, 2 / cloud.frequency, 200)
    # Explicit illustrative convention, not a claim about the producer's sign.
    values = np.real(
        cloud.electric[point, :, None] * np.exp(1j * 2 * np.pi * cloud.frequency * times)
    )
    for component, label in enumerate(("Ex", "Ey", "Ez")):
        ax.plot(times * 1e9, values[component], label=label)
    ax.set(xlabel="t, нс", ylabel="E, V/m", title="Гармоническая реконструкция, exp(+iωt)")
    ax.legend()
    fig.tight_layout()
    return fig
