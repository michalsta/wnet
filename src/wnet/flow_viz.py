"""Interactive, draggable visualisation of min-cost-flow networks.

Three distinct schemes, one for each thing you actually want to look at:

* :func:`draw_network`  — the *structure*: nodes, arcs, capacities and unit
  costs.  Neutral colours; this is the blueprint before any flow is pushed.
* :func:`draw_flow`     — a *solved* network: edge thickness and colour encode
  how much flow each arc carries, saturated (bottleneck) arcs are flagged red,
  unused arcs fade into dashed grey.
* :func:`draw_residual` — the *residual* network of a solved flow: forward
  (remaining-capacity) arcs and reverse (flow-cancelling, negated-cost) arcs
  are drawn in different colours and line styles, curved apart so a forward /
  reverse pair between the same two nodes never overlaps.

All three render as a `vis.js` (via ``pyvis``) widget that works inline in
Jupyter: nodes are **draggable** (pull them apart to uncover overlap), every
node and edge has a **mouseover tooltip**, and the layered source → empirical →
theoretical → sink structure is laid out left-to-right to keep crossings down.

The functions accept either a ``networkx.DiGraph`` following the attribute
convention below, or anything with an ``as_networkx()`` method (e.g. a
``SubgraphWrapper``), in which case ``.draw()`` / ``.draw_flow()`` /
``.draw_residual()`` on that object are the shorter path.

Graph attribute convention
--------------------------
Node attributes (all optional except that *some* role signal is nice):
    ``type``        one of ``SourceNode`` / ``SinkNode`` / ``EmpiricalNode`` /
                    ``TheoreticalNode`` (case-insensitive substrings also work),
                    or ``trash`` for a dedicated trash node.
    ``layer``       column index 0..3 (source..sink).  Inferred from ``type``
                    when absent.
    ``intensity``   scales node size.
    ``peak_idx`` / ``spectrum_id``   shown in the tooltip / node label.

Edge attributes:
    ``weight`` or ``cost``   unit transport cost (either key works).
    ``capacity``             arc capacity; ``None`` / missing means infinite.
    ``flow``                 flow on the arc (required only by ``draw_flow``).
    ``kind``                 ``forward`` / ``reverse`` (only read by
                             ``draw_residual``; otherwise computed for you).
"""

from __future__ import annotations

import math
from typing import Optional


# --------------------------------------------------------------------------- #
# Visual grammar — kept consistent across all three schemes so a reader who
# learns the colours once can read every picture.
# --------------------------------------------------------------------------- #

_ROLE_STYLE = {
    # role            colour     border      shape        base size
    "source":      ("#43a047", "#1b5e20", "diamond",    22),
    "sink":        ("#e53935", "#8e0000", "diamond",    22),
    "empirical":   ("#1e88e5", "#0d47a1", "dot",        16),
    "theoretical": ("#fb8c00", "#e65100", "square",     16),
    "trash":       ("#9e9e9e", "#424242", "triangleDown", 18),
    "other":       ("#b0bec5", "#546e7a", "dot",        14),
}

_LAYER_OF_ROLE = {"source": 0, "empirical": 1, "theoretical": 2, "sink": 3}

# Trash arcs span the whole layout (source → sink and friends); a big-roundness
# curve swings them right over the top of the node cloud instead of hiding them
# as a straight line drawn behind everything.
_TRASH_SMOOTH = {"type": "curvedCCW", "roundness": 0.65}

# Palette for the structure view, keyed on the semantic kind of an arc.
_STRUCT_EDGE_STYLE = {
    "supply":    ("#90a4ae", False),   # source -> empirical  (grey, solid)
    "demand":    ("#90a4ae", False),   # theoretical -> sink  (grey, solid)
    "match":     ("#5c6bc0", False),   # empirical -> theoretical (indigo)
    "trash":     ("#8d6e63", True),    # any trash arc (brown, dashed)
    "chain":     ("#ab47bc", True),    # 1D chain adjacency arc (purple, dashed)
    "other":     ("#b0bec5", False),
}


_SHAPE_CHAR = {"diamond": "◆", "dot": "●", "square": "■", "triangleDown": "▼"}

_ROLE_LABEL = {
    "source": "source",
    "sink": "sink",
    "empirical": "empirical peak",
    "theoretical": "theoretical peak",
    "trash": "trash node",
    "other": "node",
}


def _role(node_type) -> str:
    """Normalise a ``type`` attribute to one of the role keys above."""
    t = str(node_type or "").lower()
    if "source" in t:
        return "source"
    if "sink" in t:
        return "sink"
    if "empirical" in t:
        return "empirical"
    if "theoretical" in t:
        return "theoretical"
    if "trash" in t:
        return "trash"
    return "other"


def _fmt(x) -> str:
    """Compact number formatting: integers stay integers, floats trim zeros."""
    if x is None:
        return "∞"  # infinity
    if isinstance(x, float):
        if math.isinf(x):
            return "∞"
        if x == int(x):
            return str(int(x))
        return f"{x:.4g}"
    return str(x)


def _cost(data) -> float:
    """Read the unit cost from an edge, accepting either key."""
    if "cost" in data:
        return data["cost"]
    return data.get("weight", 0.0)


def _edge_kind(g, u, v) -> str:
    """Classify a structural arc from the roles of its endpoints."""
    su = _role(g.nodes[u].get("type"))
    sv = _role(g.nodes[v].get("type"))
    if su == "source" and sv == "empirical":
        return "supply"
    if su == "theoretical" and sv == "sink":
        return "demand"
    if su == "empirical" and sv == "theoretical":
        return "match"
    if sv == "sink" and su in ("source", "empirical") or sv == "trash" or su == "trash":
        return "trash"
    if su == "source" and sv in ("sink", "theoretical"):
        return "trash"
    if su == sv and su in ("empirical", "theoretical"):
        return "chain"
    return "other"


# --------------------------------------------------------------------------- #
# Node / tooltip construction
# --------------------------------------------------------------------------- #

def _node_label(node_id, role, data) -> str:
    if role == "source":
        return "source"
    if role == "sink":
        return "sink"
    if role == "trash":
        return "trash"
    pk = data.get("peak_idx")
    if role == "empirical":
        return f"e{pk}" if pk is not None else f"e·{node_id}"
    if role == "theoretical":
        sid = data.get("spectrum_id")
        if sid is not None and pk is not None:
            return f"t{sid}:{pk}"
        return f"t·{node_id}"
    return str(node_id)


def _node_tooltip(node_id, role, data) -> str:
    # vis.js sets string titles via element.innerText, which renders HTML tags
    # literally but honours "\n" as line breaks — so tooltips are plain text.
    rows = [f"{data.get('type', role)}", f"id {node_id}"]
    for key in ("spectrum_id", "peak_idx", "intensity", "layer"):
        if data.get(key) is not None:
            rows.append(f"{key}: {_fmt(data[key])}")
    pos = data.get("pos") or data.get("position")
    if pos is not None:
        rows.append(f"pos: {pos}")
    return "\n".join(rows)


def _intensity_size(intensity, imax, base) -> float:
    if not intensity or not imax or imax <= 0:
        return base
    # sqrt so a 10x intensity is ~3x area, not 10x radius (keeps big peaks sane)
    return base + 26.0 * math.sqrt(max(intensity, 0.0) / imax)


def _add_nodes(net, g):
    intensities = [
        d.get("intensity") for _, d in g.nodes(data=True) if d.get("intensity")
    ]
    imax = max(intensities) if intensities else 0.0
    for node_id, data in g.nodes(data=True):
        role = _role(data.get("type"))
        colour, border, shape, base = _ROLE_STYLE[role]
        layer = data.get("layer", _LAYER_OF_ROLE.get(role, 1))
        size = (
            base
            if role in ("source", "sink")
            else _intensity_size(data.get("intensity"), imax, base)
        )
        net.add_node(
            node_id,
            label=_node_label(node_id, role, data),
            title=_node_tooltip(node_id, role, data),
            color={"background": colour, "border": border},
            shape=shape,
            size=size,
            level=int(layer),
            borderWidth=2,
        )


# --------------------------------------------------------------------------- #
# pyvis network construction + inline rendering
# --------------------------------------------------------------------------- #

def _new_network(height, width, hierarchical, physics, smooth_roundness=0.0):
    from pyvis.network import Network

    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#ffffff",
        font_color="#263238",
        cdn_resources="in_line",  # inline JS so the widget works offline
        notebook=False,
    )
    layout = (
        {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "levelSeparation": 200,
                "nodeSpacing": 85,
                "treeSpacing": 120,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            }
        }
        if hierarchical
        else {"improvedLayout": True}
    )
    options = {
        "layout": layout,
        "physics": {
            "enabled": bool(physics),
            "barnesHut": {"springLength": 140, "gravitationalConstant": -6000},
            "stabilization": {"iterations": 300},
        },
        "interaction": {
            "hover": True,
            "tooltipDelay": 80,
            "navigationButtons": True,
            "keyboard": False,
            "dragNodes": True,
            "multiselect": True,
            "hideEdgesOnDrag": False,
        },
        "nodes": {"font": {"size": 15, "face": "monospace"}},
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "font": {"size": 11, "align": "middle", "background": "#ffffff"},
            # Smoothing stays *enabled* with a consistent curvedCCW type even
            # when the base roundness is 0 (which renders as a straight line):
            # that keeps the curve machinery alive under the hierarchical layout
            # so per-edge `roundness` overrides are honoured.  Two effects ride
            # on this: trash arcs override to a big roundness to swing right over
            # the graph, and the residual view raises the base roundness so every
            # u→v arc and its v→u reverse bow to opposite sides (no overplot).
            "smooth": {
                "enabled": True,
                "type": "curvedCCW",
                "roundness": smooth_roundness,
            },
        },
    }
    import json

    net.set_options(json.dumps(options))
    return net


def _render(net, height, filename: Optional[str], legend_html: str = ""):
    """Return an inline Jupyter widget, or write a standalone HTML file.

    ``legend_html`` (if given) is a caption card placed above the graph; it is
    embedded in the saved file too so the HTML export is self-describing.
    """
    if filename is not None:
        net.write_html(filename, notebook=False)
        if legend_html:
            with open(filename, "r", encoding="utf-8") as fh:
                doc = fh.read()
            doc = doc.replace("<body>", f"<body>{legend_html}", 1)
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(doc)
        return filename
    html = net.generate_html(notebook=False)
    try:
        from IPython.display import HTML
    except Exception:  # pragma: no cover - non-Jupyter fallback
        return legend_html + html
    # Sandbox the whole widget in an <iframe srcdoc> so its inlined scripts run
    # regardless of notebook trust settings (the reason we don't hand raw HTML
    # to the cell).  IPython warns "use IFrame instead" for exactly this shape;
    # the warning doesn't apply to a self-contained srcdoc, so silence it.
    srcdoc = html.replace("&", "&amp;").replace('"', "&quot;")
    # Square off the iframe's top corners when a legend card sits flush above it.
    radius = "0 0 8px 8px" if legend_html else "8px"
    iframe = (
        f'<iframe srcdoc="{srcdoc}" '
        f'style="width:100%;height:{height};border:1px solid #cfd8dc;'
        f'border-radius:{radius};background:#fff"></iframe>'
    )
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consider using IPython.display.IFrame")
        return HTML(legend_html + iframe)


def _to_graph(obj):
    """Accept a networkx DiGraph or anything exposing ``as_networkx()``."""
    if hasattr(obj, "as_networkx"):
        return obj.as_networkx()
    return obj


# --------------------------------------------------------------------------- #
# Legends — rendered as an HTML caption card *above* the graph iframe (so they
# live in the notebook output, not inside the vis canvas).
# --------------------------------------------------------------------------- #

def _sw_node(shape, colour) -> str:
    char = _SHAPE_CHAR.get(shape, "●")
    return f'<span style="color:{colour};font-size:15px;line-height:1">{char}</span>'


def _sw_edge(colour, dashed=False, curved=False, width=3) -> str:
    style = "dashed" if dashed else "solid"
    arc = "↷ " if curved else ""
    return (
        f'<span style="color:{colour};margin-right:2px">{arc}</span>'
        f'<span style="display:inline-block;width:24px;'
        f'border-top:{width}px {style} {colour};vertical-align:middle"></span>'
    )


def _node_legend_items(g):
    """Node swatches, restricted to the roles actually present in ``g``."""
    present = {_role(d.get("type")) for _, d in g.nodes(data=True)}
    items = []
    for role in ("source", "empirical", "theoretical", "sink", "trash"):
        if role in present:
            colour, _, shape, _ = _ROLE_STYLE[role]
            items.append((_sw_node(shape, colour), _ROLE_LABEL[role]))
    return items


def _legend_html(groups, note=None):
    """``groups``: list of (label, [(swatch_html, text), ...])."""
    rows = []
    for label, items in groups:
        if not items:
            continue
        cells = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f'margin-right:15px;white-space:nowrap">{sw}'
            f'<span>{txt}</span></span>'
            for sw, txt in items
        )
        rows.append(
            f'<div style="margin:2px 0"><b style="color:#455a64;'
            f'margin-right:8px">{label}</b>{cells}</div>'
        )
    note_html = (
        f'<div style="margin-top:3px;color:#78909c;font-style:italic">{note}</div>'
        if note
        else ""
    )
    return (
        '<div style="font:13px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;'
        "color:#263238;background:#fafafa;border:1px solid #cfd8dc;"
        "border-bottom:none;border-radius:8px 8px 0 0;padding:8px 12px\">"
        f"{''.join(rows)}{note_html}</div>"
    )


# --------------------------------------------------------------------------- #
# Scheme 1 — structure
# --------------------------------------------------------------------------- #

def draw_network(
    graph,
    *,
    height: str = "620px",
    width: str = "100%",
    edge_labels: bool = False,
    hierarchical: bool = True,
    physics: bool = False,
    filename: Optional[str] = None,
):
    """Draw the network *structure*: nodes, arcs, capacities and unit costs.

    Colours encode arc semantics (supply/demand grey, matching indigo, trash
    brown-dashed, chain purple-dashed); every arc's tooltip carries its unit
    cost and capacity (``∞`` = uncapacitated).  No flow is shown — use
    :func:`draw_flow` for that.

    Parameters
    ----------
    graph : networkx.DiGraph or object with ``as_networkx()``
    edge_labels : if True, print ``c=…/κ=…`` on each arc (off by default to
        keep dense graphs readable — the tooltip always has it).
    hierarchical / physics : layout controls.  Hierarchical LR is the default;
        set ``physics=True`` (optionally ``hierarchical=False``) for a
        force-directed layout you can shake apart.
    filename : if given, write a standalone HTML file and return its path
        instead of an inline widget.
    """
    g = _to_graph(graph)
    net = _new_network(height, width, hierarchical, physics)
    _add_nodes(net, g)
    kinds = set()
    for u, v, data in g.edges(data=True):
        kind = data.get("_kind") or _edge_kind(g, u, v)
        kinds.add(kind)
        colour, dashed = _STRUCT_EDGE_STYLE.get(kind, _STRUCT_EDGE_STYLE["other"])
        cost = _cost(data)
        cap = data.get("capacity")
        title = (
            f"{kind}: {u} → {v}"
            f"\nunit cost: {_fmt(cost)}"
            f"\ncapacity: {_fmt(cap)}"
        )
        label = f"c={_fmt(cost)}/κ={_fmt(cap)}" if edge_labels else None
        kwargs = dict(color=colour, dashes=dashed, width=1.6, title=title, label=label)
        if kind == "trash":
            kwargs["smooth"] = _TRASH_SMOOTH
        net.add_edge(u, v, **kwargs)
    edge_items = []
    if kinds & {"supply", "demand"}:
        edge_items.append((_sw_edge("#90a4ae"), "supply / demand"))
    if "match" in kinds:
        edge_items.append((_sw_edge("#5c6bc0"), "matching"))
    if "chain" in kinds:
        edge_items.append((_sw_edge("#ab47bc", dashed=True), "chain (1D)"))
    if "trash" in kinds:
        edge_items.append((_sw_edge("#8d6e63", dashed=True, curved=True), "trash (discard)"))
    legend = _legend_html(
        [("nodes", _node_legend_items(g)), ("arcs", edge_items)],
        note="hover for cost / capacity  ·  drag nodes to separate  ·  node size ∝ intensity",
    )
    return _render(net, height, filename, legend)


# --------------------------------------------------------------------------- #
# Scheme 2 — solved flow
# --------------------------------------------------------------------------- #

def _flow_edge_color(flow, cap):
    """Colour arcs by how loaded they are; grey+dashed when unused."""
    if flow <= 0:
        return "#cfd8dc", True                      # unused: faint, dashed
    if cap is not None and flow >= cap:
        return "#d81b60", False                     # saturated bottleneck: pink-red
    return "#1e88e5", False                          # carrying flow: blue


def draw_flow(
    graph,
    *,
    height: str = "620px",
    width: str = "100%",
    edge_labels: bool = True,
    hierarchical: bool = True,
    physics: bool = False,
    filename: Optional[str] = None,
):
    """Draw a *solved* network, emphasising where the flow goes.

    Arc **width** grows with flow; **colour/style** encodes load — unused arcs
    fade to dashed grey, saturated (flow == capacity) arcs turn pink-red to
    mark bottlenecks, and partially-loaded arcs are blue.  Tooltips give
    ``flow / capacity``, unit cost, and the arc's cost contribution
    (``flow × cost``).  Requires a ``flow`` attribute on the edges.
    """
    g = _to_graph(graph)
    flows = [d.get("flow", 0) or 0 for _, _, d in g.edges(data=True)]
    fmax = max(flows) if flows else 0
    net = _new_network(height, width, hierarchical, physics)
    _add_nodes(net, g)
    has_trash = False
    for u, v, data in g.edges(data=True):
        flow = data.get("flow", 0) or 0
        cap = data.get("capacity")
        cost = _cost(data)
        kind = data.get("_kind") or _edge_kind(g, u, v)
        has_trash = has_trash or kind == "trash"
        colour, dashed = _flow_edge_color(flow, cap)
        width_px = 1.2 + (6.5 * flow / fmax if fmax else 0.0)
        sat = " (SATURATED)" if (cap is not None and flow >= cap and flow > 0) else ""
        title = (
            f"{u} → {v}{sat}"
            f"\nflow: {_fmt(flow)} / {_fmt(cap)}"
            f"\nunit cost: {_fmt(cost)}"
            f"\ncost contribution: {_fmt(flow * cost)}"
        )
        label = _fmt(flow) if (edge_labels and flow > 0) else None
        kwargs = dict(
            color=colour, dashes=dashed, width=width_px, title=title, label=label
        )
        if kind == "trash":
            kwargs["smooth"] = _TRASH_SMOOTH
        net.add_edge(u, v, **kwargs)
    edge_items = [
        (_sw_edge("#1e88e5"), "carrying flow"),
        (_sw_edge("#d81b60"), "saturated (at capacity)"),
        (_sw_edge("#cfd8dc", dashed=True), "unused (flow 0)"),
    ]
    if has_trash:
        edge_items.append(
            (_sw_edge("#8d6e63", dashed=True, curved=True), "trash flow (discarded)")
        )
    legend = _legend_html(
        [("nodes", _node_legend_items(g)), ("arcs", edge_items)],
        note="arc width ∝ flow  ·  hover for flow / capacity and cost  ·  node size ∝ intensity",
    )
    return _render(net, height, filename, legend)


# --------------------------------------------------------------------------- #
# Scheme 3 — residual network
# --------------------------------------------------------------------------- #

def residual_from_flow(graph):
    """Build the residual network of a *solved* flow as a ``networkx.MultiDiGraph``.

    Richer than a plain residual: every arc is tagged with ``kind``
    (``forward`` = remaining capacity, ``reverse`` = flow that can be
    cancelled), the residual capacity (``rescap``), and the residual ``cost``
    (negated on reverse arcs).  Optimal min-cost flows leave no negative cycle
    here — that's the property this view lets you eyeball.

    A MultiDiGraph is required: with antiparallel original arcs (e.g. the 1D
    chain factory's bidirectional chain edges) a forward residual and the
    reverse residual of the opposite arc share the same (u, v) pair, and a
    plain DiGraph would silently drop one of them.
    """
    import networkx as nx

    g = _to_graph(graph)
    r = nx.MultiDiGraph()
    for n, d in g.nodes(data=True):
        r.add_node(n, **d)
    for u, v, d in g.edges(data=True):
        flow = d.get("flow", 0) or 0
        cap = d.get("capacity")
        cost = _cost(d)
        rescap = math.inf if cap is None else cap - flow
        if cap is None or rescap > 0:
            r.add_edge(u, v, kind="forward", cost=cost, rescap=rescap)
        if flow > 0:
            r.add_edge(v, u, kind="reverse", cost=-cost, rescap=flow)
    return r


def draw_residual(
    graph,
    *,
    height: str = "620px",
    width: str = "100%",
    edge_labels: bool = False,
    hierarchical: bool = True,
    physics: bool = False,
    filename: Optional[str] = None,
):
    """Draw the *residual* network of a solved flow.

    Accepts a solved network (residual is computed via :func:`residual_from_flow`)
    or an already-residual graph whose edges carry ``kind`` in
    ``{forward, reverse}``.

    * **Forward** arcs (remaining capacity) — solid teal.
    * **Reverse** arcs (flow you could cancel, at negated cost) — dashed orange.
    * Every arc is curved (consistent handedness), so a forward/reverse pair
      between the same two nodes bows to opposite sides and never overplots.
    * Negative-cost arcs are flagged (``⚠``) in the tooltip.
    """
    g = _to_graph(graph)
    edge_view = list(g.edges(data=True))
    is_residual = any(d.get("kind") in ("forward", "reverse") for _, _, d in edge_view)
    r = g if is_residual else residual_from_flow(g)

    # A pronounced global curve (curvedCCW for every arc) makes each forward
    # arc and its reverse bow to opposite sides, so bidirectional pairs never
    # overplot; forward vs. reverse stay tellable apart by colour and dash.
    net = _new_network(height, width, hierarchical, physics, smooth_roundness=0.3)
    _add_nodes(net, r)
    for u, v, data in r.edges(data=True):
        kind = data.get("kind", "forward")
        cost = _cost(data)
        rescap = data.get("rescap")
        if kind == "reverse":
            colour, dashed = "#fb8c00", True     # cancellable flow: orange dashed
            head = "cancel up to"
        else:
            colour, dashed = "#00897b", False    # spare capacity: teal solid
            head = "push up to"
        neg = "  ⚠ negative cost" if cost < 0 else ""
        title = (
            f"{kind} residual{neg}"
            f"\n{u} → {v}"
            f"\n{head}: {_fmt(rescap)}"
            f"\nresidual cost: {_fmt(cost)}"
        )
        label = _fmt(cost) if edge_labels else None
        net.add_edge(u, v, color=colour, dashes=dashed, width=2.0, title=title, label=label)
    edge_items = [
        (_sw_edge("#00897b"), "forward — spare capacity"),
        (_sw_edge("#fb8c00", dashed=True), "reverse — cancellable flow (−cost)"),
    ]
    legend = _legend_html(
        [("nodes", _node_legend_items(r)), ("arcs", edge_items)],
        note="every arc curves; a forward/reverse pair bows to opposite sides  ·  "
        "an optimal flow leaves no negative-cost cycle (⚠ in tooltips)",
    )
    return _render(net, height, filename, legend)
