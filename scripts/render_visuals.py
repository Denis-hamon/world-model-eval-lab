"""Generate the SVG illustrations shipped under `docs/assets/`.

Stdlib-only on purpose: this is hand-rolled SVG, not matplotlib. The repo
keeps zero ML/plotting dependencies at runtime, and even the visual assets
are reproducible from the same constraint.

The illustrations include classes, IDs, and data attributes that the live
Pages site uses for:
- scroll-triggered reveal animations (CSS via `docs/assets/css/style.css`),
- hover tooltips on the sweep chart (JS via `docs/assets/js/interactive.js`),
- an SVG-native `<animateMotion>` agent walking the maze path (no JS needed).

Run from the repo root:

    python -m scripts.render_visuals

Outputs:
- docs/assets/architecture.svg              evaluation contract flow (animatable)
- docs/assets/horizon_sweep.svg             success rate + latency vs plan horizon (interactive)
- docs/assets/horizon_sweep_compare.svg     same chart, oracle dynamics vs learned MLP overlaid
- docs/assets/maze.svg                      7x7 maze with an animated agent
- docs/assets/policy_comparison.svg         three-up: random, greedy, world model
- docs/assets/favicon.svg                   small square favicon for the Pages site
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets"

PALETTE = {
    "ink": "#1f2933",
    "muted": "#52606d",
    "accent": "#0f5fbf",
    "accent_light": "#7fb1f0",
    "warn": "#b34a00",
    "warn_light": "#f5a26b",
    "wall": "#1f2933",
    "free": "#f7f7f7",
    "start": "#0f5fbf",
    "goal": "#b34a00",
    "stroke": "#c4cdd5",
    "bg": "#ffffff",
}


def _svg_open(width: int, height: int, extra: str = "") -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="ui-sans-serif, system-ui, '
        f'-apple-system, Segoe UI, sans-serif"{extra}>'
    )


def render_architecture() -> str:
    """Encoder -> Latent -> Predictor -> Future Latent -> Planner -> Action.

    Boxes are tagged with class="arch-box" and a `data-step` attribute so the
    layout can fade them in sequentially. Arrows get class="arch-arrow".
    """
    width, height = 980, 180
    nodes = [
        ("Observation", 60),
        ("Encoder", 200),
        ("Latent state", 360),
        ("Action-conditioned\npredictor", 540),
        ("Future latent state", 720),
        ("Planner", 860),
    ]
    box_w, box_h = 130, 60

    parts = [_svg_open(width, height, ' class="figure figure-architecture"')]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')

    cy = 90
    for i, (label, cx) in enumerate(nodes):
        x = cx - box_w // 2
        y = cy - box_h // 2
        parts.append(f'<g class="arch-box" data-step="{i}">')
        parts.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="white" stroke="{PALETTE["accent"]}" stroke-width="1.5"/>'
        )
        lines = label.split("\n")
        line_y = y + box_h // 2 - 5 * (len(lines) - 1) + 4
        for line in lines:
            parts.append(
                f'<text x="{cx}" y="{line_y}" font-size="13" fill="{PALETTE["ink"]}" '
                f'text-anchor="middle">{line}</text>'
            )
            line_y += 14
        parts.append('</g>')

    parts.append(
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["muted"]}"/></marker></defs>'
    )
    for i in range(len(nodes) - 1):
        x1 = nodes[i][1] + box_w // 2
        x2 = nodes[i + 1][1] - box_w // 2
        parts.append(
            f'<line class="arch-arrow" data-step="{i}" '
            f'x1="{x1}" y1="{cy}" x2="{x2 - 2}" y2="{cy}" '
            f'stroke="{PALETTE["muted"]}" stroke-width="1.8" marker-end="url(#arrow)"/>'
        )

    parts.append(
        f'<line class="arch-arrow" data-step="{len(nodes) - 1}" '
        f'x1="{nodes[-1][1] + box_w // 2}" y1="{cy}" x2="{width - 20}" y2="{cy}" '
        f'stroke="{PALETTE["muted"]}" stroke-width="1.8" marker-end="url(#arrow)"/>'
    )
    parts.append(
        f'<text x="{width - 18}" y="{cy - 8}" font-size="13" font-weight="600" '
        f'fill="{PALETTE["accent"]}" text-anchor="end">action</text>'
    )

    parts.append(
        f'<text x="{width // 2}" y="{height - 14}" font-size="12" fill="{PALETTE["muted"]}" '
        f'text-anchor="middle">The evaluation contract every adapter must implement.</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def render_horizon_sweep(report_path: Path) -> str:
    """Interactive dual-axis chart: success rate (left) and per-call latency (right) vs plan_horizon.

    Each horizon gets a `<g class="data-slice" data-...>` group containing the
    two data circles plus a wide invisible `<rect class="hover-target">` so
    the layout's JS can render a custom tooltip on hover.
    """
    with report_path.open() as f:
        report = json.load(f)

    points = report["points"]
    horizons = [p["plan_horizon"] for p in points]
    successes = [p["scorecard"]["success_rate"] for p in points]
    success_lo = [p["success_ci"][0] for p in points]
    success_hi = [p["success_ci"][1] for p in points]
    latencies = [p["scorecard"]["average_planning_latency_ms"] for p in points]
    latency_lo = [p["latency_ci_ms"][0] for p in points]
    latency_hi = [p["latency_ci_ms"][1] for p in points]
    computes = [
        p["scorecard"].get("average_compute_per_decision") for p in points
    ]
    steps = [p["scorecard"].get("average_steps_to_success") for p in points]

    width, height = 720, 380
    pad_l, pad_r, pad_t, pad_b = 70, 70, 40, 60
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    x_min, x_max = min(horizons), max(horizons)
    y1_min, y1_max = 0.0, 1.0
    lat_top = max(latency_hi) * 1.15 if latency_hi else 1.0
    y2_min, y2_max = 0.0, lat_top

    def x_of(h: float) -> float:
        return pad_l + (h - x_min) / (x_max - x_min) * plot_w

    def y1_of(s: float) -> float:
        return pad_t + (1.0 - (s - y1_min) / (y1_max - y1_min)) * plot_h

    def y2_of(latency: float) -> float:
        return pad_t + (1.0 - (latency - y2_min) / (y2_max - y2_min)) * plot_h

    parts = [_svg_open(width, height, ' class="figure figure-sweep"')]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')

    parts.append(
        f'<text x="{width // 2}" y="22" font-size="15" font-weight="600" '
        f'fill="{PALETTE["ink"]}" text-anchor="middle">Planning-horizon sweep (maze toy, '
        f'tabular world model)</text>'
    )

    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y1_of(s)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="{PALETTE["stroke"]}" stroke-dasharray="2,3" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="{PALETTE["muted"]}" '
            f'text-anchor="end">{s:.2f}</text>'
        )

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        latency = y2_min + frac * (y2_max - y2_min)
        y = y2_of(latency)
        parts.append(
            f'<text x="{width - pad_r + 8}" y="{y + 4:.1f}" font-size="11" '
            f'fill="{PALETTE["warn"]}" text-anchor="start">{latency:.2f}</text>'
        )

    for h in horizons:
        x = x_of(h)
        parts.append(
            f'<line x1="{x:.1f}" y1="{pad_t + plot_h}" x2="{x:.1f}" '
            f'y2="{pad_t + plot_h + 5}" stroke="{PALETTE["muted"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t + plot_h + 20}" font-size="11" '
            f'fill="{PALETTE["muted"]}" text-anchor="middle">{h}</text>'
        )

    parts.append(
        f'<text x="{pad_l - 50}" y="{pad_t + plot_h // 2}" font-size="12" '
        f'fill="{PALETTE["accent"]}" transform="rotate(-90 {pad_l - 50},{pad_t + plot_h // 2})" '
        f'text-anchor="middle">success rate</text>'
    )
    parts.append(
        f'<text x="{width - pad_r + 50}" y="{pad_t + plot_h // 2}" font-size="12" '
        f'fill="{PALETTE["warn"]}" transform="rotate(90 {width - pad_r + 50},{pad_t + plot_h // 2})" '
        f'text-anchor="middle">latency per call (ms)</text>'
    )
    parts.append(
        f'<text x="{width // 2}" y="{height - 14}" font-size="12" fill="{PALETTE["muted"]}" '
        f'text-anchor="middle">plan_horizon</text>'
    )

    # Success rate CI band.
    band_pts_lo = [f"{x_of(h):.1f},{y1_of(s):.1f}" for h, s in zip(horizons, success_lo)]
    band_pts_hi = [f"{x_of(h):.1f},{y1_of(s):.1f}" for h, s in reversed(list(zip(horizons, success_hi)))]
    band_pts = " ".join(band_pts_lo + band_pts_hi)
    parts.append(
        f'<polygon class="success-band" points="{band_pts}" fill="{PALETTE["accent_light"]}" '
        f'fill-opacity="0.25" stroke="none"/>'
    )

    # Success rate line (drawable).
    success_line = " ".join(f"{x_of(h):.1f},{y1_of(s):.1f}" for h, s in zip(horizons, successes))
    parts.append(
        f'<polyline class="series-line success-line" points="{success_line}" fill="none" '
        f'stroke="{PALETTE["accent"]}" stroke-width="2.5"/>'
    )

    # Latency CI band.
    lat_band_lo = [f"{x_of(h):.1f},{y2_of(l):.1f}" for h, l in zip(horizons, latency_lo)]
    lat_band_hi = [f"{x_of(h):.1f},{y2_of(l):.1f}" for h, l in reversed(list(zip(horizons, latency_hi)))]
    lat_band = " ".join(lat_band_lo + lat_band_hi)
    parts.append(
        f'<polygon class="latency-band" points="{lat_band}" fill="{PALETTE["warn_light"]}" '
        f'fill-opacity="0.25" stroke="none"/>'
    )

    # Latency line (drawable).
    latency_line = " ".join(f"{x_of(h):.1f},{y2_of(l):.1f}" for h, l in zip(horizons, latencies))
    parts.append(
        f'<polyline class="series-line latency-line" points="{latency_line}" fill="none" '
        f'stroke="{PALETTE["warn"]}" stroke-width="2.5" stroke-dasharray="4,3"/>'
    )

    # Per-horizon data slices: invisible hover target + the two data dots, all
    # grouped so the layout's JS can find them by `data-h`.
    half_x_step = plot_w / max(1, len(horizons) - 1) / 2
    for h, s, lat, comp, st in zip(horizons, successes, latencies, computes, steps):
        x = x_of(h)
        sy = y1_of(s)
        ly = y2_of(lat)
        comp_str = "n/a" if comp is None else f"{comp:.1f}"
        st_str = "n/a" if st is None else f"{st:.1f}"
        parts.append(
            f'<g class="data-slice" data-h="{h}" data-success="{s:.3f}" '
            f'data-latency="{lat:.3f}" data-compute="{comp_str}" data-steps="{st_str}">'
        )
        parts.append(
            f'<rect class="hover-target" x="{x - half_x_step:.1f}" y="{pad_t}" '
            f'width="{2 * half_x_step:.1f}" height="{plot_h}" fill="transparent"/>'
        )
        parts.append(
            f'<circle class="point success-point" cx="{x:.1f}" cy="{sy:.1f}" r="4" '
            f'fill="{PALETTE["accent"]}" stroke="white" stroke-width="1.5"/>'
        )
        parts.append(
            f'<circle class="point latency-point" cx="{x:.1f}" cy="{ly:.1f}" r="3.5" '
            f'fill="{PALETTE["warn"]}" stroke="white" stroke-width="1.5"/>'
        )
        parts.append(
            f'<title>plan_horizon = {h} | success = {s:.3f} | latency = {lat:.3f} ms/call | '
            f'compute/decision = {comp_str}</title>'
        )
        parts.append('</g>')

    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="{PALETTE["muted"]}" stroke-width="1"/>'
    )

    leg_y = pad_t + 12
    parts.append(
        f'<line x1="{pad_l + 16}" y1="{leg_y}" x2="{pad_l + 48}" y2="{leg_y}" '
        f'stroke="{PALETTE["accent"]}" stroke-width="2.5"/>'
    )
    parts.append(
        f'<text x="{pad_l + 54}" y="{leg_y + 4}" font-size="11" fill="{PALETTE["ink"]}">success rate (95% CI band)</text>'
    )
    parts.append(
        f'<line x1="{pad_l + 230}" y1="{leg_y}" x2="{pad_l + 262}" y2="{leg_y}" '
        f'stroke="{PALETTE["warn"]}" stroke-width="2.5" stroke-dasharray="4,3"/>'
    )
    parts.append(
        f'<text x="{pad_l + 268}" y="{leg_y + 4}" font-size="11" fill="{PALETTE["ink"]}">planning latency per call (95% CI band)</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def render_maze() -> str:
    """7x7 maze with start, goal, walls, and an animated agent walking the optimal path."""
    layout = [
        "#######",
        "#S#...#",
        "#.#.#.#",
        "#.#.#.#",
        "#...#.#",
        "#.###G#",
        "#######",
    ]
    height_cells = len(layout)
    width_cells = len(layout[0])
    cell = 48
    margin = 20
    width = width_cells * cell + 2 * margin
    height = height_cells * cell + 2 * margin + 36

    parts = [_svg_open(width, height, ' class="figure figure-maze"')]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')

    def cell_xy(col: int, row: int) -> tuple[int, int]:
        return margin + col * cell, margin + row * cell

    for row, line in enumerate(layout):
        for col, ch in enumerate(line):
            x, y = cell_xy(col, row)
            fill = PALETTE["wall"] if ch == "#" else PALETTE["free"]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="{PALETTE["stroke"]}" stroke-width="1"/>'
            )
            if ch == "S":
                cx, cy = x + cell // 2, y + cell // 2
                parts.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{cell // 3}" fill="{PALETTE["start"]}"/>'
                )
                parts.append(
                    f'<text x="{cx}" y="{cy + 4}" font-size="14" font-weight="600" fill="white" '
                    f'text-anchor="middle">S</text>'
                )
            elif ch == "G":
                cx, cy = x + cell // 2, y + cell // 2
                parts.append(
                    f'<circle cx="{cx}" cy="{cy}" r="{cell // 3}" fill="{PALETTE["goal"]}"/>'
                )
                parts.append(
                    f'<text x="{cx}" y="{cy + 4}" font-size="14" font-weight="600" fill="white" '
                    f'text-anchor="middle">G</text>'
                )

    path_cells = [
        (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4),
        (3, 3), (3, 2), (3, 1), (4, 1), (5, 1),
        (5, 2), (5, 3), (5, 4), (5, 5),
    ]
    centers = [
        (margin + c * cell + cell // 2, margin + r * cell + cell // 2)
        for c, r in path_cells
    ]
    path_d = "M " + " L ".join(f"{x},{y}" for x, y in centers)
    parts.append(
        f'<path id="agent-path" d="{path_d}" fill="none" stroke="{PALETTE["accent"]}" '
        f'stroke-width="3" stroke-opacity="0.45" stroke-linecap="round" '
        f'stroke-linejoin="round" stroke-dasharray="6,4"/>'
    )

    # The animated agent: an orange dot that loops along the optimal path.
    # animateMotion is supported across modern browsers; no JS required.
    parts.append(
        f'<g class="maze-agent">'
        f'<circle r="9" fill="{PALETTE["goal"]}" stroke="white" stroke-width="2">'
        f'<animateMotion dur="5s" repeatCount="indefinite" rotate="auto">'
        f'<mpath href="#agent-path" />'
        f'</animateMotion>'
        f'</circle>'
        f'<circle r="3" fill="white">'
        f'<animateMotion dur="5s" repeatCount="indefinite" rotate="auto">'
        f'<mpath href="#agent-path" />'
        f'</animateMotion>'
        f'</circle>'
        f'</g>'
    )

    caption_y = height - 12
    parts.append(
        f'<text x="{width // 2}" y="{caption_y}" font-size="12" fill="{PALETTE["muted"]}" '
        f'text-anchor="middle">Two-room maze, 7x7. Optimal path = 14 actions. Naive greedy gets stuck on walls.</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def render_horizon_sweep_compare(oracle_path: Path, learned_path: Path) -> str:
    """Two stacked panels: success-rate identical, latency dramatically different.

    Reads `horizon_sweep_report.json` (oracle dynamics, stdlib) and
    `learned_horizon_sweep_report.json` (PyTorch MLP dynamics). Shows that
    the framework's evaluation contract holds for both kinds of dynamics:
    success rates coincide because the learned model recovers the oracle
    transition table, but per-call planning latency diverges by an order
    of magnitude because of torch invocation overhead.
    """
    with oracle_path.open() as f:
        oracle = json.load(f)
    with learned_path.open() as f:
        learned = json.load(f)

    o_pts = oracle["points"]
    l_pts = learned["points"]
    horizons = [p["plan_horizon"] for p in o_pts]
    o_success = [p["scorecard"]["success_rate"] for p in o_pts]
    l_success = [p["scorecard"]["success_rate"] for p in l_pts]
    o_lat = [p["scorecard"]["average_planning_latency_ms"] for p in o_pts]
    l_lat = [p["scorecard"]["average_planning_latency_ms"] for p in l_pts]

    width, height = 760, 540
    title_h = 50
    panel_gap = 36
    pad_l, pad_r, pad_b = 70, 50, 50
    panel_h = (height - title_h - pad_b - panel_gap) // 2

    parts = [_svg_open(width, height, ' class="figure figure-sweep-compare"')]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')

    parts.append(
        f'<text x="{width // 2}" y="22" font-size="15" font-weight="600" '
        f'fill="{PALETTE["ink"]}" text-anchor="middle">Same contract, two dynamics</text>'
    )
    parts.append(
        f'<text x="{width // 2}" y="40" font-size="12" fill="{PALETTE["muted"]}" '
        f'text-anchor="middle">Oracle stdlib vs PyTorch-learned MLP, maze toy, 30 episodes per point.</text>'
    )

    x_min, x_max = min(horizons), max(horizons)
    plot_w = width - pad_l - pad_r

    def x_of(h: float) -> float:
        return pad_l + (h - x_min) / (x_max - x_min) * plot_w

    def panel_y_of(value: float, vmin: float, vmax: float, panel_top: int) -> float:
        return panel_top + (1.0 - (value - vmin) / (vmax - vmin)) * panel_h

    # ----- Panel A: success rate -----
    panel_a_top = title_h + 8
    y_min_a, y_max_a = 0.0, 1.05

    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = panel_y_of(s, y_min_a, y_max_a, panel_a_top)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="{PALETTE["stroke"]}" stroke-dasharray="2,3" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="{PALETTE["muted"]}" '
            f'text-anchor="end">{s:.2f}</text>'
        )

    parts.append(
        f'<text x="{pad_l - 50}" y="{panel_a_top + panel_h // 2}" font-size="12" '
        f'fill="{PALETTE["accent"]}" transform="rotate(-90 {pad_l - 50},{panel_a_top + panel_h // 2})" '
        f'text-anchor="middle">success rate</text>'
    )
    parts.append(
        f'<text x="{pad_l + 6}" y="{panel_a_top + 16}" font-size="11" '
        f'font-weight="500" fill="{PALETTE["muted"]}">Success</text>'
    )

    # Oracle line + dots (solid blue).
    line_pts = " ".join(f"{x_of(h):.1f},{panel_y_of(s, y_min_a, y_max_a, panel_a_top):.1f}"
                       for h, s in zip(horizons, o_success))
    parts.append(
        f'<polyline points="{line_pts}" fill="none" stroke="{PALETTE["accent"]}" stroke-width="2.5"/>'
    )
    for h, s in zip(horizons, o_success):
        parts.append(
            f'<circle cx="{x_of(h):.1f}" cy="{panel_y_of(s, y_min_a, y_max_a, panel_a_top):.1f}" '
            f'r="4" fill="{PALETTE["accent"]}" stroke="white" stroke-width="1.5"/>'
        )

    # Learned line + dots (dashed warn).
    line_pts_l = " ".join(f"{x_of(h):.1f},{panel_y_of(s, y_min_a, y_max_a, panel_a_top):.1f}"
                         for h, s in zip(horizons, l_success))
    parts.append(
        f'<polyline points="{line_pts_l}" fill="none" stroke="{PALETTE["warn"]}" '
        f'stroke-width="2" stroke-dasharray="5,3" stroke-opacity="0.85"/>'
    )
    for h, s in zip(horizons, l_success):
        parts.append(
            f'<circle cx="{x_of(h):.1f}" cy="{panel_y_of(s, y_min_a, y_max_a, panel_a_top):.1f}" '
            f'r="3" fill="{PALETTE["warn"]}" stroke="white" stroke-width="1"/>'
        )

    parts.append(
        f'<rect x="{pad_l}" y="{panel_a_top}" width="{plot_w}" height="{panel_h}" '
        f'fill="none" stroke="{PALETTE["muted"]}" stroke-width="1"/>'
    )

    # Note on the panel: curves overlap.
    parts.append(
        f'<text x="{width - pad_r - 6}" y="{panel_a_top + 16}" font-size="11" '
        f'fill="{PALETTE["muted"]}" text-anchor="end">Both dynamics reach 100% success at h>=15. '
        f'The curves overlap.</text>'
    )

    # ----- Panel B: latency per call -----
    panel_b_top = panel_a_top + panel_h + panel_gap
    y_max_b = max(l_lat) * 1.15
    y_min_b = 0.0

    grid_b_vals = [y_min_b + frac * (y_max_b - y_min_b) for frac in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for v in grid_b_vals:
        y = panel_y_of(v, y_min_b, y_max_b, panel_b_top)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" '
            f'stroke="{PALETTE["stroke"]}" stroke-dasharray="2,3" stroke-width="0.8"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="{PALETTE["muted"]}" '
            f'text-anchor="end">{v:.0f}</text>'
        )

    parts.append(
        f'<text x="{pad_l - 50}" y="{panel_b_top + panel_h // 2}" font-size="12" '
        f'fill="{PALETTE["warn"]}" transform="rotate(-90 {pad_l - 50},{panel_b_top + panel_h // 2})" '
        f'text-anchor="middle">latency / call (ms)</text>'
    )
    parts.append(
        f'<text x="{pad_l + 6}" y="{panel_b_top + 16}" font-size="11" '
        f'font-weight="500" fill="{PALETTE["muted"]}">Latency per plan() call</text>'
    )

    # X axis ticks at the bottom of panel B.
    for h in horizons:
        x = x_of(h)
        parts.append(
            f'<line x1="{x:.1f}" y1="{panel_b_top + panel_h}" x2="{x:.1f}" '
            f'y2="{panel_b_top + panel_h + 5}" stroke="{PALETTE["muted"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{panel_b_top + panel_h + 20}" font-size="11" '
            f'fill="{PALETTE["muted"]}" text-anchor="middle">{h}</text>'
        )
    parts.append(
        f'<text x="{width // 2}" y="{panel_b_top + panel_h + 38}" font-size="12" '
        f'fill="{PALETTE["muted"]}" text-anchor="middle">plan_horizon</text>'
    )

    # Oracle latency line (solid accent).
    line_o = " ".join(f"{x_of(h):.1f},{panel_y_of(l, y_min_b, y_max_b, panel_b_top):.1f}"
                     for h, l in zip(horizons, o_lat))
    parts.append(
        f'<polyline points="{line_o}" fill="none" stroke="{PALETTE["accent"]}" stroke-width="2.5"/>'
    )
    for h, l in zip(horizons, o_lat):
        parts.append(
            f'<circle cx="{x_of(h):.1f}" cy="{panel_y_of(l, y_min_b, y_max_b, panel_b_top):.1f}" '
            f'r="3.5" fill="{PALETTE["accent"]}" stroke="white" stroke-width="1.5"/>'
        )

    # Learned latency line (warn, thicker, dashed).
    line_l = " ".join(f"{x_of(h):.1f},{panel_y_of(l, y_min_b, y_max_b, panel_b_top):.1f}"
                     for h, l in zip(horizons, l_lat))
    parts.append(
        f'<polyline points="{line_l}" fill="none" stroke="{PALETTE["warn"]}" '
        f'stroke-width="2.5" stroke-dasharray="5,3"/>'
    )
    for h, l in zip(horizons, l_lat):
        parts.append(
            f'<circle cx="{x_of(h):.1f}" cy="{panel_y_of(l, y_min_b, y_max_b, panel_b_top):.1f}" '
            f'r="3.5" fill="{PALETTE["warn"]}" stroke="white" stroke-width="1.5"/>'
        )

    # Annotation: range of per-horizon ratios. Honest about the spread; an
    # earlier draft used only the rightmost-horizon ratio and was off by a
    # third at the smaller horizons.
    ratios = [
        l / o for o, l in zip(o_lat, l_lat) if o > 0
    ]
    if ratios:
        rmin, rmax = min(ratios), max(ratios)
        annotation = f"Learned MLP costs {rmin:.0f}-{rmax:.0f}x more per call."
    else:
        annotation = "Learned MLP latency unavailable."
    parts.append(
        f'<text x="{width - pad_r - 6}" y="{panel_b_top + 16}" font-size="11" '
        f'font-weight="600" fill="{PALETTE["warn"]}" text-anchor="end">{annotation}</text>'
    )

    parts.append(
        f'<rect x="{pad_l}" y="{panel_b_top}" width="{plot_w}" height="{panel_h}" '
        f'fill="none" stroke="{PALETTE["muted"]}" stroke-width="1"/>'
    )

    # Legend at the bottom.
    legend_y = height - 16
    legend_x0 = pad_l + 10
    parts.append(
        f'<line x1="{legend_x0}" y1="{legend_y}" x2="{legend_x0 + 28}" y2="{legend_y}" '
        f'stroke="{PALETTE["accent"]}" stroke-width="2.5"/>'
    )
    parts.append(
        f'<text x="{legend_x0 + 34}" y="{legend_y + 4}" font-size="11" '
        f'fill="{PALETTE["ink"]}">Oracle dynamics (stdlib)</text>'
    )
    parts.append(
        f'<line x1="{legend_x0 + 220}" y1="{legend_y}" x2="{legend_x0 + 248}" y2="{legend_y}" '
        f'stroke="{PALETTE["warn"]}" stroke-width="2.5" stroke-dasharray="5,3"/>'
    )
    parts.append(
        f'<text x="{legend_x0 + 254}" y="{legend_y + 4}" font-size="11" '
        f'fill="{PALETTE["ink"]}">Learned MLP dynamics (PyTorch)</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def render_policy_comparison() -> str:
    """Three side-by-side mini-mazes, one per policy that ran on the maze toy.

    Each panel embeds the same 7x7 layout with a different agent animation:
    - random:  jitters near the start, never reaching the goal.
    - greedy (no waypoint):  bumps the wall, gets stuck.
    - tabular world model:  walks the optimal path to the goal.

    The numbers shown under each panel come from `examples/maze_toy/sample_report.json`
    and must be kept in sync when the report is regenerated.
    """
    layout = [
        "#######",
        "#S#...#",
        "#.#.#.#",
        "#.#.#.#",
        "#...#.#",
        "#.###G#",
        "#######",
    ]
    cell = 22
    margin = 10
    panel_w = len(layout[0]) * cell + 2 * margin
    panel_inner_h = len(layout) * cell + 2 * margin
    label_h = 28
    verdict_h = 78
    panel_h = label_h + panel_inner_h + verdict_h

    gap = 26
    width = panel_w * 3 + gap * 2 + 2 * margin
    height = panel_h + 2 * margin

    panels = [
        {
            "key": "random",
            "label": "Random policy",
            "verdict": "Wanders near the start.",
            "subverdict": "Goal stays out of reach.",
            "metrics": "success 0%   |   0.03 ms / call",
            "color": "#b34a00",
            "agent_color": "#b34a00",
            "path_cells": [(1, 1), (1, 2), (2, 2), (1, 2), (1, 1), (2, 1)],
            "path_dur": "2.6s",
            "stuck": False,
        },
        {
            "key": "greedy",
            "label": "Greedy (no waypoint)",
            "verdict": "Walks into the wall.",
            "subverdict": "Plan diverges from env, stuck.",
            "metrics": "success 0%   |   0.001 ms / call",
            "color": "#52606d",
            "agent_color": "#52606d",
            "path_cells": [(1, 1), (1, 1)],
            "path_dur": "1.2s",
            "stuck": True,
        },
        {
            "key": "wm",
            "label": "Tabular world model",
            "verdict": "Finds the corridor.",
            "subverdict": "Reaches the goal in ~34 steps.",
            "metrics": "success 100%   |   3.12 ms / call",
            "color": "#0f5fbf",
            "agent_color": "#0f5fbf",
            "path_cells": [
                (1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4),
                (3, 3), (3, 2), (3, 1), (4, 1), (5, 1),
                (5, 2), (5, 3), (5, 4), (5, 5),
            ],
            "path_dur": "4.5s",
            "stuck": False,
        },
    ]

    parts = [_svg_open(width, height, ' class="figure figure-comparison"')]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')

    for idx, p in enumerate(panels):
        x0 = margin + idx * (panel_w + gap)
        y0 = margin

        parts.append(
            f'<text x="{x0 + panel_w // 2}" y="{y0 + 18}" font-size="13" font-weight="600" '
            f'fill="{p["color"]}" text-anchor="middle">{p["label"]}</text>'
        )

        maze_y0 = y0 + label_h
        for row, line in enumerate(layout):
            for col, ch in enumerate(line):
                cx = x0 + margin + col * cell
                cy = maze_y0 + margin + row * cell
                fill = PALETTE["wall"] if ch == "#" else PALETTE["free"]
                parts.append(
                    f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" '
                    f'fill="{fill}" stroke="{PALETTE["stroke"]}" stroke-width="0.6"/>'
                )
                if ch == "S":
                    parts.append(
                        f'<circle cx="{cx + cell // 2}" cy="{cy + cell // 2}" r="{cell // 3}" '
                        f'fill="{PALETTE["start"]}"/>'
                    )
                elif ch == "G":
                    parts.append(
                        f'<circle cx="{cx + cell // 2}" cy="{cy + cell // 2}" r="{cell // 3}" '
                        f'fill="{PALETTE["goal"]}"/>'
                    )

        path_d_centres = [
            (x0 + margin + c * cell + cell // 2, maze_y0 + margin + r * cell + cell // 2)
            for c, r in p["path_cells"]
        ]

        agent_id = f"agent-path-{p['key']}"
        if p["stuck"]:
            # No visible track for greedy. The agent dot shakes horizontally in place
            # via an <animate> on cx instead of animateMotion.
            cx0, cy0 = path_d_centres[0]
            parts.append(
                f'<g class="agent-dot agent-stuck">'
                f'<circle cx="{cx0}" cy="{cy0}" r="{cell // 3}" fill="{p["agent_color"]}" '
                f'stroke="white" stroke-width="2">'
                f'<animate attributeName="cx" '
                f'values="{cx0};{cx0 + 4};{cx0};{cx0 + 4};{cx0}" '
                f'dur="{p["path_dur"]}" repeatCount="indefinite"/>'
                f'</circle>'
                f'</g>'
            )
        else:
            path_d = "M " + " L ".join(f"{cx},{cy}" for cx, cy in path_d_centres)
            parts.append(
                f'<path id="{agent_id}" d="{path_d}" fill="none" stroke="{p["color"]}" '
                f'stroke-width="2" stroke-opacity="0.35" stroke-linecap="round" '
                f'stroke-linejoin="round" stroke-dasharray="3,3"/>'
            )
            parts.append(
                f'<g class="agent-dot">'
                f'<circle r="{cell // 3}" fill="{p["agent_color"]}" stroke="white" stroke-width="2">'
                f'<animateMotion dur="{p["path_dur"]}" repeatCount="indefinite">'
                f'<mpath href="#{agent_id}" />'
                f'</animateMotion>'
                f'</circle>'
                f'</g>'
            )

        verdict_y = maze_y0 + panel_inner_h + 16
        # Three text lines: bold verdict (active voice), muted subverdict
        # (consequence), monospace metrics. The parallel structure across the
        # three panels makes the comparison readable left-to-right.
        parts.append(
            f'<text x="{x0 + panel_w // 2}" y="{verdict_y}" font-size="12" font-weight="600" '
            f'fill="{PALETTE["ink"]}" text-anchor="middle">{p["verdict"]}</text>'
        )
        parts.append(
            f'<text x="{x0 + panel_w // 2}" y="{verdict_y + 16}" font-size="11" font-weight="400" '
            f'fill="{PALETTE["muted"]}" text-anchor="middle">{p["subverdict"]}</text>'
        )
        parts.append(
            f'<text x="{x0 + panel_w // 2}" y="{verdict_y + 38}" font-size="11" '
            f'fill="{PALETTE["muted"]}" text-anchor="middle" font-family="ui-monospace, '
            f'SFMono-Regular, Menlo, monospace">{p["metrics"]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_favicon() -> str:
    """A 32x32 square favicon: blue rounded square with 'wm' lettering."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" '
        'width="32" height="32">'
    ]
    parts.append(f'<rect width="32" height="32" rx="6" fill="{PALETTE["accent"]}"/>')
    parts.append(
        f'<text x="16" y="22" font-family="ui-sans-serif, system-ui, sans-serif" '
        f'font-size="15" font-weight="700" fill="white" text-anchor="middle">wm</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    (ASSETS / "architecture.svg").write_text(render_architecture() + "\n")
    print(f"wrote {ASSETS / 'architecture.svg'}")

    report_path = REPO_ROOT / "examples" / "maze_toy" / "horizon_sweep_report.json"
    if not report_path.exists():
        raise SystemExit(
            f"Missing {report_path}. Run `python -m examples.maze_toy.run_horizon_sweep` first."
        )
    (ASSETS / "horizon_sweep.svg").write_text(render_horizon_sweep(report_path) + "\n")
    print(f"wrote {ASSETS / 'horizon_sweep.svg'}")

    (ASSETS / "maze.svg").write_text(render_maze() + "\n")
    print(f"wrote {ASSETS / 'maze.svg'}")

    (ASSETS / "policy_comparison.svg").write_text(render_policy_comparison() + "\n")
    print(f"wrote {ASSETS / 'policy_comparison.svg'}")

    learned_path = REPO_ROOT / "examples" / "maze_toy" / "learned_horizon_sweep_report.json"
    if learned_path.exists():
        (ASSETS / "horizon_sweep_compare.svg").write_text(
            render_horizon_sweep_compare(report_path, learned_path) + "\n"
        )
        print(f"wrote {ASSETS / 'horizon_sweep_compare.svg'}")
    else:
        print(
            f"skipped horizon_sweep_compare.svg (missing {learned_path}; run "
            "`python -m examples.maze_toy.run_learned_sweep` first)"
        )

    (ASSETS / "favicon.svg").write_text(render_favicon() + "\n")
    print(f"wrote {ASSETS / 'favicon.svg'}")


if __name__ == "__main__":
    main()
