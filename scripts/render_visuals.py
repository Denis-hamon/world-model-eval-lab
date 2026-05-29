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


# ============================================================================
# Paper figures (HTML mirror of the LaTeX TikZ/pgfplots figures).
#
# The paper PDF compiles its own TikZ figures via texlive-latex-extra. The
# HTML mirror at docs/paper.md needs the same content in SVG so a web reader
# sees the same visual story. These four functions reproduce the LaTeX
# figures in stdlib SVG, matching the numeric data cell-for-cell.
# ============================================================================

_FIG_PALETTE = {
    "red": "#c45c3a",
    "red_dark": "#8a3a1f",
    "blue": "#5083c2",
    "blue_dark": "#2f5a8a",
    "orange": "#d68a3c",
    "orange_dark": "#9a5e1f",
    "axis": "#52606d",
    "grid": "#d8e0e8",
    "ink": "#1f2933",
    "muted": "#52606d",
    "bg": "#ffffff",
    "zero": "#7b8794",
}


def _axis_line(x1: float, y1: float, x2: float, y2: float, stroke: str = None) -> str:
    s = stroke or _FIG_PALETTE["axis"]
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{s}" stroke-width="1"/>'


def _gridline(x1: float, y1: float, x2: float, y2: float) -> str:
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{_FIG_PALETTE["grid"]}" stroke-dasharray="2,3" stroke-width="0.8"/>'
    )


def _tick_text(x: float, y: float, text: str, anchor: str = "middle", size: int = 11) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'fill="{_FIG_PALETTE["muted"]}" text-anchor="{anchor}">{text}</text>'
    )


def _error_bar_asym(x: float, y: float, y_top: float, y_bot: float, color: str, cap: float = 4.0) -> str:
    """Asymmetric vertical error bar centered at (x, y), spanning (y_top, y_bot)."""
    return (
        f'<line x1="{x:.1f}" y1="{y_top:.1f}" x2="{x:.1f}" y2="{y_bot:.1f}" stroke="{color}" stroke-width="1.4"/>'
        f'<line x1="{x - cap:.1f}" y1="{y_top:.1f}" x2="{x + cap:.1f}" y2="{y_top:.1f}" stroke="{color}" stroke-width="1.4"/>'
        f'<line x1="{x - cap:.1f}" y1="{y_bot:.1f}" x2="{x + cap:.1f}" y2="{y_bot:.1f}" stroke="{color}" stroke-width="1.4"/>'
    )


def render_paper_fig1_cpg_vs_data() -> str:
    """Figure 1: val MSE plummets 150x while CPG stays flat (n=150 pooled).

    Twin-axis chart, log-scale val MSE on the left, CPG with asymmetric AC
    error bars on the right. Three points across training-set size.
    """
    import math

    W, H = 640, 420
    L, R, T, B = 70, 70, 50, 60
    plot_w = W - L - R
    plot_h = H - T - B

    # Data (matches paper/figures/cpg_vs_data.tex):
    sizes = [200, 2000, 20000]
    val_mse = [0.0651, 0.0233, 0.00042]
    cpg = 0.267
    eplus, eminus = 0.068, 0.076

    # x in log-scale: 200=0, 2000=1, 20000=2
    def x_of(size: int) -> float:
        t = math.log10(size / 200) / math.log10(20000 / 200)
        return L + t * plot_w

    # Left y-axis: val_mse, log scale, 1e-4 to 1e-1
    y_log_min, y_log_max = -4, -1

    def y_left(v: float) -> float:
        t = (math.log10(v) - y_log_min) / (y_log_max - y_log_min)
        return T + (1 - t) * plot_h

    # Right y-axis: CPG, linear, -0.1 to 0.6
    cpg_min, cpg_max = -0.1, 0.6

    def y_right(v: float) -> float:
        t = (v - cpg_min) / (cpg_max - cpg_min)
        return T + (1 - t) * plot_h

    parts = [_svg_open(W, H, ' class="paper-fig paper-fig-1"')]
    parts.append(f'<rect width="{W}" height="{H}" fill="{_FIG_PALETTE["bg"]}"/>')
    # Title
    parts.append(
        f'<text x="{W // 2}" y="22" font-size="14" font-weight="600" '
        f'fill="{_FIG_PALETTE["ink"]}" text-anchor="middle">'
        f'Val MSE drops 150x; CPG stays flat at +0.267</text>'
    )
    parts.append(
        f'<text x="{W // 2}" y="40" font-size="11" fill="{_FIG_PALETTE["muted"]}" '
        f'text-anchor="middle">DMC Acrobot, MLP world model, n=150 pooled per arm. Source: results/dmc_acrobot/cpg_sweep.json</text>'
    )
    # Gridlines (left axis, log)
    for exp in range(y_log_min, y_log_max + 1):
        y = y_left(10 ** exp)
        parts.append(_gridline(L, y, W - R, y))
        parts.append(_tick_text(L - 8, y + 4, f"10^{{{exp}}}", anchor="end"))
    # Axis frame
    parts.append(_axis_line(L, T, L, T + plot_h))
    parts.append(_axis_line(W - R, T, W - R, T + plot_h))
    parts.append(_axis_line(L, T + plot_h, W - R, T + plot_h))
    # Left y-axis label
    parts.append(
        f'<text x="20" y="{T + plot_h // 2}" font-size="11" fill="{_FIG_PALETTE["red_dark"]}" '
        f'text-anchor="middle" transform="rotate(-90 20 {T + plot_h // 2})">Val MSE (log)</text>'
    )
    # Right y-axis label
    parts.append(
        f'<text x="{W - 20}" y="{T + plot_h // 2}" font-size="11" fill="{_FIG_PALETTE["blue_dark"]}" '
        f'text-anchor="middle" transform="rotate(90 {W - 20} {T + plot_h // 2})">CPG (oracle - learned)</text>'
    )
    # Right y-axis ticks
    for v in (0.0, 0.2, 0.4, 0.6):
        y = y_right(v)
        parts.append(_tick_text(W - R + 8, y + 4, f"{v:+.1f}", anchor="start"))
    # x-axis ticks + labels
    for s in sizes:
        x = x_of(s)
        parts.append(_axis_line(x, T + plot_h, x, T + plot_h + 5))
        label = "200" if s == 200 else ("2 000" if s == 2000 else "20 000")
        parts.append(_tick_text(x, T + plot_h + 20, label, anchor="middle"))
    parts.append(
        f'<text x="{L + plot_w // 2}" y="{H - 12}" font-size="11" '
        f'fill="{_FIG_PALETTE["muted"]}" text-anchor="middle">Training-set size (transitions, log scale)</text>'
    )
    # Zero line on right axis
    y_zero = y_right(0)
    parts.append(
        f'<line x1="{L}" y1="{y_zero:.1f}" x2="{W - R}" y2="{y_zero:.1f}" '
        f'stroke="{_FIG_PALETTE["zero"]}" stroke-dasharray="3,3" stroke-width="0.8"/>'
    )

    # Val MSE series: red squares + line
    pts_left = [(x_of(s), y_left(v)) for s, v in zip(sizes, val_mse)]
    parts.append(
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts_left)}" '
        f'fill="none" stroke="{_FIG_PALETTE["red"]}" stroke-width="2"/>'
    )
    for (x, y), v in zip(pts_left, val_mse):
        parts.append(
            f'<rect x="{x - 4:.1f}" y="{y - 4:.1f}" width="8" height="8" '
            f'fill="{_FIG_PALETTE["red"]}" stroke="{_FIG_PALETTE["red_dark"]}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" font-size="10" '
            f'fill="{_FIG_PALETTE["red_dark"]}">{v:.4f}</text>'
        )

    # CPG series: blue triangles + line + asymmetric error bars
    pts_right = [(x_of(s), y_right(cpg)) for s in sizes]
    parts.append(
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts_right)}" '
        f'fill="none" stroke="{_FIG_PALETTE["blue"]}" stroke-width="2"/>'
    )
    for x, y in pts_right:
        y_top = y_right(cpg + eplus)
        y_bot = y_right(cpg - eminus)
        parts.append(_error_bar_asym(x, y, y_top, y_bot, _FIG_PALETTE["blue_dark"]))
        # Triangle marker
        parts.append(
            f'<polygon points="{x:.1f},{y - 5:.1f} {x - 5:.1f},{y + 4:.1f} {x + 5:.1f},{y + 4:.1f}" '
            f'fill="{_FIG_PALETTE["blue"]}" stroke="{_FIG_PALETTE["blue_dark"]}" stroke-width="1"/>'
        )

    # Legend
    lx, ly = L + 16, T + 12
    parts.append(
        f'<rect x="{lx - 8}" y="{ly - 12}" width="200" height="44" rx="4" '
        f'fill="white" stroke="{_FIG_PALETTE["grid"]}"/>'
    )
    parts.append(
        f'<rect x="{lx}" y="{ly - 4}" width="10" height="10" fill="{_FIG_PALETTE["red"]}" stroke="{_FIG_PALETTE["red_dark"]}"/>'
    )
    parts.append(
        f'<text x="{lx + 16}" y="{ly + 4}" font-size="10" fill="{_FIG_PALETTE["ink"]}">Val MSE (MLP, held-out)</text>'
    )
    parts.append(
        f'<polygon points="{lx + 5},{ly + 14} {lx},{ly + 22} {lx + 10},{ly + 22}" '
        f'fill="{_FIG_PALETTE["blue"]}" stroke="{_FIG_PALETTE["blue_dark"]}"/>'
    )
    parts.append(
        f'<text x="{lx + 16}" y="{ly + 22}" font-size="10" fill="{_FIG_PALETTE["ink"]}">CPG (AC 95% CI)</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _grouped_bars(
    *,
    width: int,
    height: int,
    title: str,
    subtitle: str,
    xlabels: list[str],
    ylabel: str,
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    series: list[dict],  # each {"name": str, "color": str, "color_dark": str, "values": [...], "eplus": [...], "eminus": [...]}
    legend_pos: str = "tr",  # tr / tl
) -> str:
    L, R, T, B = 70, 50, 60, 70
    plot_w = width - L - R
    plot_h = height - T - B

    n_groups = len(xlabels)
    n_series = len(series)
    group_w = plot_w / n_groups
    bar_w = group_w / (n_series + 1)

    def y_of(v: float) -> float:
        t = (v - y_min) / (y_max - y_min)
        return T + (1 - t) * plot_h

    parts = [_svg_open(width, height, ' class="paper-fig"')]
    parts.append(f'<rect width="{width}" height="{height}" fill="{_FIG_PALETTE["bg"]}"/>')
    parts.append(
        f'<text x="{width // 2}" y="22" font-size="14" font-weight="600" '
        f'fill="{_FIG_PALETTE["ink"]}" text-anchor="middle">{title}</text>'
    )
    if subtitle:
        parts.append(
            f'<text x="{width // 2}" y="40" font-size="11" fill="{_FIG_PALETTE["muted"]}" '
            f'text-anchor="middle">{subtitle}</text>'
        )

    for t in y_ticks:
        y = y_of(t)
        parts.append(_gridline(L, y, width - R, y))
        parts.append(_tick_text(L - 8, y + 4, f"{t:+.2f}" if t < 0 else f"{t:.2f}", anchor="end"))
    parts.append(_axis_line(L, T, L, T + plot_h))
    parts.append(_axis_line(L, T + plot_h, width - R, T + plot_h))
    y_zero = y_of(0)
    if y_min < 0 < y_max:
        parts.append(
            f'<line x1="{L}" y1="{y_zero:.1f}" x2="{width - R}" y2="{y_zero:.1f}" '
            f'stroke="{_FIG_PALETTE["zero"]}" stroke-dasharray="3,3" stroke-width="0.8"/>'
        )
    parts.append(
        f'<text x="20" y="{T + plot_h // 2}" font-size="11" fill="{_FIG_PALETTE["muted"]}" '
        f'text-anchor="middle" transform="rotate(-90 20 {T + plot_h // 2})">{ylabel}</text>'
    )

    for i, label in enumerate(xlabels):
        cx = L + (i + 0.5) * group_w
        parts.append(_tick_text(cx, T + plot_h + 20, label, anchor="middle"))

    for gi in range(n_groups):
        for si, s in enumerate(series):
            cx = L + gi * group_w + (si + 0.7) * bar_w
            v = s["values"][gi]
            y_top_bar = y_of(max(v, 0))
            y_bot_bar = y_of(min(v, 0))
            parts.append(
                f'<rect x="{cx - bar_w / 2 + 1:.1f}" y="{y_top_bar:.1f}" '
                f'width="{bar_w - 2:.1f}" height="{abs(y_bot_bar - y_top_bar):.1f}" '
                f'fill="{s["color"]}" stroke="{s["color_dark"]}" stroke-width="1"/>'
            )
            eplus = s["eplus"][gi]
            eminus = s["eminus"][gi]
            if eplus > 0 or eminus > 0:
                yp = y_of(v + eplus)
                yn = y_of(v - eminus)
                parts.append(_error_bar_asym(cx, y_of(v), yp, yn, s["color_dark"]))

    if legend_pos == "tr":
        lx, ly = width - R - 180, T + 12
    else:
        lx, ly = L + 16, T + 12
    legend_h = 18 + 16 * n_series
    parts.append(
        f'<rect x="{lx - 8}" y="{ly - 12}" width="180" height="{legend_h}" rx="4" '
        f'fill="white" stroke="{_FIG_PALETTE["grid"]}"/>'
    )
    for si, s in enumerate(series):
        yoff = ly - 4 + si * 16
        parts.append(
            f'<rect x="{lx}" y="{yoff}" width="12" height="10" '
            f'fill="{s["color"]}" stroke="{s["color_dark"]}"/>'
        )
        parts.append(
            f'<text x="{lx + 18}" y="{yoff + 9}" font-size="10" fill="{_FIG_PALETTE["ink"]}">{s["name"]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_paper_fig2_coverage_histogram() -> str:
    """Figure 2: uprightness coverage histogram, random vs oracle planner."""
    # buckets: 8 bins of width 0.5 from -2 to +2
    labels = ["[-2,-1.5)", "[-1.5,-1)", "[-1,-0.5)", "[-0.5,0)",
              "[0,0.5)", "[0.5,1)", "[1,1.5)", "[1.5,2)"]
    random_pct = [13.6, 23.9, 13.8, 13.6, 17.1, 17.9, 0.0, 0.0]
    oracle_pct = [8.3, 7.7, 13.6, 11.8, 14.8, 23.6, 8.0, 12.2]
    return _grouped_bars(
        width=720, height=420,
        title="Coverage: random rollouts vs oracle planner",
        subtitle="Uprightness u = cos(theta_1) + cos(theta_2). Upright pose at +2. Source: results/dmc_acrobot/coverage.json.",
        xlabels=labels,
        ylabel="Fraction of states (%)",
        y_min=0, y_max=28,
        y_ticks=[0, 5, 10, 15, 20, 25],
        series=[
            {"name": "Random rollouts (n=2000)",
             "color": _FIG_PALETTE["red"], "color_dark": _FIG_PALETTE["red_dark"],
             "values": random_pct, "eplus": [0] * 8, "eminus": [0] * 8},
            {"name": "Oracle planner (n=846)",
             "color": _FIG_PALETTE["blue"], "color_dark": _FIG_PALETTE["blue_dark"],
             "values": oracle_pct, "eplus": [0] * 8, "eminus": [0] * 8},
        ],
        legend_pos="tl",
    )


def render_paper_fig3_cross_env() -> str:
    """Figure 3: cross-env CPG, Acrobot vs Cartpole (size=5) on the same four arms."""
    labels = ["RS x MLP", "RS x TD-MPC2", "CEM x MLP", "CEM x TD-MPC2"]
    return _grouped_bars(
        width=720, height=440,
        title="Cross-env CPG: Acrobot vs Cartpole (model_size = 5)",
        subtitle="Asymmetric Agresti-Caffo 95% CI per bar. Sources: results/dmc_acrobot/ + results/dmc_cartpole/*_size5_pooled.json.",
        xlabels=labels,
        ylabel="Counterfactual Planning Gap",
        y_min=-0.15, y_max=1.05,
        y_ticks=[0.0, 0.25, 0.5, 0.75, 1.0],
        series=[
            {"name": "Acrobot",
             "color": _FIG_PALETTE["blue"], "color_dark": _FIG_PALETTE["blue_dark"],
             "values": [0.300, 0.300, 0.880, 0.880],
             "eplus":  [0.259, 0.259, 0.043, 0.043],
             "eminus": [0.359, 0.359, 0.066, 0.066]},
            {"name": "Cartpole (size=5)",
             "color": _FIG_PALETTE["orange"], "color_dark": _FIG_PALETTE["orange_dark"],
             "values": [0.900, 0.700, 0.500, 0.367],
             "eplus":  [0.073, 0.140, 0.152, 0.191],
             "eminus": [0.186, 0.227, 0.215, 0.237]},
        ],
        legend_pos="tr",
    )


def render_paper_fig4_cartpole_capacity() -> str:
    """Figure 4: Cartpole capacity sweep (size=5 vs size=1), 4 arms each.

    The rightmost orange bar (CEM x TD-MPC2 size=1) crosses zero, the visible
    signal of the INCONCLUSIVE verdict.
    """
    labels = ["RS x MLP", "RS x TD-MPC2", "CEM x MLP", "CEM x TD-MPC2"]
    return _grouped_bars(
        width=720, height=440,
        title="Cartpole capacity sweep: model_size = 5 vs model_size = 1",
        subtitle="Same protocol, identical 10^6-step budget. INCONCLUSIVE on the rightmost orange bar. n=30 pooled.",
        xlabels=labels,
        ylabel="Counterfactual Planning Gap",
        y_min=-0.35, y_max=1.05,
        y_ticks=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        series=[
            {"name": "model_size = 5",
             "color": _FIG_PALETTE["blue"], "color_dark": _FIG_PALETTE["blue_dark"],
             "values": [0.900, 0.700, 0.500, 0.367],
             "eplus":  [0.073, 0.140, 0.152, 0.191],
             "eminus": [0.186, 0.227, 0.215, 0.237]},
            {"name": "model_size = 1",
             "color": _FIG_PALETTE["orange"], "color_dark": _FIG_PALETTE["orange_dark"],
             "values": [0.900, 0.400, 0.433, -0.033],
             "eplus":  [0.073, 0.183, 0.174, 0.247],
             "eminus": [0.186, 0.233, 0.227, 0.243]},
        ],
        legend_pos="tr",
    )


def render_paper_fig5_power_detectability() -> str:
    """Figure 5: CPG verdict detectability map (HTML mirror of the TikZ fig).

    Log-x curve = smallest oracle-minus-learned success-rate gap whose AC CI
    clears zero at a given per-arm n (oracle fixed at 0.94). Region below the
    curve is INCONCLUSIVE (shaded); above is DECIDABLE. Two annotated points
    at n=100: the 0.94-vs-0.92 near-tie (in the grey zone) and the 0.94-vs-0.78
    gap (decidable). Boundary baked from wmel.metrics.detectable_gap_at_n.
    """
    import math

    W, H = 640, 400
    L, R, T, B = 70, 30, 30, 56
    plot_w = W - L - R
    plot_h = H - T - B

    # (n, min detectable gap) boundary, from detectable_gap_at_n(0.94, 0.94-g, n).
    boundary = [
        (10, 0.415), (20, 0.260), (30, 0.195), (50, 0.140), (75, 0.110),
        (100, 0.090), (150, 0.070), (200, 0.060), (300, 0.045), (500, 0.035),
        (1000, 0.025),
    ]
    n_min, n_max = 10, 1000
    y_max = 0.45

    def x_of(n: float) -> float:
        t = math.log10(n / n_min) / math.log10(n_max / n_min)
        return L + t * plot_w

    def y_of(g: float) -> float:
        return T + (1 - g / y_max) * plot_h

    parts = [_svg_open(W, H, ' class="paper-fig paper-fig-5"')]
    parts.append(f'<rect width="{W}" height="{H}" fill="{_FIG_PALETTE["bg"]}"/>')
    parts.append(
        f'<text x="{W // 2}" y="20" font-size="14" font-weight="600" '
        f'fill="{_FIG_PALETTE["ink"]}" text-anchor="middle">'
        f'Verdict detectability map (oracle rate 0.94)</text>'
    )
    # y gridlines + ticks
    for g in (0.0, 0.1, 0.2, 0.3, 0.4):
        y = y_of(g)
        parts.append(_gridline(L, y, W - R, y))
        parts.append(_tick_text(L - 8, y + 4, f"{g:.1f}", anchor="end"))
    # x ticks (log)
    for n in (10, 30, 100, 300, 1000):
        x = x_of(n)
        parts.append(_axis_line(x, T + plot_h, x, T + plot_h + 5))
        parts.append(_tick_text(x, T + plot_h + 19, str(n), anchor="middle"))
    parts.append(_axis_line(L, T, L, T + plot_h))
    parts.append(_axis_line(L, T + plot_h, W - R, T + plot_h))
    parts.append(
        f'<text x="{L + plot_w // 2}" y="{H - 10}" font-size="11" '
        f'fill="{_FIG_PALETTE["muted"]}" text-anchor="middle">Episodes per arm n (log scale)</text>'
    )
    parts.append(
        f'<text x="16" y="{T + plot_h // 2}" font-size="11" fill="{_FIG_PALETTE["muted"]}" '
        f'text-anchor="middle" transform="rotate(-90 16 {T + plot_h // 2})">Success-rate gap (oracle - learned)</text>'
    )
    # INCONCLUSIVE region: polygon under the boundary down to y=0 (axis).
    pts = [(x_of(n), y_of(g)) for n, g in boundary]
    y_axis = y_of(0.0)
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    poly += f" {pts[-1][0]:.1f},{y_axis:.1f} {pts[0][0]:.1f},{y_axis:.1f}"
    parts.append(f'<polygon points="{poly}" fill="#9aa3ad" fill-opacity="0.18"/>')
    # Boundary curve.
    parts.append(
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" '
        f'fill="none" stroke="{_FIG_PALETTE["blue"]}" stroke-width="2"/>'
    )
    # Region labels.
    parts.append(
        f'<text x="{x_of(320):.1f}" y="{y_of(0.05):.1f}" font-size="11" '
        f'fill="{_FIG_PALETTE["muted"]}" text-anchor="middle">INCONCLUSIVE</text>'
    )
    parts.append(
        f'<text x="{x_of(45):.1f}" y="{y_of(0.34):.1f}" font-size="11" '
        f'fill="{_FIG_PALETTE["blue_dark"]}">DECIDABLE</text>'
    )
    # Annotated leaderboard points at n=100.
    xp = x_of(100)
    parts.append(
        f'<polygon points="{xp:.1f},{y_of(0.02) - 5:.1f} {xp - 5:.1f},{y_of(0.02) + 4:.1f} '
        f'{xp + 5:.1f},{y_of(0.02) + 4:.1f}" fill="{_FIG_PALETTE["red"]}" stroke="{_FIG_PALETTE["red_dark"]}"/>'
    )
    parts.append(
        f'<text x="{xp + 9:.1f}" y="{y_of(0.02) + 4:.1f}" font-size="10" '
        f'fill="{_FIG_PALETTE["red_dark"]}">near-tie 0.94 vs 0.92</text>'
    )
    parts.append(
        f'<rect x="{xp - 4:.1f}" y="{y_of(0.16) - 4:.1f}" width="8" height="8" '
        f'fill="#3a8a4f" stroke="#256634"/>'
    )
    parts.append(
        f'<text x="{xp + 9:.1f}" y="{y_of(0.16) + 4:.1f}" font-size="10" '
        f'fill="#256634">gap 0.94 vs 0.78</text>'
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

    # Paper figures (HTML mirror of paper/figures/*.tex):
    (ASSETS / "paper_fig1_cpg_vs_data.svg").write_text(render_paper_fig1_cpg_vs_data() + "\n")
    print(f"wrote {ASSETS / 'paper_fig1_cpg_vs_data.svg'}")
    (ASSETS / "paper_fig2_coverage_histogram.svg").write_text(render_paper_fig2_coverage_histogram() + "\n")
    print(f"wrote {ASSETS / 'paper_fig2_coverage_histogram.svg'}")
    (ASSETS / "paper_fig3_cross_env.svg").write_text(render_paper_fig3_cross_env() + "\n")
    print(f"wrote {ASSETS / 'paper_fig3_cross_env.svg'}")
    (ASSETS / "paper_fig4_cartpole_capacity.svg").write_text(render_paper_fig4_cartpole_capacity() + "\n")
    print(f"wrote {ASSETS / 'paper_fig4_cartpole_capacity.svg'}")
    (ASSETS / "paper_fig5_power_detectability.svg").write_text(render_paper_fig5_power_detectability() + "\n")
    print(f"wrote {ASSETS / 'paper_fig5_power_detectability.svg'}")


if __name__ == "__main__":
    main()
