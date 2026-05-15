"""Generate the SVG illustrations shipped under `docs/assets/`.

Stdlib-only on purpose: this is hand-rolled SVG, not matplotlib. The repo
keeps zero ML/plotting dependencies at runtime, and even the visual assets
are reproducible from the same constraint.

Run from the repo root:

    python -m scripts.render_visuals

Outputs:
- docs/assets/architecture.svg       evaluation contract flow
- docs/assets/horizon_sweep.svg      success rate + latency vs plan horizon
- docs/assets/maze.svg               default 7x7 maze layout
- docs/assets/favicon.svg            small square favicon for the Pages site
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


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="ui-sans-serif, system-ui, '
        f'-apple-system, Segoe UI, sans-serif">'
    )


def render_architecture() -> str:
    """Encoder -> Latent -> Predictor -> Future Latent -> Planner -> Action."""
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

    parts = [_svg_open(width, height)]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')

    cy = 90
    for label, cx in nodes:
        x = cx - box_w // 2
        y = cy - box_h // 2
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

    # Connecting arrows.
    parts.append(
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{PALETTE["muted"]}"/></marker></defs>'
    )
    for i in range(len(nodes) - 1):
        x1 = nodes[i][1] + box_w // 2
        x2 = nodes[i + 1][1] - box_w // 2
        parts.append(
            f'<line x1="{x1}" y1="{cy}" x2="{x2 - 2}" y2="{cy}" '
            f'stroke="{PALETTE["muted"]}" stroke-width="1.8" marker-end="url(#arrow)"/>'
        )

    # Action arrow on the right.
    parts.append(
        f'<line x1="{nodes[-1][1] + box_w // 2}" y1="{cy}" x2="{width - 20}" y2="{cy}" '
        f'stroke="{PALETTE["muted"]}" stroke-width="1.8" marker-end="url(#arrow)"/>'
    )
    parts.append(
        f'<text x="{width - 18}" y="{cy - 8}" font-size="13" font-weight="600" '
        f'fill="{PALETTE["accent"]}" text-anchor="end">action</text>'
    )

    # Caption.
    parts.append(
        f'<text x="{width // 2}" y="{height - 14}" font-size="12" fill="{PALETTE["muted"]}" '
        f'text-anchor="middle">The evaluation contract every adapter must implement.</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def render_horizon_sweep(report_path: Path) -> str:
    """Dual-axis chart: success rate (left) and per-call latency (right) vs plan_horizon."""
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

    parts = [_svg_open(width, height)]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')

    # Title.
    parts.append(
        f'<text x="{width // 2}" y="22" font-size="15" font-weight="600" '
        f'fill="{PALETTE["ink"]}" text-anchor="middle">Planning-horizon sweep (maze toy, '
        f'tabular world model)</text>'
    )

    # Grid lines / left axis ticks (0, 0.25, 0.5, 0.75, 1.0).
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

    # Right axis ticks (latency).
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        latency = y2_min + frac * (y2_max - y2_min)
        y = y2_of(latency)
        parts.append(
            f'<text x="{width - pad_r + 8}" y="{y + 4:.1f}" font-size="11" '
            f'fill="{PALETTE["warn"]}" text-anchor="start">{latency:.2f}</text>'
        )

    # X axis ticks.
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

    # Axis labels.
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
        f'<polygon points="{band_pts}" fill="{PALETTE["accent_light"]}" fill-opacity="0.25" stroke="none"/>'
    )

    # Success rate line.
    success_line = " ".join(f"{x_of(h):.1f},{y1_of(s):.1f}" for h, s in zip(horizons, successes))
    parts.append(
        f'<polyline points="{success_line}" fill="none" stroke="{PALETTE["accent"]}" stroke-width="2.5"/>'
    )
    for h, s in zip(horizons, successes):
        parts.append(
            f'<circle cx="{x_of(h):.1f}" cy="{y1_of(s):.1f}" r="4" fill="{PALETTE["accent"]}" stroke="white" stroke-width="1.5"/>'
        )

    # Latency CI band.
    lat_band_lo = [f"{x_of(h):.1f},{y2_of(l):.1f}" for h, l in zip(horizons, latency_lo)]
    lat_band_hi = [f"{x_of(h):.1f},{y2_of(l):.1f}" for h, l in reversed(list(zip(horizons, latency_hi)))]
    lat_band = " ".join(lat_band_lo + lat_band_hi)
    parts.append(
        f'<polygon points="{lat_band}" fill="{PALETTE["warn_light"]}" fill-opacity="0.25" stroke="none"/>'
    )

    # Latency line.
    latency_line = " ".join(f"{x_of(h):.1f},{y2_of(l):.1f}" for h, l in zip(horizons, latencies))
    parts.append(
        f'<polyline points="{latency_line}" fill="none" stroke="{PALETTE["warn"]}" '
        f'stroke-width="2.5" stroke-dasharray="4,3"/>'
    )
    for h, l in zip(horizons, latencies):
        parts.append(
            f'<circle cx="{x_of(h):.1f}" cy="{y2_of(l):.1f}" r="3.5" fill="{PALETTE["warn"]}" stroke="white" stroke-width="1.5"/>'
        )

    # Plot border.
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="{PALETTE["muted"]}" stroke-width="1"/>'
    )

    # Legend.
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
    """Default 7x7 maze layout with start, goal, walls, and the optimal path."""
    # Layout is read top-down (highest y first), matching DEFAULT_LAYOUT in
    # examples/maze_toy/environment.py.
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
    height = height_cells * cell + 2 * margin + 36  # extra room for caption

    parts = [_svg_open(width, height)]
    parts.append(f'<rect width="{width}" height="{height}" fill="{PALETTE["bg"]}"/>')

    def cell_xy(col: int, row: int) -> tuple[int, int]:
        return margin + col * cell, margin + row * cell

    # Draw cells.
    for row, line in enumerate(layout):
        for col, ch in enumerate(line):
            x, y = cell_xy(col, row)
            if ch == "#":
                fill = PALETTE["wall"]
            else:
                fill = PALETTE["free"]
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

    # Optimal path from (1,5) -> ... -> (5,1), tracing the corridor.
    # Each tuple is (col, row_from_top), matching layout indexing.
    path_cells = [
        (1, 1),  # S row from top
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (3, 4),
        (3, 3),
        (3, 2),
        (3, 1),
        (4, 1),
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (5, 5),  # G
    ]
    centers = [
        (margin + c * cell + cell // 2, margin + r * cell + cell // 2)
        for c, r in path_cells
    ]
    path_d = "M " + " L ".join(f"{x},{y}" for x, y in centers)
    parts.append(
        f'<path d="{path_d}" fill="none" stroke="{PALETTE["accent"]}" stroke-width="3" '
        f'stroke-opacity="0.5" stroke-linecap="round" stroke-linejoin="round" '
        f'stroke-dasharray="6,4"/>'
    )

    # Caption.
    caption_y = height - 12
    parts.append(
        f'<text x="{width // 2}" y="{caption_y}" font-size="12" fill="{PALETTE["muted"]}" '
        f'text-anchor="middle">Two-room maze, 7x7. Optimal path = 14 actions. Naive greedy gets stuck on walls.</text>'
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

    (ASSETS / "favicon.svg").write_text(render_favicon() + "\n")
    print(f"wrote {ASSETS / 'favicon.svg'}")


if __name__ == "__main__":
    main()
