"""Small inline SVG diagrams for the page.

Each function returns one `<svg>` element. The diagrams are deliberately
simple: a few shapes and a few words. Colors and text styles come from the
page's CSS (`.fig` classes), so they follow its theme. The numbers in them
(file size, vector lengths) are passed in from the build, not typed by hand.
"""

from __future__ import annotations

import math

WIDTH = 600
BLUE = "#03a0f1"  # the logo's "picture" color
PINK = "#e1096f"  # the logo's "records" color
SEED = "#e07a5f"
NEIGHBOR = "#79c2d0"
DENSE = "#ffd66e"
SPARSE = "#5c7cb8"
DOT = "#8a9aa8"
GITHUB_LIMIT_MB = 100 * 1024 * 1024 / 1e6  # 100 MiB in MB (about 105)
MORPHEM_INDEX_MB = 33  # measured: the size of a HNSW index on the 384-d vectors


def _svg(name: str, height: int, title: str, body: str) -> str:
    """Wrap shapes in an accessible SVG with a shared arrow marker."""

    return (
        f'<svg class="fig" viewBox="0 0 {WIDTH} {height}" role="img" '
        f'aria-labelledby="{name}-title">'
        f'<title id="{name}-title">{title}</title>'
        f'<defs><marker id="{name}-arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="#c9d4dc"/></marker></defs>'
        f"{body}</svg>"
    )


def _text(x: float, y: float, words: str, cls: str = "") -> str:
    return f'<text x="{x:g}" y="{y:g}" class="{cls}">{words}</text>'


def _arrow(  # noqa: PLR0913
    name: str, x1: float, y1: float, x2: float, y2: float, *, color: str
) -> str:
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" '
        f'stroke-width="2.5" marker-end="url(#{name}-arrow)"/>'
    )


def _dots(
    seed: int, count: int, x: tuple[float, float], y: tuple[float, float]
) -> list[tuple[float, float]]:
    """Repeatable pseudo-random points (a tiny generator, so no imports)."""

    state = seed
    points = []
    for _ in range(count):
        state = (state * 1664525 + 1013904223) % 4294967296
        px = x[0] + (state / 4294967296) * (x[1] - x[0])
        state = (state * 1664525 + 1013904223) % 4294967296
        py = y[0] + (state / 4294967296) * (y[1] - y[0])
        points.append((px, py))
    return points


def polyglot() -> str:
    name = "fig-polyglot"
    body = (
        '<rect x="20" y="34" width="240" height="40" class="box" '
        'style="fill:#12324a"/>'
        '<rect x="260" y="34" width="320" height="40" class="box" '
        'style="fill:#3a1f33"/>'
        '<line x1="470" y1="34" x2="470" y2="74" class="line"/>'
        '<line x1="530" y1="34" x2="530" y2="74" class="line"/>'
        + _text(140, 59, "JPEG picture", "t-ink t-mid")
        + _text(365, 59, "database.duckdb", "t-ink t-mid")
        + _text(500, 59, "README", "t-small t-mid")
        + _text(555, 51, "table of", "t-small t-mid")
        + _text(555, 66, "contents", "t-small t-mid")
        + '<line x1="260" y1="26" x2="260" y2="82" stroke="#f6f7f2" '
        'stroke-width="2" stroke-dasharray="3 3"/>'
        + _text(260, 20, "FF D9", "t-small t-mid")
        + _arrow(name, 24, 108, 254, 108, color=BLUE)
        + _text(140, 132, "A picture viewer reads from the start", "t-small t-mid")
        + _text(140, 150, "and stops at FF D9.", "t-small t-mid")
        + _arrow(name, 576, 108, 266, 108, color=PINK)
        + _text(422, 132, "A ZIP tool reads from the end and", "t-small t-mid")
        + _text(422, 150, "finds the table of contents first.", "t-small t-mid")
    )
    return _svg(
        name,
        168,
        "One file, two readers: a picture viewer reads from the start and stops "
        "at FF D9, and a ZIP tool reads from the end",
        body,
    )


def size_limit(jpeg_bytes: int) -> str:
    name = "fig-size"
    this_mb = jpeg_bytes / 1e6
    with_index = this_mb + MORPHEM_INDEX_MB
    scale = 480 / max(with_index * 1.05, GITHUB_LIMIT_MB * 1.05)
    x0 = 20
    limit_x = x0 + GITHUB_LIMIT_MB * scale
    bar_a = this_mb * scale
    bar_b = with_index * scale
    body = (
        f'<rect x="{x0}" y="46" width="{bar_a:.1f}" height="30" rx="3" '
        f'fill="{BLUE}" fill-opacity="0.85"/>'
        + _text(x0 + 10, 66, f"This file: {this_mb:.0f} MB", "t-dark")
        + f'<rect x="{x0}" y="92" width="{bar_b:.1f}" height="30" rx="3" '
        f'fill="{PINK}" fill-opacity="0.85"/>'
        + _text(
            x0 + 10, 112, f"With a MorphEm index: {with_index:.0f} MB", "t-dark"
        )
        + f'<line x1="{limit_x:.1f}" y1="34" x2="{limit_x:.1f}" y2="134" '
        'stroke="#f6f7f2" stroke-width="2" stroke-dasharray="5 4"/>'
        + _text(
            limit_x, 24, "GitHub limit: 100 MiB (about 105 MB)", "t-small t-mid"
        )
        + _text(x0 + bar_b + 8, 112, "too big", "t-small")
    )
    return _svg(
        name,
        146,
        f"Bars compare this {this_mb:.0f} MB file, and the {with_index:.0f} MB "
        "it would be with a MorphEm index, against GitHub's 100 MiB limit",
        body,
    )


def index_vs_scan() -> str:
    name = "fig-index"
    # left: an exact scan compares the query with every point
    query = (50, 130)
    points = _dots(7, 15, (105, 270), (62, 186))
    scan = ""
    nearest = sorted(points, key=lambda p: math.dist(p, query))[:3]
    for px, py in points:
        scan += (
            f'<line x1="{query[0]}" y1="{query[1]}" x2="{px:.1f}" y2="{py:.1f}" '
            'stroke="#4b5c6a" stroke-width="1"/>'
        )
    for px, py in points:
        color = NEIGHBOR if (px, py) in nearest else DOT
        scan += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{color}"/>'
    scan += f'<circle cx="{query[0]}" cy="{query[1]}" r="8" fill="{SEED}"/>'

    # right: HNSW layers, with the path a search takes
    layer2 = [380, 470, 560]
    layer1 = [350, 400, 445, 490, 535, 575]
    layer0 = [340, 365, 390, 415, 440, 465, 490, 515, 540, 565, 590]
    rows = ((80, layer2), (125, layer1), (170, layer0))
    graph = ""
    for y, xs in rows:
        graph += (
            f'<line x1="{xs[0]}" y1="{y}" x2="{xs[-1]}" y2="{y}" '
            'stroke="#4b5c6a" stroke-width="1.5"/>'
        )
        for x in xs:
            graph += f'<circle cx="{x}" cy="{y}" r="4.5" fill="{DOT}"/>'
    graph += (
        f'<line x1="380" y1="80" x2="470" y2="80" stroke="{NEIGHBOR}" '
        'stroke-width="3"/>'
        f'<line x1="470" y1="80" x2="445" y2="125" stroke="{NEIGHBOR}" '
        'stroke-width="3"/>'
        f'<line x1="445" y1="125" x2="465" y2="170" stroke="{NEIGHBOR}" '
        'stroke-width="3"/>'
        f'<circle cx="470" cy="80" r="5.5" fill="{NEIGHBOR}"/>'
        f'<circle cx="445" cy="125" r="5.5" fill="{NEIGHBOR}"/>'
        f'<circle cx="465" cy="170" r="7" fill="{NEIGHBOR}" stroke="#f6f7f2" '
        'stroke-width="2"/>'
        f'<circle cx="380" cy="80" r="8" fill="{SEED}"/>'
    )
    labels = (
        _text(312, 84, "top", "t-tiny")
        + _text(312, 129, "mid", "t-tiny")
        + _text(312, 174, "all", "t-tiny")
    )
    body = (
        _text(20, 26, "Exact scan", "t-ink")
        + _text(20, 46, "compares with every nucleus", "t-small")
        + scan
        + _text(320, 26, "HNSW index", "t-ink")
        + _text(320, 46, "hops down the layers", "t-small")
        + graph
        + labels
        + _text(170, 208, "visits every dot", "t-small t-mid")
        + _text(465, 208, "visits a few dots", "t-small t-mid")
    )
    return _svg(
        name,
        218,
        "An exact scan compares one nucleus with every other nucleus. An HNSW "
        "index hops down layers of links and visits only a few",
        body,
    )


def three_maps() -> str:
    name = "fig-maps"
    strip = ""
    colors = ["#5c7cb8", "#7fbf9f", "#ffd66e", "#e1096f", "#03a0f1", "#8a9aa8"]
    for i in range(12):
        strip += (
            f'<rect x="28" y="{46 + i * 9}" width="64" height="7" rx="1" '
            f'fill="{colors[(i * 5) % len(colors)]}" fill-opacity="0.9"/>'
        )
    body = (
        strip
        + _text(60, 172, "384 numbers", "t-small t-mid")
        + _text(60, 188, "per nucleus", "t-small t-mid")
        + _arrow(name, 104, 90, 150, 90, color="#f6f7f2")
    )
    centers = (235, 365, 495)
    # PCA: a cloud stretched along one direction
    for px, py in _dots(3, 34, (-1, 1), (-1, 1)):
        x = centers[0] + px * 44
        y = 90 + px * -14 + py * 12
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{BLUE}"/>'
    body += (
        f'<line x1="{centers[0] - 52}" y1="105" x2="{centers[0] + 52}" y2="75" '
        'stroke="#f6f7f2" stroke-width="1.5" stroke-dasharray="4 3"/>'
    )
    # UMAP: dots along a curved ribbon
    for i in range(30):
        t = i / 29
        x = centers[1] - 46 + t * 92
        y = 90 + math.sin(t * math.pi * 2) * 26 + ((i * 7) % 5 - 2) * 2.2
        body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{DENSE}"/>'
    # t-SNE: a few tight clumps
    for cx, cy, seed in ((-26, -14, 5), (22, -10, 9), (0, 26, 13)):
        for px, py in _dots(seed, 11, (-1, 1), (-1, 1)):
            x = centers[2] + cx + px * 11
            y = 90 + cy + py * 11
            body += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{PINK}"/>'
    label_centers = (210, 365, 520)
    body += (
        _text(label_centers[0], 152, "PCA", "t-ink t-mid")
        + _text(label_centers[0], 171, "keeps the biggest", "t-small t-mid")
        + _text(
            label_centers[0],
            187,
            "spread (a straight view)",
            "t-small t-mid",
        )
        + _text(label_centers[1], 152, "UMAP", "t-ink t-mid")
        + _text(label_centers[1], 171, "keeps neighborhoods", "t-small t-mid")
        + _text(
            label_centers[1],
            187,
            "and some of the shape",
            "t-small t-mid",
        )
        + _text(label_centers[2], 152, "t-SNE", "t-ink t-mid")
        + _text(label_centers[2], 171, "keeps near neighbors", "t-small t-mid")
        + _text(label_centers[2], 187, "tight (makes clumps)", "t-small t-mid")
    )
    return _svg(
        name,
        204,
        "A vector of 384 numbers is squeezed into three numbers three ways: PCA "
        "stretches a cloud along one direction, UMAP draws a curved sheet, and "
        "t-SNE makes tight clumps",
        body,
    )


def neighbor_density() -> str:
    name = "fig-density"
    body = ""
    for cx, radius, halo, label_a, label_b in (
        (150, 30, DENSE, "6 neighbors close by", "short average distance: yellow"),
        (450, 66, SPARSE, "6 neighbors far away", "long average distance: blue"),
    ):
        body += (
            f'<circle cx="{cx}" cy="98" r="{radius + 20}" fill="{halo}" '
            'fill-opacity="0.16"/>'
        )
        for k in range(6):
            angle = math.radians(20 + k * 60)
            nx = cx + math.cos(angle) * radius
            ny = 98 + math.sin(angle) * radius
            body += (
                f'<line x1="{cx}" y1="98" x2="{nx:.1f}" y2="{ny:.1f}" '
                'stroke="#f6f7f2" stroke-opacity="0.6" stroke-width="1.5"/>'
                f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="5.5" fill="{NEIGHBOR}"/>'
            )
        body += f'<circle cx="{cx}" cy="98" r="8" fill="{SEED}"/>'
        body += _text(cx, 186, label_a, "t-ink t-mid t-small")
        body += _text(cx, 204, label_b, "t-small t-mid")
    body += _text(150, 22, "Dense neighborhood", "t-ink t-mid") + _text(
        450, 22, "Sparse neighborhood", "t-ink t-mid"
    )
    return _svg(
        name,
        216,
        "Neighbor density: a nucleus with six close neighbors has a dense "
        "neighborhood, and one with six far neighbors has a sparse one",
        body,
    )


def fusion(cp_dim: int, morphem_dim: int, fused_dim: int) -> str:
    name = "fig-fusion"
    cell = 6
    cp_label_y = 30
    cp_bar_y = 38
    morphem_label_y = 98
    morphem_label_detail_y = 114
    morphem_bar_y = 126
    arrow_y = 78
    out_x = 296
    out_label_y = 58
    out_bar_y = 68
    out_caption_y = 112
    body = _text(20, cp_label_y, f"cp_measure: {cp_dim} numbers", "t-small")
    for i in range(cp_dim):
        body += (
            f'<rect x="{20 + i * cell}" y="{cp_bar_y}" width="{cell - 1}" height="16" '
            f'fill="{BLUE}" fill-opacity="0.9"/>'
        )
    body += _text(20, morphem_label_y, f"MorphEm: {morphem_dim} numbers", "t-small")
    body += _text(20, morphem_label_detail_y, "(384 reduced with PCA)", "t-small")
    for i in range(morphem_dim):
        body += (
            f'<rect x="{20 + i * cell}" y="{morphem_bar_y}" width="{cell - 1}" '
            'height="16" '
            f'fill="{PINK}" fill-opacity="0.9"/>'
        )
    body += _arrow(name, 226, arrow_y, 282, arrow_y, color="#f6f7f2")
    body += _text(
        out_x,
        out_label_y,
        f"Fused: {fused_dim} numbers, side by side",
        "t-small",
    )
    for i in range(fused_dim):
        color = BLUE if i < cp_dim else PINK
        body += (
            f'<rect x="{out_x + i * cell}" y="{out_bar_y}" '
            f'width="{cell - 1}" height="16" '
            f'fill="{color}" fill-opacity="0.9"/>'
        )
    body += _text(out_x, out_caption_y, "one vector per nucleus", "t-small")
    return _svg(
        name,
        158,
        f"The fused vector puts the {cp_dim} cp_measure numbers next to the "
        f"{morphem_dim} reduced MorphEm numbers, {fused_dim} in all",
        body,
    )


def is_and_is_not() -> str:
    name = "fig-not"
    left = ("An interactive experiment", "Data you can query in a browser",
            "Three maps to compare side by side")
    right = ("A benchmark score", "Biological ground truth",
             "A recommendation or a standard")
    body = (
        '<rect x="20" y="16" width="270" height="130" class="box"/>'
        '<rect x="310" y="16" width="270" height="130" class="box"/>'
        + _text(36, 44, "It is", "t-ink")
        + _text(326, 44, "It is not", "t-ink")
    )
    for i, words in enumerate(left):
        body += (
            f'<text x="36" y="{74 + i * 26}" class="mark-yes">&#10003;</text>'
            + _text(58, 74 + i * 26, words, "t-small")
        )
    for i, words in enumerate(right):
        body += (
            f'<text x="326" y="{74 + i * 26}" class="mark-no">&#10005;</text>'
            + _text(348, 74 + i * 26, words, "t-small")
        )
    return _svg(
        name,
        162,
        "This page is an interactive experiment. It is not a benchmark score, "
        "biological ground truth, or a recommended format",
        body,
    )


def what_it_carries() -> str:
    name = "fig-carries"
    body = (
        '<rect x="20" y="58" width="170" height="64" rx="4" class="box"/>'
        + _text(105, 86, "database.jpg", "t-ink t-mid")
        + _text(105, 106, "one file", "t-small t-mid")
    )
    rows = (
        (18, "Image viewer", "shows the picture", BLUE),
        (73, "DuckDB", "opens the database", DENSE),
        (128, "Web browser", "runs this page", PINK),
    )
    for y, head, sub, color in rows:
        body += (
            f'<rect x="340" y="{y}" width="240" height="44" rx="4" class="box"/>'
            + _text(360, y + 19, head, "t-ink")
            + _text(360, y + 36, sub, "t-small")
            + _arrow(name, 194, 90, 334, y + 22, color=color)
        )
    return _svg(
        name,
        188,
        "One file, database.jpg, opens as a picture in an image viewer, as a "
        "database in DuckDB, and as this interactive page in a web browser",
        body,
    )


def all_diagrams(
    jpeg_bytes: int, cp_dim: int, morphem_dim: int, fused_dim: int
) -> dict[str, str]:
    """Map each `__DIAGRAM_*__` placeholder in the page to its SVG."""

    return {
        "__DIAGRAM_POLYGLOT__": polyglot(),
        "__DIAGRAM_SIZE__": size_limit(jpeg_bytes),
        "__DIAGRAM_INDEX__": index_vs_scan(),
        "__DIAGRAM_MAPS__": three_maps(),
        "__DIAGRAM_DENSITY__": neighbor_density(),
        "__DIAGRAM_FUSION__": fusion(cp_dim, morphem_dim, fused_dim),
        "__DIAGRAM_NOT__": is_and_is_not(),
        "__DIAGRAM_CARRIES__": what_it_carries(),
    }
