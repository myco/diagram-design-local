#!/usr/bin/env python3
"""Run the repository's automated diagram gates on one generated file.

Wraps the two checkers SKILL.md §9 tells an agent to run — the packaged
``self_check.py`` (accessible-SVG contract, single-file safety) and
``scripts/verify-geometry.py`` (label masks clipped by later nodes) — and
adds the checks a model most often gets wrong: leftover template
placeholders, a viewBox off the preset, a node outside the canvas, a label
wider than its box, an arrow that starts or ends where no node is, a
connector through a node it does not join, two connectors crossing without a
hop, a node nothing connects to, and a planned node that was never drawn.
Every check is calibrated against the shipped examples. Findings are plain
sentences the model can act on in a repair round.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parent.parent
SELF_CHECK = ROOT / "skills" / "diagram-design" / "scripts" / "self_check.py"
GEOMETRY = ROOT / "scripts" / "verify-geometry.py"

PLACEHOLDERS = (
    "[diagram-slug]",
    "[Diagram title]",
    "[Type]",
    "[One sentence describing what the diagram shows]",
    "Draw arrows first, then nodes. Replace with your content.",
)
FENCE_RE = re.compile(r"```(?:html)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
VIEWBOX_RE = re.compile(r'viewBox="([^"]+)"')


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_html(reply: str) -> str | None:
    """The HTML document inside a model reply, fenced or bare."""
    fenced = FENCE_RE.findall(reply)
    candidates = [block for block in fenced if "<svg" in block.casefold()] or fenced
    if candidates:
        text = max(candidates, key=len).strip()
    else:
        text = reply.strip()
    start = text.casefold().find("<!doctype html")
    if start < 0:
        start = text.casefold().find("<html")
    end = text.casefold().rfind("</html>")
    if start < 0 or end < 0:
        return None
    return text[start : end + len("</html>")] + "\n"


RECT_RE = re.compile(
    r'<rect\b[^>]*?\bx="(-?[\d.]+)"\s+y="(-?[\d.]+)"\s+width="([\d.]+)"\s+height="([\d.]+)"', re.IGNORECASE
)
SHAPE_RE = re.compile(r"<(rect|polygon|ellipse|circle)\b([^>]*)>", re.IGNORECASE)
ARROW_RE = re.compile(r'<(line|path|polyline)\b([^>]*\bmarker-end="url\(#[^"]+\)"[^>]*)>', re.IGNORECASE)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
TEXT_RE = re.compile(r'<text\b([^>]*)>([^<]*)</text>', re.IGNORECASE)
# Average advance of Geist at weight 600, as a fraction of font-size; deliberately
# low so only clear overflows are reported.
GLYPH_EM = 0.5
NODE_MIN = (60.0, 40.0)
ARROW_TOLERANCE = 24.0
HAIRLINE_RE = re.compile(r'<line\b([^>]*)/?>', re.IGNORECASE)
# Types whose arrows join boxes. Sequence messages end on lifelines and axis
# arrows on charts point into space, so the dangling-arrow check skips them.
ARROW_TYPES = frozenset({
    "architecture", "data-flow", "db-schema", "dependency", "deployment", "dp-integration",
    "er", "flowchart", "high-level", "it-state", "layers", "loop", "medallion", "nested",
    "org-chart", "process", "state", "swimlane", "tree", "uml-class",
})


def _attr(attrs: str, name: str) -> str | None:
    match = re.search(rf'\b{name}="([^"]*)"', attrs, re.IGNORECASE)
    return match.group(1) if match else None


def _shape_boxes(svg: str) -> list[tuple[float, float, float, float]]:
    """Bounding boxes of everything an arrow could legitimately end on."""
    boxes = []
    for tag, attrs in SHAPE_RE.findall(svg):
        tag = tag.lower()
        try:
            if tag == "rect":
                x, y, w, h = (float(_attr(attrs, k) or 0) for k in ("x", "y", "width", "height"))
            elif tag == "polygon":
                nums = [float(n) for n in NUM_RE.findall(_attr(attrs, "points") or "")]
                xs, ys = nums[0::2], nums[1::2]
                if not xs:
                    continue
                x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
            elif tag == "circle":
                cx, cy, r = (float(_attr(attrs, k) or 0) for k in ("cx", "cy", "r"))
                x, y, w, h = cx - r, cy - r, 2 * r, 2 * r
            else:
                cx, cy, rx, ry = (float(_attr(attrs, k) or 0) for k in ("cx", "cy", "rx", "ry"))
                x, y, w, h = cx - rx, cy - ry, 2 * rx, 2 * ry
        except ValueError:
            continue
        # Label masks are 8-14px tall; anything an arrow may end on is taller.
        # Circles are never masks, so a state machine's initial dot counts.
        if (w >= 24 and h >= 20) or (tag in ("circle", "ellipse") and w >= 8):
            boxes.append((x, y, w, h))
    return boxes


def _arrow_span(tag: str, attrs: str) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Start and end of an arrow, for the absolute-command paths the templates use."""
    try:
        if tag == "line":
            x1, y1, x2, y2 = (float(_attr(attrs, k) or 0) for k in ("x1", "y1", "x2", "y2"))
            return (x1, y1), (x2, y2)
        data = _attr(attrs, "d") if tag == "path" else _attr(attrs, "points")
        if not data or re.search(r"[a-zHV]", data):
            return None  # relative or H/V commands: the end point is not the last pair
        nums = [float(n) for n in NUM_RE.findall(data)]
        if len(nums) < 4:
            return None
        return (nums[0], nums[1]), (nums[-2], nums[-1])
    except ValueError:
        return None


def _legend_top(svg: str, min_x: float, width: float, min_y: float, height: float) -> float:
    """The y of the legend hairline (SKILL.md §6), below which arrows are samples."""
    best = None
    for attrs in HAIRLINE_RE.findall(svg):
        if "marker-end" in attrs:
            continue
        try:
            x1, y1, x2, y2 = (float(_attr(attrs, k) or 0) for k in ("x1", "y1", "x2", "y2"))
        except ValueError:
            continue
        if abs(y1 - y2) < 1 and abs(x2 - x1) >= 0.6 * width and y1 > min_y + height / 2:
            best = y1 if best is None else max(best, y1)
    # Without a hairline, only the bottom strip the legend would occupy is exempt.
    return best if best is not None else min_y + height - 60


def _near_box(point: tuple[float, float], boxes: list[tuple[float, float, float, float]]) -> bool:
    px, py = point
    for x, y, w, h in boxes:
        if x - ARROW_TOLERANCE <= px <= x + w + ARROW_TOLERANCE and y - ARROW_TOLERANCE <= py <= y + h + ARROW_TOLERANCE:
            return True
    return False


def _on_border(point: tuple[float, float], boxes: list[tuple[float, float, float, float]], slack: float = 8.0) -> bool:
    """Within `slack` of a box's outline — how an arrow into a zone lands."""
    px, py = point
    for x, y, w, h in boxes:
        inside = x - slack <= px <= x + w + slack and y - slack <= py <= y + h + slack
        deep = x + slack < px < x + w - slack and y + slack < py < y + h - slack
        if inside and not deep:
            return True
    return False


def _attached(point: tuple[float, float], leaves: list, zones: list) -> bool:
    """A connector end sits on a node, or on the outline of a zone."""
    return _near_box(point, leaves) or _on_border(point, zones)


def layout_findings(html: str, type_slug: str | None = None) -> list[str]:
    """Defects the shipped gates do not see: overflow and dangling arrows."""
    findings = []
    boxes = VIEWBOX_RE.findall(html)
    if not boxes:
        return findings
    try:
        min_x, min_y, width, height = (float(v) for v in boxes[0].split())
    except ValueError:
        return findings
    max_x, max_y = min_x + width, min_y + height
    svg_match = re.search(r"<svg\b.*?</svg>", html, re.DOTALL | re.IGNORECASE)
    svg = svg_match.group(0) if svg_match else html
    for x, y, w, h in RECT_RE.findall(svg):
        x, y, w, h = float(x), float(y), float(w), float(h)
        if w >= NODE_MIN[0] and h >= NODE_MIN[1] and (x + w > max_x + 0.5 or y + h > max_y + 0.5 or x < min_x - 0.5 or y < min_y - 0.5):
            findings.append(
                f"node rect at ({x:g},{y:g}) {w:g}x{h:g} extends beyond the viewBox {boxes[0]}; "
                "shrink the layout or its gaps so every node keeps a 40px margin"
            )
    nodes = [(float(x), float(y), float(w), float(h)) for x, y, w, h in RECT_RE.findall(svg)
             if float(w) >= NODE_MIN[0] and float(h) >= NODE_MIN[1]]
    for attrs, text in TEXT_RE.findall(svg):
        label = re.sub(r"\s+", " ", text).strip()
        size = _attr(attrs, "font-size")
        anchor = (_attr(attrs, "text-anchor") or "").lower()
        if not label or not size or anchor != "middle":
            continue
        try:
            fs, tx, ty = float(size), float(_attr(attrs, "x") or "nan"), float(_attr(attrs, "y") or "nan")
        except ValueError:
            continue
        if fs < 11 or tx != tx or ty != ty:
            continue
        holders = [n for n in nodes if n[0] <= tx <= n[0] + n[2] and n[1] <= ty <= n[1] + n[3]]
        if not holders:
            continue
        box = min(holders, key=lambda n: n[2] * n[3])
        estimate = len(label) * fs * GLYPH_EM
        if estimate > box[2] - 8:
            findings.append(
                f"text {label!r} at {fs:g}px is about {estimate:.0f}px wide but its node box is only {box[2]:g}px; "
                "widen the box, shorten the label, or break it into two <text> lines"
            )
    if type_slug not in ARROW_TYPES:
        return findings  # connector checks need a known node-and-arrow type
    # A backdrop covering the canvas is paper, not something an arrow can reach.
    shapes = [s for s in _shape_boxes(svg) if s[2] * s[3] < 0.9 * width * height]
    legend_top = _legend_top(svg, min_x, width, min_y, height)
    big = sorted({s for s in shapes if s[2] >= NODE_MIN[0] and s[3] >= NODE_MIN[1]})
    zones = [s for s in big if _is_container(s, big)]
    targets = [s for s in shapes if s not in zones]
    for tag, attrs in ARROW_RE.findall(svg):
        span = _arrow_span(tag.lower(), attrs)
        if span is None:
            continue
        start, end = span
        if start[1] >= legend_top - 2 and end[1] >= legend_top - 2:
            continue  # legend key sample
        if not _attached(end, targets, zones):
            findings.append(
                f"arrow ({tag.lower()} with marker-end) ends at ({end[0]:g},{end[1]:g}) where there is no node; "
                "extend it to the border of its target box or remove it"
            )
        if not _attached(start, targets, zones):
            findings.append(
                f"arrow ({tag.lower()} with marker-end) starts at ({start[0]:g},{start[1]:g}) in empty space; "
                "start it on the border of its source box"
            )
    findings.extend(_connector_findings(svg, shapes, legend_top, type_slug))
    return findings


def coverage_findings(html: str, names: list[str]) -> list[str]:
    """Planned nodes that never made it into the drawing."""
    svg_match = re.search(r"<svg\b.*?</svg>", html, re.DOTALL | re.IGNORECASE)
    svg = svg_match.group(0) if svg_match else html
    words = set(re.findall(r"[a-z0-9]+", " ".join(t for _, t in TEXT_RE.findall(svg)).casefold()))
    findings = []
    for name in names:
        tokens = [w for w in re.findall(r"[a-z0-9]+", str(name).casefold()) if len(w) >= 3] or re.findall(r"[a-z0-9]+", str(name).casefold())
        if tokens and not all(token in words for token in tokens):
            findings.append(f"planned node {name!r} does not appear anywhere in the diagram; draw it")
    return findings


CONNECTOR_RE = re.compile(r"<(line|path|polyline)\b([^>]*)>", re.IGNORECASE)
# Types where every node takes part in at least one connection. Layer stacks,
# nested containment and IT landscapes carry meaning by placement alone.
CONNECTED_TYPES = ARROW_TYPES - {"layers", "nested", "it-state", "swimlane", "medallion"}
CROSS_MARGIN = 6.0


PATH_CMD_RE = re.compile(r"([MLHVCSQTAZ])([^MLHVCSQTAZ]*)", re.IGNORECASE)
ARGS_PER_CMD = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


def _walk_path(data: str) -> list[list[tuple[float, float]]] | None:
    """Points of an absolute path, one list per pen-down stroke.

    Straight commands (L, H, V) add a point the caller can turn into a
    segment; curve commands (Q, C, A, …) only move the pen, so a rounded
    elbow contributes its endpoints but no straight piece. Relative
    commands return None: their geometry is not worth guessing.
    """
    strokes: list[list[tuple[float, float]]] = []
    current = (0.0, 0.0)
    for cmd, body in PATH_CMD_RE.findall(data):
        if cmd.islower():
            return None
        nums = [float(n) for n in NUM_RE.findall(body)]
        width = ARGS_PER_CMD[cmd]
        groups = [nums[i : i + width] for i in range(0, len(nums), width)] if width else [[]]
        for args in groups:
            if width and len(args) < width:
                break
            if cmd == "M":
                current = (args[0], args[1])
                strokes.append([current])
                cmd = "L"  # subsequent pairs are implicit line-tos
                continue
            if cmd == "Z":
                continue
            if cmd == "H":
                current = (args[0], current[1])
            elif cmd == "V":
                current = (current[0], args[0])
            else:
                current = (args[-2], args[-1])
            if not strokes:
                strokes.append([current])
            elif cmd in ("L", "H", "V"):
                strokes[-1].append(current)
            else:
                strokes[-1].append(("curve", current))  # type: ignore[arg-type]
    return strokes


def _connector_geometry(tag: str, attrs: str) -> tuple[list, list] | None:
    """(straight axis-aligned segments, stroke endpoints) of one connector."""
    try:
        if tag == "line":
            x1, y1, x2, y2 = (float(_attr(attrs, k) or 0) for k in ("x1", "y1", "x2", "y2"))
            strokes = [[(x1, y1), (x2, y2)]]
        elif tag == "polyline":
            nums = [float(n) for n in NUM_RE.findall(_attr(attrs, "points") or "")]
            strokes = [list(zip(nums[0::2], nums[1::2]))]
        else:
            walked = _walk_path(_attr(attrs, "d") or "")
            if walked is None:
                return None
            strokes = walked
    except ValueError:
        return None
    segments = []
    endpoints = []
    for stroke in strokes:
        points = [p[1] if p and p[0] == "curve" else p for p in stroke]  # type: ignore[index]
        if len(points) < 2:
            continue
        endpoints.extend([points[0], points[-1]])
        for a, b, raw in zip(points, points[1:], stroke[1:]):
            if isinstance(raw[0], str):
                continue  # curve: no straight piece
            if abs(a[0] - b[0]) < 1 or abs(a[1] - b[1]) < 1:
                segments.append((a, b))
    return segments, endpoints


def _encloses(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    x, y, w, h = outer
    return outer != inner and x <= inner[0] and y <= inner[1] and x + w >= inner[0] + inner[2] and y + h >= inner[1] + inner[3]


def _is_part(box: tuple[float, float, float, float], nodes: list[tuple[float, float, float, float]]) -> bool:
    """A header or row rect stacked inside a compound node of the same width."""
    return any(_encloses(other, box) and abs(other[0] - box[0]) < 1 and abs(other[2] - box[2]) < 1 for other in nodes)


def _is_container(box: tuple[float, float, float, float], nodes: list[tuple[float, float, float, float]]) -> bool:
    """A zone: it encloses a node that is not merely one of its own parts."""
    return any(_encloses(box, other) and not _is_part(other, [box]) for other in nodes)


def _node_name(svg: str, box: tuple[float, float, float, float]) -> str:
    x, y, w, h = box
    best = ("", 0.0)
    for attrs, text in TEXT_RE.findall(svg):
        try:
            tx, ty, fs = float(_attr(attrs, "x") or "nan"), float(_attr(attrs, "y") or "nan"), float(_attr(attrs, "font-size") or 0)
        except ValueError:
            continue
        if x <= tx <= x + w and y <= ty <= y + h and fs > best[1]:
            best = (re.sub(r"\s+", " ", text).strip(), fs)
    return best[0] or f"at ({x:g},{y:g})"


def _connector_findings(svg: str, shapes: list, legend_top: float, type_slug: str | None) -> list[str]:
    """A connector through a node it does not end on; a node nothing connects to."""
    findings: list[str] = []
    nodes = sorted({s for s in shapes if s[2] >= NODE_MIN[0] and s[3] >= NODE_MIN[1] and s[1] < legend_top})
    # A node holds a label; a rect with no text inside is a backdrop or a zone.
    leaves = [n for n in nodes if not _is_container(n, nodes) and not _is_part(n, nodes) and not _node_name(svg, n).startswith("at (")]
    segments = []
    owners: list[int] = []
    arrows: list[bool] = []
    endpoints = []
    unparsed = 0
    for index, (tag, attrs) in enumerate(CONNECTOR_RE.findall(svg)):
        geometry = _connector_geometry(tag.lower(), attrs)
        if geometry is None:
            unparsed += 1
            continue
        pieces, ends = geometry
        if not ends:
            continue
        if all(point[1] >= legend_top - 2 for point in ends):
            continue  # legend samples and the hairline itself
        span = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in ends for b in ends)
        if span < 12:
            continue  # tick marks and crow's-feet
        if any(all(x - 1 <= px <= x + w + 1 and y - 1 <= py <= y + h + 1 for px, py in ends) for x, y, w, h in leaves):
            continue  # a divider drawn inside one node (an entity's header rule)
        endpoints.extend(ends)
        segments.extend(pieces)
        owners.extend([index] * len(pieces))
        arrows.extend(["marker-end" in attrs] * len(pieces))

    for i, first in enumerate(leaves):
        for second in leaves[i + 1 :]:
            dx = min(first[0] + first[2], second[0] + second[2]) - max(first[0], second[0])
            dy = min(first[1] + first[3], second[1] + second[3]) - max(first[1], second[1])
            if dx > 4 and dy > 4 and not _encloses(first, second) and not _encloses(second, first):
                findings.append(
                    f"nodes {_node_name(svg, first)!r} and {_node_name(svg, second)!r} overlap by "
                    f"{dx:g}x{dy:g}px; space them at least 24px apart"
                )

    seen: set[tuple] = set()
    for (ax, ay), (bx, by) in segments:
        for box in leaves:
            x, y, w, h = box
            if _near_box((ax, ay), [box]) or _near_box((bx, by), [box]):
                continue  # this node is an endpoint of the connector
            if abs(ay - by) < 1:  # horizontal
                crosses = y + CROSS_MARGIN < ay < y + h - CROSS_MARGIN and min(ax, bx) < x + CROSS_MARGIN and max(ax, bx) > x + w - CROSS_MARGIN
            else:  # vertical
                crosses = x + CROSS_MARGIN < ax < x + w - CROSS_MARGIN and min(ay, by) < y + CROSS_MARGIN and max(ay, by) > y + h - CROSS_MARGIN
            if crosses and (ax, ay, bx, by, box) not in seen:
                seen.add((ax, ay, bx, by, box))
                findings.append(
                    f"connector segment ({ax:g},{ay:g})→({bx:g},{by:g}) passes through node {_node_name(svg, box)!r} "
                    "which is not one of its endpoints; route it around with a rounded elbow"
                )

    # Only true connectors take part: arrows, or the marker-less relationship
    # lines of ER-family types. Lane dividers and org-chart buses are structure.
    relational = type_slug in {"er", "db-schema", "uml-class"}
    counts = [arrows[k] or relational for k in range(len(segments))]
    crossings: set[tuple[float, float]] = set()
    for i, ((ax, ay), (bx, by)) in enumerate(segments):
        if abs(ay - by) >= 1 or not counts[i]:
            continue  # take each crossing from its horizontal member
        for j, ((cx, cy), (dx, dy)) in enumerate(segments):
            if owners[i] == owners[j] or abs(cx - dx) >= 1 or not counts[j]:
                continue
            if min(ax, bx) + 4 < cx < max(ax, bx) - 4 and min(cy, dy) + 4 < ay < max(cy, dy) - 4:
                point = (round(cx), round(ay))
                if point not in crossings:
                    crossings.add(point)
                    findings.append(
                        f"two connectors cross at ({point[0]},{point[1]}) with no hop; "
                        "add a bridge arc on one of them or reroute so they do not cross"
                    )

    # The orphan check needs every connector accounted for, so it stands down
    # when any path used relative commands the walker declines to interpret.
    if type_slug in CONNECTED_TYPES and len(endpoints) >= 4 and unparsed == 0:
        canvas_width = max((x + w for x, _, w, _ in nodes), default=0.0)
        for box in leaves:
            if box[2] > 0.45 * canvas_width:
                continue  # a band that wide is a note or a lane, not a node
            if not any(_near_box(point, [box]) for point in endpoints):
                findings.append(
                    f"node {_node_name(svg, box)!r} has no connector touching it; "
                    "draw its relationship or remove the node"
                )
    return findings


def structural_findings(html: str, expected_view_box: str | None = None) -> list[str]:
    findings = []
    for placeholder in PLACEHOLDERS:
        if placeholder in html:
            findings.append(f"template placeholder left in the file: {placeholder!r}")
    svgs = re.findall(r"<svg\b", html, re.IGNORECASE)
    if len(svgs) != 1:
        findings.append(f"expected exactly one <svg> element, found {len(svgs)}")
    boxes = VIEWBOX_RE.findall(html)
    if not boxes:
        findings.append("the <svg> has no viewBox attribute")
    elif expected_view_box:
        try:
            ex, ey, ew, eh = (int(float(v)) for v in expected_view_box.split())
            x, y, w, h = (int(float(v)) for v in boxes[0].split())
        except ValueError:
            findings.append(f"viewBox {boxes[0]!r} is not four numbers")
        else:
            if (x, y, w) != (ex, ey, ew) or not eh <= h <= eh + 120:
                findings.append(
                    f'viewBox is "{boxes[0]}" but the size preset requires "{expected_view_box}" '
                    "(height may grow by up to 120px for a legend strip)"
                )
    if "<script" in html.casefold() and "data-motion-mode" not in html:
        findings.append("static diagrams must not contain a <script> element")
    return findings


def gate_findings(path: Path) -> list[str]:
    findings = []
    try:
        self_check = _load(SELF_CHECK, "self_check")
        findings.extend(f"self_check: {error}" for error in self_check.verify(path))
    except Exception as exc:  # noqa: BLE001 - a crashed gate is itself a finding
        findings.append(f"self_check crashed: {exc}")
    if GEOMETRY.is_file():
        try:
            geometry = _load(GEOMETRY, "verify_geometry")
            findings.extend(f"geometry: {line}" for line in geometry.check(path))
        except Exception as exc:  # noqa: BLE001
            findings.append(f"verify-geometry crashed: {exc}")
    return findings


def verify_file(
    path: Path,
    expected_view_box: str | None = None,
    type_slug: str | None = None,
    planned_nodes: list[str] | None = None,
) -> list[str]:
    html = path.read_text(encoding="utf-8")
    findings = (
        structural_findings(html, expected_view_box)
        + coverage_findings(html, planned_nodes or [])
        + layout_findings(html, type_slug)
        + gate_findings(path)
    )
    return list(dict.fromkeys(findings))  # a box drawn as fill + stroke reports once


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: verify.py <diagram.html> [expected viewBox] [type]")
        return 2
    path = Path(sys.argv[1])
    findings = verify_file(path, sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else None)
    for finding in findings:
        print(f"  - {finding}")
    print(("FAIL " if findings else "OK ") + str(path))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
