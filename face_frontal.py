#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frontal face detector: hand-crafted 11-channel features + Depth-2 AdaBoost cascade + 68-point LBF regression.

Design goals: Readability first, performance is not the priority.
- Pyramid scaling uses cv2.resize directly, without fixed-point math or pointer patching.
- 11-channel features use numpy slicing to make "which pixel subtracts which" obvious at a glance.
- Cascade evaluation retains the stage early-rejection semantics, but every branch is explicitly written as if/else.
- Clustering uses a concise Disjoint-Set Union to demonstrate the SimilarRects(eps=0.1) concept, avoiding distraction by library function threshold semantics.

Usage:
  python face_detector_edu.py test6_480x640.jpg face_detector_cascade.bin out_edu.jpg \
      --min-size 48 --scale 1.2 --min-neighbors 2
"""

from __future__ import annotations

import argparse
import os
import struct
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

WIN_W = 24
WIN_H = 24
N_CHANNELS = 11

LANDMARK_POINTS = 68
LBF_STAGES = 5
FERNS_PER_STAGE = 340
FERN_LEAVES = 32
WEIGHT_ROW = 21760
WEIGHT_HALF = WEIGHT_ROW // 2
WEIGHT_SCALE = 0.0001  # DLL constant xmmword_18001BE80 = 0x38D1B717

# A fern node: 4 float offsets + int16 threshold + int32 left/right children (heap-style indices)
FERN_NODE_DTYPE = np.dtype(
    [("offset", "<f4", (4,)), ("thr", "<i2"), ("pad", "<i2"), ("left", "<i4"), ("right", "<i4")]
)


@dataclass(frozen=True)
class Node:
    # A node in a Depth-2 tree. The root uses left/right to point to child nodes; leaf nodes output left/right scores.
    x1: int
    y1: int
    x2: int
    y2: int
    ch: int
    thr: int
    left: int
    right: int
    score_left: int
    score_right: int


@dataclass
class Stage:
    threshold: int
    trees: List[Tuple[Node, Node, Node]]  # Each tree has exactly 3 nodes: root, left_child, right_child


def read_i32(buf: bytes, off: int) -> Tuple[int, int]:
    return struct.unpack_from("<i", buf, off)[0], off + 4


def load_cascade(path: str) -> List[Stage]:
    """Reads the cascade in the reverse-engineered format: global header (stages,24,24), followed by stages."""
    buf = open(path, "rb").read()
    off = 0
    n_stages, off = read_i32(buf, off)
    win_w, off = read_i32(buf, off)
    win_h, off = read_i32(buf, off)
    if (win_w, win_h) != (WIN_W, WIN_H):
        raise ValueError(f"unexpected window size: {win_w}x{win_h}")

    stages: List[Stage] = []
    for _ in range(n_stages):
        n_trees, off = read_i32(buf, off)
        trees: List[Tuple[Node, Node, Node]] = []
        for _ in range(n_trees):
            nodes: List[Node] = []
            for _ in range(3):
                vals = struct.unpack_from("<10i", buf, off)
                off += 40
                off += 16  # 16-byte padding at node tail, filled with pointers by the HPC version at runtime; not needed for the educational version.
                nodes.append(Node(*vals))
            off += 8  # Tree tail block: reject_threshold + reserved, not used in educational inference
            trees.append((nodes[0], nodes[1], nodes[2]))
        stage_thr, off = read_i32(buf, off)  # The first int32 of the stage tail block is the stage threshold
        off += 12  # The rest of the tail block bytes are dump residues, with no semantics
        stages.append(Stage(stage_thr, trees))
    return stages


def compute_11_channels(gray: np.ndarray) -> List[np.ndarray]:
    """
    11-channel hand-crafted features. Computed using int16 intermediates, clipped back to uint8 at the end.
    Channel meanings:
      ch0: Original grayscale
      ch1: 2x2 box mean
      ch2: Smoothing with a 2-pixel gap
      ch3/ch4: Horizontal/vertical short-range gradients of ch1
      ch5/ch6: Main/anti-diagonal short-range gradients of ch1
      ch7~ch10: Long-range gradients of ch2 spanning 4 pixels
    """
    h, w = gray.shape
    ch = [np.zeros((h, w), dtype=np.uint8) for _ in range(N_CHANNELS)]
    ch[0] = gray

    # ch1[y,x] = avg( ch0[y,x], ch0[y+1,x], ch0[y,x+1], ch0[y+1,x+1] )
    g = ch[0].astype(np.int16)
    ch[1][: h - 1, : w - 1] = (
        ((g[: h - 1, : w - 1] + g[1:h, : w - 1] + 1) // 2)
        + ((g[: h - 1, 1:w] + g[1:h, 1:w] + 1) // 2)
        + 1
    ) // 2

    c1 = ch[1].astype(np.int16)
    # Short-range feature source pixels: (y,x), (y,x+2), (y+2,x), (y+2,x+2)
    p00 = c1[: h - 3, : w - 3]
    p02 = c1[: h - 3, 2 : w - 1]
    p20 = c1[2 : h - 1, : w - 3]
    p22 = c1[2 : h - 1, 2 : w - 1]

    # ch2[y,x] = avg(avg(ch1[y,x], ch1[y+2,x]), avg(ch1[y+2,x+2], ch1[y,x+2]))
    ch[2][: h - 3, : w - 3] = (((p00 + p20) // 2) + ((p22 + p02 + 1) // 2) + 1) // 2

    # Gradients are uniformly mapped to [0,255]: (diff + 255) / 2
    ch[3][: h - 3, : w - 3] = (p02 - p00 + 255) // 2  # Horizontal
    ch[4][: h - 3, : w - 3] = (p20 - p00 + 255) // 2  # Vertical
    ch[5][: h - 3, : w - 3] = (p22 - p00 + 255) // 2  # Main diagonal
    ch[6][: h - 3, : w - 3] = (p20 - p02 + 255) // 2  # Anti-diagonal

    c2 = ch[2].astype(np.int16)
    # Long-range feature source pixels: (y,x), (y,x+4), (y+4,x), (y+4,x+4)
    q00 = c2[: h - 7, : w - 7]
    q04 = c2[: h - 7, 4 : w - 3]
    q40 = c2[4 : h - 3, : w - 7]
    q44 = c2[4 : h - 3, 4 : w - 3]
    ch[7][: h - 7, : w - 7] = (q04 - q00 + 255) // 2
    ch[8][: h - 7, : w - 7] = (q40 - q00 + 255) // 2
    ch[9][: h - 7, : w - 7] = (q44 - q00 + 255) // 2
    ch[10][: h - 7, : w - 7] = (q40 - q04 + 255) // 2
    return ch


def eval_window(ch: Sequence[np.ndarray], stages: Sequence[Stage], x: int, y: int) -> int:
    """
    Returns: >0 if passes all stages (final margin), 0 if rejected by stage 0, -k if rejected by stage k.
    Semantics match the engineering version: diff > thr goes right; sum all trees in a stage before checking against the threshold.
    """
    for si, stage in enumerate(stages):
        score_sum = 0
        for root, left_child, right_child in stage.trees:
            f = ch[root.ch]
            diff = int(f[y + root.y1, x + root.x1]) - int(f[y + root.y2, x + root.x2])
            node = right_child if diff > root.thr else left_child
            ff = ch[node.ch]
            d2 = int(ff[y + node.y1, x + node.x1]) - int(ff[y + node.y2, x + node.x2])
            score_sum += node.score_right if d2 > node.thr else node.score_left
        if score_sum < stage.threshold:
            return 0 if si == 0 else -si
        margin = score_sum - stage.threshold + 1
    return margin


def similar_rect(a: Sequence[int], b: Sequence[int], eps: float = 0.1) -> bool:
    # Same relative displacement concept as OpenCV SimilarRects: allows 10% bounding box size difference
    delta = min(a[2], b[2]) + min(a[3], b[3])
    return (
        abs(a[0] - b[0]) <= eps * delta
        and abs(a[1] - b[1]) <= eps * delta
        and abs(a[0] + a[2] - b[0] - b[2]) <= eps * delta
        and abs(a[1] + a[3] - b[1] - b[3]) <= eps * delta
    )


def group_rectangles(
    raw: Sequence[Tuple[int, int, int, int, int]], eps: float, min_neighbors: int
) -> List[Tuple[int, int, int, int, int, int]]:
    """Small Disjoint-Set Union clustering: The educational version doesn't pursue speed, clearly demonstrating how the neighbor count fuses overlapping boxes."""
    n = len(raw)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if similar_rect(raw[i][:4], raw[j][:4], eps):
                union(i, j)

    clusters: dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    out: List[Tuple[int, int, int, int, int, int]] = []
    for idxs in clusters.values():
        if len(idxs) < min_neighbors:
            continue
        xs = [raw[i][0] for i in idxs]
        ys = [raw[i][1] for i in idxs]
        ws = [raw[i][2] for i in idxs]
        hs = [raw[i][3] for i in idxs]
        ss = [raw[i][4] for i in idxs]
        out.append(
            (
                int(round(sum(xs) / len(idxs))),
                int(round(sum(ys) / len(idxs))),
                int(round(sum(ws) / len(idxs))),
                int(round(sum(hs) / len(idxs))),
                len(idxs),
                int(round(sum(ss) / len(idxs))),
            )
        )
    return out


def detect_faces(
    gray: np.ndarray,
    stages: Sequence[Stage],
    min_size: int = 48,
    max_size: int = 0,
    scale: float = 1.2,
    min_neighbors: int = 2,
) -> List[Tuple[int, int, int, int, int, int]]:
    """Returns [(x,y,w,h,neighbors,score)]. The educational version uses float scale and direct cv2.resize."""
    h, w = gray.shape
    start = max(WIN_W, min_size)
    limit = min(w, (WIN_W * h) // WIN_H)
    if max_size > 0:
        limit = min(limit, max_size)

    raw: List[Tuple[int, int, int, int, int]] = []  # x,y,w,h,margin
    win = start
    while win <= limit:
        s = win / WIN_W
        sw = max(WIN_W, int(round(w / s)))
        sh = max(WIN_H, int(round(h / s)))
        scaled = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
        ch = compute_11_channels(scaled)

        # The educational version skips step stride tricks: scans pixel by pixel to avoid engineering pitfalls like "missing faces on odd/even coordinates" distracting from understanding.
        for y in range(0, sh - WIN_H + 1):
            for x in range(0, sw - WIN_W + 1):
                ret = eval_window(ch, stages, x, y)
                if ret > 0:
                    raw.append((int(x * s), int(y * s), int(round(WIN_W * s)), int(round(WIN_H * s)), ret))

        nxt = int(round(win * scale))
        win = nxt if nxt > win else win + 1

    if not raw:
        return []
    return group_rectangles(raw, eps=0.1, min_neighbors=min_neighbors)


@dataclass
class LandmarkModels:
    mean_shape: np.ndarray  # (68,2) float32, mean shape, range roughly [-1,1]
    ferns: np.ndarray       # (5,340,31) structured nodes
    weights: np.ndarray     # (5,68,21760) int8, first 10880 are dx, last 10880 are dy


def load_landmark_models(mean_path: str, ferns_path: str, weights_path: str) -> LandmarkModels:
    mean_shape = np.fromfile(mean_path, dtype="<f4").reshape(LANDMARK_POINTS, 2)
    ferns = np.fromfile(ferns_path, dtype=FERN_NODE_DTYPE).reshape(LBF_STAGES, FERNS_PER_STAGE, 31)
    weights = np.fromfile(weights_path, dtype=np.int8).reshape(LBF_STAGES, LANDMARK_POINTS, WEIGHT_ROW)
    return LandmarkModels(mean_shape, ferns, weights)


def estimate_similarity(current: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Aligns the current shape to the mean shape using a 2x2 similarity transform, returns [s0,s1,s2,s3]."""
    cur0 = current - current.mean(axis=0, keepdims=True)
    base0 = mean - mean.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(cur0) / (np.linalg.norm(base0) + 1e-7)
    a = float((base0 * cur0).sum())                                         # A = u*x + v*y
    b = float((base0[:, 0] * cur0[:, 1] - base0[:, 1] * cur0[:, 0]).sum())  # B = u*y - v*x
    norm = np.hypot(a, b) + 1e-7
    cos_t, sin_t = a / norm, b / norm
    return np.array([scale * cos_t, -scale * sin_t, scale * sin_t, scale * cos_t], dtype=np.float32)


def extract_fern_indices(
    gray: np.ndarray, bbox: Tuple[int, int, int, int], current: np.ndarray, sim: np.ndarray, fern_stage: np.ndarray
) -> np.ndarray:
    """Computes leaf indices for 68 points * 5 ferns per point, returns (340,) int32."""
    h, w = gray.shape
    gi = gray.astype(np.int16)
    x, y, bw, bh = bbox
    half_w, half_h = bw * 0.5, bh * 0.5
    cx, cy = x + half_w, y + half_h
    s0w, s1w, s2h, s3h = half_w * sim[0], half_w * sim[1], half_h * sim[2], half_h * sim[3]

    indices = np.empty(LANDMARK_POINTS * 5, dtype=np.int32)
    out = 0
    for lm in range(LANDMARK_POINTS):
        anchor_x = half_w * current[lm, 0] + cx
        anchor_y = half_h * current[lm, 1] + cy
        for _ in range(5):
            node_idx = 0
            for _depth in range(5):
                node = fern_stage[out, node_idx]  # 'out' is the global fern ID: landmark*5 + fern_id
                f0, f1, f2, f3 = node["offset"]
                p1x = int(s0w * f0 + f1 * s1w + anchor_x)
                p1y = int(s3h * f1 + s2h * f0 + anchor_y)
                p2x = int(s0w * f2 + f3 * s1w + anchor_x)
                p2y = int(s3h * f3 + s2h * f2 + anchor_y)
                p1x = min(max(p1x, 0), w - 1)
                p1y = min(max(p1y, 0), h - 1)
                p2x = min(max(p2x, 0), w - 1)
                p2y = min(max(p2y, 0), h - 1)
                diff = int(gi[p1y, p1x]) - int(gi[p2y, p2x])
                node_idx = int(node["right"]) if diff >= int(node["thr"]) else int(node["left"])
            indices[out] = out * FERN_LEAVES + (node_idx - 31)
            out += 1
    return indices


def regress_landmarks(
    gray: np.ndarray, bbox: Tuple[int, int, int, int], models: LandmarkModels
) -> np.ndarray:
    """5-stage LBF/ESR regression, returns image coordinates (68,2) float32."""
    current = models.mean_shape.copy()
    for stage in range(LBF_STAGES):
        sim = estimate_similarity(current, models.mean_shape)
        leaf_idx = extract_fern_indices(gray, bbox, current, sim, models.ferns[stage])
        s0, s1, s2, s3 = sim * WEIGHT_SCALE
        for p in range(LANDMARK_POINTS):
            row = models.weights[stage, p]
            dx = row[:WEIGHT_HALF].astype(np.int32)[leaf_idx].sum()
            dy = row[WEIGHT_HALF:].astype(np.int32)[leaf_idx].sum()
            current[p, 0] += dx * s0 + dy * s1
            current[p, 1] += dx * s2 + dy * s3

    x, y, bw, bh = bbox
    half_w, half_h = bw * 0.5, bh * 0.5
    pts = np.empty_like(current)
    pts[:, 0] = half_w * current[:, 0] + x + half_w
    pts[:, 1] = half_h * current[:, 1] + y + half_h
    return pts


def draw_landmarks(color: np.ndarray, pts: np.ndarray) -> None:
    # BGR color scheme, matching the C++ example: Jaw/Brow/Nose/Eye/Lip groupings
    groups = [
        (0, 16, False, (208, 224, 72)),
        (17, 21, False, (64, 176, 255)),
        (22, 26, False, (96, 120, 255)),
        (27, 30, False, (80, 232, 255)),
        (31, 35, False, (64, 214, 255)),
        (36, 41, True, (255, 205, 80)),
        (42, 47, True, (255, 132, 190)),
        (48, 59, True, (122, 78, 255)),
        (60, 67, True, (198, 176, 255)),
    ]
    overlay = color.copy()
    for begin, end, closed, bgr in groups:
        poly = np.round(pts[begin : end + 1]).astype(np.int32)
        cv2.polylines(overlay, [poly], closed, bgr, 2, cv2.LINE_AA)
        for p in poly:
            cv2.circle(overlay, tuple(p), 2, bgr, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.78, color, 0.22, 0, color)


def default_landmark_paths(cascade_path: str) -> Tuple[str, str, str] | None:
    root = os.path.dirname(os.path.abspath(cascade_path)) or "."
    paths = (
        os.path.join(root, "face_landmark_mean_shape.bin"),
        os.path.join(root, "face_landmark_ferns.bin"),
        os.path.join(root, "face_landmark_weights.bin"),
    )
    return paths if all(os.path.exists(p) for p in paths) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("cascade")
    ap.add_argument("output")
    ap.add_argument("--min-size", type=int, default=48)
    ap.add_argument("--max-size", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.2)
    ap.add_argument("--min-neighbors", type=int, default=2)
    ap.add_argument("--mean-shape", default=None, help="Default infers from the cascade directory")
    ap.add_argument("--ferns", default=None, help="Default infers from the cascade directory")
    ap.add_argument("--weights", default=None, help="Default infers from the cascade directory")
    ap.add_argument("--no-landmark", action="store_true", help="Run AdaBoost detection only, skip 68-point regression")
    args = ap.parse_args()

    color = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if color is None:
        raise SystemExit(f"cannot read image: {args.image}")
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    stages = load_cascade(args.cascade)

    lm_models = None
    if not args.no_landmark:
        explicit = (args.mean_shape, args.ferns, args.weights)
        paths = explicit if all(explicit) else default_landmark_paths(args.cascade)
        if paths is not None:
            lm_models = load_landmark_models(*paths)

    faces = detect_faces(
        gray,
        stages,
        min_size=args.min_size,
        max_size=args.max_size,
        scale=args.scale,
        min_neighbors=args.min_neighbors,
    )
    print(f"{len(faces)} faces")
    for i, (x, y, w, h, nb, score) in enumerate(faces):
        print(f"face {i}: rect=[{x},{y},{w},{h}] neighbors={nb} score={score}")
        cv2.rectangle(color, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            color,
            f"n={nb} s={score}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        if lm_models is not None:
            pts = regress_landmarks(gray, (x, y, w, h), lm_models)
            draw_landmarks(color, pts)
            print(
                "  landmarks: eyeL=(%.1f,%.1f) eyeR=(%.1f,%.1f) nose=(%.1f,%.1f) mouth=(%.1f,%.1f)"
                % (pts[39, 0], pts[39, 1], pts[46, 0], pts[46, 1], pts[30, 0], pts[30, 1], pts[51, 0], pts[51, 1])
            )
    cv2.imwrite(args.output, color)


if __name__ == "__main__":
    main()
