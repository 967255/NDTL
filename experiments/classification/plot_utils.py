"""Shared matrix loading, cluster labels, and spectral coordinates."""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
from sklearn.manifold import SpectralEmbedding

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
    "#59a14f",
    "#edc948",
    "#af7aa1",
    "#76b7b2",
    "#e15759",
]


def read_matrix_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0][0] != "system":
        raise ValueError(f"Unexpected matrix header: {path}")
    names = rows[0][1:]
    matrix = np.zeros((len(names), len(names)), dtype=float)
    for i, row in enumerate(rows[1:]):
        if row[0] != names[i]:
            raise ValueError(f"Matrix row mismatch in {path}: {row[0]} vs {names[i]}")
        matrix[i, :] = [float(x) for x in row[1:]]
    return (names, matrix)


def display_name(name: str) -> str:
    return name.replace("_", " ")


def symmetrize_distance(matrix: np.ndarray, reject_negative: bool = True) -> np.ndarray:
    distance = 0.5 * (
        np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T
    )
    np.fill_diagonal(distance, 0.0)
    if np.any(~np.isfinite(distance)):
        raise ValueError("distance matrix contains non-finite values")
    if reject_negative and np.any(distance < -1e-12):
        raise ValueError("distance matrix contains negative values")
    return distance


def remap_labels_by_leaf_order(labels: np.ndarray, order: list[int]) -> np.ndarray:
    mapping: dict[int, int] = {}
    next_label = 1
    for idx in order:
        old = int(labels[int(idx)])
        if old not in mapping:
            mapping[old] = next_label
            next_label += 1
    return np.array([mapping[int(x)] for x in labels], dtype=int)


def cluster_size_profile(labels: np.ndarray) -> list[int]:
    return sorted(
        [int(np.sum(labels == lab)) for lab in np.unique(labels)], reverse=True
    )


def cluster_size_profile_text(labels: np.ndarray) -> str:
    return ";".join(str(size) for size in cluster_size_profile(labels))


def cluster_rgb_image(
    assignments, order: list[int], k_values: list[int] | None = None
) -> np.ndarray:
    rows = assignments if k_values is None else [assignments[k] for k in k_values]
    image = np.ones((len(rows), len(order), 3), dtype=float)
    for row, labels in enumerate(rows):
        for col, idx in enumerate(order):
            label = int(labels[int(idx)])
            image[row, col, :] = matplotlib.colors.to_rgb(
                PALETTE[(label - 1) % len(PALETTE)]
            )
    return image


def distance_to_affinity(
    distance: np.ndarray, sigma_scale: float
) -> tuple[np.ndarray, float]:
    upper = distance[np.triu_indices_from(distance, k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        raise ValueError("distance matrix has no positive off-diagonal values")
    sigma = float(np.median(positive) * sigma_scale)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    affinity = np.exp(-(distance**2) / (2.0 * sigma**2))
    np.fill_diagonal(affinity, 1.0)
    return (affinity, sigma)


def spectral_coordinates(affinity: np.ndarray, random_state: int) -> np.ndarray:
    coords = SpectralEmbedding(
        n_components=2, affinity="precomputed", random_state=random_state
    ).fit_transform(affinity)
    coords = np.asarray(coords, dtype=float)
    mins = coords.min(axis=0)
    spans = np.maximum(coords.max(axis=0) - mins, 1e-12)
    return 2.0 * ((coords - mins) / spans) - 1.0


def read_symmetric_matrix_csv(path):
    names, matrix = read_matrix_csv(path)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)
    if np.any(~np.isfinite(matrix)):
        raise ValueError("distance matrix contains non-finite values")
    return names, matrix
