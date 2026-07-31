"""Exploratory clustering of exact-orthogonal answer geometry.

This replaces the former per-model, strength-0.85 clustering script. A single
candidate partition is learned from the six-model consensus after exact
question projection removal. Candidate counts are evaluated with held-out
model views and disjoint model halves, while topic metadata is used only as a
diagnostic and never to choose the number of clusters.

Even a stable partition may be a reproducible cut through a continuous
manifold. These outputs are descriptive candidate partitions, not evidence of
discrete moral theories.
"""

from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

from ..tools.data import MODELS, FrameworkDataset
from ..tools.geometry import (
    cosine_matrix,
    cross_topic_mask,
    normalize_rows,
)


TOKEN_STOPWORDS = {
    "and",
    "ethical",
    "ethics",
    "individual",
    "personal",
    "professional",
    "public",
    "social",
    "the",
    "versus",
    "vs",
}


def consensus_features(
    dataset: FrameworkDataset,
    models: tuple[str, ...] = MODELS,
) -> np.ndarray:
    """Concatenate model residuals so dot products equal mean model cosine."""

    if not models:
        raise ValueError("At least one model is required")
    return np.concatenate(
        [dataset.residuals[model] for model in models],
        axis=1,
    ) / np.sqrt(len(models))


def _kmeans_labels(
    features: np.ndarray,
    clusters: int,
    random_state: int,
) -> np.ndarray:
    return KMeans(
        n_clusters=clusters,
        n_init=100,
        random_state=random_state,
    ).fit_predict(features)


def _symmetric_min_coassignment_retention(
    discovery: np.ndarray,
    validation: np.ndarray,
) -> float:
    """Least within-cluster pair co-assignment retained across two views."""

    def directional(first: np.ndarray, second: np.ndarray) -> float:
        strengths = []
        for cluster in np.unique(first):
            positions = np.flatnonzero(first == cluster)
            if len(positions) < 2:
                strengths.append(0.0)
                continue
            second_labels = second[positions]
            same = second_labels[:, None] == second_labels[None, :]
            triangle = np.triu_indices(len(positions), k=1)
            strengths.append(float(np.mean(same[triangle])))
        return float(np.min(strengths))

    return min(
        directional(discovery, validation),
        directional(validation, discovery),
    )


def _canonicalize_labels(
    labels: np.ndarray,
    similarity: np.ndarray,
    question_ids: np.ndarray,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    """Make arbitrary cluster IDs stable by ordering cluster medoid IDs."""

    medoids = {}
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        within = similarity[np.ix_(positions, positions)]
        medoid_position = int(positions[np.argmax(np.mean(within, axis=1))])
        medoids[int(label)] = medoid_position
    ordered = sorted(
        medoids,
        key=lambda label: int(question_ids[medoids[label]]),
    )
    mapping = {old: new for new, old in enumerate(ordered, start=1)}
    canonical = np.asarray([mapping[int(label)] for label in labels], dtype=int)
    canonical_medoids = {
        mapping[old]: medoid for old, medoid in medoids.items()
    }
    return canonical, mapping, canonical_medoids


def _distinctive_conflict_terms(
    dataset: FrameworkDataset,
    labels: np.ndarray,
    cluster: int,
    *,
    limit: int = 8,
) -> list[str]:
    """Recurring annotation terms enriched inside one cluster."""

    inside = np.flatnonzero(labels == cluster)
    outside = np.flatnonzero(labels != cluster)

    def counts(positions: np.ndarray) -> Counter[str]:
        output: Counter[str] = Counter()
        for position in positions:
            for token in re.findall(
                r"[A-Za-z][A-Za-z'-]+",
                dataset.conflicts[int(position)].lower(),
            ):
                if token not in TOKEN_STOPWORDS and len(token) > 2:
                    output[token] += 1
        return output

    inside_counts = counts(inside)
    outside_counts = counts(outside)
    scored = []
    for token, count in inside_counts.items():
        if count < 2:
            continue
        inside_rate = (count + 0.5) / (len(inside) + 1.0)
        outside_rate = (outside_counts[token] + 0.5) / (len(outside) + 1.0)
        scored.append((np.log(inside_rate / outside_rate), count, token))
    scored.sort(reverse=True)
    return [token for _, _, token in scored[:limit]]


def _topic_diagnostics(
    dataset: FrameworkDataset,
    labels: np.ndarray,
) -> dict[str, float]:
    question_silhouette = silhouette_score(
        dataset.question_embeddings,
        labels,
        metric="cosine",
    )
    return {
        "question_embedding_silhouette": float(question_silhouette),
        "domain_adjusted_mutual_information": float(
            adjusted_mutual_info_score(dataset.domains, labels)
        ),
        "source_adjusted_mutual_information": float(
            adjusted_mutual_info_score(dataset.sources, labels)
        ),
    }


def _evaluate_k(
    dataset: FrameworkDataset,
    *,
    clusters: int,
    strict_mask: np.ndarray,
    random_state: int,
) -> tuple[dict[str, object], np.ndarray]:
    full_features = consensus_features(dataset)
    full_similarity = full_features @ full_features.T
    labels = _kmeans_labels(full_features, clusters, random_state)
    cluster_sizes = np.bincount(labels, minlength=clusters)

    held_out_rows = []
    for model_index, held_out_model in enumerate(MODELS):
        training_models = tuple(
            model for model in MODELS if model != held_out_model
        )
        training_labels = _kmeans_labels(
            consensus_features(dataset, training_models),
            clusters,
            random_state,
        )
        held_out_labels = _kmeans_labels(
            dataset.residuals[held_out_model],
            clusters,
            random_state,
        )
        held_out_similarity = cosine_matrix(dataset.residuals[held_out_model])
        strict_triangle = np.triu(strict_mask, k=1)
        same_cluster = (
            training_labels[:, None] == training_labels[None, :]
        )
        within = held_out_similarity[strict_triangle & same_cluster]
        between = held_out_similarity[strict_triangle & ~same_cluster]
        held_out_rows.append(
            {
                "model": held_out_model,
                "silhouette_using_training_labels": float(
                    silhouette_score(
                        dataset.residuals[held_out_model],
                        training_labels,
                        metric="cosine",
                    )
                ),
                "held_out_view_partition_ari": float(
                    adjusted_rand_score(training_labels, held_out_labels)
                ),
                "strict_cross_topic_similarity_contrast": float(
                    np.mean(within) - np.mean(between)
                ),
                "strict_within_pair_count": int(len(within)),
                "strict_between_pair_count": int(len(between)),
            }
        )

    split_rows = []
    first_model = MODELS[0]
    for split_index, partners in enumerate(combinations(MODELS[1:], 2)):
        first_half = (first_model, *partners)
        second_half = tuple(
            model for model in MODELS if model not in first_half
        )
        first_labels = _kmeans_labels(
            consensus_features(dataset, first_half),
            clusters,
            random_state,
        )
        second_labels = _kmeans_labels(
            consensus_features(dataset, second_half),
            clusters,
            random_state,
        )
        split_rows.append(
            {
                "half_1": list(first_half),
                "half_2": list(second_half),
                "adjusted_rand_index": float(
                    adjusted_rand_score(first_labels, second_labels)
                ),
                "symmetric_min_coassignment_retention": (
                    _symmetric_min_coassignment_retention(
                        first_labels,
                        second_labels,
                    )
                ),
            }
        )

    sources = np.asarray(dataset.sources, dtype=object)
    source_omission_aris = []
    for source in sorted(set(dataset.sources)):
        keep = sources != source
        if np.sum(keep) <= clusters:
            continue
        omitted_labels = _kmeans_labels(
            full_features[keep],
            clusters,
            random_state,
        )
        source_omission_aris.append(
            float(adjusted_rand_score(labels[keep], omitted_labels))
        )

    distance = np.clip(1.0 - full_similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    agglomerative_labels = AgglomerativeClustering(
        n_clusters=clusters,
        metric="precomputed",
        linkage="average",
    ).fit_predict(distance)

    held_out_silhouettes = np.asarray(
        [
            row["silhouette_using_training_labels"]
            for row in held_out_rows
        ],
        dtype=np.float64,
    )
    split_aris = np.asarray(
        [row["adjusted_rand_index"] for row in split_rows],
        dtype=np.float64,
    )
    split_coassignment = np.asarray(
        [
            row["symmetric_min_coassignment_retention"]
            for row in split_rows
        ],
        dtype=np.float64,
    )
    diagnostics = _topic_diagnostics(dataset, labels)
    row = {
        "k": clusters,
        "cluster_sizes": cluster_sizes.tolist(),
        "minimum_cluster_size": int(np.min(cluster_sizes)),
        "consensus_silhouette_in_sample": float(
            silhouette_score(full_features, labels, metric="cosine")
        ),
        "mean_held_out_silhouette": float(np.mean(held_out_silhouettes)),
        "held_out_silhouette_standard_error": float(
            np.std(held_out_silhouettes, ddof=1)
            / np.sqrt(len(held_out_silhouettes))
        ),
        "mean_held_out_view_partition_ari": float(
            np.mean(
                [
                    row["held_out_view_partition_ari"]
                    for row in held_out_rows
                ]
            )
        ),
        "mean_strict_cross_topic_similarity_contrast": float(
            np.mean(
                [
                    row["strict_cross_topic_similarity_contrast"]
                    for row in held_out_rows
                ]
            )
        ),
        "mean_split_half_ari": float(np.mean(split_aris)),
        "minimum_split_half_ari": float(np.min(split_aris)),
        "median_split_min_coassignment_retention": float(
            np.median(split_coassignment)
        ),
        "minimum_split_min_coassignment_retention": float(
            np.min(split_coassignment)
        ),
        "minimum_leave_one_source_out_ari": float(
            np.min(source_omission_aris)
        ),
        "agglomerative_vs_kmeans_ari": float(
            adjusted_rand_score(labels, agglomerative_labels)
        ),
        "agglomerative_cluster_sizes": np.bincount(
            agglomerative_labels,
            minlength=clusters,
        ).tolist(),
        **diagnostics,
        "held_out_models": held_out_rows,
        "split_half_views": split_rows,
    }
    return row, labels


def run_clustering_analysis(
    dataset: FrameworkDataset,
    *,
    min_clusters: int = 2,
    max_clusters: int = 8,
    random_state: int = 42,
) -> dict[str, object]:
    """Find a parsimonious, view-stable candidate consensus partition."""

    if not 2 <= min_clusters <= max_clusters < len(dataset.question_ids):
        raise ValueError("Require 2 <= min_clusters <= max_clusters < questions")

    strict_mask, strict_cutoff = cross_topic_mask(
        dataset,
        question_similarity_quantile=0.25,
        require_different_source=True,
    )
    scan = []
    labels_by_k = {}
    for clusters in range(min_clusters, max_clusters + 1):
        row, labels = _evaluate_k(
            dataset,
            clusters=clusters,
            strict_mask=strict_mask,
            random_state=random_state,
        )
        scan.append(row)
        labels_by_k[clusters] = labels

    eligible = [
        row
        for row in scan
        if row["minimum_cluster_size"] >= 5
        and row["mean_split_half_ari"] >= 0.50
    ]
    if eligible:
        best = max(eligible, key=lambda row: row["mean_held_out_silhouette"])
        threshold = (
            best["mean_held_out_silhouette"]
            - best["held_out_silhouette_standard_error"]
        )
        plateau = [
            row for row in eligible
            if row["mean_held_out_silhouette"] >= threshold
        ]
        selected_k = int(min(row["k"] for row in plateau))
        selection_status = "KMeans candidate partition found"
    else:
        selected_k = None
        selection_status = "no partition passed the stability/size gate"

    assignments = []
    cluster_profiles = []
    if selected_k is not None:
        full_features = consensus_features(dataset)
        full_similarity = full_features @ full_features.T
        canonical, _, medoids = _canonicalize_labels(
            labels_by_k[selected_k],
            full_similarity,
            dataset.question_ids,
        )
        question_similarity = cosine_matrix(dataset.question_embeddings)
        strict_triangle = np.triu(strict_mask, k=1)
        for cluster in sorted(np.unique(canonical)):
            positions = np.flatnonzero(canonical == cluster)
            medoid_position = medoids[cluster]
            medoid_similarity = full_similarity[medoid_position, positions]
            exemplar_positions = positions[
                np.argsort(-medoid_similarity, kind="stable")[:3]
            ]
            source_counts = Counter(
                dataset.sources[position] for position in positions
            )
            within = np.ix_(positions, positions)
            strict_within = (
                strict_triangle
                & (canonical[:, None] == cluster)
                & (canonical[None, :] == cluster)
            )
            cluster_profiles.append(
                {
                    "cluster": int(cluster),
                    "size": int(len(positions)),
                    "medoid_question_id": int(
                        dataset.question_ids[medoid_position]
                    ),
                    "distinctive_researcher_conflict_terms": (
                        _distinctive_conflict_terms(
                            dataset,
                            canonical,
                            int(cluster),
                        )
                    ),
                    "unique_domains": len(
                        {dataset.domains[position] for position in positions}
                    ),
                    "unique_sources": len(source_counts),
                    "largest_source_share": float(
                        max(source_counts.values()) / len(positions)
                    ),
                    "mean_residual_similarity": float(
                        np.mean(
                            full_similarity[within][
                                np.triu_indices(len(positions), k=1)
                            ]
                        )
                    ),
                    "mean_question_similarity": float(
                        np.mean(
                            question_similarity[within][
                                np.triu_indices(len(positions), k=1)
                            ]
                        )
                    ),
                    "strict_cross_topic_within_pair_count": int(
                        np.sum(strict_within)
                    ),
                    "exemplar_questions": [
                        {
                            "question_id": int(dataset.question_ids[position]),
                            "domain": dataset.domains[position],
                            "conflict": dataset.conflicts[position],
                            "question": dataset.question_texts[position],
                        }
                        for position in exemplar_positions
                    ],
                }
            )
        centroids = {
            cluster: normalize_rows(
                np.mean(
                    full_features[canonical == cluster],
                    axis=0,
                    keepdims=True,
                )
            )[0]
            for cluster in np.unique(canonical)
        }
        for position, cluster in enumerate(canonical):
            assignments.append(
                {
                    "question_id": int(dataset.question_ids[position]),
                    "cluster": int(cluster),
                    "cosine_distance_to_cluster_centroid": float(
                        1.0 - np.dot(full_features[position], centroids[cluster])
                    ),
                    "domain": dataset.domains[position],
                    "source": dataset.sources[position],
                    "conflict": dataset.conflicts[position],
                    "question": dataset.question_texts[position],
                }
            )

    selected_metrics = (
        next(row for row in scan if row["k"] == selected_k)
        if selected_k is not None
        else None
    )
    return {
        "method": (
            "exploratory KMeans consensus clustering with held-out model-view "
            "stability diagnostics"
        ),
        "projection_strength": 1.0,
        "candidate_k_range": [min_clusters, max_clusters],
        "strict_question_similarity_cutoff": strict_cutoff,
        "strict_cross_topic_pair_count": int(
            np.sum(np.triu(strict_mask, k=1))
        ),
        "selection_rule": (
            "Among k with minimum cluster size >=5 and mean 3v3-model ARI "
            ">=0.50, choose the smallest k within one standard error of the "
            "best mean held-out silhouette. Topic metadata is not used."
        ),
        "eligible_k": [int(row["k"]) for row in eligible],
        "selection_status": selection_status,
        "selected_k": selected_k,
        "selected_metrics": selected_metrics,
        "k_scan": scan,
        "cluster_profiles": cluster_profiles,
        "assignments": assignments,
        "interpretation_warning": (
            "A stable partition can still be an arbitrary cut through a "
            "continuous semantic manifold. These are candidate partitions of "
            "this fixed response corpus, not discrete ethical frameworks."
        ),
    }
