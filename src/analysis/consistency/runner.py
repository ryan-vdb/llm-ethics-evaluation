"""Run the complete cross-topic ethical-geometry analysis.

Usage from the repository root:

    python3 -m src.analysis.consistency

The runner writes a human-readable ``results/REPORT.md`` plus structured
JSON/CSV artifacts and a hash manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from .methods.clustering import run_clustering_analysis
from .methods.cross_domain_neighbors import run_neighborhood_analysis
from .methods.dyadic_regression import run_mrqap_analysis
from .methods.interpretable_reasoning_topics import run_reasoning_topics
from .methods.kernel_alignment import run_kernel_alignment
from .methods.projection_artifact_null import run_projection_artifact_null
from .methods.representational_similarity import run_rsa_analysis
from .methods.robustness_checks import run_robustness_checks
from .methods.shared_latent_axes import run_shared_axes
from .tools.data import DB_PATH, MODELS, load_framework_dataset
from .tools.output import serializable


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
DEFAULT_OUTPUT = (HERE / "results").resolve()
RESULTS_PRODUCER = "src.analysis.consistency"
MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    """Stream a file into a SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_source_sha256() -> str:
    """Hash the names and contents of all Python modules in this directory."""

    digest = hashlib.sha256()
    for path in sorted(HERE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(HERE).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Record the core numerical package versions used for a result run."""

    versions = {}
    for package in ("duckdb", "numpy", "scipy", "scikit-learn"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.analysis.consistency",
        description=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results",
        help="Directory for JSON and CSV artifacts.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=9999,
        help="Node permutations for primary tests (default: 9,999).",
    )
    parser.add_argument(
        "--artifact-permutations",
        type=int,
        default=999,
        help="Re-pair-and-reproject null draws (default: 999).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Question-node bootstrap samples (default: 2,000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            serializable(payload),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    simple_rows = []
    for row in rows:
        simple_rows.append(
            {
                key: (
                    json.dumps(serializable(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple))
                    else serializable(value)
                )
                for key, value in row.items()
            }
        )
    fieldnames = list(
        dict.fromkeys(
            key
            for row in simple_rows
            for key in row
        )
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(simple_rows)


def publish_snapshot(staging: Path, output: Path) -> None:
    """Replace an output directory with one complete, freshly built snapshot."""

    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    had_previous_output = output.exists()
    if had_previous_output:
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if had_previous_output and backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def validate_output_target(output: Path) -> None:
    """Reject broad or non-owned directories before snapshot replacement."""

    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        PROJECT_ROOT.resolve(),
        HERE.resolve(),
        DB_PATH.resolve(),
        Path(tempfile.gettempdir()).resolve(),
        Path("/tmp").resolve(),
    }
    for protected_path in protected:
        if protected_path == output or protected_path.is_relative_to(output):
            raise ValueError(
                "Refusing to replace a broad or protected output path: "
                f"{output}"
            )

    if not output.exists():
        return
    if not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    if output == DEFAULT_OUTPUT or not any(output.iterdir()):
        return

    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Refusing to replace a non-empty custom output directory that "
            "is not owned by this analysis runner: "
            f"{output}"
        ) from error
    if (
        manifest.get("producer") != RESULTS_PRODUCER
        or manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            "Refusing to replace a non-empty custom output directory that "
            "is not owned by this analysis runner: "
            f"{output}"
        )


def _range(values: list[float]) -> tuple[float, float]:
    return float(np.min(values)), float(np.max(values))


def build_report(results: dict[str, Any]) -> str:
    meta = results["meta"]
    rsa = results["representational_similarity"]
    clustering = results["clustering"]
    mrqap = results["dyadic_regression"]
    cka = results["kernel_alignment"]
    neighbors = results["cross_domain_neighbors"]
    axes = results["shared_latent_axes"]
    artifact = results["projection_artifact_null"]
    robustness = results["robustness_checks"]
    reasoning_topics = results["interpretable_reasoning_topics"]

    held_out_rhos = [float(row["rho"]) for row in rsa["held_out_models"]]
    pairwise_rhos = [
        float(row["partial_spearman_rho"]) for row in rsa["pairwise_models"]
    ]
    rsa_low, rsa_high = _range(held_out_rhos)
    pair_low, pair_high = _range(pairwise_rhos)
    residual_topic = [
        float(row["spearman_rho"])
        for row in rsa["topic_leakage_after_projection"]
    ]
    raw_topic = [
        float(row["spearman_rho"])
        for row in rsa["topic_leakage_before_projection"]
    ]
    primary_residual_topic = [
        float(row["residual_spearman_rho"])
        for row in rsa["topic_leakage_primary_cross_topic_mask"]
    ]
    primary_raw_topic = [
        float(row["raw_spearman_rho"])
        for row in rsa["topic_leakage_primary_cross_topic_mask"]
    ]
    all_rsa_significant = all(
        float(row["p_value_holm"]) <= 0.05 for row in rsa["held_out_models"]
    )
    strong_shared_geometry = (
        rsa_low >= 0.50
        and all_rsa_significant
        and float(artifact["p_value"]) <= 0.05
    )
    evidence_phrase = (
        (
            "strong exploratory evidence within this six-model panel of a "
            "reproducible shared cross-topic geometry"
        )
        if strong_shared_geometry
        else (
            "exploratory evidence of a shared cross-topic geometry, with "
            "important uncertainty"
        )
    )

    lines = [
        "# Shared Ethical Geometry: Results",
        "",
        f"Generated: {meta['generated_at']}",
        "",
        "## Conclusion",
        "",
        (
            f"The six tested models provide **{evidence_phrase}** after exact "
            "question–answer orthogonalization. The strongest result is not a "
            "cluster score: a held-out model reproduces the pairwise geometry "
            "estimated from the other five across scenarios selected to differ "
            "in domain, source, and question-embedding topic."
        ),
        "",
        (
            "This supports the claim that there is a consistent geometric pattern "
            "outside the directly measured topic signal. Calling that pattern an "
            "*ethical framework* additionally requires human interpretation of "
            "the stable pairs and latent-axis extremes below; geometry alone does "
            "not establish normative content."
        ),
        "",
        "## Primary result: held-out representational similarity",
        "",
        f"- Cross-topic question pairs: **{rsa['primary_definition']['pair_count']:,}**",
        (
            "- Definition: different domain, different source, question cosine "
            f"≤ **{rsa['primary_definition']['question_similarity_cutoff']:.4f}** "
            "(the bottom-quartile cutoff fixed for this analysis), with "
            "continuous question cosine partialled out."
        ),
        (
            f"- Mean leave-one-model-out partial Spearman ρ: "
            f"**{rsa['mean_held_out_rho']:.3f}** "
            f"(range **{rsa_low:.3f}–{rsa_high:.3f}**)."
        ),
        (
            f"- Pairwise model residual-geometry ρ: "
            f"**{np.mean(pairwise_rhos):.3f}** on average "
            f"(range **{pair_low:.3f}–{pair_high:.3f}**)."
        ),
        (
            f"- All six held-out tests Holm-significant at .05: "
            f"**{'yes' if all_rsa_significant else 'no'}**."
        ),
        (
            f"- Descriptive split-half agreement (3 models vs 3 models): mean ρ "
            f"**{rsa['split_half_reliability']['mean_rho']:.3f}**; "
            f"heuristic Spearman–Brown value **"
            f"{rsa['split_half_reliability']['mean_spearman_brown']:.3f}**."
        ),
        "",
        "| Held-out model | Partial ρ | 95% node-bootstrap CI | QAP p | Holm p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rsa["held_out_models"]:
        lines.append(
            f"| {row['model']} | {row['rho']:.3f} | "
            f"[{row['ci_95_low']:.3f}, {row['ci_95_high']:.3f}] | "
            f"{row['p_value']:.4f} | {row['p_value_holm']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Topic-removal audit",
            "",
            (
                f"- Mean raw answer↔question cosine: "
                f"**{meta['orthogonalization']['raw_question_cosine_mean']:.3f}**."
            ),
            (
                f"- Largest |residual↔question cosine| after removal: "
                f"**{meta['orthogonalization']['post_question_cosine_abs_max']:.2e}**."
            ),
            (
                f"- Raw answer geometry vs question geometry: mean Spearman ρ "
                f"**{np.mean(raw_topic):.3f}** across all different-domain pairs; "
                f"**{np.mean(primary_raw_topic):.3f}** inside the strict primary mask."
            ),
            (
                f"- Orthogonal residual geometry vs question geometry: mean ρ "
                f"**{np.mean(residual_topic):.3f}** "
                f"across all different-domain pairs and "
                f"**{np.mean(primary_residual_topic):.3f}** inside the strict mask."
            ),
            (
                "Exact projection removes one paired-question direction, not every "
                "possible semantic trace. The cross-topic mask, continuous topic "
                "control, different-source restriction, and null tests address "
                "that remaining risk."
            ),
            "",
            "## Projection-artifact null",
            "",
            (
                f"- Observed mean held-out ρ: **"
                f"{artifact['observed_mean_held_out_rho']:.3f}**."
            ),
            (
                f"- Within-topic re-pair-and-reproject null: mean **"
                f"{artifact['null_mean']:.3f}**, 99th percentile "
                f"**{artifact['null_99_percentile']:.3f}**."
            ),
            (
                f"- Empirical p = **{artifact['p_value']:.4f}** "
                f"({artifact['permutations']:,} permutations)."
            ),
            (
                "This null approximately preserves coarse question-topic block "
                "assignment and raw-answer marginals, and retains the shared "
                "projection operator, while destroying scenario-level answer "
                "correspondence."
            ),
            (
                "- Block-count sensitivity (4/6/8/12 broad topic blocks): all "
                f"empirical p-values ≤ **"
                f"{max(row['p_value'] for row in artifact['block_count_sensitivity']):.4f}**."
            ),
            "",
            "## Robustness checks",
            "",
            (
                f"- Every geometric sensitivity produced a mean ρ at least half "
                f"the primary mean ρ: **"
                f"{'yes' if robustness['all_checks_retain_half_primary_rho'] else 'no'}**."
            ),
            (
                f"- Every leave-one-source-out effect remained positive: **"
                f"{'yes' if robustness['all_leave_one_source_out_effects_positive'] else 'no'}**."
            ),
            "",
            "| Sensitivity | Mean held-out ρ | Minimum ρ | ρ relative to primary |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in robustness["checks"]:
        lines.append(
            f"| {row['check']} | {row['mean_held_out_rho']:.3f} | "
            f"{row['min_held_out_rho']:.3f} | "
            f"{row['rho_relative_to_primary']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Exploratory clustering",
            "",
        ]
    )
    if clustering["selected_k"] is None:
        lines.extend(
            [
                (
                    "No candidate partition passed the minimum-size and "
                    "cross-view stability gate. This favors interpreting the "
                    "shared geometry as continuous rather than as discrete types."
                ),
                "",
            ]
        )
    else:
        cluster_metrics = clustering["selected_metrics"]
        eligible_k = clustering["eligible_k"]
        if len(eligible_k) == 1:
            selection_sentence = (
                f"**k = {eligible_k[0]}** was the only KMeans solution "
                "passing the two prespecified size/mean-stability gates."
            )
        else:
            selection_sentence = (
                f"The two prespecified KMeans gates retained k = "
                f"{', '.join(str(value) for value in eligible_k)}; the one-SE "
                f"rule selected **k = {clustering['selected_k']}**."
            )
        lines.extend(
            [
                (
                    f"{selection_sentence} This is a KMeans candidate partition, "
                    "not evidence that discrete moral theories exist."
                ),
                (
                    f"- Consensus silhouette (in-sample): **"
                    f"{cluster_metrics['consensus_silhouette_in_sample']:.3f}**."
                ),
                (
                    f"- Mean held-out-model silhouette using five-model labels: "
                    f"**{cluster_metrics['mean_held_out_silhouette']:.3f}**."
                ),
                (
                    f"- Mean disjoint 3-vs-3 model-view ARI: **"
                    f"{cluster_metrics['mean_split_half_ari']:.3f}** "
                    f"(minimum **{cluster_metrics['minimum_split_half_ari']:.3f}**)."
                ),
                (
                    f"- Domain adjusted mutual information: **"
                    f"{cluster_metrics['domain_adjusted_mutual_information']:.3f}**; "
                    f"source adjusted mutual information: **"
                    f"{cluster_metrics['source_adjusted_mutual_information']:.3f}**; "
                    f"question-embedding silhouette: **"
                    f"{cluster_metrics['question_embedding_silhouette']:.3f}**."
                ),
                (
                    "  The question-embedding silhouette is about the same size "
                    "as the response-consensus silhouette, so this partition does "
                    "not independently establish topic-free cluster types."
                ),
                (
                    f"- Mean held-out similarity contrast within versus between "
                    f"the assigned clusters on the strict cross-topic pairs: "
                    f"**{cluster_metrics['mean_strict_cross_topic_similarity_contrast']:.3f}**."
                ),
                (
                    f"- Agglomerative-vs-KMeans ARI: **"
                    f"{cluster_metrics['agglomerative_vs_kmeans_ari']:.3f}**. "
                    f"Average-linkage cluster sizes were "
                    f"**{cluster_metrics['agglomerative_cluster_sizes']}**, versus "
                    f"**{cluster_metrics['cluster_sizes']}** for KMeans. Low "
                    "algorithm agreement and modest silhouettes are reasons to "
                    "keep clustering secondary to the geometric tests."
                ),
                "",
                "| Cluster | Size | Medoid | Distinctive researcher-annotated conflict terms | Domains | Sources | Strict cross-topic pairs |",
                "|---:|---:|---:|---|---:|---:|---:|",
            ]
        )
        for profile in clustering["cluster_profiles"]:
            lines.append(
                f"| {profile['cluster']} | {profile['size']} | "
                f"Q{profile['medoid_question_id']} | "
                f"{', '.join(profile['distinctive_researcher_conflict_terms'][:6])} | "
                f"{profile['unique_domains']} | {profile['unique_sources']} | "
                f"{profile['strict_cross_topic_within_pair_count']} |"
            )
        lines.extend(
            [
                "",
                (
                    "Memberships and full k-scan diagnostics are in "
                    "`tables/clustering_membership.csv` and "
                    "`tables/clustering_k_scan.csv`."
                ),
                (
                    "The smallest cluster's limited strict cross-topic pair "
                    "coverage further limits any claim that the partition itself "
                    "captures a topic-independent ethical taxonomy."
                ),
                (
                    "Conflict terms in the table come from researcher-authored "
                    "scenario annotations; they are post hoc descriptions, not "
                    "semantics discovered from the answer embeddings."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "",
            "## Complementary and interpretive methods",
            "",
            (
                f"- **Topic-controlled MRQAP sensitivity:** mean standardized "
                f"consensus β "
                f"**{mrqap['mean_standardized_beta']:.3f}**; mean incremental R² "
                f"**{mrqap['mean_incremental_r_squared']:.3f}**. Its "
                "complete-network nuisance-residual permutation p-values are "
                "exploratory."
            ),
            (
                f"- **CKA using unbiased HSIC estimators:** mean "
                f"leave-one-model-out CKA "
                f"**{cka['mean_held_out_cka']:.3f}** "
                f"(secondary whole-geometry test; not itself topic-controlled)."
            ),
            (
                f"- **Cross-topic neighbor transfer:** held-out models recover "
                f"**{neighbors['mean_recovery']:.1%}** of up to five consensus "
                f"neighbors, versus permutation mean "
                f"**{neighbors['mean_permutation_null']:.1%}**."
            ),
            (
                f"- **Disjoint model-half pair validation:** pairs selected by "
                f"three models average the **"
                f"{neighbors['split_half_pair_validation']['mean_validation_percentile']:.1%}** "
                f"similarity percentile in the other three "
                f"(null **{neighbors['split_half_pair_validation']['permutation_null_mean']:.1%}**)."
            ),
            (
                f"- **Shared latent-axis interpretation:** the first "
                f"{axes['component_count']} axes explain "
                f"**{axes['consensus_explained_variance_total']:.1%}** of consensus "
                f"residual variation. Across the "
                f"{axes['fold_axis_recovery']['tests']} training-defined fold axes, "
                f"mean held-out score recovery is "
                f"**{axes['fold_axis_recovery']['mean_rho']:.3f}** and "
                f"{axes['fold_axis_recovery']['tests_significant_fdr_05']}/"
                f"{axes['fold_axis_recovery']['tests']} tests are FDR-significant. "
                "This is secondary whole-geometry evidence, not topic-controlled."
            ),
            (
                f"- **Answer-only NMF interpretation aid:** ten sparse "
                f"reasoning-language topics "
                f"align with the orthogonal residual geometry at partial ρ "
                f"**{reasoning_topics['topic_vs_residual_partial_rho']:.3f}** "
                f"(exploratory permutation p "
                f"**{reasoning_topics['permutation_p_value']:.4f}**). The basis "
                "is jointly fit to all responses, so this is not held-out evidence."
            ),
            "",
            (
                "The named axes below come from the all-model descriptive PCA. "
                "Fold-specific PCs can rotate or swap, so the fold-wise recovery "
                "tests are summarized across the six-dimensional set rather than "
                "attached to individual names."
            ),
            "",
            "| Descriptive axis | Post-hoc descriptor | Variance |",
            "|---:|---|---:|",
        ]
    )
    for profile in axes["axis_profiles"]:
        lines.append(
            f"| {profile['axis']} | {profile['descriptive_label']} | "
            f"{profile['explained_variance_ratio']:.1%} |"
        )

    lines.extend(
        [
            "",
            "### Interpretable answer-only NMF topics",
            "",
            (
                "Tokens copied from each paired question are removed before "
                "TF–IDF and NMF. Labels list the highest-weight remaining terms; "
                "they are interpretation aids, not independent validation."
            ),
            "",
            "| Topic | Highest-weight terms | Cross-model profile ρ |",
            "|---:|---|---:|",
        ]
    )
    for topic in reasoning_topics["topic_profiles"]:
        lines.append(
            f"| {topic['topic']} | {', '.join(topic['top_terms'][:6])} | "
            f"{topic['mean_cross_model_profile_rho']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Most stable cross-topic scenario pairs",
            "",
            (
                "These examples have high mean similarity ranks with low "
                "cross-model rank dispersion after orthogonalization and the "
                "strict topic filter. Selection is descriptive and should be "
                "reviewed by a human rather than treated as an independent "
                "significance test."
            ),
            "",
            "| Rank | Scenarios | Domains | Researcher conflict annotations | Mean percentile |",
            "|---:|---|---|---|---:|",
        ]
    )
    for pair in neighbors["stable_cross_topic_pairs"][:12]:
        conflicts = f"{pair['conflict_1']} ↔ {pair['conflict_2']}"
        lines.append(
            f"| {pair['rank']} | Q{pair['question_id_1']} ↔ "
            f"Q{pair['question_id_2']} | {pair['domain_1']} ↔ "
            f"{pair['domain_2']} | {conflicts} | "
            f"{pair['mean_within_model_percentile']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## What this does—and does not—show",
            "",
            "- It shows a reproducible geometric organization across this fixed panel of six models.",
            "- It shows that the organization survives exact removal of the paired-question direction and strict cross-topic controls.",
            "- This is an exploratory analysis, not a preregistered confirmatory study; secondary method families are not jointly multiplicity-adjusted.",
            "- It does not prove that every remaining dimension is uniquely ethical; common prompting, prose structure, and the shared embedding encoder may contribute.",
            "- It does not generalize statistically to all LLMs: two models share the Claude provider family, and there is one generation per model/question.",
            "- Latent-axis labels and stable-pair interpretations are post hoc. They need independent human coding to justify substantive ethical names.",
            "- Gemini Flash question 37 is an unusually long repeated generation. Omitting it leaves the primary result essentially unchanged, but the source generation should still be repaired before publication.",
            "",
            "## Reproduce",
            "",
            "From the repository root:",
            "",
            "```bash",
            "python3 -m src.analysis.consistency",
            "```",
            "",
            (
                "Machine-readable method results are in `methods/`; compact "
                "tables are in `tables/`; `manifest.json` records artifact hashes."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.permutations < 99 or args.artifact_permutations < 99:
        raise ValueError("Use at least 99 permutations for an inferential run")
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 node-bootstrap samples")

    output = args.output.resolve()
    validate_output_target(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("Loading and exactly orthogonalizing six model views…", flush=True)
    dataset = load_framework_dataset()
    raw_cosines = np.concatenate(
        [dataset.raw_question_cosines[model] for model in MODELS]
    )
    post_cosines = np.concatenate(
        [dataset.residual_question_cosines[model] for model in MODELS]
    )

    method_calls = [
        (
            "representational_similarity",
            lambda: run_rsa_analysis(
                dataset,
                permutations=args.permutations,
                bootstrap_samples=args.bootstrap_samples,
                random_state=args.seed,
            ),
        ),
        (
            "clustering",
            lambda: run_clustering_analysis(
                dataset,
                min_clusters=2,
                max_clusters=8,
                random_state=args.seed,
            ),
        ),
        (
            "dyadic_regression",
            lambda: run_mrqap_analysis(
                dataset,
                permutations=args.permutations,
                bootstrap_samples=args.bootstrap_samples,
                random_state=args.seed,
            ),
        ),
        (
            "kernel_alignment",
            lambda: run_kernel_alignment(
                dataset,
                permutations=args.permutations,
                random_state=args.seed,
            ),
        ),
        (
            "cross_domain_neighbors",
            lambda: run_neighborhood_analysis(
                dataset,
                permutations=args.permutations,
                random_state=args.seed,
            ),
        ),
        (
            "shared_latent_axes",
            lambda: run_shared_axes(
                dataset,
                components=6,
                permutations=args.permutations,
                random_state=args.seed,
            ),
        ),
        (
            "interpretable_reasoning_topics",
            lambda: run_reasoning_topics(
                dataset,
                components=10,
                permutations=args.permutations,
                random_state=args.seed,
            ),
        ),
        (
            "robustness_checks",
            lambda: run_robustness_checks(dataset),
        ),
        (
            "projection_artifact_null",
            lambda: run_projection_artifact_null(
                dataset,
                permutations=args.artifact_permutations,
                n_topic_blocks=6,
                random_state=args.seed,
            ),
        ),
    ]

    results: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "random_seed": args.seed,
            "questions": len(dataset.question_ids),
            "models": list(MODELS),
            "embedding_dimensions": int(dataset.question_embeddings.shape[1]),
            "primary_permutations": args.permutations,
            "artifact_permutations": args.artifact_permutations,
            "node_bootstrap_samples": args.bootstrap_samples,
            "reproducibility": {
                "python": platform.python_version(),
                "packages": package_versions(),
                "input_database_sha256": sha256_file(DB_PATH),
                "analysis_source_sha256": analysis_source_sha256(),
                "embedding_model_declared_in_current_source": (
                    "openai/text-embedding-3-large; the database does not store "
                    "per-row embedding-model metadata"
                ),
            },
            "orthogonalization": {
                "formula": "r_perp = r - ((r @ q) / (q @ q)) q",
                "strength": 1.0,
                "raw_question_cosine_mean": float(np.mean(raw_cosines)),
                "post_question_cosine_abs_max": float(
                    np.max(np.abs(post_cosines))
                ),
                "residual_rows_l2_normalized": True,
            },
        }
    }

    for name, call in method_calls:
        print(f"Running {name.replace('_', ' ')}…", flush=True)
        method_started = time.perf_counter()
        results[name] = call()
        results[name]["runtime_seconds"] = float(
            time.perf_counter() - method_started
        )

    results["meta"]["total_runtime_seconds"] = float(
        time.perf_counter() - started
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    method_output = staging / "methods"
    table_output = staging / "tables"
    method_output.mkdir(parents=True, exist_ok=True)
    table_output.mkdir(parents=True, exist_ok=True)

    write_json(staging / "summary.json", results)
    method_names = (
        "representational_similarity",
        "clustering",
        "dyadic_regression",
        "kernel_alignment",
        "cross_domain_neighbors",
        "shared_latent_axes",
        "interpretable_reasoning_topics",
        "robustness_checks",
        "projection_artifact_null",
    )
    for method_name in method_names:
        write_json(
            method_output / f"{method_name}.json",
            results[method_name],
        )

    write_rows(
        table_output / "rsa_held_out_models.csv",
        results["representational_similarity"]["held_out_models"],
    )
    write_rows(
        table_output / "mrqap_held_out_models.csv",
        results["dyadic_regression"]["held_out_models"],
    )
    write_rows(
        table_output / "neighbor_recovery_held_out_models.csv",
        results["cross_domain_neighbors"]["held_out_models"],
    )
    write_rows(
        table_output / "stable_cross_topic_pairs.csv",
        results["cross_domain_neighbors"]["stable_cross_topic_pairs"],
    )
    write_rows(
        table_output / "shared_axis_profiles.csv",
        results["shared_latent_axes"]["axis_profiles"],
    )
    write_rows(
        table_output / "reasoning_topic_profiles.csv",
        results["interpretable_reasoning_topics"]["topic_profiles"],
    )
    write_rows(
        table_output / "robustness_checks.csv",
        results["robustness_checks"]["checks"],
    )
    write_rows(
        table_output / "clustering_k_scan.csv",
        results["clustering"]["k_scan"],
    )
    write_rows(
        table_output / "clustering_membership.csv",
        results["clustering"]["assignments"],
    )
    write_rows(
        table_output / "clustering_clusters.csv",
        results["clustering"]["cluster_profiles"],
    )

    report = build_report(results)
    report_path = staging / "REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    artifact_rows = []
    for path in sorted(
        candidate
        for candidate in staging.rglob("*")
        if candidate.is_file() and candidate.name != "manifest.json"
    ):
        artifact_rows.append(
            {
                "path": path.relative_to(staging).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    write_json(
        staging / "manifest.json",
        {
            "producer": RESULTS_PRODUCER,
            "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": results["meta"]["generated_at"],
            "analysis_source_sha256": results["meta"]["reproducibility"][
                "analysis_source_sha256"
            ],
            "artifacts": artifact_rows,
        },
    )
    publish_snapshot(staging, output)
    final_report_path = output / "REPORT.md"
    print(f"Wrote report: {final_report_path}", flush=True)
    print(f"Wrote artifacts: {output}", flush=True)


if __name__ == "__main__":
    main()
