"""Public API for CopyKAT-style GPU CNV inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import torch

from .backend import resolve_device
from .kernels import (
    chromosome_coverage_mask,
    cluster_ward_d2,
    copykat_bin_classification,
    aneuploidy_score,
    freeman_tukey_normalize,
    hg20_cell_cycle_genes,
    make_copykat_hg20_bins,
    make_genomic_bins,
    segment_profiles,
    segment_profiles_copykat_mcmc,
    smooth_ordered_expression,
    dlm_poly_smooth,
)
from .reference import load_gene_coordinates


@dataclass(frozen=True)
class CopyKATResult:
    """Results returned by :func:`copykat`."""

    prediction: pd.DataFrame
    cna: pd.DataFrame
    gene_cna: pd.DataFrame
    clusters: pd.Series
    breakpoints: pd.DataFrame
    device: str
    diagnostics: pd.DataFrame | None = None


def _read_expression(expression: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(expression, pd.DataFrame):
        matrix = expression.copy()
    else:
        path = Path(expression)
        matrix = pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",", index_col=0)
    if matrix.empty:
        raise ValueError("Expression matrix is empty.")
    if matrix.index.has_duplicates:
        matrix = matrix.groupby(level=0, sort=False).sum()
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if (matrix < 0).any().any():
        raise ValueError("Expression values must be non-negative.")
    return matrix


def _cluster_for_segmentation(
    smoothed: torch.Tensor,
    n_clusters: int,
    metric: str,
    seed: int,
) -> torch.Tensor:
    del seed
    return cluster_ward_d2(smoothed, max_clusters=n_clusters, min_cluster_size=10, metric=metric)


def _known_normal_indices(cells: pd.Index, known_normal_cells: Sequence[str] | None, device: torch.device) -> torch.Tensor:
    if not known_normal_cells:
        return torch.empty(0, device=device, dtype=torch.long)
    positions = cells.get_indexer(pd.Index(map(str, known_normal_cells)))
    positions = positions[positions >= 0]
    return torch.as_tensor(positions, device=device, dtype=torch.long)


def _baseline(
    smoothed: torch.Tensor,
    cells: pd.Index,
    labels: torch.Tensor,
    known_normal_cells: Sequence[str] | None,
    cell_line: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    normal_indices = _known_normal_indices(cells, known_normal_cells, smoothed.device)
    if cell_line:
        synthetic = smoothed.std(dim=1) * torch.randn(smoothed.shape[0], device=smoothed.device, dtype=smoothed.dtype)
        return synthetic, normal_indices
    if normal_indices.numel():
        return smoothed.index_select(1, normal_indices).median(dim=1).values, normal_indices
    unique_labels = torch.unique(labels, sorted=True)
    scores = torch.stack([smoothed[:, labels == label].median(dim=1).values.abs().median() for label in unique_labels])
    normal_indices = torch.where(labels == unique_labels[scores.argmin()])[0]
    return smoothed.index_select(1, normal_indices).median(dim=1).values, normal_indices


def _normal_mad_calls(scores: torch.Tensor, normal_indices: torch.Tensor, multiplier: float) -> tuple[torch.Tensor, torch.Tensor]:
    if normal_indices.numel() == 0:
        raise ValueError("normal_mad classification requires known_normal_cells or inferred normals.")
    normal_scores = scores.index_select(0, normal_indices)
    median = normal_scores.median()
    mad = (normal_scores - median).abs().median().clamp_min(torch.finfo(scores.dtype).eps)
    threshold = median + multiplier * 1.4826 * mad
    return scores >= threshold, threshold


def copykat(
    expression: str | Path | pd.DataFrame,
    gene_coordinates: str | Path | pd.DataFrame,
    *,
    gene_id_type: Literal["symbol", "ensembl"] = "symbol",
    known_normal_cells: Sequence[str] | None = None,
    cell_line: bool = False,
    min_genes_per_cell: int = 200,
    low_detection_rate: float = 0.05,
    segmentation_detection_rate: float = 0.10,
    genes_per_chromosome: int = 5,
    smoothing_window: int = 101,
    smoothing: Literal["dlm", "moving_average"] = "dlm",
    segmentation_window: int = 25,
    breakpoint_z: float = 1.5,
    ks_cut: float = 0.1,
    posterior_samples: int = 1000,
    segmentation: Literal["copykat_mcmc", "change"] = "copykat_mcmc",
    bin_size: int = 220_000,
    binning: Literal["copykat_hg20", "coordinate"] = "copykat_hg20",
    n_clusters: int = 6,
    distance: Literal["euclidean", "pearson", "emd"] = "euclidean",
    aneuploidy_threshold: float = 0.08,
    classification: Literal["normal_mad", "copykat_cluster", "threshold"] = "normal_mad",
    normal_mad_multiplier: float = 0.0,
    device: str | torch.device | None = "auto",
    dtype: torch.dtype = torch.float32,
    seed: int = 1234,
) -> CopyKATResult:
    """Infer CNVs with GPU numerical kernels and CopyKAT-style post-processing.

    CopyKAT's Poisson-Gamma segmentation is executed with batched GPU Gamma
    posterior sampling and empirical KS break detection. Quality control,
    supplied-normal baselines, bin re-centering, chromosome coverage filtering,
    and cluster diagnostics are retained. ``normal_mad`` is the default
    classifier because it calibrates the continuous score to the current
    normal-reference distribution. Its default median cutoff preserves
    CopyKAT's sensitivity to broad, low-amplitude CNAs while remaining
    data-adaptive; raise ``normal_mad_multiplier`` for a stricter cutoff.
    """
    torch.manual_seed(seed)
    active_device = resolve_device(device)
    counts = _read_expression(expression)
    input_cells = pd.Index(counts.columns, name="cell")
    coordinates = load_gene_coordinates(gene_coordinates, gene_id_type=gene_id_type)

    counts = counts.loc[:, (counts > 0).sum(axis=0) >= min_genes_per_cell]
    if counts.shape[1] == 0:
        raise ValueError("No cells pass min_genes_per_cell.")
    counts = counts.loc[(counts > 0).mean(axis=1) >= low_detection_rate]
    if counts.shape[0] == 0:
        raise ValueError("No genes pass low_detection_rate.")
    joined = coordinates.set_index("gene").join(counts, how="inner")
    if joined.empty:
        raise ValueError("Expression genes and coordinate reference have no overlap.")
    joined = joined.assign(
        _abspos=(joined["start"] + joined["end"]) / 2,
        _gene_order=joined.index.astype(str),
    ).sort_values(["chromosome", "_abspos", "_gene_order"], kind="stable").drop(
        columns=["_abspos", "_gene_order"]
    )
    valid_chromosomes = joined.groupby("chromosome").size().loc[lambda sizes: sizes >= genes_per_chromosome].index
    joined = joined.loc[joined["chromosome"].isin(valid_chromosomes)]
    excluded_genes = hg20_cell_cycle_genes()
    joined = joined.loc[
        ~joined.index.isin(excluded_genes)
        & ~joined.index.to_series().str.startswith("HLA-").to_numpy()
    ]
    if joined.empty:
        raise ValueError("No annotated chromosomes pass genes_per_chromosome.")

    def expression_tensor(table: pd.DataFrame) -> tuple[pd.Index, torch.Tensor, torch.Tensor]:
        cell_index = pd.Index(table.columns[3:], name="cell")
        values = torch.as_tensor(table.loc[:, cell_index].to_numpy(dtype=np.float32, copy=True), device=active_device, dtype=dtype)
        chromosome = torch.as_tensor(table["chromosome"].to_numpy(), device=active_device)
        return cell_index, values, chromosome

    cells, values, chromosome = expression_tensor(joined)
    primary_mask = chromosome_coverage_mask(values, chromosome, genes_per_chromosome)
    cells = cells[primary_mask.detach().cpu().numpy()]
    joined = joined.loc[:, ["chromosome", "start", "end", *cells]]
    cells, values, chromosome = expression_tensor(joined)
    if values.shape[1] == 0:
        raise ValueError("No cells pass chromosome-coverage quality control.")

    normalized = freeman_tukey_normalize(values)
    smoothed = (
        dlm_poly_smooth(normalized)
        if smoothing == "dlm"
        else smooth_ordered_expression(normalized, window=smoothing_window)
    )
    segmentation_clusters = _cluster_for_segmentation(smoothed, min(n_clusters, smoothed.shape[1]), distance, seed)
    baseline, _ = _baseline(smoothed, cells, segmentation_clusters, known_normal_cells, cell_line)
    relative = smoothed - baseline[:, None]
    retained_genes = (values > 0).float().mean(dim=1) >= segmentation_detection_rate
    if not retained_genes.any():
        retained_genes = torch.ones_like(retained_genes, dtype=torch.bool)
    joined = joined.iloc[retained_genes.detach().cpu().numpy()]
    cells, values, chromosome = expression_tensor(joined)
    relative = relative[retained_genes]

    final_cell_mask = chromosome_coverage_mask(values, chromosome, genes_per_chromosome)
    cells = cells[final_cell_mask.detach().cpu().numpy()]
    joined = joined.loc[:, ["chromosome", "start", "end", *cells]]
    cells, values, chromosome = expression_tensor(joined)
    relative = relative[:, final_cell_mask]
    if values.shape[1] == 0:
        raise ValueError("No cells pass segmentation chromosome-coverage quality control.")
    normal_indices = _known_normal_indices(cells, known_normal_cells, active_device)
    if normal_indices.numel() == 0:
        normal_indices = torch.where(segmentation_clusters[final_cell_mask] == segmentation_clusters[final_cell_mask][0])[0]

    if segmentation == "copykat_mcmc":
        segmented, breakpoint_indices = segment_profiles_copykat_mcmc(
            relative,
            segmentation_clusters[final_cell_mask],
            window=segmentation_window,
            ks_cut=ks_cut,
            posterior_samples=posterior_samples,
        )
    else:
        segmented, breakpoint_indices = segment_profiles(
            relative,
            segmentation_clusters[final_cell_mask],
            window=segmentation_window,
            z_threshold=breakpoint_z,
        )
    segmented = segmented - segmented.mean(dim=0, keepdim=True)
    if binning == "copykat_hg20":
        binned, copykat_bins = make_copykat_hg20_bins(
            segmented,
            joined.index.to_numpy(),
            chromosome,
            joined["start"].to_numpy(),
            joined["end"].to_numpy(),
        )
        bin_chromosome = torch.as_tensor(copykat_bins["chrom"].to_numpy(), device=active_device)
        bin_start = torch.as_tensor(copykat_bins["chrompos"].to_numpy(), device=active_device)
        bin_end = bin_start
    else:
        positions = torch.as_tensor(((joined["start"] + joined["end"]) // 2).to_numpy(), device=active_device)
        binned, bin_chromosome, bin_start, bin_end = make_genomic_bins(segmented, chromosome, positions, bin_size=bin_size)
    raw_scores = aneuploidy_score(segmented)
    adjusted_bins, final_clusters, cluster_calls, adjusted_scores, reference_cluster = copykat_bin_classification(
        binned,
        segmented,
        normal_indices,
        max_clusters=min(4, n_clusters),
        min_cluster_size=10,
        metric=distance,
        seed=seed,
    )
    if classification == "threshold":
        aneuploid_calls = raw_scores >= aneuploidy_threshold
        threshold = torch.tensor(aneuploidy_threshold, device=active_device, dtype=dtype)
    elif classification == "copykat_cluster":
        aneuploid_calls = cluster_calls
        threshold = torch.tensor(float("nan"), device=active_device, dtype=dtype)
    else:
        aneuploid_calls, threshold = _normal_mad_calls(raw_scores, normal_indices, normal_mad_multiplier)
    calls = np.where(aneuploid_calls.detach().cpu().numpy(), "aneuploid", "diploid")
    normal_position_array = normal_indices.detach().cpu().numpy()
    calls[normal_position_array] = "diploid"

    final_prediction = pd.DataFrame(index=input_cells)
    final_prediction.index.name = "cell"
    final_prediction["copykat_prediction"] = "not.defined"
    final_prediction["aneuploidy_score"] = np.nan
    final_prediction["cluster"] = pd.Series(pd.NA, index=input_cells, dtype="Int64")
    final_prediction["is_reference_normal"] = input_cells.isin(cells[normal_position_array])
    final_prediction["is_reference_cluster"] = False
    final_prediction.loc[cells, "copykat_prediction"] = calls
    final_prediction.loc[cells, "aneuploidy_score"] = raw_scores.detach().cpu().numpy()
    final_prediction.loc[cells, "cluster"] = final_clusters.detach().cpu().numpy() + 1
    final_prediction.loc[cells, "is_reference_cluster"] = final_clusters.detach().cpu().numpy() == int(reference_cluster.detach().cpu())

    cna = pd.DataFrame(adjusted_bins.detach().cpu().numpy(), columns=cells)
    cna.insert(0, "end", bin_end.detach().cpu().numpy())
    cna.insert(0, "start", bin_start.detach().cpu().numpy())
    cna.insert(0, "chromosome", bin_chromosome.detach().cpu().numpy())
    gene_cna = pd.DataFrame(segmented.detach().cpu().numpy(), columns=cells)
    gene_cna.insert(0, "end", joined["end"].to_numpy())
    gene_cna.insert(0, "start", joined["start"].to_numpy())
    gene_cna.insert(0, "chromosome", joined["chromosome"].to_numpy())
    gene_cna.insert(0, "gene", joined.index.to_numpy())
    breakpoint_array = breakpoint_indices.detach().cpu().numpy()
    breakpoints = pd.DataFrame({"gene_index": breakpoint_array, "gene": joined.index.to_numpy()[np.minimum(breakpoint_array, len(joined) - 1)]})
    diagnostics = pd.DataFrame({
        "cell": cells,
        "cluster": final_clusters.detach().cpu().numpy() + 1,
        "is_reference_normal": cells.isin(cells[normal_position_array]),
        "is_reference_cluster": final_clusters.detach().cpu().numpy() == int(reference_cluster.detach().cpu()),
        "classification": classification,
        "classification_threshold": float(threshold.detach().cpu()),
    })
    clusters = pd.Series(final_clusters.detach().cpu().numpy() + 1, index=cells, name="cluster")
    return CopyKATResult(final_prediction, cna, gene_cna, clusters, breakpoints, str(active_device), diagnostics)
