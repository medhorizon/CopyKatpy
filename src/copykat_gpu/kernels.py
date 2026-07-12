"""Numerical kernels executed with PyTorch on CPU or CUDA."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional


@lru_cache(maxsize=1)
def _scipy_hierarchy():
    """Load SciPy only for the small CPU hierarchical-clustering stage."""
    try:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
    except ImportError as error:
        raise ImportError(
            "Ward.D2 CopyKAT clustering requires scipy. Install copykat-gpu with its base dependencies."
        ) from error
    return fcluster, linkage, squareform

def freeman_tukey_normalize(counts: torch.Tensor) -> torch.Tensor:
    """Apply CopyKAT's Freeman--Tukey transform and per-cell centering."""
    transformed = torch.log(torch.sqrt(counts) + torch.sqrt(counts + 1.0))
    return transformed - transformed.mean(dim=0, keepdim=True)

def smooth_ordered_expression(values: torch.Tensor, window: int = 101) -> torch.Tensor:
    """Smooth each cell along genomic gene order using a GPU convolution."""
    if window < 3 or values.shape[0] < 3:
        return values - values.mean(dim=0, keepdim=True)
    maximum = values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1
    window = min(window if window % 2 else window - 1, maximum)
    if window < 3:
        return values - values.mean(dim=0, keepdim=True)
    kernel = torch.ones((1, 1, window), dtype=values.dtype, device=values.device) / window
    padded = functional.pad(values.T.unsqueeze(1), (window // 2, window // 2), mode="replicate")
    smoothed = functional.conv1d(padded, kernel).squeeze(1).T
    return smoothed - smoothed.mean(dim=0, keepdim=True)


def dlm_poly_smooth(
    values: torch.Tensor,
    observation_variance: float = 0.16,
    evolution_variance: float = 0.001,
    initial_variance: float = 1e7,
) -> torch.Tensor:
    """GPU Rauch--Tung--Striebel smoother matching ``dlmModPoly(order=1)``.

    CopyKAT configures R ``dlm`` as a local-level model with observation noise
    ``dV=0.16``, random-walk evolution noise ``dW=0.001``, zero initial state,
    and the package default diffuse initial variance. Every state update is
    batched over cells and remains on the active PyTorch device.
    """
    n_genes, n_cells = values.shape
    if n_genes == 0:
        return values
    dtype, device = values.dtype, values.device
    observation = torch.as_tensor(observation_variance, device=device, dtype=dtype)
    evolution = torch.as_tensor(evolution_variance, device=device, dtype=dtype)
    covariance = torch.as_tensor(initial_variance, device=device, dtype=dtype)
    state = torch.zeros(n_cells, device=device, dtype=dtype)
    filtered_state = torch.empty_like(values)
    predicted_state = torch.empty_like(values)
    filtered_covariance = torch.empty(n_genes, device=device, dtype=dtype)
    predicted_covariance = torch.empty(n_genes, device=device, dtype=dtype)

    for index in range(n_genes):
        covariance = covariance + evolution
        predicted_state[index] = state
        predicted_covariance[index] = covariance
        innovation_variance = covariance + observation
        gain = covariance / innovation_variance
        state = state + gain * (values[index] - state)
        covariance = (1.0 - gain) * covariance
        filtered_state[index] = state
        filtered_covariance[index] = covariance

    smoothed = torch.empty_like(values)
    smoothed[-1] = filtered_state[-1]
    for index in range(n_genes - 2, -1, -1):
        gain = filtered_covariance[index] / predicted_covariance[index + 1]
        smoothed[index] = filtered_state[index] + gain * (smoothed[index + 1] - predicted_state[index + 1])
    return smoothed - smoothed.mean(dim=0, keepdim=True)

def pairwise_distance(values: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """Calculate a cell-by-cell distance matrix on the active device."""
    metric = metric.lower()
    cells_by_features = values.T
    if metric == "euclidean":
        return torch.cdist(cells_by_features, cells_by_features)
    if metric in {"pearson", "correlation"}:
        centered = cells_by_features - cells_by_features.mean(dim=1, keepdim=True)
        normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(torch.finfo(values.dtype).eps)
        return (1.0 - normalized @ normalized.T).clamp_min(0)
    if metric == "emd":
        ordered = torch.sort(cells_by_features, dim=1).values
        return torch.cdist(ordered, ordered, p=1) / ordered.shape[1]
    raise ValueError("metric must be 'euclidean', 'pearson', or 'emd'.")

def cluster_medoids(distance: torch.Tensor, n_clusters: int, max_iter: int = 100, seed: int = 1234) -> torch.Tensor:
    """Deterministic GPU k-medoids clustering."""
    n_cells = distance.shape[0]
    n_clusters = max(1, min(n_clusters, n_cells))
    generator = torch.Generator(device=distance.device)
    generator.manual_seed(seed)
    medoids = torch.randperm(n_cells, generator=generator, device=distance.device)[:n_clusters]
    for _ in range(max_iter):
        labels = distance[:, medoids].argmin(dim=1)
        new_medoids = medoids.clone()
        for cluster_index in range(n_clusters):
            members = torch.where(labels == cluster_index)[0]
            if members.numel() == 0:
                new_medoids[cluster_index] = distance[:, medoids].amin(dim=1).argmax()
                continue
            within = distance.index_select(0, members).index_select(1, members)
            new_medoids[cluster_index] = members[within.sum(dim=1).argmin()]
        if torch.equal(torch.sort(new_medoids).values, torch.sort(medoids).values):
            break
        medoids = new_medoids
    return distance[:, medoids].argmin(dim=1)


def cluster_ward_d2(
    values: torch.Tensor,
    max_clusters: int,
    min_cluster_size: int = 10,
    metric: str = "euclidean",
) -> torch.Tensor:
    """Apply R ``hclust(..., method='ward.D2')`` with CopyKAT's k policy.

    R's implementation clusters a precomputed Euclidean distance matrix with
    Ward.D2.  The pairwise matrix is computed on the requested Torch device;
    linkage construction and tree cutting are intentionally delegated to SciPy
    because this O(cells²) control path is small and must match R semantics.
    """
    n_cells = values.shape[1]
    if n_cells < 2:
        return torch.zeros(n_cells, dtype=torch.long, device=values.device)
    if metric != "euclidean":
        return cluster_medoids(pairwise_distance(values, metric=metric), n_clusters=min(max_clusters, n_cells))
    fcluster, linkage, squareform = _scipy_hierarchy()
    distances = pairwise_distance(values, metric="euclidean").detach().cpu().numpy().astype(np.float64, copy=False)
    linkage_matrix = linkage(squareform(distances, checks=False), method="ward", optimal_ordering=False)
    for cluster_count in range(min(max_clusters, n_cells), 1, -1):
        labels = fcluster(linkage_matrix, cluster_count, criterion="maxclust") - 1
        if np.bincount(labels, minlength=cluster_count).min() > min_cluster_size:
            return torch.as_tensor(labels, dtype=torch.long, device=values.device)
    return torch.zeros(n_cells, dtype=torch.long, device=values.device)


def cluster_ward_d(
    values: torch.Tensor,
    max_clusters: int,
    min_cluster_size: int = 10,
    metric: str = "euclidean",
) -> torch.Tensor:
    """Reproduce R ``hclust(..., method='ward.D')`` for CopyKAT bin calls.

    R's legacy Ward.D update operates on unsquared dissimilarities, unlike
    Ward.D2/SciPy ``ward``.  CopyKAT uses Ward.D for both bin-level trees, so
    this compact CPU control-path implementation retains that exact
    Lance--Williams recurrence while keeping distance construction on Torch.
    """
    n_cells = values.shape[1]
    if n_cells < 2:
        return torch.zeros(n_cells, dtype=torch.long, device=values.device)
    if metric != "euclidean":
        return cluster_medoids(pairwise_distance(values, metric=metric), n_clusters=min(max_clusters, n_cells))
    distances = pairwise_distance(values, metric="euclidean").detach().cpu().numpy().astype(np.float64, copy=False)
    distances = np.pad(distances, ((0, n_cells), (0, n_cells)), constant_values=np.inf)
    np.fill_diagonal(distances, np.inf)
    sizes = np.ones(2 * n_cells - 1, dtype=np.float64)
    members = [np.array([index], dtype=np.int64) for index in range(n_cells)] + [None] * (n_cells - 1)
    active = np.zeros(2 * n_cells - 1, dtype=bool)
    active[:n_cells] = True
    partitions: dict[int, np.ndarray] = {n_cells: np.arange(n_cells, dtype=np.int64)}

    for merge_index in range(n_cells - 1):
        flat_index = np.argmin(distances)
        left, right = np.unravel_index(flat_index, distances.shape)
        if left > right:
            left, right = right, left
        new = n_cells + merge_index
        merged_members = np.concatenate((members[left], members[right]))
        members[new] = merged_members
        active_indices = np.flatnonzero(active)
        remaining = active_indices[(active_indices != left) & (active_indices != right)]
        if remaining.size:
            denominator = sizes[left] + sizes[right] + sizes[remaining]
            updated = (
                (sizes[remaining] + sizes[left]) * distances[left, remaining]
                + (sizes[remaining] + sizes[right]) * distances[right, remaining]
                - sizes[remaining] * distances[left, right]
            ) / denominator
            distances[new, remaining] = updated
            distances[remaining, new] = updated
        distances[left, :] = np.inf
        distances[:, left] = np.inf
        distances[right, :] = np.inf
        distances[:, right] = np.inf
        distances[new, new] = np.inf
        active[left] = False
        active[right] = False
        active[new] = True
        sizes[new] = sizes[left] + sizes[right]
        cluster_count = n_cells - merge_index - 1
        if 2 <= cluster_count <= max_clusters:
            labels = np.empty(n_cells, dtype=np.int64)
            for label, node in enumerate(np.flatnonzero(active)):
                labels[members[node]] = label
            partitions[cluster_count] = labels

    for cluster_count in range(min(max_clusters, n_cells), 1, -1):
        labels = partitions[cluster_count]
        if np.bincount(labels, minlength=cluster_count).min() > min_cluster_size:
            return torch.as_tensor(labels, dtype=torch.long, device=values.device)
    return torch.zeros(n_cells, dtype=torch.long, device=values.device)

def _boundary_scores(cluster_profiles: torch.Tensor, window: int) -> torch.Tensor:
    n_genes = cluster_profiles.shape[0]
    if n_genes < 2 * window:
        return torch.empty(0, device=cluster_profiles.device, dtype=cluster_profiles.dtype)
    windows = cluster_profiles.unfold(0, window, 1)
    left, right = windows[:-window].mean(dim=-1), windows[window:].mean(dim=-1)
    pooled = torch.cat((left, right), dim=1).std(dim=1, keepdim=True).clamp_min(1e-6)
    return (left - right).abs().div(pooled).amax(dim=1)

def segment_profiles(relative_expression: torch.Tensor, cluster_labels: torch.Tensor, window: int, z_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment profiles with GPU batched breakpoint detection and levels."""
    unique_labels = torch.unique(cluster_labels, sorted=True)
    cluster_profiles = torch.stack(
        [relative_expression[:, cluster_labels == label].median(dim=1).values for label in unique_labels],
        dim=1,
    )
    scores = _boundary_scores(cluster_profiles, window)
    candidate_breaks = torch.where(scores >= z_threshold)[0] + window
    n_genes = relative_expression.shape[0]
    breaks = torch.unique(torch.cat((torch.tensor([0, n_genes], device=relative_expression.device), candidate_breaks))).sort().values
    segmented = torch.empty_like(relative_expression)
    for start, end in zip(breaks[:-1].tolist(), breaks[1:].tolist()):
        segmented[start:end] = relative_expression[start:end].mean(dim=0, keepdim=True)
    return segmented, breaks

def make_genomic_bins(segmented: torch.Tensor, chromosome: torch.Tensor, positions: torch.Tensor, bin_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Median-aggregate segmented genes into fixed genomic bins."""
    chromosome, positions = chromosome.long(), positions.long()
    bin_ids = chromosome * (10**10) + torch.div(positions, bin_size, rounding_mode="floor")
    _, inverse = torch.unique_consecutive(bin_ids, return_inverse=True)
    bin_values, bin_chromosome, bin_start, bin_end = [], [], [], []
    for index in range(int(inverse.max()) + 1):
        mask = inverse == index
        bin_values.append(segmented[mask].mean(dim=0))
        bin_chromosome.append(chromosome[mask][0])
        bin_start.append(positions[mask].min())
        bin_end.append(positions[mask].max())
    return torch.stack(bin_values), torch.stack(bin_chromosome), torch.stack(bin_start), torch.stack(bin_end)


@lru_cache(maxsize=1)
def hg20_dna_bins() -> pd.DataFrame:
    """Load CopyKAT's bundled hg20 DNA target-bin reference."""
    path = Path(__file__).with_name("data") / "dna_hg20_bins.tsv"
    bins = pd.read_csv(path, sep="\t")
    return bins.loc[bins["chrom"].ne(24), ["chrom", "chrompos", "abspos"]].reset_index(drop=True)


@lru_cache(maxsize=1)
def hg20_cell_cycle_genes() -> frozenset[str]:
    """Load the R CopyKAT hg20 cell-cycle gene exclusion list."""
    path = Path(__file__).with_name("data") / "hg20_cell_cycle_genes.tsv"
    return frozenset(pd.read_csv(path, sep="\t")["gene"].astype(str))


def make_copykat_hg20_bins(
    segmented: torch.Tensor,
    gene_symbols: np.ndarray,
    chromosome: torch.Tensor,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[torch.Tensor, pd.DataFrame]:
    """Reproduce ``convert.all.bins.hg20`` fixed-bin median and fill rules."""
    dna_bins = hg20_dna_bins()
    gene_table = pd.DataFrame(
        {
            "gene": gene_symbols.astype(str),
            "chromosome": chromosome.detach().cpu().numpy(),
            "center": (starts.astype(np.int64) + ends.astype(np.int64)) / 2,
        }
    )
    bin_indices = np.full(len(gene_table), -1, dtype=np.int64)
    previous_end = 0.0
    for bin_index, (chromosome_value, end) in enumerate(dna_bins[["chrom", "chrompos"]].itertuples(index=False, name=None)):
        mask = (
            gene_table["chromosome"].eq(chromosome_value)
            & gene_table["center"].ge(previous_end)
            & gene_table["center"].le(end)
        ).to_numpy()
        bin_indices[mask] = bin_index
        previous_end = float(end)

    output = torch.full(
        (len(dna_bins), segmented.shape[1]),
        torch.nan,
        dtype=segmented.dtype,
        device=segmented.device,
    )
    for bin_index in np.unique(bin_indices[bin_indices >= 0]):
        selected = torch.as_tensor(np.flatnonzero(bin_indices == bin_index), device=segmented.device)
        output[bin_index] = segmented.index_select(0, selected).median(dim=0).values

    missing = torch.isnan(output[:, 0])
    if missing.any():
        known = torch.where(~missing)[0]
        if known.numel() == 0:
            raise ValueError("No genes overlap CopyKAT hg20 DNA target bins.")
        targets = torch.where(missing)[0]
        nearest = known[(known[:, None] - targets[None, :]).abs().argmin(dim=0)]
        output[targets] = output.index_select(0, nearest)
    return output, dna_bins

def aneuploidy_score(binned_expression: torch.Tensor) -> torch.Tensor:
    """Robust per-cell genome-wide CNV amplitude."""
    medians = binned_expression.median(dim=0, keepdim=True).values
    return (binned_expression - medians).abs().median(dim=0).values


def _copykat_cluster_labels(
    values: torch.Tensor,
    max_clusters: int = 4,
    min_cluster_size: int = 10,
    metric: str = "euclidean",
    seed: int = 1234,
) -> torch.Tensor:
    """Cluster cells using GPU distances and CopyKAT's 4-to-2 cluster policy."""
    return cluster_ward_d2(
        values,
        max_clusters=max_clusters,
        min_cluster_size=min_cluster_size,
        metric=metric,
    )


def _reference_cluster(labels: torch.Tensor, normal_indices: torch.Tensor) -> torch.Tensor:
    """Return the cluster enriched for supplied normal-reference cells."""
    unique_labels = torch.unique(labels, sorted=True)
    if normal_indices.numel() == 0:
        return unique_labels[0]
    normal_labels = labels.index_select(0, normal_indices)
    fractions = torch.stack([(normal_labels == label).float().mean() for label in unique_labels])
    return unique_labels[fractions.argmax()]


def wasserstein_1d(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Equal-weight 1D Wasserstein distance used by CopyKAT's cluster decision."""
    return (torch.sort(left).values - torch.sort(right).values).abs().mean()


def copykat_bin_classification(
    binned_expression: torch.Tensor,
    gene_expression: torch.Tensor,
    normal_indices: torch.Tensor,
    *,
    max_clusters: int = 4,
    min_cluster_size: int = 10,
    metric: str = "euclidean",
    seed: int = 1234,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate CopyKAT's bin-level re-baselining and cluster-based calling.

    GPU kernels perform distances, medians, profile shrinkage, and Wasserstein
    comparisons. The procedure follows the R workflow's initial normal-cluster
    selection, normal-profile re-centering, near-baseline shrinkage, and final
    cluster-to-diploid/aneuploid assignment.
    """
    initial_labels = cluster_ward_d(
        binned_expression,
        max_clusters=max_clusters,
        min_cluster_size=min_cluster_size,
        metric=metric,
    )
    initial_reference = _reference_cluster(initial_labels, normal_indices)
    initial_diploid = initial_labels == initial_reference
    if not initial_diploid.any():
        initial_diploid = torch.ones_like(initial_labels, dtype=torch.bool)

    reference_mean = binned_expression[:, initial_diploid].mean(dim=1, keepdim=True)
    recentered = binned_expression - reference_mean
    recentered = recentered - recentered.mean(dim=0, keepdim=True)
    normal_values = recentered[:, initial_diploid]
    normal_sd = normal_values.std(dim=1, unbiased=True, keepdim=True).clamp_min(1e-6)
    normal_mean = normal_values.mean(dim=1, keepdim=True)
    cell_means = recentered.mean(dim=0, keepdim=True)
    shrink_mask = (recentered - normal_mean).abs() <= 0.25 * normal_sd
    adjusted = torch.where(shrink_mask, cell_means.expand_as(recentered), recentered)
    adjusted = adjusted - adjusted.mean(dim=0, keepdim=True)

    final_labels = cluster_ward_d(
        adjusted,
        max_clusters=max_clusters,
        min_cluster_size=min_cluster_size,
        metric=metric,
    )
    reference_label = _reference_cluster(final_labels, normal_indices)
    unique_labels = torch.unique(final_labels, sorted=True)
    reference_profile = gene_expression[:, final_labels == reference_label].median(dim=1).values

    if unique_labels.numel() == 1:
        calls = torch.zeros_like(final_labels, dtype=torch.bool)
    else:
        normal_fractions = torch.stack([
            (final_labels.index_select(0, normal_indices) == label).float().mean()
            if normal_indices.numel()
            else torch.tensor(0.0, device=adjusted.device)
            for label in unique_labels
        ])
        aneuploid_label = unique_labels[normal_fractions.argmin()]
        aneuploid_profile = gene_expression[:, final_labels == aneuploid_label].median(dim=1).values
        reference_centered = reference_profile - reference_profile.mean()
        aneuploid_centered = aneuploid_profile - aneuploid_profile.mean()
        correlation = (reference_centered * aneuploid_centered).mean() / (
            reference_centered.std(unbiased=False).clamp_min(1e-6)
            * aneuploid_centered.std(unbiased=False).clamp_min(1e-6)
        )
        calls = torch.zeros_like(final_labels, dtype=torch.bool)
        if correlation < 0.6:
            reference_distance = wasserstein_1d(reference_profile, aneuploid_profile)
            for label in unique_labels:
                profile = gene_expression[:, final_labels == label].median(dim=1).values
                calls[final_labels == label] = wasserstein_1d(profile, reference_profile) >= wasserstein_1d(
                    profile, aneuploid_profile
                )
            calls[final_labels == reference_label] = False

    scores = aneuploidy_score(adjusted)
    return adjusted, final_labels, calls, scores, reference_label


def chromosome_coverage_mask(
    values: torch.Tensor,
    chromosome: torch.Tensor,
    minimum_genes: int,
) -> torch.Tensor:
    """Keep cells with sufficient detected genes on every represented chromosome."""
    chromosome = chromosome.long()
    unique_chromosomes, inverse = torch.unique_consecutive(chromosome, return_inverse=True)
    detected = values > 0
    coverage = torch.stack(
        [detected[inverse == index].sum(dim=0) for index in range(unique_chromosomes.numel())]
    )
    return coverage.min(dim=0).values >= minimum_genes


def _copykat_window_boundaries(n_genes: int, window: int, device: torch.device) -> torch.Tensor:
    """Reproduce CopyKAT's 1-indexed initial MCMC window boundaries."""
    if window < 1 or n_genes < 2:
        return torch.tensor([0, max(n_genes - 1, 0)], device=device, dtype=torch.long)
    stop = (n_genes // window - 1) * window
    starts = torch.arange(0, max(stop, 0), window, device=device, dtype=torch.long)
    if starts.numel() == 0 or starts[0] != 0:
        starts = torch.cat((torch.zeros(1, device=device, dtype=torch.long), starts))
    # Store zero-based *inclusive* positions.  R's ``breks`` contains one-based
    # gene indices and each final CNA segment is assigned over ``BR[i]:BR[i+1]``.
    return torch.unique(
        torch.cat((starts, torch.tensor([n_genes - 1], device=device, dtype=torch.long)))
    ).sort().values


def _gamma_posterior_samples(
    values: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """Sample CopyKAT's Gamma posterior for Poisson observations.

    R uses ``MCpoissongamma(y, alpha=mean(y), beta=1, mc=1000)``; its posterior
    is Gamma(mean(y) + sum(y), 1 + length(y)).
    """
    alpha = values.mean(dim=0).clamp_min(0.001)
    shape = alpha + values.sum(dim=0)
    rate = values.shape[0] + 1.0
    distribution = torch.distributions.Gamma(shape, torch.full_like(shape, rate))
    return distribution.rsample((sample_count,))


def gamma_posterior_mean(values: torch.Tensor) -> torch.Tensor:
    """Return CopyKAT's Poisson-Gamma posterior expectation per cell."""
    alpha = values.mean(dim=0).clamp_min(0.001)
    return (alpha + values.sum(dim=0)) / (values.shape[0] + 1.0)


def _two_sample_ks_from_gamma(
    left: torch.Tensor,
    right: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    """GPU empirical KS statistics for batched CopyKAT Gamma posteriors."""
    left_samples = _gamma_posterior_samples(left, sample_count).transpose(0, 1)
    right_samples = _gamma_posterior_samples(right, sample_count).transpose(0, 1)
    combined = torch.cat((left_samples, right_samples), dim=1)
    order = combined.argsort(dim=1)
    left_origin = order < sample_count
    left_cdf = left_origin.cumsum(dim=1).to(combined.dtype) / sample_count
    right_cdf = (~left_origin).cumsum(dim=1).to(combined.dtype) / sample_count
    return (left_cdf - right_cdf).abs().amax(dim=1)


def _copykat_mcmc_breaks(
    relative_expression: torch.Tensor,
    cluster_labels: torch.Tensor,
    window: int,
    ks_cut: float,
    posterior_samples: int,
) -> torch.Tensor:
    """Build CopyKAT's union of cluster-level MCMC/KS breakpoints on GPU."""
    n_genes = relative_expression.shape[0]
    boundaries = _copykat_window_boundaries(n_genes, window, relative_expression.device)
    if boundaries.numel() < 3:
        return torch.tensor([0, n_genes], device=relative_expression.device, dtype=torch.long)
    cluster_profiles = torch.stack(
        [relative_expression[:, cluster_labels == label].median(dim=1).values for label in torch.unique(cluster_labels, sorted=True)],
        dim=1,
    )
    expression = cluster_profiles.exp()
    selected = []
    for index in range(boundaries.numel() - 2):
        left_start, left_end, right_end = boundaries[index], boundaries[index + 1], boundaries[index + 2]
        # CopyKAT includes the shared boundary in the left window and starts the
        # right window at the following gene.
        left = expression[left_start : left_end + 1]
        right = expression[left_end + 1 : right_end + 1]
        if left.shape[1] == 0 or right.shape[1] == 0:
            continue
        statistics = _two_sample_ks_from_gamma(left, right, posterior_samples)
        if (statistics > ks_cut).any():
            selected.append(left_end)
    base = torch.tensor([0, n_genes - 1], device=relative_expression.device, dtype=torch.long)
    if selected:
        base = torch.cat((base, torch.stack(selected)))
    return torch.unique(base).sort().values


def segment_profiles_copykat_mcmc(
    relative_expression: torch.Tensor,
    cluster_labels: torch.Tensor,
    window: int,
    ks_cut: float = 0.1,
    posterior_samples: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPU counterpart of CopyKAT's ``CNA.MCMC`` segmentation.

    The method keeps R's Gamma posterior/empirical-KS breakpoint rule,
    including the 0.10 -> 0.05 -> 0.025 fallback when too few breakpoints are
    found. ``posterior_samples`` controls the R-compatible empirical KS draws.
    Segment levels use the exact Gamma posterior expectation rather than a
    second Monte Carlo draw, eliminating avoidable cross-runtime RNG noise
    while matching the expectation of R's posterior sample mean.
    """
    for threshold in (ks_cut, 0.5 * ks_cut, 0.25 * ks_cut):
        breaks = _copykat_mcmc_breaks(
            relative_expression,
            cluster_labels,
            window,
            threshold,
            posterior_samples,
        )
        if breaks.numel() >= 25 or threshold == 0.25 * ks_cut:
            break
    n_genes, n_cells = relative_expression.shape
    segmented = torch.empty_like(relative_expression)
    for start, end in zip(breaks[:-1].tolist(), breaks[1:].tolist()):
        # ``CNA.MCMC`` deliberately includes each internal breakpoint in both
        # adjacent posterior estimates.  The later assignment overwrites that
        # shared gene with the following segment's value, exactly as in R.
        values = relative_expression[start : end + 1].exp()
        if values.shape[0] == 0:
            continue
        posterior = gamma_posterior_mean(values).clamp_min(1e-6).log()
        segmented[start : end + 1] = posterior.expand(end - start + 1, n_cells)
    return segmented, breaks
