import pandas as pd
import torch
from copykat_gpu import copykat
from copykat_gpu import kernels
from copykat_gpu.kernels import cluster_ward_d, cluster_ward_d2, gamma_posterior_mean, hg20_cell_cycle_genes, make_copykat_hg20_bins, pairwise_distance, smooth_ordered_expression

def test_gpu_kernels_preserve_shapes():
    values = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    assert smooth_ordered_expression(values, window=5).shape == values.shape
    assert pairwise_distance(values, "euclidean").shape == (5, 5)
    assert torch.allclose(pairwise_distance(values, "pearson").diagonal(), torch.zeros(5))


def test_ward_d2_matches_r_reference_partition():
    values = torch.tensor(
        [
            [0.1990, -0.7960, -0.5440, 0.5760, -0.6740, -0.3640],
            [0.9940, -0.7980, -0.2810, -0.1960, -0.3300, -0.1910],
            [0.7340, 0.8820, 0.6010, 0.1880, 0.9230, 0.7540],
            [-0.1070, 0.6990, 0.4780, -1.1420, -1.0540, 0.0520],
            [0.6120, -0.0470, -0.8580, 0.3070, -1.1480, 0.3370],
            [0.5630, -0.0660, -0.7950, -0.0340, -1.2530, 0.8580],
            [-0.1810, 0.0370, 0.3290, 0.0200, -0.1590, 0.2940],
        ]
    )
    labels = cluster_ward_d2(values, max_clusters=3, min_cluster_size=0)
    partition = {frozenset(torch.where(labels == label)[0].tolist()) for label in torch.unique(labels)}
    assert partition == {frozenset({0, 3, 5}), frozenset({1, 2}), frozenset({4})}


def test_ward_d_matches_r_reference_partition():
    values = torch.tensor(
        [
            [0.1990, -0.7960, -0.5440, 0.5760, -0.6740, -0.3640],
            [0.9940, -0.7980, -0.2810, -0.1960, -0.3300, -0.1910],
            [0.7340, 0.8820, 0.6010, 0.1880, 0.9230, 0.7540],
            [-0.1070, 0.6990, 0.4780, -1.1420, -1.0540, 0.0520],
            [0.6120, -0.0470, -0.8580, 0.3070, -1.1480, 0.3370],
            [0.5630, -0.0660, -0.7950, -0.0340, -1.2530, 0.8580],
            [-0.1810, 0.0370, 0.3290, 0.0200, -0.1590, 0.2940],
        ]
    )
    labels = cluster_ward_d(values, max_clusters=3, min_cluster_size=0)
    partition = {frozenset(torch.where(labels == label)[0].tolist()) for label in torch.unique(labels)}
    assert partition == {frozenset({0, 3, 5}), frozenset({1, 2}), frozenset({4})}


def test_copykat_hg20_bins_use_medians_and_nearest_bin_fill():
    segmented = torch.tensor([[1.0], [3.0], [10.0]])
    values, bins = make_copykat_hg20_bins(
        segmented,
        __import__("numpy").array(["G1", "G2", "G3"]),
        torch.tensor([1, 1, 1]),
        __import__("numpy").array([1_040_000, 1_050_000, 1_510_000]),
        __import__("numpy").array([1_050_000, 1_060_000, 1_520_000]),
    )
    assert bins.iloc[:3][["chrom", "chrompos"]].to_numpy().tolist() == [[1, 1_042_457], [1, 1_265_484], [1, 1_519_859]]
    assert torch.allclose(values[:4, 0], torch.tensor([1.0, 1.0, 10.0, 10.0]))


def test_hg20_cell_cycle_reference_matches_r_exclusion_basics():
    genes = hg20_cell_cycle_genes()
    assert len(genes) == 1316
    assert {"PCNA", "MCM2", "RTEL1"}.issubset(genes)


def test_gamma_posterior_mean_matches_copykat_parameterization():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([(2.0 + 4.0) / 3.0, (3.0 + 6.0) / 3.0])
    assert torch.allclose(gamma_posterior_mean(values), expected)

def test_copykat_runs_on_small_data():
    genes = [f"G{i}" for i in range(24)]
    cells = [f"cell_{i}" for i in range(8)]
    expression = pd.DataFrame(3, index=genes, columns=cells, dtype=float)
    expression.loc["G0":"G11", "cell_4":] = 25
    coordinates = pd.DataFrame({"gene": genes, "chromosome": [1] * 12 + [2] * 12, "start": list(range(1_000, 13_000, 1_000)) * 2, "end": list(range(1_500, 13_500, 1_000)) * 2})
    result = copykat(expression, coordinates, min_genes_per_cell=1, low_detection_rate=0.0, segmentation_detection_rate=0.0, genes_per_chromosome=2, smoothing_window=5, segmentation_window=3, n_clusters=2, device="cpu")
    assert result.device == "cpu"
    assert result.prediction.shape[0] == len(cells)
    assert set(result.prediction["copykat_prediction"]) <= {"diploid", "aneuploid", "not.defined"}
    assert result.diagnostics is not None
    assert result.cna.shape[0] > 0
    assert result.gene_cna.shape[0] == len(genes)


def test_chromosome_qc_returns_not_defined_for_filtered_cells():
    genes = [f"G{i}" for i in range(12)]
    expression = pd.DataFrame(4.0, index=genes, columns=["normal", "tumor", "missing_chr2"])
    expression.loc["G6":, "missing_chr2"] = 0.0
    expression.loc["G0":"G5", "tumor"] = 20.0
    coordinates = pd.DataFrame(
        {
            "gene": genes,
            "chromosome": [1] * 6 + [2] * 6,
            "start": list(range(1_000, 7_000, 1_000)) * 2,
            "end": list(range(1_500, 7_500, 1_000)) * 2,
        }
    )

    result = copykat(
        expression,
        coordinates,
        known_normal_cells=["normal"],
        min_genes_per_cell=1,
        low_detection_rate=0.0,
        segmentation_detection_rate=0.0,
        genes_per_chromosome=2,
        smoothing_window=5,
        segmentation_window=3,
        n_clusters=2,
        device="cpu",
    )

    assert result.prediction.loc["missing_chr2", "copykat_prediction"] == "not.defined"
    assert result.prediction.loc["normal", "copykat_prediction"] == "diploid"


def test_mcmc_segmentation_uses_r_style_overlapping_breakpoints(monkeypatch):
    def fixed_breaks(*_args, **_kwargs):
        return torch.tensor([0, 2, 4])

    def posterior_mean(values, sample_count):
        return values.mean(dim=0).expand(sample_count, -1)

    monkeypatch.setattr(kernels, "_copykat_mcmc_breaks", fixed_breaks)
    monkeypatch.setattr(kernels, "_gamma_posterior_samples", posterior_mean)
    relative_expression = torch.log(torch.tensor([[1.0], [1.0], [10.0], [10.0], [10.0]]))
    segmented, breaks = kernels.segment_profiles_copykat_mcmc(
        relative_expression,
        torch.zeros(1, dtype=torch.long),
        window=2,
        posterior_samples=2,
    )

    assert breaks.tolist() == [0, 2, 4]
    assert torch.allclose(segmented[:, 0], torch.log(torch.tensor([4.0, 4.0, 10.0, 10.0, 10.0])))
