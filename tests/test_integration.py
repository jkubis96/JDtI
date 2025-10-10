import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from jdti import COMPsc, volcano_plot

# integration tests


@pytest.fixture(scope="session")
def jseq_object():
    obj = COMPsc.project_dir("data", ["set1", "set2"])
    obj.load_sparse_from_projects(normalized_data=True)
    return obj


def test_gene_histogram(jseq_object):
    fig = jseq_object.gene_histograme(bins=100)
    assert isinstance(fig, mpl.figure.Figure)


def test_gene_threshold(jseq_object):

    jseq_object.gene_threshold(min_n=50, max_n=3000)
    bin_df = jseq_object.normalized_data.copy()
    bin_df[bin_df > 0] = 1

    bin_sum = bin_df.sum(axis=0)

    assert not any(bin_sum < 50)
    assert not any(bin_sum > 3000)


def test_reduce_full(jseq_object):
    jseq_object.reduce_cols(full="0", inc_set=False)
    assert "10" in list(jseq_object.input_metadata["cell_names"])
    assert "0" not in list(jseq_object.input_metadata["cell_names"])


def test_reduce_reg(jseq_object):
    jseq_object.reduce_cols(reg="0", inc_set=False)

    assert "10" not in jseq_object.input_metadata["cell_names"]


def test_reduce_rows(jseq_object):

    assert "KIT" in jseq_object.normalized_data.index
    assert "MITF" in jseq_object.normalized_data.index

    jseq_object.reduce_rows(["KIT", "MITF"])

    assert "KIT" not in jseq_object.normalized_data.index
    assert "MITF" not in jseq_object.normalized_data.index


def test_cell_histogram(jseq_object):
    fig = jseq_object.cell_histograme(name_slot="cell_names")
    assert isinstance(fig, mpl.figure.Figure)


def test_cluster_threshold(jseq_object):
    jseq_object.cluster_threshold(min_n=20, name_slot="cell_names")
    assert not any(jseq_object.cells_calc["n"] < 20)


def test_difference_markers(jseq_object):
    jseq_object.calculate_difference_markers(
        min_exp=0, min_pct=0.25, n_proc=2, force=True
    )
    assert isinstance(jseq_object.var_data, pd.DataFrame)


def test_similarity(jseq_object):
    jseq_object.estimating_similarity(method="pearson", p_val=0.05, top_n=5)
    fig = jseq_object.similarity_plot(
        split_sets=True, set_info=True, cmap="seismic", width=6, height=4
    )
    assert isinstance(jseq_object.similarity, pd.DataFrame)
    assert isinstance(fig, mpl.figure.Figure)


def test_similarity_cell_to_cell(jseq_object):

    fig = jseq_object.cell_regression(
        cell_x="11",
        cell_y="6",
        set_x="set1",
        set_y="set2",
        threshold=6,
        image_width=12,
        image_high=7,
        color="black",
    )

    assert isinstance(fig, mpl.figure.Figure)


def test_spatial_similarity(jseq_object):
    fig = jseq_object.spatial_similarity(
        set_info=True,
        bandwidth=1,
        n_neighbors=5,
        min_dist=0.1,
        legend_split=4,
        point_size=100,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        width=10,
        height=8,
    )

    assert isinstance(fig, mpl.figure.Figure)


def test_pca_umap(jseq_object):
    jseq_object.clustering_features(
        name_slot="cell_names",
        features_list=None,
        p_val=0.05,
        top_n=10,
        adj_mean=False,
        beta=0.2,
    )

    jseq_object.perform_PCA(pc_num=10)
    jseq_object.harmonize_sets()
    jseq_object.perform_UMAP(factorize=True, umap_num=2, pc_num=5, harmonized=True)
    fig = jseq_object.UMAP_vis(
        names_slot="cell_names",
        set_sep=True,
        point_size=1,
        font_size=6,
        legend_split_col=2,
        width=6,
        height=4,
        inc_num=True,
    )

    assert isinstance(jseq_object.clustering_data, pd.DataFrame)
    assert isinstance(jseq_object.explained_var, np.ndarray)
    assert isinstance(jseq_object.umap, pd.DataFrame)
    assert isinstance(fig, mpl.figure.Figure)


# tutaj !
def test_umap_feature(jseq_object):
    fig = jseq_object.UMAP_feature(
        features_data=jseq_object.get_data(set_info=False),
        feature_name="MAP1B",
        point_size=0.6,
        font_size=6,
        width=6,
        height=4,
        palette="light",
    )

    assert isinstance(fig, mpl.figure.Figure)


def test_statistics_and_volcano_and_scatter(jseq_object):
    stats = jseq_object.statistic(
        cells=None, sets="All", min_exp=0, min_pct=0.05, n_proc=2
    )
    assert not stats.empty
    fig = volcano_plot(stats)

    assert isinstance(fig, mpl.figure.Figure)

    stats_5 = (
        stats.sort_values(
            ["valid_group", "esm", "log(FC)"], ascending=[True, False, False]
        )
        .groupby("valid_group")
        .head(5)
    )

    fig = jseq_object.scatter_plot(
        names=None,
        features=list(set(stats_5["feature"])),
        name_slot="cell_names",
        scale=False,
        colors="viridis",
        hclust="complete",
        img_width=8,
        img_high=3,
        label_size=10,
        size_scale=200,
        y_lab="Genes",
        legend_lab="log(CPM + 1)",
        set_box_size=5,
        set_box_high=0.1,
        bbox_to_anchor_scale=25,
        bbox_to_anchor_perc=(0.90, 0.5),
        bbox_to_anchor_group=(0.9, 0.3),
    )

    assert isinstance(fig, mpl.figure.Figure)


def test_compositions(jseq_object):
    jseq_object.data_composition(
        features_count=None, name_slot="cell_names", set_sep=True
    )
    fig = jseq_object.composition_pie(
        width=6,
        height=6,
        font_size=10,
        cmap="tab20",
        legend_split_col=1,
        offset_labels=0.5,
        legend_bbox=(1.15, 0.95),
    )

    assert isinstance(fig, mpl.figure.Figure)

    fig = jseq_object.bar_composition(
        cmap="tab20b",
        width=2,
        height=6,
        font_size=10,
        legend_split_col=1,
        legend_bbox=(1.3, 1),
    )

    assert isinstance(fig, mpl.figure.Figure)


def test_metadata(jseq_object):
    met = jseq_object.input_metadata
    data = jseq_object.get_data(set_info=True)
    metadata = jseq_object.get_metadata()
    assert met is not None
    assert data is not None
    assert metadata is not None
