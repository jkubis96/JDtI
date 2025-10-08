import matplotlib as mpl
import pandas as pd
import pytest

from scdiff.scdiff import Clustering, load_sparse


@pytest.fixture(scope="session")
def clusters_obj():
    data, metadata = load_sparse(path="data/set1", name="set1")
    clusters = Clustering.add_data_frame(data, metadata)
    return clusters


def test_clustering_data_and_metadata(clusters_obj):
    assert isinstance(clusters_obj.clustering_data, pd.DataFrame)
    assert isinstance(clusters_obj.clustering_metadata, pd.DataFrame)
    assert len(clusters_obj.clustering_data) > 0
    assert len(clusters_obj.clustering_metadata) > 0


def test_perform_pca(clusters_obj):
    clusters_obj.perform_PCA(pc_num=10, width=4, height=3)
    pca_data = clusters_obj.get_pca_data()
    assert isinstance(pca_data, pd.DataFrame)
    assert "PC1" in pca_data.columns


def test_knee_plot_pca(clusters_obj):
    fig = clusters_obj.knee_plot_PCA(width=4, height=3)
    assert isinstance(fig, mpl.figure.Figure)


def test_harmonize_sets(clusters_obj):
    clusters_obj.harmonize_sets()
    assert isinstance(clusters_obj.harmonized_pca, pd.DataFrame)


def test_find_clusters_pca(clusters_obj):
    clusters_obj.find_clusters_PCA(
        pc_num=5, eps=0.5, min_samples=5, width=4, height=3, harmonized=False
    )
    assert "PCA_clusters" in clusters_obj.clustering_metadata.columns


def test_perform_umap(clusters_obj):
    clusters_obj.perform_UMAP(factorize=False, umap_num=10, pc_num=5, harmonized=False)
    umap_data = clusters_obj.get_umap_data()
    assert isinstance(umap_data, pd.DataFrame)


def test_knee_plot_umap(clusters_obj):
    fig = clusters_obj.knee_plot_umap(eps=0.5, min_samples=5)
    assert isinstance(fig, mpl.figure.Figure)


def test_find_clusters_umap(clusters_obj):
    clusters_obj.find_clusters_UMAP(umap_n=5, eps=0.5, min_samples=5, width=4, height=3)
    assert "UMAP_clusters" in clusters_obj.clustering_metadata.columns


def test_umap_vis(clusters_obj):
    fig = clusters_obj.UMAP_vis(names_slot="cell_names", set_sep=True, point_size=0.6)
    assert isinstance(fig, mpl.figure.Figure)


def test_umap_feature(clusters_obj):
    fig = clusters_obj.UMAP_feature(
        feature_name="KIT", features_data=None, point_size=0.6
    )
    assert isinstance(fig, mpl.figure.Figure)


def test_get_umap_data(clusters_obj):
    umap_data = clusters_obj.get_umap_data()
    assert isinstance(umap_data, pd.DataFrame)


def test_get_pca_data(clusters_obj):
    pca_data = clusters_obj.get_pca_data()
    assert isinstance(pca_data, pd.DataFrame)


def test_return_clusters_umap(clusters_obj):
    clusters = clusters_obj.return_clusters(clusters="umap")
    assert isinstance(clusters, pd.Series)
    assert clusters.nunique() > 1
