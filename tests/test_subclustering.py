import os

import matplotlib as mpl
import pytest

from jdti import Clustering, COMPsc


@pytest.fixture(scope="session")
def jseq_object():
    obj = COMPsc.project_dir(os.path.join("data"), ["set1"])
    obj.load_sparse_from_projects(normalized_data=True)
    return obj


def test_normalized_data_loaded(jseq_object):
    assert hasattr(jseq_object, "normalized_data")
    assert not jseq_object.normalized_data.empty


def test_subcluster_prepare(jseq_object):
    jseq_object.subcluster_prepare(features=["HMGCS1", "MAP1B", "SOX4"], cluster="0")

    assert isinstance(jseq_object.subclusters_, Clustering)
    assert set(["HMGCS1", "MAP1B", "SOX4"]).issubset(
        jseq_object.subclusters_.current_features
    )


def test_define_subclusters(jseq_object):
    jseq_object.define_subclusters(
        umap_num=2,
        eps=1,
        min_samples=5,
        n_neighbors=5,
        min_dist=0.1,
        spread=1.0,
        set_op_mix_ratio=1.0,
        local_connectivity=1,
        repulsion_strength=1.0,
        negative_sample_rate=5,
        width=8,
        height=6,
    )

    assert isinstance(jseq_object.subclusters_.subclusters, list)


def test_rename_subclusters(jseq_object):
    mapping = {"old_name": ["0", "1", "2"], "new_name": ["0", "0", "0"]}
    assert "1" in jseq_object.subclusters_.subclusters
    assert "2" in jseq_object.subclusters_.subclusters
    jseq_object.rename_subclusters(mapping)
    assert "1" not in jseq_object.subclusters_.subclusters
    assert "2" not in jseq_object.subclusters_.subclusters


def test_subcluster_DEG_scatter(jseq_object):
    fig = jseq_object.subcluster_DEG_scatter(
        top_n=3,
        min_exp=0,
        min_pct=0.1,
        p_val=0.05,
        colors="viridis",
        hclust="complete",
        img_width=3,
        img_high=5,
        label_size=6,
        size_scale=70,
        y_lab="Genes",
        legend_lab="normalized",
        n_proc=2,
    )

    assert isinstance(fig, mpl.figure.Figure)


def test_accept_subclusters(jseq_object):
    jseq_object.accept_subclusters()
    assert "0.0" in list(jseq_object.input_metadata["cell_names"])
