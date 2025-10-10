import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

from jdti import (
    adjust_cells_to_group_mean,
    average,
    calc_DEG,
    development_clust,
    features_scatter,
    find_features,
    find_names,
    get_color_palette,
    load_sparse,
    make_unique_list,
    occurrence,
    reduce_data,
    volcano_plot,
)

# utils tests


@pytest.fixture(scope="session")
def input_data():
    data, metadata = load_sparse(path="data/set1", name="set1")
    return data, metadata


def test_find_features(input_data):

    data, _ = input_data
    result = find_features(data, features=["KIT", "MC1", "EDNRB", "PAX3"])
    assert len(result["included"]) == 3
    assert len(result["not_included"]) == 1


def test_find_names(input_data):
    data, _ = input_data

    result = find_names(data, names=["0", "1", "2", "10", "1&"])

    assert len(result["included"]) == 4
    assert len(result["not_included"]) == 1


def test_reduce_and_calc_DEG(input_data):
    data, _ = input_data

    res1 = find_features(data, features=["KIT", "MC1R", "EDNRB", "PAX3"])
    res2 = find_names(data, names=["0", "1", "2", "10"])

    data_reduced = reduce_data(data, features=res1["included"], names=res2["included"])

    compare_dict = {"g1": ["0", "1"], "g2": ["2", "10"]}

    DEG = calc_DEG(
        data_reduced,
        metadata_list=None,
        entities=compare_dict,
        sets=None,
        min_exp=0,
        min_pct=0.1,
        n_proc=1,
    )

    assert isinstance(data_reduced, pd.DataFrame)
    assert data_reduced.shape == (4, 444)
    assert isinstance(DEG, pd.DataFrame)
    assert all(col in DEG.columns for col in ["feature", "p_val", "log(FC)"])


def test_DEG_and_volcano_plot(input_data):
    data, metadata = input_data

    compare_dict = {"g1": ["0", "1"], "g2": ["2", "10"]}

    DEG = calc_DEG(
        data,
        metadata_list=metadata["sets"],
        entities=compare_dict,
        sets=None,
        min_exp=0,
        min_pct=0.1,
        n_proc=1,
    )

    fig = volcano_plot(DEG, p_adj=True, top=25)

    assert isinstance(DEG, pd.DataFrame)
    assert all(col in DEG.columns for col in ["feature", "p_val", "log(FC)"])
    assert isinstance(fig, mpl.figure.Figure)


def test_get_color_palette():
    palette = get_color_palette(["A", "B", "C"], palette_name="tab10")
    assert isinstance(palette, dict)
    assert len(palette) == 3


def test_average_occurrence(input_data):
    data, _ = input_data

    avg = average(data)
    occ = occurrence(data)
    assert isinstance(avg, pd.DataFrame)
    assert isinstance(occ, pd.DataFrame)


def test_features_scatter(input_data):
    data, _ = input_data

    res1 = find_features(data, features=["KIT", "MC1R", "EDNRB", "PAX3"])
    res2 = find_names(data, names=["0", "1", "2", "10"])

    data_reduced = reduce_data(data, features=res1["included"], names=res2["included"])

    avg = average(data_reduced)
    occ = occurrence(data_reduced)
    fig = features_scatter(expression_data=avg, occurence_data=occ)

    assert isinstance(fig, mpl.figure.Figure)


def test_make_unique_list():
    lst = ["0", "0", "1", "2", "2"]
    unique = make_unique_list(lst)
    assert len(set(unique)) == len(lst)


def test_development_clust(input_data):
    data, _ = input_data

    res1 = find_features(data, features=["KIT", "MC1R", "EDNRB", "PAX3"])
    res2 = find_names(data, names=["0", "1", "2", "10"])

    data_reduced = reduce_data(data, features=res1["included"], names=res2["included"])

    avg = average(data_reduced)

    fig = development_clust(data=avg, method="ward")

    assert isinstance(fig, mpl.figure.Figure)


def test_adjust_cells_to_group_mean(input_data):
    data, _ = input_data

    avg = average(data)
    adj = adjust_cells_to_group_mean(data=data, data_avg=avg, beta=0.5)
    assert isinstance(adj, pd.DataFrame)
    assert adj.shape == data.shape
