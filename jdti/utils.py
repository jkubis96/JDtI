import os
import re

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from adjustText import adjust_text
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.io import mmread
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


def load_sparse(path: str, name: str):
    """
    Load a sparse matrix dataset along with associated gene
    and cell metadata, and return it as a dense DataFrame.

    This function expects the input directory to contain three files in standard
    10x Genomics format:
      - "matrix.mtx": the gene expression matrix in Matrix Market format
      - "genes.tsv": tab-separated file containing gene identifiers
      - "barcodes.tsv": tab-separated file containing cell barcodes / names

    Parameters
    ----------
    path : str
        Path to the directory containing the matrix and annotation files.

    name : str
        Label or dataset identifier to be assigned to all cells in the metadata.

    Returns
    -------
    data : pandas.DataFrame
        Dense expression matrix where rows correspond to genes and columns
        correspond to cells.
    metadata : pandas.DataFrame
        Metadata DataFrame with two columns:
          - "cell_names": the names of the cells (barcodes)
          - "sets": the dataset label assigned to each cell (from `name`)

    Notes
    -----
    The function converts the sparse matrix into a dense DataFrame. This may
    require a large amount of memory for datasets with many cells and genes.
    """

    data = mmread(os.path.join(path, "matrix.mtx"))
    data = pd.DataFrame(data.todense())
    genes = pd.read_csv(os.path.join(path, "genes.tsv"), header=None, sep="\t")
    names = pd.read_csv(os.path.join(path, "barcodes.tsv"), header=None, sep="\t")
    data.columns = [str(x) for x in names[0]]
    data.index = list(genes[0])

    names = list(data.columns)
    sets = [name] * len(names)

    metadata = pd.DataFrame({"cell_names": names, "sets": sets})

    return data, metadata


def volcano_plot(
    deg_data: pd.DataFrame,
    p_adj: bool = True,
    top: int = 25,
    top_rank: str = "p_value",
    p_val: float | int = 0.05,
    lfc: float | int = 0.25,
    rescale_adj: bool = True,
    image_width: int = 12,
    image_high: int = 12,
):
    """
    Generate a volcano plot from differential expression results.

    A volcano plot visualizes the relationship between statistical significance
    (p-values or standarized p-value) and log(fold change) for each gene, highlighting
    genes that pass significance thresholds.

    Parameters
    ----------
    deg_data : pandas.DataFrame
        DataFrame containing differential expression results from calc_DEG() function.

    p_adj : bool, default=True
        If True, use adjusted p-values. If False, use raw p-values.

    top : int, default=25
        Number of top significant genes to highlight on the plot.

    top_rank : str, default='p_value'
        Statistic used primarily to determine the top significant genes to highlight on the plot. ['p_value' or 'FC']

    p_val : float | int, default=0.05
        Significance threshold for p-values (or adjusted p-values).

    lfc : float | int, default=0.25
        Threshold for absolute log fold change.

    rescale_adj : bool, default=True
        If True, rescale p-values to avoid long breaks caused by outlier values.

    image_width : int, default=12
        Width of the generated plot in inches.

    image_high : int, default=12
        Height of the generated plot in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The generated volcano plot figure.

    """

    if top_rank.upper() not in ["FC", "P_VALUE"]:
        raise ValueError("top_rank must be either 'FC' or 'p_value'")

    if p_adj:
        pv = "adj_pval"
    else:
        pv = "p_val"

    deg_df = deg_data.copy()

    shift = 0.25

    p_val_scale = "-log(p_val)"

    min_minus = min(deg_df[pv][(deg_df[pv] != 0) & (deg_df["log(FC)"] < 0)])
    min_plus = min(deg_df[pv][(deg_df[pv] != 0) & (deg_df["log(FC)"] > 0)])

    zero_p_plus = deg_df[(deg_df[pv] == 0) & (deg_df["log(FC)"] > 0)]
    zero_p_plus = zero_p_plus.sort_values(by="log(FC)", ascending=False).reset_index(
        drop=True
    )
    zero_p_plus[pv] = [
        (shift * x) * min_plus for x in range(1, len(zero_p_plus.index) + 1)
    ]

    zero_p_minus = deg_df[(deg_df[pv] == 0) & (deg_df["log(FC)"] < 0)]
    zero_p_minus = zero_p_minus.sort_values(by="log(FC)", ascending=True).reset_index(
        drop=True
    )
    zero_p_minus[pv] = [
        (shift * x) * min_minus for x in range(1, len(zero_p_minus.index) + 1)
    ]

    tmp_p = deg_df[
        ((deg_df[pv] != 0) & (deg_df["log(FC)"] < 0))
        | ((deg_df[pv] != 0) & (deg_df["log(FC)"] > 0))
    ]

    del deg_df

    deg_df = pd.concat([zero_p_plus, tmp_p, zero_p_minus], ignore_index=True)

    deg_df[p_val_scale] = -np.log10(deg_df[pv])

    deg_df["top100"] = None

    if rescale_adj:

        deg_df = deg_df.sort_values(by=p_val_scale, ascending=False)

        deg_df = deg_df.reset_index(drop=True)

        eps = 1e-300
        doubled = []
        ratio = []
        for n, i in enumerate(deg_df.index):
            for j in range(1, 6):
                if (
                    n + j < len(deg_df.index)
                    and (deg_df[p_val_scale][n] + eps)
                    / (deg_df[p_val_scale][n + j] + eps)
                    >= 2
                ):
                    doubled.append(n)
                    ratio.append(
                        (deg_df[p_val_scale][n + j] + eps)
                        / (deg_df[p_val_scale][n] + eps)
                    )

        df = pd.DataFrame({"doubled": doubled, "ratio": ratio})
        df = df[df["doubled"] < 100]

        df["ratio"] = (1 - df["ratio"]) / 5
        df = df.reset_index(drop=True)

        df = df.sort_values("doubled")

        if len(df["doubled"]) == 1 and 0 in df["doubled"]:
            df = df
        else:
            doubled2 = []

            for l in df["doubled"]:
                if l + 1 != len(doubled) and l + 1 - l == 1:
                    doubled2.append(l)
                    doubled2.append(l + 1)
                else:
                    break

            doubled2 = sorted(set(doubled2), reverse=True)

        if len(doubled2) > 1:
            df = df[df["doubled"].isin(doubled2)]
            df = df.sort_values("doubled", ascending=False)
            df = df.reset_index(drop=True)
            for c in df.index:
                deg_df.loc[df["doubled"][c], p_val_scale] = deg_df.loc[
                    df["doubled"][c] + 1, p_val_scale
                ] * (1 + df["ratio"][c])

    deg_df.loc[(deg_df["log(FC)"] <= 0) & (deg_df[pv] <= p_val), "top100"] = "red"
    deg_df.loc[(deg_df["log(FC)"] > 0) & (deg_df[pv] <= p_val), "top100"] = "blue"
    deg_df.loc[deg_df[pv] > p_val, "top100"] = "lightgray"

    if lfc > 0:
        deg_df.loc[
            (deg_df["log(FC)"] <= lfc) & (deg_df["log(FC)"] >= -lfc), "top100"
        ] = "lightgray"

    down_int = len(
        deg_df["top100"][(deg_df["log(FC)"] <= lfc * -1) & (deg_df[pv] <= p_val)]
    )
    up_int = len(deg_df["top100"][(deg_df["log(FC)"] > lfc) & (deg_df[pv] <= p_val)])

    deg_df_up = deg_df[deg_df["log(FC)"] > 0]

    if top_rank.upper() == "P_VALUE":
        deg_df_up = deg_df_up.sort_values([pv, "log(FC)"], ascending=[True, False])
    elif top_rank.upper() == "FC":
        deg_df_up = deg_df_up.sort_values(["log(FC)", pv], ascending=[False, True])

    deg_df_up = deg_df_up.reset_index(drop=True)

    n = -1
    l = 0
    while True:
        n += 1
        if deg_df_up["log(FC)"][n] > lfc:
            deg_df_up.loc[n, "top100"] = "green"
            l += 1
        if l == top or deg_df_up[pv][n] > p_val:
            break

    deg_df_down = deg_df[deg_df["log(FC)"] <= 0]

    if top_rank.upper() == "P_VALUE":
        deg_df_down = deg_df_down.sort_values([pv, "log(FC)"], ascending=[True, True])
    elif top_rank.upper() == "FC":
        deg_df_down = deg_df_down.sort_values(["log(FC)", pv], ascending=[True, True])

    deg_df_down = deg_df_down.reset_index(drop=True)

    n = -1
    l = 0
    while True:
        n += 1
        if deg_df_down["log(FC)"][n] < lfc * -1:
            deg_df_down.loc[n, "top100"] = "yellow"

            l += 1
        if l == top or deg_df_down[pv][n] > p_val:
            break

    deg_df = pd.concat([deg_df_up, deg_df_down])

    que = ["lightgray", "red", "blue", "yellow", "green"]

    deg_df = deg_df.sort_values(
        by="top100", key=lambda x: x.map({v: i for i, v in enumerate(que)})
    )

    deg_df = deg_df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(image_width, image_high))

    plt.scatter(
        x=deg_df["log(FC)"], y=deg_df[p_val_scale], color=deg_df["top100"], zorder=2
    )

    plt.plot(
        [max(deg_df["log(FC)"]) * -1.1, max(deg_df["log(FC)"]) * 1.1],
        [-np.log10(p_val), -np.log10(p_val)],
        linestyle="--",
        linewidth=3,
        color="lightgray",
        zorder=1,
    )

    if lfc > 0:
        plt.plot(
            [lfc * -1, lfc * -1],
            [-3, max(deg_df[p_val_scale]) * 1.1],
            linestyle="--",
            linewidth=3,
            color="lightgray",
            zorder=1,
        )
        plt.plot(
            [lfc, lfc],
            [-3, max(deg_df[p_val_scale]) * 1.1],
            linestyle="--",
            linewidth=3,
            color="lightgray",
            zorder=1,
        )

    plt.xlabel("log(FC)")
    plt.ylabel(p_val_scale)
    plt.title("Volcano plot")

    plt.ylim(min(deg_df[p_val_scale]) - 5, max(deg_df[p_val_scale]) * 1.25)

    texts = [
        ax.text(deg_df["log(FC)"][i], deg_df[p_val_scale][i], deg_df["feature"][i])
        for i in deg_df.index
        if deg_df["top100"][i] in ["green", "yellow"]
    ]

    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5))

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="top-upregulated",
            markerfacecolor="green",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="top-downregulated",
            markerfacecolor="yellow",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="upregulated",
            markerfacecolor="blue",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="downregulated",
            markerfacecolor="red",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="non-significant",
            markerfacecolor="lightgray",
            markersize=10,
        ),
    ]

    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(visible=False)

    ax.annotate(
        f"\nmin {pv} = " + str(p_val),
        xy=(0.025, 0.975),
        xycoords="axes fraction",
        fontsize=12,
    )

    if lfc > 0:
        ax.annotate(
            "\nmin log(FC) = " + str(lfc),
            xy=(0.025, 0.95),
            xycoords="axes fraction",
            fontsize=12,
        )

    ax.annotate(
        "\nDownregulated: " + str(down_int),
        xy=(0.025, 0.925),
        xycoords="axes fraction",
        fontsize=12,
        color="red",
    )

    ax.annotate(
        "\nUpregulated: " + str(up_int),
        xy=(0.025, 0.9),
        xycoords="axes fraction",
        fontsize=12,
        color="blue",
    )

    plt.show()

    return fig


def find_features(data: pd.DataFrame, features: list):
    """
    Identify features (rows) from a DataFrame that match a given list of features,
    ignoring case sensitivity.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with features in the index (rows).

    features : list
        List of feature names to search for.

    Returns
    -------
    dict
        Dictionary with keys:
        - "included": list of features found in the DataFrame index.
        - "not_included": list of requested features not found in the DataFrame index.
        - "potential": list of features in the DataFrame that may be similar.
    """

    features_upper = [str(x).upper() for x in features]

    index_set = set(data.index)

    features_in = [x for x in index_set if x.upper() in features_upper]
    features_in_upper = [x.upper() for x in features_in]
    features_out_upper = [x for x in features_upper if x not in features_in_upper]
    features_out = [x for x in features if x.upper() in features_out_upper]
    similar_features = [
        idx for idx in index_set if any(x in idx.upper() for x in features_out_upper)
    ]

    return {
        "included": features_in,
        "not_included": features_out,
        "potential": similar_features,
    }


def find_names(data: pd.DataFrame, names: list):
    """
    Identify names (columns) from a DataFrame that match a given list of names,
    ignoring case sensitivity.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with names in the columns.

    names : list
        List of names to search for.

    Returns
    -------
    dict
        Dictionary with keys:
        - "included": list of names found in the DataFrame columns.
        - "not_included": list of requested names not found in the DataFrame columns.
        - "potential": list of names in the DataFrame that may be similar.
    """

    names_upper = [str(x).upper() for x in names]

    columns = set(data.columns)

    names_in = [x for x in columns if x.upper() in names_upper]
    names_in_upper = [x.upper() for x in names_in]
    names_out_upper = [x for x in names_upper if x not in names_in_upper]
    names_out = [x for x in names if x.upper() in names_out_upper]
    similar_names = [
        idx for idx in columns if any(x in idx.upper() for x in names_out_upper)
    ]

    return {"included": names_in, "not_included": names_out, "potential": similar_names}


def reduce_data(data: pd.DataFrame, features: list = [], names: list = []):
    """
    Subset a DataFrame based on selected features (rows) and/or names (columns).

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame with features as rows and names as columns.

    features : list
        List of features to include (rows). Default is an empty list.
        If empty, all rows are returned.

    names : list
        List of names to include (columns). Default is an empty list.
        If empty, all columns are returned.

    Returns
    -------
    pandas.DataFrame
        Subset of the input DataFrame containing only the selected rows
        and/or columns.

    Raises
    ------
    ValueError
        If both `features` and `names` are empty.
    """

    if len(features) > 0 and len(names) > 0:
        fet = find_features(data=data, features=features)

        nam = find_names(data=data, names=names)

        data_to_return = data.loc[fet["included"], nam["included"]]

    elif len(features) > 0 and len(names) == 0:
        fet = find_features(data=data, features=features)

        data_to_return = data.loc[fet["included"], :]

    elif len(features) == 0 and len(names) > 0:

        nam = find_names(data=data, names=names)

        data_to_return = data.loc[:, nam["included"]]

    else:

        raise ValueError("features and names have zero length!")

    return data_to_return


def make_unique_list(lst):
    """
    Generate a list where duplicate items are renamed to ensure uniqueness.

    Each duplicate is appended with a suffix ".n", where n indicates the
    occurrence count (starting from 1).

    Parameters
    ----------
    lst : list
        Input list of items (strings or other hashable types).

    Returns
    -------
    list
        List with unique values.

    Examples
    --------
    >>> make_unique_list(["A", "B", "A", "A"])
    ['A', 'B', 'A.1', 'A.2']
    """
    seen = {}
    result = []
    for item in lst:
        if item not in seen:
            seen[item] = 0
            result.append(item)
        else:
            seen[item] += 1
            result.append(f"{item}.{seen[item]}")
    return result


def get_color_palette(variable_list, palette_name="tab10"):
    n = len(variable_list)
    cmap = plt.get_cmap(palette_name)
    colors = [cmap(i % cmap.N) for i in range(n)]
    return dict(zip(variable_list, colors))


def features_scatter(
    expression_data: pd.DataFrame,
    occurence_data: pd.DataFrame | None = None,
    scale: bool = False,
    features: list | None = None,
    metadata_list: list | None = None,
    colors: str = "viridis",
    hclust: str | None = "complete",
    img_width: int = 8,
    img_high: int = 5,
    label_size: int = 10,
    size_scale: int = 100,
    y_lab: str = "Genes",
    legend_lab: str = "log(CPM + 1)",
    set_box_size: float | int = 5,
    set_box_high: float | int = 5,
    bbox_to_anchor_scale: int = 25,
    bbox_to_anchor_perc: tuple = (0.91, 0.63),
    bbox_to_anchor_group: tuple = (1.01, 0.4),
):
    """
    Create a bubble scatter plot of selected features across samples.

    Each point represents a feature-sample pair, where the color encodes the
    expression value and the size encodes occurrence or relative abundance.
    Optionally, hierarchical clustering can be applied to order rows and columns.

    Parameters
    ----------
    expression_data : pandas.DataFrame
        Expression values (mean) with features as rows and samples as columns derived from average() function.

    occurence_data : pandas.DataFrame or None
        DataFrame with occurrence/frequency values (same shape as `expression_data`) derived from occurrence() function.
        If None, bubble sizes are based on expression values.

    scale: bool, default False
        If True, expression_data will be scaled (0–1) across the rows (features).

    features : list or None
        List of features (rows) to display. If None, all features are used.

    metadata_list : list or None, optional
        Metadata grouping for samples (same length as number of columns).
        Used to add group colors and separators in the plot.

    colors : str, default='viridis'
        Colormap for expression values.

    hclust : str or None, default='complete'
        Linkage method for hierarchical clustering. If None, no clustering
        is performed.

    img_width : int or float, default=8
        Width of the plot in inches.

    img_high : int or float, default=5
        Height of the plot in inches.

    label_size : int, default=10
        Font size for axis labels and ticks.

    size_scale : int or float, default=100
        Scaling factor for bubble sizes.

    y_lab : str, default='Genes'
        Label for the x-axis.

    legend_lab : str, default='log(CPM + 1)'
        Label for the colorbar legend.

    bbox_to_anchor_scale : int, default=25
        Vertical scale (percentage) for positioning the colorbar.

    bbox_to_anchor_perc : tuple, default=(0.91, 0.63)
        Anchor position for the size legend (percent bubble legend).

    bbox_to_anchor_group : tuple, default=(1.01, 0.4)
        Anchor position for the group legend.

    Returns
    -------
    matplotlib.figure.Figure
        The generated scatter plot figure.

    Raises
    ------
    ValueError
        If `metadata_list` is provided but its length does not match
        the number of columns in `expression_data`.

    Notes
    -----
    - Colors represent expression values normalized to the colormap.
    - Bubble sizes represent occurrence values (or expression values if
      `occurence_data` is None).
    - If `metadata_list` is given, groups are indicated with colors and
      dashed vertical separators.
    """

    scatter_df = expression_data.copy()

    if scale:

        legend_lab = "Scaled\n" + legend_lab

        scaler = MinMaxScaler(feature_range=(0, 1))
        scatter_df = pd.DataFrame(
            scaler.fit_transform(scatter_df.T).T,
            index=scatter_df.index,
            columns=scatter_df.columns,
        )

    metadata = {}

    metadata["primary_names"] = [str(x) for x in scatter_df.columns]

    if metadata_list is not None:
        metadata["sets"] = metadata_list

        if len(metadata["primary_names"]) != len(metadata["sets"]):

            raise ValueError(
                "Metadata list and DataFrame columns must have the same length."
            )

    else:

        metadata["sets"] = [""] * len(metadata["primary_names"])

    metadata = pd.DataFrame(metadata)
    if features is not None:
        scatter_df = scatter_df.loc[
            find_features(data=scatter_df, features=features)["included"],
        ]
    scatter_df.columns = metadata["primary_names"] + "#" + metadata["sets"]

    if occurence_data is not None:
        if features is not None:
            occurence_data = occurence_data.loc[
                find_features(data=occurence_data, features=features)["included"],
            ]
        occurence_data.columns = metadata["primary_names"] + "#" + metadata["sets"]

    # check duplicated names

    tmp_columns = scatter_df.columns

    new_cols = make_unique_list(list(tmp_columns))

    scatter_df.columns = new_cols

    if hclust is not None and len(expression_data.index) != 1:

        Z = linkage(scatter_df, method=hclust)

        # Get the order of features based on the dendrogram
        order_of_features = dendrogram(Z, no_plot=True)["leaves"]

        indexes_sort = list(scatter_df.index)
        sorted_list_rows = []
        for n in order_of_features:
            sorted_list_rows.append(indexes_sort[n])

        scatter_df = scatter_df.transpose()

        Z = linkage(scatter_df, method=hclust)

        # Get the order of features based on the dendrogram
        order_of_features = dendrogram(Z, no_plot=True)["leaves"]

        indexes_sort = list(scatter_df.index)
        sorted_list_columns = []
        for n in order_of_features:
            sorted_list_columns.append(indexes_sort[n])

        scatter_df = scatter_df.transpose()

        scatter_df = scatter_df.loc[sorted_list_rows, sorted_list_columns]

        if occurence_data is not None:
            occurence_data = occurence_data.loc[sorted_list_rows, sorted_list_columns]

        metadata["sets"] = [re.sub(".*#", "", x) for x in scatter_df.columns]

    scatter_df.columns = [re.sub("#.*", "", x) for x in scatter_df.columns]

    if occurence_data is not None:
        occurence_data.columns = [re.sub("#.*", "", x) for x in occurence_data.columns]

    fig, ax = plt.subplots(figsize=(img_width, img_high))

    norm = plt.Normalize(0, np.max(scatter_df))

    cmap = plt.get_cmap(colors)

    # Bubble scatter
    for i, _ in enumerate(scatter_df.index):
        for j, _ in enumerate(scatter_df.columns):
            if occurence_data is not None:
                value_e = scatter_df.iloc[i, j]
                value_o = occurence_data.iloc[i, j]
                ax.scatter(
                    j,
                    i,
                    s=value_o * size_scale,
                    c=[cmap(norm(value_e))],
                    edgecolors="k",
                    linewidths=0.3,
                )
            else:
                value = scatter_df.iloc[i, j]
                ax.scatter(
                    j,
                    i,
                    s=value * size_scale,
                    c=[cmap(norm(value))],
                    edgecolors="k",
                    linewidths=0.3,
                )

    ax.set_yticks(range(len(scatter_df.index)))
    ax.set_yticklabels(scatter_df.index, fontsize=label_size * 0.8)
    ax.set_ylabel(y_lab, fontsize=label_size)
    ax.set_xticks(range(len(scatter_df.columns)))
    ax.set_xticklabels(scatter_df.columns, fontsize=label_size * 0.8, rotation=90)

    # color bar
    cax = inset_axes(
        ax,
        width="1%",
        height=f"{bbox_to_anchor_scale}%",
        bbox_to_anchor=(1, 0, 1, 1),
        bbox_transform=ax.transAxes,
        loc="upper left",
    )
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)
    cb.set_label(legend_lab, fontsize=label_size * 0.65)
    cb.ax.tick_params(labelsize=label_size * 0.7)

    if metadata_list is not None:

        metadata_list = list(metadata["sets"])
        group_colors = get_color_palette(list(set(metadata_list)), palette_name="tab10")

        for i, group in enumerate(metadata_list):
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.5, len(scatter_df.index) - 0.1 * set_box_high),
                    1,
                    0.1 * set_box_size,
                    color=group_colors[group],
                    transform=ax.transData,
                    clip_on=False,
                )
            )

        for i in range(1, len(metadata_list)):
            if metadata_list[i] != metadata_list[i - 1]:
                ax.axvline(i - 0.5, color="black", linestyle="--", lw=1)

        group_patches = [
            mpatches.Patch(color=color, label=label)
            for label, color in group_colors.items()
        ]
        fig.legend(
            handles=group_patches,
            title="Group",
            fontsize=label_size * 0.7,
            title_fontsize=label_size * 0.7,
            loc="center left",
            bbox_to_anchor=bbox_to_anchor_group,
            frameon=False,
        )

    # second legend (size)
    if occurence_data is not None:
        size_values = [0.25, 0.5, 1]
        legend2_handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markersize=np.sqrt(v * size_scale * 0.5),
                color="gray",
                alpha=0.6,
                label=f"{v * 100:.1f}",
            )
            for v in size_values
        ]

        fig.legend(
            handles=legend2_handles,
            title="Percent [%]",
            fontsize=label_size * 0.7,
            title_fontsize=label_size * 0.7,
            loc="center left",
            bbox_to_anchor=bbox_to_anchor_perc,
            frameon=False,
        )

    _, ymax = ax.get_ylim()

    ax.set_xlim(-0.5, len(scatter_df.columns) - 0.5)
    ax.set_ylim(-0.5, ymax + 0.5)

    return fig


def calc_DEG(
    data,
    metadata_list: list | None = None,
    entities: str | list | dict | None = None,
    sets: str | list | dict | None = None,
    min_exp: int | float = 0,
    min_pct: int | float = 0.1,
    n_proc: int = 10,
):
    """
    Perform differential gene expression (DEG) analysis on gene expression data.

    The function compares groups of cells or samples (defined by `entities` or
    `sets`) using the Mann–Whitney U test. It computes p-values, adjusted
    p-values, fold changes, standardized effect sizes, and other statistics.

    Parameters
    ----------
    data : pandas.DataFrame
        Expression matrix with features (e.g., genes) as rows and samples/cells
        as columns.

    metadata_list : list or None, optional
        Metadata grouping corresponding to the columns in `data`. Required for
        comparisons based on sets. Default is None.

    entities : list, str, dict, or None, optional
        Defines the comparison strategy:
        - list of sample names → compare selected cells to the rest.
        - 'All' → compare each sample/cell to all others.
        - dict → user-defined groups for pairwise comparison.
        - None → must be combined with `sets`.

    sets : str, dict, or None, optional
        Defines group-based comparisons:
        - 'All' → compare each set/group to all others.
        - dict with two groups → perform pairwise set comparison.
        - None → must be combined with `entities`.

    min_exp : float | int, default=0
        Minimum expression threshold for filtering features.

    min_pct : float | int, default=0.1
        Minimum proportion of samples within the target group that must express
        a feature for it to be tested.

    n_proc : int, default=10
        Number of parallel processes to use for statistical testing.

    Returns
    -------
    pandas.DataFrame or dict
        Results of the differential expression analysis:
        - If `entities` is a list → dict with keys: 'valid_cells',
          'control_cells', and 'DEG' (results DataFrame).
        - If `entities == 'All'` or `sets == 'All'` → DataFrame with results
          for all groups.
        - If pairwise comparison (dict for `entities` or `sets`) → DataFrame
          with results for the specified groups.

        The results DataFrame contains:
        - 'feature': feature name
        - 'p_val': raw p-value
        - 'adj_pval': adjusted p-value (multiple testing correction)
        - 'pct_valid': fraction of target group expressing the feature
        - 'pct_ctrl': fraction of control group expressing the feature
        - 'avg_valid': mean expression in target group
        - 'avg_ctrl': mean expression in control group
        - 'sd_valid': standard deviation in target group
        - 'sd_ctrl': standard deviation in control group
        - 'esm': effect size metric
        - 'FC': fold change
        - 'log(FC)': log2-transformed fold change
        - 'norm_diff': difference in mean expression

    Raises
    ------
    ValueError
        - If `metadata_list` is provided but its length does not match
          the number of columns in `data`.
        - If neither `entities` nor `sets` is provided.

    Notes
    -----
    - Mann–Whitney U test is used for group comparisons.
    - Multiple testing correction is applied using a simple
      Benjamini–Hochberg-like method.
    - Features expressed below `min_exp` or in fewer than `min_pct` of target
      samples are filtered out.
    - Parallelization is handled by `joblib.Parallel`.

    Examples
    --------
    Compare a selected list of cells against all others:

    >>> result = calc_DEG(data, entities=["cell1", "cell2", "cell3"])

    Compare each group to others (based on metadata):

    >>> result = calc_DEG(data, metadata_list=group_labels, sets="All")

    Perform pairwise comparison between two predefined sets:

    >>> sets = {"GroupA": ["A1", "A2"], "GroupB": ["B1", "B2"]}
    >>> result = calc_DEG(data, sets=sets)
    """

    metadata = {}

    metadata["primary_names"] = [str(x) for x in data.columns]

    if metadata_list is not None:
        metadata["sets"] = metadata_list

        if len(metadata["primary_names"]) != len(metadata["sets"]):

            raise ValueError(
                "Metadata list and DataFrame columns must have the same length."
            )

    else:

        metadata["sets"] = [""] * len(metadata["primary_names"])

    metadata = pd.DataFrame(metadata)

    def stat_calc(choose, feature_name):
        target_values = choose.loc[choose["DEG"] == "target", feature_name]
        rest_values = choose.loc[choose["DEG"] == "rest", feature_name]

        pct_valid = (target_values > 0).sum() / len(target_values)
        pct_rest = (rest_values > 0).sum() / len(rest_values)

        avg_valid = np.mean(target_values)
        avg_ctrl = np.mean(rest_values)
        sd_valid = np.std(target_values, ddof=1)
        sd_ctrl = np.std(rest_values, ddof=1)
        esm = (avg_valid - avg_ctrl) / np.sqrt(((sd_valid**2 + sd_ctrl**2) / 2))

        if np.sum(target_values) == np.sum(rest_values):
            p_val = 1.0
        else:
            _, p_val = stats.mannwhitneyu(
                target_values, rest_values, alternative="two-sided"
            )

        return {
            "feature": feature_name,
            "p_val": p_val,
            "pct_valid": pct_valid,
            "pct_ctrl": pct_rest,
            "avg_valid": avg_valid,
            "avg_ctrl": avg_ctrl,
            "sd_valid": sd_valid,
            "sd_ctrl": sd_ctrl,
            "esm": esm,
        }

    def prepare_and_run_stat(choose, valid_group, min_exp, min_pct, n_proc):

        tmp_dat = choose[choose["DEG"] == "target"]
        tmp_dat = tmp_dat.drop("DEG", axis=1)

        counts = (tmp_dat > min_exp).sum(axis=0)

        total_count = tmp_dat.shape[0]

        info = pd.DataFrame(
            {"feature": list(tmp_dat.columns), "pct": list(counts / total_count)}
        )

        del tmp_dat

        drop_col = info["feature"][info["pct"] <= min_pct]

        if len(drop_col) + 1 == len(choose.columns):
            drop_col = info["feature"][info["pct"] == 0]

        del info

        choose = choose.drop(list(drop_col), axis=1)

        results = Parallel(n_jobs=n_proc)(
            delayed(stat_calc)(choose, feature)
            for feature in tqdm(choose.columns[choose.columns != "DEG"])
        )

        if len(results) > 0:
            df = pd.DataFrame(results)

            df = df[(df["avg_valid"] > 0) | (df["avg_ctrl"] > 0)]

            df["valid_group"] = valid_group
            df.sort_values(by="p_val", inplace=True)

            num_tests = len(df)
            df["adj_pval"] = np.minimum(
                1, (df["p_val"] * num_tests) / np.arange(1, num_tests + 1)
            )

            valid_factor = df["avg_valid"].min() / 2
            ctrl_factor = df["avg_ctrl"].min() / 2

            valid = df["avg_valid"].where(
                df["avg_valid"] != 0, df["avg_valid"] + valid_factor
            )
            ctrl = df["avg_ctrl"].where(
                df["avg_ctrl"] != 0, df["avg_ctrl"] + ctrl_factor
            )

            df["FC"] = valid / ctrl

            df["log(FC)"] = np.log2(df["FC"])
            df["norm_diff"] = df["avg_valid"] - df["avg_ctrl"]

        else:
            columns = [
                "feature",
                "valid_group",
                "p_val",
                "adj_pval",
                "avg_valid",
                "avg_ctrl",
                "FC",
                "log(FC)",
                "norm_diff",
            ]
            df = pd.DataFrame(columns=columns)
        return df

    choose = data.T

    final_results = []

    if isinstance(entities, list) and sets is None:
        print("\nAnalysis started...\nComparing selected cells to the whole set...")

        if metadata_list is None:
            choose.index = metadata["primary_names"]
        else:
            choose.index = metadata["primary_names"] + " # " + metadata["sets"]

            if "#" not in entities[0]:
                choose.index = metadata["primary_names"]
                print(
                    "You provided 'metadata_list', but did not include the set info (name # set) "
                    "in the 'entities' list. "
                    "Only the names will be compared, without considering the set information."
                )

        labels = ["target" if idx in entities else "rest" for idx in choose.index]
        valid = list(
            set(choose.index[[i for i, x in enumerate(labels) if x == "target"]])
        )

        choose["DEG"] = labels
        choose = choose[choose["DEG"] != "drop"]

        result_df = prepare_and_run_stat(
            choose.reset_index(drop=True),
            valid_group=valid,
            min_exp=min_exp,
            min_pct=min_pct,
            n_proc=n_proc,
        )

        return {"valid": valid, "control": "rest", "DEG": result_df}

    elif entities == "All" and sets is None:
        print("\nAnalysis started...\nComparing each type of cell to others...")

        if metadata_list is None:
            choose.index = metadata["primary_names"]
        else:
            choose.index = metadata["primary_names"] + " # " + metadata["sets"]

        unique_labels = set(choose.index)

        for label in tqdm(unique_labels):
            print(f"\nCalculating statistics for {label}")
            labels = ["target" if idx == label else "rest" for idx in choose.index]
            choose["DEG"] = labels
            choose = choose[choose["DEG"] != "drop"]
            result_df = prepare_and_run_stat(
                choose.copy(),
                valid_group=label,
                min_exp=min_exp,
                min_pct=min_pct,
                n_proc=n_proc,
            )
            final_results.append(result_df)

        final_results = pd.concat(final_results, ignore_index=True)

        if metadata_list is None:
            final_results["valid_group"] = [
                re.sub(" # ", "", x) for x in final_results["valid_group"]
            ]

        return final_results

    elif entities is None and sets == "All":
        print("\nAnalysis started...\nComparing each set/group to others...")
        choose.index = metadata["sets"]
        unique_sets = set(choose.index)

        for label in tqdm(unique_sets):
            print(f"\nCalculating statistics for {label}")
            labels = ["target" if idx == label else "rest" for idx in choose.index]

            choose["DEG"] = labels
            choose = choose[choose["DEG"] != "drop"]
            result_df = prepare_and_run_stat(
                choose.copy(),
                valid_group=label,
                min_exp=min_exp,
                min_pct=min_pct,
                n_proc=n_proc,
            )
            final_results.append(result_df)

        return pd.concat(final_results, ignore_index=True)

    elif entities is None and isinstance(sets, dict):
        print("\nAnalysis started...\nComparing groups...")
        choose.index = metadata["sets"]

        group_list = list(sets.keys())
        if len(group_list) != 2:
            print("Only pairwise group comparison is supported.")
            return None

        labels = [
            (
                "target"
                if idx in sets[group_list[0]]
                else "rest" if idx in sets[group_list[1]] else "drop"
            )
            for idx in choose.index
        ]
        choose["DEG"] = labels
        choose = choose[choose["DEG"] != "drop"]

        result_df = prepare_and_run_stat(
            choose.reset_index(drop=True),
            valid_group=group_list[0],
            min_exp=min_exp,
            min_pct=min_pct,
            n_proc=n_proc,
        )
        return result_df

    elif isinstance(entities, dict) and sets is None:
        print("\nAnalysis started...\nComparing groups...")

        if metadata_list is None:
            choose.index = metadata["primary_names"]
        else:
            choose.index = metadata["primary_names"] + " # " + metadata["sets"]
            if "#" not in entities[list(entities.keys())[0]][0]:
                choose.index = metadata["primary_names"]
                print(
                    "You provided 'metadata_list', but did not include the set info (name # set) "
                    "in the 'entities' dict. "
                    "Only the names will be compared, without considering the set information."
                )

        group_list = list(entities.keys())
        if len(group_list) != 2:
            print("Only pairwise group comparison is supported.")
            return None

        labels = [
            (
                "target"
                if idx in entities[group_list[0]]
                else "rest" if idx in entities[group_list[1]] else "drop"
            )
            for idx in choose.index
        ]

        choose["DEG"] = labels
        choose = choose[choose["DEG"] != "drop"]

        result_df = prepare_and_run_stat(
            choose.reset_index(drop=True),
            valid_group=group_list[0],
            min_exp=min_exp,
            min_pct=min_pct,
            n_proc=n_proc,
        )

        return result_df.reset_index(drop=True)

    else:
        raise ValueError(
            "You must specify either 'entities' or 'sets'. None were provided, which is not allowed for this analysis."
        )


def average(data):
    """
    Compute the column-wise average of a DataFrame, aggregating by column names.

    If multiple columns share the same name, their values are averaged.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame with numeric values. Columns with identical names
        will be aggregated by their mean.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same rows as the input but with unique columns,
        where duplicate columns have been replaced by their mean values.
    """

    wide_data = data

    aggregated_df = wide_data.T.groupby(level=0).mean().T

    return aggregated_df


def occurrence(data):
    """
    Calculate the occurrence frequency of features in a DataFrame.

    Converts the input DataFrame to binary (presence/absence) and computes
    the proportion of non-zero entries for each feature, aggregating by
    column names if duplicates exist.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame with numeric values. Each column represents a feature.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same rows as the input, where each value represents
        the proportion of samples in which the feature is present (non-zero).
        Columns with identical names are aggregated.
    """

    binary_data = (data > 0).astype(int)

    counts = binary_data.columns.value_counts()

    binary_data = binary_data.T.groupby(level=0).sum().T
    binary_data = binary_data.astype(float)

    for i in counts.index:
        binary_data.loc[:, i] = (binary_data.loc[:, i] / counts[i]).astype(float)

    return binary_data


def add_subnames(names_list: list, parent_name: str, new_clusters: list):
    """
    Append sub-cluster names to a parent name within a list of names.

    This function replaces occurrences of `parent_name` in `names_list` with
    a concatenation of the parent name and corresponding sub-cluster name
    from `new_clusters` (formatted as "parent.subcluster"). Non-matching names
    are left unchanged.

    Parameters
    ----------
    names_list : list
        Original list of names (e.g., column names or cluster labels).

    parent_name : str
        Name of the parent cluster to which sub-cluster names will be added.
        Must exist in `names_list`.

    new_clusters : list
        List of sub-cluster names. Its length must match the number of times
        `parent_name` occurs in `names_list`.

    Returns
    -------
    list
        Updated list of names with sub-cluster names appended to the parent name.

    Raises
    ------
    ValueError
        - If `parent_name` is not found in `names_list`.
        - If `new_clusters` length does not match the number of occurrences of
          `parent_name`.

    Examples
    --------
    >>> add_subnames(['A', 'B', 'A'], 'A', ['1', '2'])
    ['A.1', 'B', 'A.2']
    """

    if str(parent_name) not in [str(x) for x in names_list]:
        raise ValueError(
            "Parent name is missing from the original dataset`s column names!"
        )

    if len(new_clusters) != len([x for x in names_list if str(x) == str(parent_name)]):
        raise ValueError(
            "New cluster names list has a different length than the number of clusters in the original dataset!"
        )

    new_names = []
    ixn = 0
    for _, i in enumerate(names_list):
        if str(i) == str(parent_name):

            new_names.append(f"{parent_name}.{new_clusters[ixn]}")
            ixn += 1

        else:
            new_names.append(i)

    return new_names


def development_clust(
    data: pd.DataFrame, method: str = "ward", img_width: int = 5, img_high: int = 5
):
    """
    Perform hierarchical clustering on the columns of a DataFrame and plot a dendrogram.

    Uses Ward's method to cluster the transposed data (columns) and generates
    a dendrogram showing the relationships between features or samples.

    Parameters
    ----------
    data : pandas.DataFrame
        Input DataFrame with features as rows and samples/columns to be clustered.

    method : str
        Method for hierarchical clustering. Options include:
       - 'ward' : minimizes the variance of clusters being merged.
       - 'single' : uses the minimum of the distances between all observations of the two sets.
       - 'complete' : uses the maximum of the distances between all observations of the two sets.
       - 'average' : uses the average of the distances between all observations of the two sets.

    img_width : int or float, default=5
        Width of the resulting figure in inches.

    img_high : int or float, default=5
        Height of the resulting figure in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The dendrogram figure.
    """

    z = linkage(data.T, method=method)

    figure, ax = plt.subplots(figsize=(img_width, img_high))

    dendrogram(z, labels=data.columns, orientation="left", ax=ax)

    return figure


def adjust_cells_to_group_mean(data, data_avg, beta=0.2):
    """
    Adjust each cell's values towards the mean of its group (centroid).

    This function moves each cell's values in `data` slightly towards the
    corresponding group mean in `data_avg`, controlled by the parameter `beta`.

    Parameters
    ----------
    data : pandas.DataFrame
        Original data with features as rows and cells/samples as columns.

    data_avg : pandas.DataFrame
        DataFrame of group averages (centroids) with features as rows and
        group names as columns.

    beta : float, default=0.2
        Weight for adjustment towards the group mean. 0 = no adjustment,
        1 = fully replaced by the group mean.

    Returns
    -------
    pandas.DataFrame
        Adjusted data with the same shape as the input `data`.
    """

    df_adjusted = data.copy()

    for group_name in data_avg.columns:
        col_idx = [
            i
            for i, c in enumerate(df_adjusted.columns)
            if str(c).startswith(group_name)
        ]
        if not col_idx:
            continue

        centroid = data_avg.loc[df_adjusted.index, group_name].to_numpy()[:, None]

        df_adjusted.iloc[:, col_idx] = (1 - beta) * df_adjusted.iloc[
            :, col_idx
        ].to_numpy() + beta * centroid

    return df_adjusted
