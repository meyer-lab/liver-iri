import scipy.cluster.hierarchy as sch


def reorder_table(df, plot_ax=None):
    """
    Reorder a table's rows using hierarchical clustering.
    Parameters:
        df (pandas.DataFrame): data to be clustered; rows are treated as samples
            to be clustered
    Returns:
        df (pandas.DataFrame): data with rows reordered via heirarchical
            clustering
    """
    y = sch.linkage(df.to_numpy(), method="centroid")

    if plot_ax:
        index = sch.dendrogram(y, orientation="right", ax=plot_ax)["leaves"]
    else:
        index = sch.dendrogram(y, orientation="right", no_plot=True)["leaves"]

    return df.iloc[index, :]
