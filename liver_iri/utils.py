import scipy.cluster.hierarchy as sch


def reorder_table(df):
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
    index = sch.dendrogram(y, orientation="right", no_plot=True)["leaves"]
    return df.iloc[index, :]
