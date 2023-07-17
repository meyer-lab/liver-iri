import mygene
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
    y = sch.linkage(df.to_numpy(), method='centroid')
    index = sch.dendrogram(y, orientation='right', no_plot=True)['leaves']
    return df.iloc[index, :]


def lookup_genes(ensembl_genes, scopes='ensembl.gene'):
    """Converts ensembl gene IDs to gene names.

    Args:
        ensembl_genes (list[str]): ensembl gene IDs.
        scopes (str): mygene scope to use.

    Returns:
        symbols (list[str]): Translated gene names.
    """
    symbols = []
    ensembl_genes = [gene.split('.')[0] for gene in ensembl_genes]

    mg = mygene.MyGeneInfo()
    queries = mg.querymany(
        ensembl_genes,
        return_all=True,
        scopes=scopes
    )

    for query in queries:
        symbol = query.get('symbol')
        if symbol is not None:
            symbols.append(symbol)

    return symbols
