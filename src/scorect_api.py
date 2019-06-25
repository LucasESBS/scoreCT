# Import packages
import pandas as pd
import numpy as np
import requests
import itertools
import re
import matplotlib.pyplot as plt
import seaborn as sns


# I/O functions
def read_markers_from_file(filepath, ext=None):
    """
    Read a cell marker file from accepted file formats.
    Currently supported: csv, tsv.
    To add: excel, gmt
    See doc for formatting.
    """
    # Get extension
    if ext is None:
        ext = filepath.split('.')[-1]
    dict_format = {"csv":read_markers_csv, "tsv":read_markers_tsv, "gmt":read_markers_gmt}
    # Read with correct function
    marker_df = dict_format[ext](filepath)
    # To dict
    return marker_df


def read_markers_csv(filepath):
    """
    Read a marker file from a csv.
    """
    marker_df = pd.read_csv(filepath, sep=',', index_col=0)
    return marker_df


def read_markers_tsv(filepath):
    """
    Read a marker file from a tsv.
    """
    marker_df = pd.read_csv(filepath, sep='\t', index_col=0)
    return marker_df


def read_markers_gmt(filepath):
    """
    Read a marker file from a gmt.
    """
    ct_dict = {}
    with open(filepath) as file_gmt:
        for line in file_gmt:
            values = line.strip().split('\t')
            ct_dict[values[0]] = values[1:]
    return pd.DataFrame({key:pd.Series(value) for key, value in ct_dict.items()})


def get_markers_from_db(species, tissue, url=None):
    """
    Get reference from Cell Marker Database
    """
    if url is None:
        url = "http://biocc.hrbmu.edu.cn/CellMarker/download/all_cell_markers.txt"
    req = requests.get(url)
    parsed_file = []
    for chunk in req.iter_lines():
        chunk = chunk.decode("utf-8").split('\t')
        parsed_file.append(chunk)

    marker_df = pd.DataFrame(columns=parsed_file[0], data=parsed_file[1:])

    # Subset for interesting information
    sub_df = marker_df[marker_df['speciesType'] == species]
    sub_df = sub_df[sub_df['tissueType'] == tissue]

    # Parse information to create new reference
    list_ct = sub_df['cellName'].unique().tolist()
    # Use re to remove special characters
    dict_marker = {
        ct: [re.split('\W+', gene) for gene in sub_df[sub_df['cellName'] == ct]['geneSymbol'].unique().tolist()]
        for ct in list_ct}

    # merge nested lists without spelling strings
    for ct in dict_marker.keys():
        dict_marker[ct] = list(
            itertools.chain.from_iterable(itertools.repeat(x, 1) if isinstance(x, str) else x for x in dict_marker[ct]))

    # Use dict to initialize df. Here order in memory doesn't mess with loading.
    ref_df = pd.DataFrame({ct: pd.Series(genes) for ct, genes in dict_marker.items()})
    return ref_df


def get_background_genes_server(species, url=None):
    """
    Get background genes from server for null-hypothesis formulation.
    """
    # Convert name to lowercase to access
    species = species.lower()
    if url is None:
        url = 'http://public.gi.ucsc.edu/~lseninge/' + species + '_genes.tsv'
    response = requests.get(url)
    gene_list = []
    lines = response.iter_lines()
    # Skip first line
    next(lines)
    for chunk in lines:
        chunk = chunk.decode("utf-8")
        gene_list.append(chunk)
    return gene_list


def get_background_genes_file(filepath):
    """
    Get background genes from local file for null-hypothesis formulation.
    """
    gene_list = []
    with open(filepath) as file_gene:
        for line in file_gene:
            gene_list.append(line.strip())
    return gene_list


def wrangle_ranks_from_anndata(anndata):
    """
    Wrangle results from the ranked_genes_groups function of Scanpy (Wolf et al., 2018) on louvain clusters.
    """

    # Get number of top ranked genes per groups
    nb_marker = len(anndata.uns['rank_genes_groups']['names'])
    print('Wrangling: Number of markers used in ranked_gene_groups: ', nb_marker)
    print('Wrangling: Groups used for ranking:', anndata.uns['rank_genes_groups']['params']['groupby'])
    # Wrangle results into a table (pandas dataframe)
    top_score = pd.DataFrame(anndata.uns['rank_genes_groups']['scores']).loc[:nb_marker]
    top_adjpval = pd.DataFrame(anndata.uns['rank_genes_groups']['pvals_adj']).loc[:nb_marker]
    top_gene = pd.DataFrame(anndata.uns['rank_genes_groups']['names']).loc[:nb_marker]
    marker_df = pd.DataFrame()
    # Order values
    for i in range(len(top_score.columns)):
        concat = pd.concat([top_score[[str(i)]], top_adjpval[str(i)], top_gene[[str(i)]]], axis=1, ignore_index=True)
        concat['cluster_number'] = i
        col = list(concat.columns)
        col[0], col[1], col[-2] = 'z_score', 'adj_pvals', 'gene'
        concat.columns = col
        marker_df = marker_df.append(concat)
    return marker_df


def ranks_from_file(filepath, ext=None):
    """
    Read ranked genes per cluster from a file. Uses the read_markers_from_file api.
    See doc for formatting.
    """
    raise NotImplementedError


# Scoring functions
def _get_score_scale(nb_bins, scale='linear'):
    """
    Return a scoring scheme for the bins.
    """
    scores = np.arange(1, nb_bins+1)[::-1]
    scale_dict = {'linear': np.array, 'square': np.square, 'log': np.log}
    return scale_dict[scale](scores)


def _score_one_celltype(nb_bins, ranked_genes, marker_list, score_scheme):
    """
    Helper function that scores one cell type for one cluster and take care of the bining.
    Returns a single score.
    """
    # Initialize score
    score = 0
    # Iterate
    size_bin = len(ranked_genes)//nb_bins
    for k in range(nb_bins):
        sub_rank = ranked_genes[k*size_bin : (k*size_bin)+size_bin]
        score += (score_scheme[k] * len(set(sub_rank).intersection(set(marker_list))))
    return score


def _score_celltypes(nb_bins, ranked_genes, marker_ref, score_scheme):
    """
    Score all celltypes in the reference for one cluster.
    The reference is a dataframe with cell types as columns.
    """
    # Initialize empty score vector
    score_cluster = np.zeros((len(list(marker_ref)),))
    # Iterate on cell types
    celltypes = list(marker_ref)
    for i in range(len(celltypes)):
        # Score each cell type
        score_ct = _score_one_celltype(nb_bins=nb_bins,
                                       ranked_genes=ranked_genes,
                                       marker_list=marker_ref[celltypes[i]],
                                       score_scheme=score_scheme)
        score_cluster[i] = score_ct

    # Implement length correction here - ignore Na value if present in the ref
    len_vect = marker_ref.count().values
    return score_cluster/len_vect


def _randomize_ranking(ranked_genes, background_genes):
    """
    Randomize genes in cluster gene rankings.
    TO IMPROVE
    """
    copy_df = ranked_genes.copy()
    # add code here to process cluster by cluster instead of the whole df in one time
    rd_pick = np.random.choice(background_genes, len(copy_df['gene']))
    copy_df['gene'] = rd_pick
    return copy_df


def _random_score_compare(ct_score_df, random_score_df):
    """
    Compare randomized ranking scores to initial scores and output binary df with 1 if random score is greater, 0 else.
    """
    binary_df = (ct_score_df < random_score_df).astype(float)
    return binary_df


def assign_celltypes(ct_pval_df, ct_score_df, cluster_assignment, cutoff=0.1):
    """
    Assign a cell type to each cell based on its cluster assignment and the scoreCT results.
    """
    # Build dict cluster:cell type according to cutoff. Use score to break ties.
    clust_to_ct = {}
    for cluster, serie in ct_pval_df.iterrows():
        min_pval = serie.min()
        min_idx = np.where(serie.values == min_pval)[0]
        if min_pval > cutoff:
            clust_to_ct[cluster] = 'NA'
        elif len(min_idx) == 1:
            clust_to_ct[cluster] = serie.index[min_idx[0]]
        else:
            # Subset the score_df
            clust_to_ct[cluster] = ct_score_df.iloc[cluster].values[min_idx].idxmax()
    # get a new pandas series with cell as indexes and cell type as value
    ct_assignments = cluster_assignment.map(clust_to_ct)
    return ct_assignments


def celltype_scores(nb_bins, ranked_genes, marker_ref, score_scheme):
    """
    Score every cluster in the ranking.
    """
    # Initalize empty array for dataframe
    cluster_unique = np.unique(ranked_genes['cluster_number'].values)
    score_array = np.zeros((len(cluster_unique),len(list(marker_ref))))
    for cluster_i in cluster_unique:
        mask = ranked_genes['cluster_number'] == cluster_i
        valid_cluster = ranked_genes[mask]
        cluster_scores = _score_celltypes(nb_bins=nb_bins,
                                          ranked_genes = valid_cluster['gene'],
                                          marker_ref=marker_ref,
                                          score_scheme=score_scheme)
        score_array[cluster_i,:] = cluster_scores
    # Array to df
    return pd.DataFrame(index=cluster_unique, data=score_array, columns=list(marker_ref))


def celltype_pvalues(nb_bins, ranked_genes, marker_ref, background_genes, scale='linear', n_iter=1000):
    """
    Main function for collection scores and pvalues from ranking and reference.
    """
    # Get a score scheme
    score_scheme = _get_score_scale(nb_bins=nb_bins, scale=scale)
    # Get initial cluster scores
    ct_scores_df = celltype_scores(nb_bins=nb_bins,
                                   ranked_genes=ranked_genes,
                                   marker_ref=marker_ref,
                                   score_scheme=score_scheme)

    # Initialize pvalue df for randomize scoring
    ct_pvalues_df = pd.DataFrame(columns=ct_scores_df.columns, index=ct_scores_df.index).fillna(0)
    for i in range(n_iter):
        random_ranking = _randomize_ranking(ranked_genes=ranked_genes, background_genes=background_genes)
        random_scores = celltype_scores(nb_bins=nb_bins,
                                        ranked_genes=random_ranking,
                                        marker_ref=marker_ref,
                                        score_scheme=score_scheme)
        ct_pvalues_df += _random_score_compare(ct_score_df=ct_scores_df, random_score_df=random_scores)

    # Divide by number of iteration and return original scores and pvalue
    return ct_pvalues_df/n_iter, ct_scores_df


# Util functions : plotting, ...
def plot_pvalue(pval_df, clusters, cutoff=0.1):
    """
    Dot plot of pvalue for each cell type in passed clusters.
    """
    # If only one cluster is input as int, convert to list
    if type(clusters) == int:
        clusters = list(clusters)
    # Iterate
    for cluster in clusters:
        sub_serie = pval_df.iloc[cluster].sort_values(ascending=True)
        # Only plot cell types below cutoff
        sub_serie = sub_serie[sub_serie.values < cutoff]
        sns.scatterplot(sub_serie.index, sub_serie.values, marker='+', color='red', s=150)
        plt.ylabel('P-value')
        plt.xticks(rotation=60)
        plt.title('P-value plot for cluster ' + str(cluster))
        plt.show()





