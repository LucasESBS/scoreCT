#!/usr/bin/env python3

# Lucas Seninge (lseninge)
# Group members: Lucas Seninge
# Last updated: 04-6-2022
# File: scorect_api.py
# Purpose: Automated scoring of cell types in scRNA-seq.
# Author: Lucas Seninge (lseninge@ucsc.edu)
# Credits for help to: Duncan McColl


# Import packages
import pandas as pd
import numpy as np
import scanpy as sc
import requests
import itertools
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns


# I/O functions
def read_markers_from_file(filepath, ext=None):
    """
    Read a cell marker file from accepted file formats.
    Currently supported: csv, tsv.
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
            ct_dict[values[0]] = values[2:]
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


def wrangle_ranks_from_anndata(anndata, cluster_key='louvain'):
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
    for i in top_score.columns:
        concat = pd.concat([top_score[[str(i)]], top_adjpval[str(i)], top_gene[[str(i)]]], axis=1, ignore_index=True)
        concat['cluster_number'] = i
        col = list(concat.columns)
        col[0], col[1], col[-2] = 'z_score', 'adj_pvals', 'gene'
        concat.columns = col
        marker_df = marker_df.append(concat)
    return marker_df


def ranks_from_file(filepath, sep='\t'):
    """
    Read ranked genes per cluster from a file. Uses the read_markers_from_file api.
    Exepcts a file formatted as followed: each column is a cluster and genes are ranked in each column from top to bottom. Expects column headers for cluster labels (or number). See example file.

    Parameters
    ----------
    filepath
        path to file containing the ranked genes
    sep
        separator
    """
    ranked_df = pd.read_csv(filepath, sep=sep, index_col=0)
    clust = []
    # Reformat to match scoreCT input format
    for c in list(ranked_df):
        clust += [c]*len(ranked_df)
    genes = ranked_df.values.flatten(order='F')
    return pd.DataFrame(index=np.arange(len(clust)), data={'cluster_number':clust, 'gene':genes})


# Scoring functions
def _get_score_scale(nb_bins, scale='linear'):
    """
    Return a scoring scheme for the bins.
    """
    scores = np.arange(1, nb_bins+1)[::-1]
    scale_dict = {'linear':np.array}
    return scale_dict[scale](scores)


def _score_one_celltype(nb_bins, ranked_genes, marker_list, background, score_scheme, ct_name, null_model, n_permutations):
    """
    Helper function that scores one cell type for one cluster and take care of the bining.
    Returns a single score and associated p-value.
    """
    # Get score
    score = _score_function(top_genes=ranked_genes, 
                            markers=marker_list,
                            m_bins=nb_bins,
                            score_scheme=score_scheme) 
    # Get pvalue associated with returned score
    N = len(background)
    K = len(ranked_genes)
    n = len(set(marker_list).intersection(set(background)))
    # Check for which null model to use - tolerance value depends on n and N (n<<N)
    if (n/N) > 3.33e-3 and null_model=='multinomial':
        print('Marker genes number for %s is too big (n=%.i) for multinomial approximation to work. Switching to permutation test.'%(ct_name, n))
        null_model='random'
    # Get pvalue for null model
    if null_model=='random':
        pval = _pval_null_permutation(score=score, all_genes=background, markers=marker_list, K_top=K, m_bins=nb_bins, n_permutations=n_permutations)
    else:
        pval = _pval_null_multinomial(score=score, N_genes=N, K_top=K, n_markers=n, m_bins=nb_bins)
    return score, pval


def _score_celltypes(nb_bins, ranked_genes, marker_ref, background, score_scheme, null_model, n_permutations):
    """
    Score all celltypes in the reference for one cluster.
    The reference is a dataframe with cell types as columns.
    """
    # Initialize empty score vector
    score_cluster = np.zeros((len(list(marker_ref)),))
    pval_cluster = np.zeros((len(list(marker_ref)),))
    # Iterate on cell types
    celltypes = list(marker_ref)
    for i,c in enumerate(celltypes):
        # Score each cell type
        score_ct, pval_ct = _score_one_celltype(nb_bins=nb_bins,
                                                ranked_genes=ranked_genes,
                                                marker_list=marker_ref[celltypes[i]],
                                                background=background,
                                                score_scheme=score_scheme,
                                                ct_name=c,
                                                null_model=null_model,
                                                n_permutations=n_permutations)
        score_cluster[i] = score_ct
        pval_cluster[i] = pval_ct

    return score_cluster, pval_cluster

def _score_function(top_genes, markers, m_bins, score_scheme=None):
    """
    scoreCT scoring function.
    """
    if score_scheme is None:
        score_scheme = [i for i in range(1,m_bins+1)][::-1]
    score = 0
    size_bin = len(top_genes)//m_bins
    for k in range(m_bins):
        sub_rank = top_genes[k*size_bin : (k*size_bin)+size_bin]
        score += (score_scheme[k] * len(set(sub_rank).intersection(set(markers))))
    return score

def _pval_null_permutation(score, all_genes, markers, K_top, m_bins, n_permutations):
    """
    Compute p-value from a permutation test. Gene ranking is randomized and score is re-computed N times to approximate the null distribution.
    """
    s_perm = []
    for n in range(n_permutations):
        perm_genes = random.sample(all_genes, len(all_genes))
        top_genes = perm_genes[:K_top]
        s = _score_function(top_genes, markers, m_bins)
        s_perm.append(s)
    return np.mean(np.array(s_perm)>=score)

def _pval_null_multinomial(score, N_genes, K_top, n_markers, m_bins):
    """
    Get one tail test p-value associated to the input score given null hypothesis. This approximation only holds well for a small amount of markers (n<100). 
    For a description of the multinomial null-model approximation see documentation.
    """
    # Define probability parameters
    multi_dist = np.array([(N_genes-n_markers)/N_genes]+[(1-((N_genes-n_markers)/N_genes))/m_bins]*m_bins)
    # Get polynomial coefficients of degree K
    coeff = np.polynomial.polynomial.polypow(multi_dist, K_top)
    return np.sum(coeff[score:])


def assign_celltypes(ct_pval_df, ct_score_df, cluster_assignment, cutoff=0.1):
    """
    Assign a cell type to each cell based on its cluster assignment and the scoreCT results.
    """
    # Make sure cluster assignments are categories and not int or something
    cluster_assignment = cluster_assignment.astype('category')
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
            clust_to_ct[cluster] = ct_score_df.loc[cluster][min_idx].idxmax()
    # get a new pandas series with cell as indexes and cell type as value
    ct_assignments = cluster_assignment.map(clust_to_ct)
    return ct_assignments


def celltype_scores(nb_bins, ranked_genes, K_top,  marker_ref, background_genes, null_model='multinomial', n_permutations=1000):
    """
    Score every cluster in the ranking.
    If a tuple is passed to K_top, it will be interpreted as (start,end). If a single integer is passed, start is 0.

    Parameters
    ----------
        nb_bins
            Number of bins to use to divide the gene ranking
        ranked_genes
            Dataframe with ranked genes for each cluster (see docs)
        K_top
            Number of top genes to include in the scoring
        marker_ref
            Reference table containing prior information on cell types and marker genes
        background_genes
            List of all genes used in the dataset
        null_model
            Null model to use for p-value computation. One of `random` or `multinomial`
        n_permutations
            Number of permutations for pvalue computation. Only used if ```null_model='random'```
    Returns
    -------
        pval_df
            Dataframe containing the pvalue associated with each celltype/cluster pair
        score_df
            Dataframe containing the scores associated with each celltype/cluster pair
    """
    assert null_model in ['random','multinomial'], "null_model must be one of ['random','multinomial']"
    # Check bounds
    if type(K_top) is tuple:
        start, end = K_top[0], K_top[1]
    elif type(K_top) is int:
        start, end = 0, K_top
    else:
        raise ValueError('K_top should be an integer or a tuple representing bounds.')
    score_scheme = _get_score_scale(nb_bins=nb_bins, scale='linear')
    # Initialize empty array for dataframe
    cluster_unique = np.unique(ranked_genes['cluster_number'].values)
    score_array = np.zeros((len(cluster_unique), len(list(marker_ref))))
    pval_array = np.zeros((len(cluster_unique), len(list(marker_ref))))
    for cluster_i in range(len(cluster_unique)):
        mask = ranked_genes['cluster_number'] == cluster_unique[cluster_i]
        valid_cluster = ranked_genes[mask][start:end]
        cluster_scores, cluster_pval = _score_celltypes(nb_bins=nb_bins,
                                                        ranked_genes=valid_cluster['gene'],
                                                        marker_ref=marker_ref,
                                                        background=background_genes,
                                                        score_scheme=score_scheme,
                                                        null_model=null_model,
                                                        n_permutations=n_permutations
                                                        )

        score_array[cluster_i, : ] = cluster_scores
        pval_array[cluster_i, : ] = cluster_pval
    # Array to df
    score_df = pd.DataFrame(index=cluster_unique, data=score_array, columns=list(marker_ref))
    pval_df = pd.DataFrame(index=cluster_unique, data=pval_array, columns=list(marker_ref))
    return pval_df, score_df



# One liner API for end user
def scorect(
            adata,
            marker_path,
            K_top,
            m_bins,
            null_model='multinomial',
            n_permutations=1000,
            cluster_key='louvain',
            pval_cutoff=0.1,
            diff_kwargs=None,
            return_copy=False
            ):
    """
    Runs scoreCT annotation procedure on input AnnData object.
    
    Parameters
    ----------
    adata
        input AnnData object
    marker_path
        path to marker genes file
    K_top
        number of top differential genes to consider
    m_bins
        number of bins to use for score
    null_model
        null model used for p-value computation. One of ['random','multinomial']
    n_permutations
        number of permutations for permutation test. Only used if null_model='random'
    cluster_key
        key in adata.obs with cluster labels/numbers
    pval_cutoff
        p-value cutoff to assign a cluster as 'NA'
    diff_kwargs
        kwargs to be passed to scanpy rank_gene_groups function (if not already ran by user)
    return_copy
        if True, a copy of adata is made and returned after scoreCT is ran
    """
    if return_copy:
        adata = adata.copy()
    adata.uns['_scorect'] = {}
    # Check if rank_gene_groups as been ran
    if 'rank_genes_groups' not in adata.uns.keys():
        print('No differential test found in input AnnData object. Running Scanpy rank_gene_groups...')
        if diff_kwargs is None:
            print('Using default values for differential test')
            diff_kwargs = {'groupby':cluster_key, 'n_genes':len(adata.raw.var), 'use_raw':True}
        sc.tl.rank_genes_groups(adata, **diff_kwargs)
    # Read marker file
    print('Reading markers...')
    markers = read_markers_from_file(marker_path)
    background = adata.raw.var.index.tolist()
    gene_df = wrangle_ranks_from_anndata(adata, cluster_key=cluster_key)
    # Run scorect procedure
    print('Scoring cell types in reference...')
    ct_pval, ct_score = celltype_scores(nb_bins=m_bins,
                                        ranked_genes=gene_df,
                                        K_top = K_top,
                                        marker_ref=markers,
                                        background_genes=background,
                                        null_model=null_model,
                                        n_permutations=n_permutations
                                        )
    # Assign cell types
    print("Assigning cell types in adata.obs['scoreCT']...")
    ct_assign = assign_celltypes(cluster_assignment=adata.obs[cluster_key],
                                 ct_pval_df=ct_pval,
                                 ct_score_df=ct_score,
                                 cutoff=pval_cutoff)
    
    adata.obs['scoreCT'] = ct_assign
    # Store output in adata
    adata.uns['_scorect']['pvals'] = ct_pval
    adata.uns['_scorect']['scores'] = ct_score
    if return_copy:
        return adata
    return


# Util functions : plotting, ...
def plot_pvalue(pval_df, clusters, cutoff=0.1, n_types=None):
    """
    Dot plot of -log10(pvalue) for each cell type in passed clusters.
    """
    # If only one cluster is input as int, convert to list
    if type(clusters) == int:
        clusters = [clusters]
    if n_types is None:
        n_types = len(list(pval_df))
    # Iterate
    for cluster in clusters:
        sub_serie = pval_df.loc[cluster].sort_values(ascending=True)
        # Only plot cell types below cutoff
        sub_serie = -np.log10(sub_serie[sub_serie.values < cutoff])
        sns.scatterplot(sub_serie.index[:n_types], sub_serie.values[:n_types], marker='o', s=80)
        plt.ylabel('-log10(p-value)')
        plt.xticks(rotation=60)
        plt.title('-log10(p-value) plot for cluster ' + str(cluster))
        plt.grid()
        plt.show()


def reassign_celltype(cluster_n, cell_type):
    """
    Function for the user to reassign a cell type. This decision can be motivated from investigating the pvalue plot
    and/or the marker list for a certain cluster.
    """
    raise NotImplementedError




