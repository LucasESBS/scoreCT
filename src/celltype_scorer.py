#!/usr/bin/env python3

# Lucas Seninge (lseninge)
# Group members: Lucas Seninge
# Last updated: 01-04-2019
# File: celltype_scorer.py
# Purpose: Automated scoring of cell types in scRNA-seq.
# Author: Lucas Seninge
# Credits for help to: NA

"""
Script description goes here.

Example usage: ./
"""

# Import packages
import pandas as pd


def wrangle_ranked_genes(anndata):
    """
    Wrangle results from the ranked_genes_groups function of Scanpy (Wolf et al., 2018) on louvain clusters.

    This function creates a pandas dataframe report of the top N genes used in the ranked_genes_groups search.

    Args:
        anndata (Anndata object): object from Scanpy analysis.

    Return:
        marker_df (pd.dataframe): dataframe with top N ranked genes per clusters with names, z-score and pval.
    """

    # Get number of top ranked genes per groups
    nb_marker = len(anndata.uns['rank_genes_groups']['names'])
    print('Number of markers used in ranked_gene_groups: ', nb_marker)

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


def score_clusters(ranked_marker, nb_marker, ref_marker, bin_size=20):
    """
    Assign a score to each cell type for each cluster in the data.

    Args:
        ranked_marker (pandas.df): A dataframe with ranked markers (from wrangle_ranked_genes()).
        nb_marker (int): number of top markers retained per cluster.
        ref_marker (pandas.df): A dataframe with a list of known markers per curated cell types.
        bin_size (int): size of bins to score.

    Return:
          dict_scores (dict): Dictionary with louvain clusters as keys and a dictionary of cell type:score as values.
          (eg: 1:{CT_1: 0, CT_2:3} ...})
    """

    # Initialize score dictionary {cluster_1: {cell_type1: X, cell_type2: Y} ...}
    dict_scores = {}
    # Scale scores -- linear
    score_list = range(1, int(nb_marker / bin_size) + 1)
    # Iterate on clusters
    for clust in set(ranked_marker['cluster_number']):
        # Initialize empty dict for given cluster
        dict_scores[clust] = {}
        sub_df = ranked_marker[ranked_marker['cluster_number'] == clust]
        # Get individual bins
        for k in range(int(nb_marker / bin_size)):
            bin_df = sub_df[k:k + bin_size]
            # Use ref to score
            # We assume format column = cell types, each row is a different marker
            for cell_type in list(ref_marker):
                # Get length of intersection between ref marker and bin , multiply by score of the associated bin
                score_i = score_list[-k] * len(set(ref_marker[cell_type]).intersection(set(bin_df['gene'])))
                dict_scores[clust][cell_type] = score_i

    return dict_scores


def assign_celltypes(anndata, dict_scores):
    """
    Given a dictionary of cell type scoring for each cluster, assign cell type with max. score
    to given cluster.

    Args:
        anndata (AnnData object): Scanpy object with analyzed data (clustered cells).
        dict_scores (dict): Dictionary of cell type scoring per cluster.

    Returns:
        anndata (AnnData object): Scanpy object with assigned cell types.
    """
    
    # Initialize new metadata column in Anndata object
    anndata.obs['Assigned type'] = ''
    # Iterate on clusters in dict_scores
    for cluster in dict_scores.keys():
        # Get cell type with best score
        max_value = max(dict_scores[cluster].values())
        if len({key for key, value in dict_scores[cluster].items() if value == max_value}) > 1:
            assign_type = 'NA'
        else:
            assign_type = max(dict_scores[cluster], key=dict_scores[cluster].get)
        # Update
        anndata.obs.loc[anndata.obs['louvain'] == str(cluster), 'Assigned type'] = assign_type

    return anndata
