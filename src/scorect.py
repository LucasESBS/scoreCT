#!/usr/bin/env python3

# Lucas Seninge (lseninge)
# Group members: Lucas Seninge
# Last updated: 01-04-2019
# File: scorect.py
# Purpose: Automated scoring of cell types in scRNA-seq.
# Author: Lucas Seninge (lseninge@ucsc.edu)
# Credits for help to: NA

"""
Script for automated cell type assignment in scRNA-seq data analyzed with the Scanpy package.
(https://github.com/theislab/scanpy)

This script contains the functions allowing to automatically assign cell types to louvain clusters infered with Scanpy,
using the gene ranking for each group. The prerequesite is therefore to have pre-analyzed scRNA-seq data as a Scanpy
object with clustering and gene ranking performed.

The method wrangle_ranked_genes() pulls out the results of the gene ranking for each cluster and wrangle it into a table
with associated rank, z-score, p-value for each gene and each cluster. This informative table is also useful for other
purposes.

This table is used to score cell types in louvain clusters using a reference. This reference is ideally a CSV with
curated markers for each cell types of interest. Choice of the table is left to the user so the method can be flexible
to any organ, species and experiment.

The method score_clusters() divide the previously mentioned table into bins of arbitrary size and create a linear
scoring scale (dependent of the number of top ranked genes retained per cluster and the size of the bin) to score each
cell type in the reference table, for each louvain cluster. A dictionary of scores for each cluster is returned.

Finally, the assign_celltypes() method takes the score dictionary and the Anndata object to assign the cell type with
the best score to each cluster. NA is assigned if there is a tie of 2 or more cell types, to avoid bias. The Anndata
object is automatically updated under Anndata.obs['Assigned type'].

Example usage:
# Import scanpy and read previously analyzed object saved to .h5
import scanpy.api as sc
import celltype_scorer as ct

adata = sc.read('/path/to/file.h5')

# Wrangle results and load reference file for markers/cell types
marker_df = ct.wrangle_ranked_genes(adata)
ref_mark = pd.read_csv('/path/to/file.csv')

# Score cell types
dict_scores = ct.score_clusters(marker_df, nb_marker=100, bin_size=20, ref_marker=ref_mark)
ct.assign_celltypes(adata, dict_score)

# Verify and plot t-SNE
print(adata.obs)
sc.pl.tsne(adata, color='Assigned type')
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


def _parse_ref(path, species, organ, context=None, comments=False):
    """
    Parses the ref file of specified species into relevant information specified by user.

    This function takes the big reference file for marker/cell types of the specified species and return a formatted
    table of the relevant fields for automated cell type annotation.

    Args:
        path (str): Path to directory with big reference.
        species (str): Specie of interest.
        organ (str): Organ of interest.
        context (str): Context of data. If None, default to 'healthy' for healthy tissue.
        comments (boolean): print comments of retained genes. Default to False.

    Returns:
        ref_df (pandas.df): Parsed reference dataframe.
    """

    import itertools

    if context is None:
        context = 'healthy'
    # Read csv of relevant specied
    specie_df = pd.read_csv(path + species + '.tsv', sep='\t', index_col=False)

    # Get relevant organ and relevant context
    sub_df = specie_df[specie_df['Organ'] == organ]
    sub_df = sub_df[sub_df['Context'] == context]

    # Create new dataframe ref_df from parsed information
    list_ct = sub_df['Cell Type/ Cell State'].unique().tolist()
    dict_marker = {
        ct: [gene for gene in sub_df[sub_df['Cell Type/ Cell State'] == ct]['Gene name(s)'].unique().tolist()]
        for ct in list_ct}

    # If several gene per row, we need to parse commas
    for ct in dict_marker.keys():
        temp_list = []
        for gene in dict_marker[ct]:
            if ',' in gene:
                temp_list.append(gene.split(','))
            else:
                temp_list.append(gene)
        # Slight modification to avoid spelling strings
        dict_marker[ct] = list(
            itertools.chain.from_iterable(itertools.repeat(x, 1) if isinstance(x, str) else x for x in temp_list))

    # Use dict to initialize df. Here order in memory doesn't mess with loading.
    ref_df = pd.DataFrame({ct: pd.Series(genes) for ct, genes in dict_marker.items()})

    # Print comments if needed
    if comments:
        for i, row in sub_df.iterrows():
            print(row['Gene name(s)'], row['Comment'])

    return ref_df


def score_clusters(ranked_marker, nb_marker, path=None,
                   species='human', organ='brain',
                   context=None, comments=False,
                   user_ref=None, bin_size=20):

    """
    Assign a score to each cell type for each cluster in the data.

    More description on usage here.

    Args:
        ranked_marker (pandas.df): A dataframe with ranked markers (from wrangle_ranked_genes()).
        nb_marker (int): number of top markers retained per cluster.
        path (str): Path to directory with provided reference (see GitHub).
        species (str): Specie of interest. Default to 'human'.
        organ (str): Organ of interest. Default to 'brain'.
        context (str): Context of data. If None, default to 'healthy' for healthy tissue.
        comments (boolean): print comments of retained genes. Default to False.
        user_ref (pandas.df): If specified, uses a custom reference file provided by user,
        as a dataframe with a list of known markers per curated cell types.
        bin_size (int): size of bins to score.

    Return:
          dict_scores (dict): Dictionary with louvain clusters as keys and a dictionary of cell type:score as values.
          (eg: 1:{CT_1: 0, CT_2:3} ...})
    """

    # Use user reference if specified, otherwise get data of specified species/organ/context
    if user_ref is not None:
        ref_marker = user_ref
    else:
        ref_marker = _parse_ref(path=path, species=species, organ=organ, context=context, comments=comments)

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
            bin_df = sub_df[(k * bin_size):bin_size + (k * bin_size)]
            # Use ref to score
            # We assume format column = cell types, each row is a different marker
            for cell_type in list(ref_marker):
                # Get length of intersection between ref marker and bin , multiply by score of the associated bin
                # score_list[-(1+k)] because k can be 0 for bin purposes
                score_i = score_list[-(1 + k)] * len(set(ref_marker[cell_type]).intersection(set(bin_df['gene'])))
                # Add score to existing score or create key
                if cell_type in dict_scores[clust]:
                    dict_scores[clust][cell_type] += score_i
                else:
                    dict_scores[clust][cell_type] = score_i

    return dict_scores


def _highlight_max(s):
    """
    Highlights the maximum in a Series yellow.
    From Pandas package tutorial (https://pandas.pydata.org/pandas-docs/stable/style.html)
    """
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


def scoring_summary(scoring_dict):
    """
    Produces a summary of the scoring function after scoring clusters with cell types.

    Args:
        scoring_dict (dict): Output from score_clusters() as a dictionary. See corresponding function.

    Returns:
        prints a pandas dataframe with highlighted best scores.
    """
    # Import matplotlib to trigger font cache stuff
    import matplotlib.pyplot

    summary_df = pd.DataFrame(scoring_dict)
    print('Rows: Cell types / Columns: Clusters')
    return summary_df.style.apply(_highlight_max)



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

    return "Cell types assigned in Anndata.obs['Assigned types']"
