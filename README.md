# scoreCT
Automated Cell Type Annotation.

Script to automate cell type annotation on scRNA-seq data analyzed with the Scanpy package (https://github.com/theislab/scanpy).
The script uses a reference CSV provided by user with markers associated to curated cell types, which is used to score and assign a cell type to each louvain cluster in the Anndata object (which encapsulates the scRNA-seq analyzed beforehand).
To run, louvain clustering and gene ranking per group must have been performed and be present in the Anndata object.


## Getting Started

scRNA-seq analysis packages allow to perform clustering and get biomarkers for inferred cluster in order to explore cell types in a population of cells. However, manual curation of cell types can be long and tidious. The goal of this repo is to provide biologist with a script to automate cell type annotation of clusters in data analyzed with Scanpy, by using their own list of markers and curated cell types. The formating of this table is left to the user, but is ideally a CSV with each row being a cell type, followed by its associated markers.

### Prerequisites

Scanpy (Wolf et al., 2018) must be installed to run the prerequesite analysis. See [Scanpy repo](https://github.com/theislab/scanpy) for tutorials on how to run Scanpy on your data.
Scanpy can be installed by running:

```
pip install scanpy
```

### Installing

Clone this repo in your home folder by running:

```
git clone https://github.com/LucasESBS/scoreCT
```

See the jupyter notebook in the example folder to run an example. Example data are in scoreCT/data/

```
cd scoreCT/example
jupyter notebook
```
