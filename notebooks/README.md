# Detailed Overview of the Notebooks



### **Notebook 01_scRNA_analysis_preprocessing.ipynb: Basic Demo Workflow**

By the completion of this notebook, a user will be able to do the following:

- Load and Preprocess the data
    - Load a sparse matrix in h5ad format using Scanpy
    - Preprocess the data, implementing standard QC metrics to assess cell and gene quality per cell, as well as per gen

- QC cells visually to understand the data
    - Users will learn how to visually inspect 5 different plots that help reflect quality control metrics for single cell data to:
        - Identify stressed or dying cells undergoing apoptosis
        - Empty droplets or dead cells
        - Cells with abnormal gene counts
        - Low quality or overly dominant cells

- Filter unusual cells
    - Users will learn how to remove cells with an extreme number of genes expressed
    - Users will filter out cells with an unusual amount of mitochondrial content

- Remove unwanted sources of variation
    - Select most variable genes to better inform analysis and increase computational efficiency
    - Regress out additional technical variation that we observed in the visual plots (Note, this can actually remove biologically relevant information, and would need to be carefully considered with a more complex data set)
    - Standardize by using a z-score transformation

- Cluster and visualize data
    - Implement PCA to reduce computational complexity. We use the GPU-accelerated PCA implementation from cuML, which significantly speeds up computation compared to CPU-based methods.
    - Identify batch effects visually by generating a UMAP plot with graph-based clustering

- Batch Correction and analysis
    - Remove assay-specific batch effects using Harmony
    - Re-compute the k-nearest neighbors graph and visualize using the UMAP.
    - Perform graph-based clustering
    - Visualize using other methods (tSNE)

- Explore the biological information from the data
    - Differential expression analysis: Identifying marker genes for cell types
        - Implement logistic regression
        - Rank genes that distinguish cell types
    - Trajectory analysis
        - Implement a diffusion map to understand the progress of cell types

### **Notebook 02_scRNA_analysis_extended.ipynb - Decoupler-GPU: Accelerated Transcriptional Regulatory Analysis**

This is an overview of methods that can be used to investigate transcriptional regulation. Although the tutorial does not dive deeply into these models, the notebook is used to reflect a difference in speed that is benefitted by GPU.

### **Notebook 03_scRNA_analysis_with_pearson_residuals.ipynb - A different way to normalize with Pearson Residuals**

By the completion of this notebook, a user will be able to remove unwanted sources of variation using Pearson Residuals to normalize. This is a different approach to analysis from Notebook 01. After filtering, we introduce a normalization step toa address the potential issues in how we previously removed unwanted sources of variation.

### **Notebook 04_scRNA_analysis_dask_out_of_core.ipynb - Scale analysis to 11M cells easily and quickly leveraging Dask**

By the completion of this notebook, a user will be able to do the following:
- Create a local DASK cluster
- Load data directly into an AnnData object using a Dask array
- Filter cells and genes without additional computation
- Log normalize the data and identify highly variable genes
- Scale gene expression
- Compute PCA using GPU acceleration

### **Notebook 05_spatial_demo.ipynb - Spatial Analysis with rapids-singlecell and Squidpy**

By the completion of this notebook, a user will be able to do the following:
- Compute spatial autocorrelation using Moran's I (global spatial structure) and Geary's C (local expression differences) to identify niche-specific expression and spatial gradients
- Perform co-occurrence analysis to measure the probability of cell-type pairs co-localizing across varying spatial distances, revealing tissue organization and cell-cell interactions
- Leverage GPU-accelerated spatial implementations in rapids-singlecell for significant speedups over CPU-based methods

### **Notebook 06_scRNA_analysis_1.0M_brain_example.ipynb - End to end example on 1M Brain Cells**

In this notebook, a user will perform similar steps as the Notebook 01 (end to end demo), but on 1M brain cells.

### **Notebook 07_perturbation_analysis_invivo_brain_example.ipynb - Perturbation Analysis: In Vivo Brain CRISPR Atlas Example**

This notebook demonstrates GPU-accelerated perturbation analysis on a whole-brain single-nucleus CRISPR atlas (~3.5M cells, targeting ~2,000 genes across hundreds of neuronal types) using RAPIDS-singlecell. By the completion of this notebook, a user will be able to do the following:

- Load a preprocessed Zarr dataset from HuggingFace into a GPU-ready AnnData object using Dask for out-of-core processing
- Configure a local CUDA cluster and tune GPU memory management with RAPIDS Memory Manager (RMM)
- Build GPU-accelerated PCA embeddings on the full dataset and visualize cell populations with UMAP
- Compute pairwise E-distances between perturbation groups and non-targeting controls across all cell types to build a global perturbation-response map
- Overlay external essential gene lists (Blomen et al., Science 2015) in select cell types and use Mann-Whitney U tests to show that essential gene perturbations produce significantly higher E-distances
