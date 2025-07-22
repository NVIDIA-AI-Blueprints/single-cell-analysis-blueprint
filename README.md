# <div align="left"><img src="https://rapids.ai/assets/images/rapids_logo.png" width="90px"/>&nbsp; <div align="left"><img src="https://canada1.discourse-cdn.com/flex035/uploads/forum11/original/1X/dfb6d71c9b8deb73aa10aa9bc47a0f8948d5304b.png" width="90px"/>&nbsp;
# **Single-Cell Analysis Blueprint**


This repository houses tutorial notebooks to run GPU-accelerated single-cell analysis workflows using [RAPIDS-singlecell](https://rapids-singlecell.readthedocs.io/en/latest/), a GPU accelerated library developed by [scverse®](https://github.com/scverse).
The goal is of this repository is to help users try out and explore  different capabilities of RAPIDS-singlecell on datasets ranging from **250 thousand to 11 million cells**. To make this as easy as possible, we set up two different GPU envinronments on [Brev](https://developer.nvidia.com/brev) that are designed to get you working with GPU-accelerated single-cell workflows as quickly as possible (see [Quickstart](#quick-start)). We've also provided instructions to run these notebooks on your own CUDA-enabled GPU systems (see [Bring your own compute](docs/bring_your_compute.md)).

These notebooks will be valuable for single-cell scientists who want to quickly evaluate ease of use as well as explore the biological interpretability of RAPIDS-singlecell results. Secondarily, scientists will find value in learning to apply these methods to very large data sets. This repository is also broadly useful for any data scientist or developer who wants to run and evaluate single cell methods leveraging RAPIDS-singlecell. Data sets used for this tutorial were made [publicly available by 10X](https://www.10xgenomics.com/datasets) as well as [CZ cellxgene](https://cellxgene.cziscience.com/).  The base container is the [25.04 RAPIDSAI Notebooks Container](https://rapids.ai), which you can freely get from [NVIDIA's NGC Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/rapidsai/containers/notebooks) following the instructions below.

If you like these notebooks and this GPU accelerated capability, and want to support scverse's efforts, please [learn more about them here](https://scverse.org/about/) as well as [consider joining their community](https://scverse.org/join/).

## Quick start

The quickest way to use these blueprints is to use one of our pre-configured [NVIDIA Brev](https://console.brev.dev/) reseources.

1. Select your resource size, and click "Deploy Now":
  * For a **Standard Instance** (L40s), click here: [![ Click here to deploy the RAPIDS Singlecell Launchable.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-2xn8JayrpXlBwGXkKASSomq6gXi)
  * For a **Large Instance** (8x H100), click here: [![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-2xn9bEVtfQgLQXHpJUlzESILpfV)

2. Click **Deploy Launchable** on the Brev.dev Launchable page

    <img src="https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/raw/main/assets/deploy_brev.png" alt="deploy_brev" width="70%"/>



3. Wait for the Container status show **Ready** (can take up to 8 minutes).  Then, click **Access GPU**

    <img src="https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/raw/main/assets/go_on_green.png" alt="go_on_green" width="70%"/>


4. On the **Instance** page, click **Open Notebook**

    <img src="https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/raw/main/assets/open_notebook.png" alt="open_notebook" width="70%"/>


You should drop into a fully installed and populated JupyterLab environment.  Open up your desired notebook from the list below, and have a great time!

## Overview

This repository contains a diverse set of notebooks to help get anyone started using RAPIDS-singlecell developed by scverse.

![layout architecture](https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/raw/main/assets/scdiagram.png)

The outline below is a suggested exploration flow.  Unless otherwise noted, you can choose any notebook to get started, as long as you have the GPU resources to run the notebook.

For those who are new to doing basic analysis for single cell data, the end to end analysis of [01_scRNA_analysis_preprocessing.ipynb](notebooks/01_scRNA_analysis_preprocessing.ipynb) is the best place to start, where you are walked through the steps of data preprocessing, cleanup, visualization, and investigation.

| Notebook         | Description |Min GPU Size /<br>Instance |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
| [01_scRNA_analysis_preprocessing.ipynb](notebooks/01_scRNA_analysis_preprocessing.ipynb)   | End to end workflow, where we understand the cells, run ETL on the data set then visiualize and explore the results. <br>This tutorial is good for all users | 24GB /<br>Standard RSC Instance |
| [02_scRNA_analysis_extended.ipynb](notebooks/02_scRNA_analysis_extended)   | This notebook continues from the outputs of [01_scRNA_analysis_preprocessing.ipynb](notebooks/01_scRNA_analysis_preprocessing.ipynb) as an overview of methods that can be used to investigate transcriptional regulation | 24GB /<br>Standard RSC Instance |
| [03_scRNA_analysis_with_pearson_residuals.ipynb](notebooks/03_scRNA_analysis_with_pearson_residuals.ipynb)  | End to end workflow, like [01_scRNA_analysis_preprocessing.ipynb](notebooks/01_scRNA_analysis_preprocessing.ipynb), but uses pearson residuals for normalization. | 24GB /<br>Standard RSC Instance |
| [04_scRNA_analysis_dask_out_of_core.ipynb](notebooks/04_scRNA_analysis_dask_out_of_core.ipynb) | In this notebook, we show the scalability of the analysis toof up to 11M cells easily by using Dask.<br>**Requires a 48GB GPU** | 48GB /<br>Standard RSC Instance |
| [05_scRNA_analysis_multi_GPU.ipynb](notebooks/05_scRNA_analysis_multi_GPU.ipynb) | This notebook enhances the 11M cell dataset analysis with Dask without exceeding memory limits.  <br>It fully scales to utilize all available GPUs, uses chunk-based execution, and efficiently manages memory<br>**Requires 8x H100s or better.  For all other GPUs systems, please run [04_scRNA_analysis_dask_out_of_core.ipynb](notebooks/04_scRNA_analysis_dask_out_of_core.ipynb) instead**| 8x 80GB /<br>Large RSC Instance |
| [06_scRNA_analysis_90k_brain_example.ipynb](notebooks/06_scRNA_analysis_90k_brain_example.ipynb) | In this notebook, show diversity in capability by run a similar workflow to [01_scRNA_analysis_preprocessing.ipynb](notebooks/01_scRNA_analysis_preprocessing.ipynb), but on brain cells | 24GB /<br>Standard RSC Instance |
| [07_scRNA_analysis_1.3M_brain_example.ipynb](notebooks/07_scRNA_analysis_1.3M_brain_example.ipynb) | In this notebook, we scale up the analysis of [06_scRNA_analysis_90k_brain_example.ipynb](notebooks/06_scRNA_analysis_90k_brain_example.ipynb) to 1 million brain cells.<br>**Requires an 80GB GPU, like an H100** |  80GB /<br>Large RSC Instance |


You can find more detail on each notebook in the [Notebooks README](notebooks/README.md).

> [!NOTE]
> To ensure you have the maximum GPU memory available, **please remember to shut down your completed notebook's kernel before starting a new notebook**.  If you don't, you may experience Out Of Memory (OOM) based errors.  To fix that, simply kill all the kernels, and the restart only the kernel for the notebook you want to run.

## Deploying this Repository

The goal of this repository is to make it easy to try GPU-accelerated single-cell analysis workflows on different compute environments and datasets. Our preferred environment is NVIDIA Brev, but you can also run these in your own GPU-connected environment. We've provided a few tutorials below on how to set this up, and the easiest place to start is to follow the Quickstart instructions.

### Using our Brev Launchable (super easy mode - highly recommended)

Follow our [Quickstart Instructions](#quick-start) above.

### Create a custom instance using Brev (knowledgeable users only)
If you want to try a compute environment on Brev that's not one of the Quickstart Launchables, you will need to create a new Launchable or Standalone Compute Instance. This will let you select your desired cloud provider and desired compute resource. Note, we have not tested this on every combination of cloud provider and instance type, so your experience may vary.

If you're interested in trying this out, please follow the instructions here: [Setting up your Custom Brev Launchable](docs/creating_custom_brev.md)


### Deploy on a CUDA compatible GPU system (knowledgeable users only)

Some people may want to have this experience off of Brev and take it with you.  Great!  We wrote a (somewhat) easy tutorial here: [Bring your own compute](docs/bring_your_compute.md)

## Support

If you have any questions about these notebooks or need support, please open an Issue on this repository and we will respond there.
