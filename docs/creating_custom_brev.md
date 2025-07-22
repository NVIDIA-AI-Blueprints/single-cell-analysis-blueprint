# Creating a custom GPU instance using Brev
The goal of this guide is to walk through the steps to create a custom GPU instance on NVIDIA Brev using the Brev Launchable interface.

Brev provides access to a wide variety of GPU instances on difference cloud environment. While this provides you flexibility on pricing and avaiablity, it makes it difficult to test this launchable on every combination of compute environment and cloud environment. As such, some tinkering may be required to get the instance to work on the specific configuration you select. To use one of our recommended environments that is tested to work, please check out the [Quickstart Instructions](../README.md#quick-start).

> [!WARNING]
> Our Docker Compose YAML files have a known issue when using GCP resources on Brev. We recommend using another cloud provider until this is resolved.

## Option 1 - Create a custom instance using the Brev Launchable Creator

1. Go to [Brev’s Launchable Creator](https://brev.nvidia.com/launchables/create) (requires account)
2. When asked **How would you like to provide your code files?** select "I have code files in a GitHub repository". Enter the URL of this repository.
3. When asked **What type of runtime environment do you need?** select "With container(s)"
4. Under **Choose a Container Configuration** select "Docker Compose" and click on the toggle to select "I have an existing docker-compose.yaml file".
5. Under **Upload Docker Compose** select "Provide GitHub/Gitlab URL" and provide a link to one of the Docker Compose YAML files in the [docker/brev](../docker/brev) directory. There is a README.md in that directory with instructions on which YAML to select. Note, you need to pass a link to the file in GitHub, not to the `raw.github.com` file (e.g. [docker-compose-nb-2504.yaml](https://github.com/NVIDIA-AI-Blueprints/single-cell-analysis-blueprint/blob/main/docker/brev/docker-compose-nb-2504.yaml)). Click "Validate".
6. On the next page, when asked **Do you want a Jupyter Notebook experience?** select "No, I don't want Jupyter (Not Recommended)". We will provide Jupyter in the Docker compose already.
7. In the section title **Do you need to expose services?** make sure that ports `8888`, `8787`, and `8786` are open. Name port 8888 `jupyter` so Brev can treat it as a jupyterlab based instance and provide an Open Notebook button.
8. Select your desired compute environment. Make sure you select sufficient disk size to download the datasets you want to work with. We recommend at least 128GB, unless you want to run the 11M cell notebook, which you will need atleast 200GB. Note, you will not be able to resize the instance once created.
9. Create a name for your launchable, and deploy.

> [!NOTE]
> Git is not installed by default in this container, but it can be installed using
> ```
> apt update
> apt install git -y
> ```

## Option 2 - Install rapids-singlecell manually in a rapids/notebook container

A second option to set up `rapids-singlecell` is to start by [Creating a Custom RAPIDS Brev Instance](https://docs.rapids.ai/deployment/stable/cloud/nvidia/brev/#option-1-setting-up-your-brev-gpu-instance), manually git cloning this `single-cell-analysis-blueprint` repository, and manually installing [the scverse `rapids-singlecell` library](https://github.com/scverse/rapids_singlecell).

1. Follow the instructions for [Creating a RAPIDS Brev Instance](https://docs.rapids.ai/deployment/stable/cloud/nvidia/brev/#option-1-setting-up-your-brev-gpu-instance).  Remember the RAPIDS version for Step 3.  Launch the instance.
2. Download this repository into the instance.
3. Download the `rapids-singlecell` repository into the instance. Follow their [installation instructions](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html). It is important to have the same RAPIDS version and CUDA Toolkit version as your installed RAPIDS versions.
 For Brev, we currently default to CUDA Tool Kit 12.8 to support Blackwell GPUs. Therefore, you can install `rapids-singelcell` using `mamba env create -f conda/rsc_<RAPIDS VERSION>.yml'
    - Example to install `rapids-singelcell` for 25.06
    ```code
    mamba env create -f conda/rsc_rapids_25.06.yml
    ```
You can check this by going to the command line and running `nvcc --version` inside the instance. You'll get an output like this:

    ```bash
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2025 NVIDIA Corporation
    Built on Fri_Feb_21_20:26:18_PST_2025
    Cuda compilation tools, release 12.8, V12.8.93
    Build cuda_12.8.r12.8/compiler.35583870_0
    ```
4. Activate the mamba environment and run the notebooks/your code.
   ```code
   mamba activate rapids-singelcell
   ```
