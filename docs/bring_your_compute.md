
If you are provisioning this on a non-Brev cloud instance, workstation, or local machine, please follow the steps below:

### 1. Provision a System Meeting These Requirements
All provisioned systems need to be RAPIDS capable. Here's what is required:

<i class="fas fa-microchip"></i> **GPU:** NVIDIA Volta™ or higher with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+
- For most of the notebooks, we recommend a **GPU with 24 GB VRAM or more**, due to the dataset size, such as the **L40S**, which can be quickly deployed above.  Some other common GPU options found in your favorite cloud service providers, your workstations, or your PC are:

| Cloud/DGX | Workstation GPU | Consumer GPU |
|---|---|---|
| A/H/B100  | A4000 ADA or better | 5090 |
| L40s/L40 | A5000 or better | 4090 |
| A10/A10s | RTX6000 or better | 3090 |

The [multi_gpu_large_data_showcase](https://github.com/clara-parabricks-workflows/single-cell-analysis-blueprint/blob/main/multi_gpu_large_data_showcase.ipynb) and the [demo_gpu-seuratv3-brain-1M](https://github.com/clara-parabricks-workflows/single-cell-analysis-blueprint/blob/main/demo_gpu-seuratv3-brain-1M.ipynb) requires a large multigpu system.  The [out-of-core_processing](https://github.com/clara-parabricks-workflows/single-cell-analysis-blueprint/blob/main/out-of-core_processing.ipynb) notebook, even using the 11 million cell dataset, and the [demo_gpu-seuratv3](https://github.com/clara-parabricks-workflows/single-cell-analysis-blueprint/blob/main/demo_gpu-seuratv3.ipynb) are respectively similar and can be run on one of the GPUs above, but a 48GB GPU is recommended.


<i class="fas fa-desktop"></i> **OS:**
- <i class="fas fa-check-circle"></i> Linux distributions with `glibc>=2.28` (released in August 2018), which include the following:
  - [Arch Linux](https://archlinux.org/), minimum version 2018-08-02
  - [Debian](https://www.debian.org/), minimum version 10.0
  - [Fedora](https://fedoraproject.org/), minimum version 29
  - [Linux Mint](https://linuxmint.com/), minimum version 20
  - [Rocky Linux](https://rockylinux.org/) / [Alma Linux](https://almalinux.org/) / [RHEL](https://www.redhat.com/en/technologies/linux-platforms/enterprise-linux), minimum version 8
  - [Ubuntu](https://ubuntu.com/), minimum version 20.04
- <i class="fas fa-check-circle"></i> Windows 11 using a [WSL2 specific install](https://docs.rapids.ai/install/#wsl2)

<i class="fas fa-download text-purple"></i> **CUDA 12 & latest NVIDIA Drivers:** Install the latest drivers for your system [HERE](https://www.nvidia.com/en-us/drivers/)

 **Note:** RAPIDS is tested with and officially supports the versions listed above. Newer CUDA and driver versions may also work with RAPIDS. See [CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/index.html) for details.

### 2. Clone This Repo:
```
git clone https://github.com/clara-parabricks-workflows/single-cell-analysis-blueprint.git
```

### 3. Install Packaging Environment Software:
RAPIDS can be installed using Pip, Conda, or Docker.  To replicate the same expereince as in the Launchable, it is recommended to use Docker for installation.
<i class="fas fa-download text-purple"></i> **3.1 Get Docker:** Use the code below to download and install Docker on your system
```
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```
<i class="fas fa-download text-purple"></i> **3.2 Get RAPIDS:** Use the code below to pull the same RAPIDS container used in the Launchable.  Example for 25.06:
```
docker run --gpus all --pull always --rm -it \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8888:8888 -p 8787:8787 -p 8786:8786
    -v ~/single-cell-analysis-blueprint:/home/rapids/notebooks/single-cell-analysis-blueprint
    nvcr.io/nvidia/rapidsai/notebooks:25.06-cuda12.8-py3.12
```
**Note:** The volume is currently assuming that you cloned the `single-cell-analysis-blueprint` repository into your `HOME` folder (`~/single-cell-analysis-blueprint`).  If you did not clone it there, please change the `~/single-cell-analysis-blueprint` portion of the above to the correct path before running the command.


### 4. Install the Extra RAPIDS-singlecell Libraries into the Running Container
Once the container is running,
- if the provisioned system is an external cloud or workstation, please use a web browser to navigate to the system's IP address and it's port 8888.  Example `http://192.168.1.2:8888`
- if the provisioned system is your sustem, then use `http://127.0.0.1:8888`.

Once the JupyterLab instance loads, open the terminal form the JupyterLab GUI, and run:
```
cd /home/rapids/notebooks/single-cell-analysis-blueprint
pip install -r requirements.txt
```

For additional information on RAPIDS-singlecell please visit the [RAPIDS-singlecell Docs](https://rapids-singlecell.readthedocs.io/)
For alternate installation instructions, please refer to the [RAPIDS-singlecell Install Guide](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html) to install using [pip](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#pypi), [Conda](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#conda), or [Docker](https://rapids-singlecell.readthedocs.io/en/latest/Installation.html#docker)

## Tips and Tricks
1. If after clicking a "Healthy" Port 8888 link in `Deploy with Brev: Step 4`, JupyterLab does not start, or the notebooks don't show, please try again in a few seconds.  There is a known issue where there system needs a minute or two Also, sometimes, the page needs to be refreshed to update the status.
2. Currently, restarting an instance, after stopping it, will start up far faster than starting a new instance.
3. If you **Stop** the instance, all the data on the main storage will be retained for the next time you start it.
4. To conserve GPU memory, please remember to shut down your completed notebook's kernel before starting a new notebook.
5. If your data download gets interrupted, please delete the files you intended to download and try again.
6. The Standard RSC Instance (L40s) has 128GB of space (~90GB user usable).
7. The Large RSC Instance (8x H100) has the same 128GB of space as the Standard RSC Instance, but there is a special `data` folder is mapped to a 1.12TB disk.  Data retention on that disk (in that folder) is not guaranteed.  YMMV.
8. If using an ARM based system, please conda install `compile` before installing `RAPIDS-singlecell` so that you can build `scikit-misc` from source/
