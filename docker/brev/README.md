# Docker Compose YAMLs

This directory contains configuration files to assist in creating a custom Brev Launchable environment. For full instructions, please consult our [Creating Custom Brev Resource](../../docs/creating_custom_brev.md) guide.

## File manifest
RAPIDS 25.04
```
docker/brev/docker-compose-nb-2504-H100-C.yaml  # Use this if you are selecting H100 instances on Crusoe and need RAPIDS 25.04
docker/brev/docker-compose-nb-2504.yaml         # Use this for any other resource and need RAPIDS 25.04
```

RAPIDS 25.06 (for future usage)
```
docker/brev/docker-compose-nb-2506-H100-C.yaml  # Use this if you are selecting H100 instances on Crusoe
docker/brev/docker-compose-nb-2506-aarch64.yaml # Use for ARM based systems like Grace Hopper, Grace Blackwell, or DGX Spark (see Known Issues #1)
docker/brev/docker-compose-nb-2506.yaml         # Use this for any other resource
```

RAPIDS 25.06 nightly (FOR PRE 25.06 RELEASE TESTING ONLY)
```
docker/brev/docker-compose-nb-2506-H100-C-nightly.yaml  # Test docker-compose for H100 instances on Crusoe for pre-release on 25.06 nightly.  Do not use after 25.06 stable is out.
docker/brev/docker-compose-nb-2506-nightly.yaml         # Test docker-compose on pre-release on 25.06 nightly.  Do not use after 25.06 stable is out.
```

## Known Issues
1. If running on an ARM Machine (`aarch64`), such as Grace Hopper, Grace Blackwell, or DGX Spark, you will hit an error as scikit-misc needs to compile as it does not have aarch64 based wheels.  To fix this, please edit your `environment` parameter in the `docker-compose` of your choice, as per the below example:
    - Original
```
environment:
        EXTRA_PIP_PACKAGES: "anndata==0.11.4 array-api-compat==1.12.0 contourpy==1.3.2 cycler==0.12.1 fonttools==4.58.0 h5py==3.13.0 imageio==2.37.0 joblib==1.5.1 kiwisolver==1.4.8 lazy-loader==0.4 legacy-api-wrap==1.4.1 llvmlite==0.44.0 matplotlib==3.10.3 natsort==8.4.0 networkx==3.4.2 numba==0.61.2 numpy==2.2.6 pandas==2.2.3 patsy==1.0.1 pillow==11.2.1 pynndescent==0.5.13 rapids-singlecell==0.12.6 scanpy==1.11.1 scikit-learn==1.5.2 scikit-image==0.25.2 scikit-misc==0.5.1 scipy==1.15.3 seaborn==0.13.2 session-info2==0.1.2 statsmodels==0.14.4 threadpoolctl==3.6.0 tifffile==2025.5.10 tqdm==4.67.1 tzdata==2025.2 umap-learn==0.5.7 wget==3.2 deprecated==1.2.18 numcodecs==0.15.1 wrapt==1.17.2 zarr==2.18.7"
```
    - Make the addition of `EXTRA_CONDA_PACKAGES` with these parameters
```
environment:
        EXTRA_CONDA_PACKAGES: "binutils==2.43,binutils_impl_linux-aarch64==2.43,binutils_linux-aarch64==2.43,c-compiler==1.9.0,compilers==1.9.0,cxx-compiler==1.9.0,fortran-compiler==1.9.0,gcc==13.3.0,gcc_impl_linux-aarch64==13.3.0,gcc_linux-aarch64==13.3.0,gfortran==13.3.0,gfortran_impl_linux-aarch64==13.3.0,gfortran_linux-aarch64==13.3.0,gxx==13.3.0,gxx_impl_linux-aarch64==13.3.0,gxx_linux-aarch64==13.3.0,kernel-headers_linux-aarch64==4.18.0,libgcc-devel_linux-aarch64==13.3.0,libsanitizer==13.3.0,libstdcxx-devel_linux-aarch64==13.3.0,sysroot_linux-aarch64==2.17"
        EXTRA_PIP_PACKAGES: "anndata==0.11.4 array-api-compat==1.12.0 contourpy==1.3.2 cycler==0.12.1 fonttools==4.58.0 h5py==3.13.0 imageio==2.37.0 joblib==1.5.1 kiwisolver==1.4.8 lazy-loader==0.4 legacy-api-wrap==1.4.1 llvmlite==0.44.0 matplotlib==3.10.3 natsort==8.4.0 networkx==3.4.2 numba==0.61.2 numpy==2.2.6 pandas==2.2.3 patsy==1.0.1 pillow==11.2.1 pynndescent==0.5.13 rapids-singlecell==0.12.6 scanpy==1.11.1 scikit-learn==1.5.2 scikit-image==0.25.2 scikit-misc==0.5.1 scipy==1.15.3 seaborn==0.13.2 session-info2==0.1.2 statsmodels==0.14.4 threadpoolctl==3.6.0 tifffile==2025.5.10 tqdm==4.67.1 tzdata==2025.2 umap-learn==0.5.7 wget==3.2 deprecated==1.2.18 numcodecs==0.15.1 wrapt==1.17.2 zarr==2.18.7"
```
