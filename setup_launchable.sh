#/bin/bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

# Install all additional libraries
cd /notebooks
uv pip install --system -r requirements.txt
mamba install -c conda-forge -c rapidsai cupy==14.0.1 numpy==2.2.6 -y

set -m

# Set missing env vars, start the primary process, and put it in the background
export CONDA_PREFIX=/opt/conda
jupyter-lab --notebook-dir=/notebooks --ip=0.0.0.0 --no-browser --NotebookApp.token='' --NotebookApp.allow_origin='*' --allow-root