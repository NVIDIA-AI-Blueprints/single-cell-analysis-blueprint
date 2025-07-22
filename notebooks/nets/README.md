# Pathway Networks

This directory contains pre-downloaded pathway network files to streamline our containerized environment.

## Background

These pathway resources were downloaded using `decoupler==2.0.4`. Starting with decoupler 2.0.0, the package has been updated to use a BSD-3-Clause license.

## Contents

- **dorothea.parquet**: Transcription factor regulatory networks
- **progeny.parquet**: Pathway activity inference networks (top 100)

## Reproducing These Files

If you need to regenerate these files or want to use different parameters, you can create them using the following Python code:

```python
import decoupler as dc

# Download DoRothEA transcription factor networks
dc.op.dorothea(license='commercial').to_parquet("dorothea.parquet")

# Download PROGENy pathway networks (top 100 pathways)
dc.op.progeny(license='commercial', top=100).to_parquet("progeny.parquet")
```
