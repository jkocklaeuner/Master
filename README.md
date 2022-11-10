# NAQS
# Master
# Master
Run NNQS calculations for quantum chemistry using NADE and RNN architectures. This repository mostly reproduces the results obtained by [Barett et al.](https://arxiv.org/pdf/2109.12606.pdf).
Please install the necessary Anaconda environment by using `conda env create -f environment.yml` and activate the environment `netket`.
A basic calculation is set up by running `bash run_MODEL.sh MOLECULE ELECTRONS MULTIPLICITY SPATIALORBITALS SEED`, the necessary parameters can be found inthe paper of Barett et al. Further modifications can be done directly in the .sh files, all available options can be listed by running `python scripts/nade.py --help`.
Since the support for RNNs is still experimental, the structure is not very flexible yet. Details on the structure of RNN wavefunction can be found in the paper of [Hibat-Allah](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.023358).

