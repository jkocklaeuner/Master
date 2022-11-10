# NAQS

## Intro

Run NNQS calculations for quantum chemistry using NADE and RNN architectures. This repository mostly reproduces the results obtained by [Barett et al.](https://arxiv.org/pdf/2109.12606.pdf).
A basic calculation is set up by running `bash run_MODEL.sh MOLECULE ELECTRONS MULTIPLICITY SPATIALORBITALS SEED`, the necessary parameters can be found inthe paper of Barett et al. Further modifications can be done directly in the .sh files, all available options can be listed by running `python scripts/nade.py --help`.
Since the support for RNNs is still experimental, the structure is not very flexible yet. Details on the structure of RNN wavefunction can be found in the paper of [Hibat-Allah](https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.2.023358).

## Setup

The calculation requires the [NetKet](https://netket.readthedocs.io/en/latest/docs/install.html) and [OpenFermion](https://quantumai.google/openfermion/install) package following the instructions in the links. Even though it is not recommended by the developers, installation with conda works quite fine.

## Monitoring

Two basic tools are provided for monitoring the optimization:
 `scripts/energy.py` outputs the variational energies of the last steps
`scripts/plot.py` creates a basic plot of the optimization run versus a reference value. 
All data is stored in Json .log files, which can be opened in python via `json.load(open(FILENAME)))` and provide a dictionary with all monitored quantities.

## Settings

A list of variable settings is obtained by a `scripts/nade.py --help` command. 

