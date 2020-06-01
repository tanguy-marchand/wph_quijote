# wph_quijote

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


This repo contains the code that was used in {{ref_to_add}} to generate the syntheses
of Large Scale Structure maps from Quijote simulations. To know more about the Quijote simulations, cf https://arxiv.org/abs/1909.05273

This repo re-use codes form this repo: https://github.com/sixin-zh/kymatio_wph and might contains code
from Kymatio (scatNet) (https://github.com/kymatio/kymatio) and kymatio phase harmonics (https://github.com/kymatio/phaseharmonics)

It uses PyTorch + Cuda on GPU

# How to use it
You need PyTorch and Cuda as well as the library listed in requirements.txt
To run a batch of syntheses, run the script run_syntheses.py. The repo already contains
the training data (i.e. 30 maps from the Quijote simulations).

You can modify the parameters of the syntheses directly in run_syntheses.py

