#!/bin/bash

conda install -c conda-forge rdkit=2022.9.5 numpy=1.26.4 scipy biopython=1.79 openbabel imageio seaborn wandb debugpy matplotlib spyrmsd qvina=2.1.0 -y
# conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
pip install torch-geometric==2.6.1 pytorch-lightning==2.5.5
conda install -c mx reduce -y

# install posecheck
pip install hydride
conda install -c conda-forge prolif datamol 
# pip install --config-settings editable-mode=strict -e .