#!/usr/bin/env bash

conda create -n llama_hw python=3.11
conda activate llama_hw

# See https://pytorch.org/get-started/previous-versions/
# Comment out nvidia for MacOS. Enable MPS later
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch
pip install tqdm==4.66.1
pip install requests==2.31.0
pip install importlib-metadata==3.7.0
pip install filelock==3.0.12
pip install scikit-learn==1.2.2
pip install numpy==1.26.3
pip install tokenizers==0.13.3
pip install sentencepiece==0.1.99
cd models
curl https://www.cs.cmu.edu/~vijayv/stories42M.pt -o stories42M.pt
cd ..