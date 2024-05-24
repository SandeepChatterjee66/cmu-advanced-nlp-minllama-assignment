#!/usr/bin/env bash

conda create -n llama_hw python=3.11
conda activate llama_hw

# See https://pytorch.org/get-started/previous-versions/
# Comment out nvidia for MacOS. Enable MPS later
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch
pip install tqdm==4.66.1
pip install requests==2.32.2
pip install importlib-metadata==4.13.0
# pip install filelock==3.13.1
pip install scikit-learn==1.4.2
# pip install numpy==1.26.4
pip install tokenizers==0.19.1
pip install sentencepiece==0.2.0
cd models
curl https://www.cs.cmu.edu/~vijayv/stories42M.pt -o stories42M.pt
cd ..