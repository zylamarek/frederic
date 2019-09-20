#!/bin/bash

conda create -n frederic-dev pip python=3.6 -y
source activate frederic-dev
conda install -c anaconda keras-gpu==2.2.4 cudnn=7.6.0=cuda9.0_0 -y
pip install -r requirements.txt
