@echo off
PATH = %PATH%;%USERPROFILE%\Miniconda3\Scripts
conda create -n cat-face-landmark-predictor pip python=3.6 -y
call activate cat-face-landmark-predictor
conda install -c anaconda keras-gpu==2.2.4 cudnn=7.6.0=cuda9.0_0 -y
pip install -r requirements.txt
