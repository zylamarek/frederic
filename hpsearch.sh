#!/bin/bash

cd tools

python train.py --loss_fn iou --flip_horizontal --crop_scale_balanced --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10
python train.py --loss_fn iou --flip_horizontal --crop_scale_balanced --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10
python train.py --loss_fn iou --flip_horizontal --crop_scale_balanced --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10
python train.py --loss_fn iou --flip_horizontal --crop_scale_balanced --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10
python train.py --loss_fn iou --flip_horizontal --crop_scale_balanced --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10

python train.py --output_type landmarks --flip_horizontal --crop --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10
python train.py --output_type landmarks --flip_horizontal --crop --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10
python train.py --output_type landmarks --flip_horizontal --crop --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10
python train.py --output_type landmarks --flip_horizontal --crop --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10
python train.py --output_type landmarks --flip_horizontal --crop --rotate_n 20 --ReduceLROnPlateau_factor 0.8 --ReduceLROnPlateau_patience 10

cd ..
