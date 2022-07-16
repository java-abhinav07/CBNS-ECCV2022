#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1

mkdir -p facescape/log/
mkdir -p facescape/log/exp/
mkdir -p facescape/log/exp/SAMPLENET64/
mkdir -p facescape/log/exp/FPS64/
mkdir -p facescape/log/exp/PointNet1024/


# Train PointNet
python3 main_.py \
    -o facescape/log/exp/PointNet1024/train \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler none \
    --num-out-points 1024 \
    --batch-size 128 \
    --train-pointnet \
    --epochs 250 \
    --base-task cls_exp

wait

# Test PointNet
python3 main_.py \
    -o facescape/log/exp/PointNet1024/test \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler none \
    --num-out-points 1024 \
    --batch-size 128 \
    --test \
    --base-task cls_exp

wait

# train samplenet
python3 main_.py \
    -o facescape/log/exp/SAMPLENET64/train \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler samplenet \
    --num-out-points 64 \
    --batch-size 128 \
    --train-samplenet \
    --epochs 100 \
    --transfer-from-pointnet facescape/log/exp/PointNet1024/train_model_best.pth \
    --base-task cls_exp

wait

# test samplenet
python3 main_.py \
    -o facescape/log/exp/SAMPLENET64/test \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler samplenet \
    --num-out-points 64 \
    --batch-size 128 \
    --test \
    --pretrained facescape/log/exp/SAMPLENET64/train_model_best.pth \
    --sampler-model facescape/log/exp/SAMPLENET64/train_sampler_best.pth \
    --base-task cls_exp

wait

# FPS test
python3 main_.py \
    -o facescape/log/exp/FPS64/test \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 128 \
    --test \
    --pretrained facescape/log/exp/PointNet1024/train_model_best.pth \
    --base-task cls_exp