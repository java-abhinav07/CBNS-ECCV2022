#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

mkdir -p facescape/log/
mkdir -p facescape/log/gender/
mkdir -p facescape/log/gender/SAMPLENET64/
mkdir -p facescape/log/gender/FPS64/
mkdir -p facescape/log/gender/PointNet1024/


# Train PointNet
python3 main_.py \
    -o facescape/log/gender/PointNet1024/train \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler none \
    --num-out-points 1024 \
    --batch-size 128 \
    --train-pointnet \
    --epochs 250 \
    --base-task cls_gender

wait

# Test PointNet
python3 main_.py \
    -o facescape/log/gender/PointNet1024/test \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler none \
    --num-out-points 1024 \
    --batch-size 128 \
    --test \
    --base-task cls_gender

wait

# train samplenet
python3 main_.py \
    -o facescape/log/gender/SAMPLENET64/train \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler samplenet \
    --num-out-points 64 \
    --batch-size 128 \
    --train-samplenet \
    --epochs 100 \
    --transfer-from-pointnet facescape/log/gender/PointNet1024/train_model_best.pth \
    --base-task cls_gender

wait

# test samplenet
python3 main_.py \
    -o facescape/log/gender/SAMPLENET64/test \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler samplenet \
    --num-out-points 64 \
    --batch-size 128 \
    --test \
    --pretrained facescape/log/gender/SAMPLENET64/train_model_best.pth \
    --sampler-model facescape/log/gender/SAMPLENET64/train_sampler_best.pth \
    --base-task cls_gender

wait

# FPS test
python3 main_.py \
    -o facescape/log/gender/FPS64/test \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 128 \
    --test \
    --pretrained facescape/log/gender/PointNet1024/train_model_best.pth \
    --base-task cls_gender