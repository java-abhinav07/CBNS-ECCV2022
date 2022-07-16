#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

mkdir -p facescape/log/as_an/
mkdir -p facescape/log/as_an/best
mkdir -p facescape/log/as_an/best/FPS64/
mkdir -p facescape/log/as_an/best/finetune/


python3 main_.py \
    -o facescape/log/as_an/best/FPS64/train \
    --task priv \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 256 \
    --train-pointnet \
    --base-task cls_exp \
    --attacker-task cls_gender \
    --std-noise 0.001 \
    --gaussian \
    --no-adv \
    --attacker facescape/log/gender/FPS64/train_model_best.pth \
    --transfer-from-pointnet facescape/log/exp/PointNet1024/train_model_best.pth

wait

python3 main_.py \
    -o facescape/log/as_an/best/FPS64/test \
    --task priv \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 256 \
    --test \
    --base-task cls_exp \
    --attacker-task cls_gender \
    --std-noise 0.001 \
    --gaussian \
    --no-adv \
    --attacker facescape/log/gender/FPS64/train_model_best.pth \
    --pretrained facescape/log/as_an/best/FPS64/train_model_best.pth

wait


python3 main_.py \
    -o facescape/log/as_an/best/finetune/train \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 128 \
    --train-pointnet \
    --epochs 1 \
    --base-task cls_gender \
    --finetune \
    --std-noise 0.001 \
    --gaussian \
    --no-adv \
    --pretrained facescape/log/gender/FPS64/train_model_best.pth


wait
    
python3 main_.py \
    -o facescape/log/as_an/best/finetune/test \
    --task priv \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 256 \
    --test \
    --base-task cls_exp \
    --attacker-task cls_gender \
    --std-noise 0.001 \
    --gaussian \
    --no-adv \
    --attacker facescape/log/as_an/best/finetune/train_model_best.pth \
    --pretrained facescape/log/as_an/best/FPS64/train_model_best.pth
