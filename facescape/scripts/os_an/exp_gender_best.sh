#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

mkdir -p facescape/log/os_an/
mkdir -p facescape/log/os_an/best
mkdir -p facescape/log/os_an/best/SAMPLENET64/
mkdir -p facescape/log/os_an/best/finetune/


python3 main_.py \
    -o facescape/log/os_an/best/SAMPLENET64/train \
    --task priv \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler samplenet \
    --num-out-points 64 \
    --batch-size 128 \
    --train-pointnet \
    --train-samplenet \
    --epochs 250 \
    --base-task cls_exp \
    --attacker-task cls_gender \
    --std-noise 0.01 \
    --gaussian \
    --no-adv \
    --pretrained facescape/log/exp/SAMPLENET64/train_model_best.pth \
    --attacker facescape/log/gender/SAMPLENET64/train_model_best.pth \
    --sampler-model facescape/log/exp/SAMPLENET64/train_sampler_best.pth


wait

python3 main_.py \
    -o facescape/log/os_an/best/SAMPLENET64/test \
    --task priv \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler samplenet \
    --num-out-points 64 \
    --batch-size 256 \
    --test \
    --base-task cls_exp \
    --attacker facescape/log/gender/SAMPLENET64/train_model_best.pth \
    --sampler-model facescape/log/os_an/best/SAMPLENET64/train_sampler_best.pth \
    --attacker-task cls_gender \
    --pretrained facescape/log/os_an/best/SAMPLENET64/train_model_best.pth \
    --std-noise 0.01 \
    --gaussian \
    --no-adv
    
wait


python3 main_.py \
    -o facescape/log/os_an/best/finetune/train \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler samplenet \
    --num-out-points 64 \
    --batch-size 128 \
    --train-pointnet \
    --epochs 100 \
    --base-task cls_gender \
    --sampler-model facescape/log/os_an/best/SAMPLENET64/train_sampler_best.pth \
    --pretrained facescape/log/gender/SAMPLENET64/train_model_best.pth \
    --finetune \
    --std-noise 0.01 \
    --gaussian \
    --no-adv

wait
    
python3 main_.py \
    -o facescape/log/os_an/best/finetune/test \
    --task priv \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler samplenet \
    --num-out-points 64 \
    --batch-size 256 \
    --test \
    --base-task cls_exp \
    --attacker facescape/log/os_an/best/finetune/train_model_best.pth \
    --sampler-model facescape/log/os_an/best/SAMPLENET64/train_sampler_best.pth \
    --attacker-task cls_gender \
    --pretrained facescape/log/os_an/best/SAMPLENET64/train_model_best.pth \
    --std-noise 0.01 \
    --gaussian \
    --no-adv
