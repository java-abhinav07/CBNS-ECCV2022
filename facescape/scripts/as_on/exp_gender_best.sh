#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0

mkdir -p facescape/log/as_on/
mkdir -p facescape/log/as_on/best/
mkdir -p facescape/log/as_on/best/finetune


python3 main_.py \
    -o facescape/log/as_on/best/train \
    --task priv \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 128 \
    --train-pointnet \
    --train-samplenet \
    --epochs 250 \
    --attacker-task cls_gender \
    --base-task cls_exp \
    --learn-noise \
    --adv-weight 0.01 \
    --pointwise-dist \
    --reg-weight 500 \
    --attacker-discrim \
    --attacker facescape/log/gender/FPS64/train_model_best.pth \
    --sampler-model facescape/log/exp/FPS64/train_sampler_best.pth

wait

python3 main_.py \
    -o facescape/log/as_on/best/test \
    --task priv \
    --base-name Faces_Data/ \
    --test \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 256 \
    --base-task cls_exp \
    --pretrained facescape/log/as_on/best/train_model_best.pth \
    --sampler-model facescape/log/as_on/best/train_sampler_best.pth \
    --attacker-task cls_gender \
    --attacker facescape/log/gender/FPS64/train_model_best.pth \
    --learn-noise \
    --adv-weight 0.01 \
    --pointwise-dist \
    --attacker-discrim
    
wait

python3 main_.py \
    -o facescape/log/as_on/best/finetune/train \
    --task vanilla \
    --base-name Faces_Data/ \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --sampler-model facescape/log/as_on/best/train_sampler_best.pth \
    --num-out-points 64 \
    --batch-size 128 \
    --train-pointnet \
    --epochs 100 \
    --pretrained facescape/log/gender/FPS64/train_model_best.pth \
    --base-task cls_gender \
    --finetune \
    --learn-noise \
    --adv-weight 0.01 \
    --pointwise-dist


wait

python3 main_.py \
    -o facescape/log/as_on/best/finetune/test \
    --task priv \
    --base-name Faces_Data/ \
    --test \
    --annot-file facescape/all_annotations1024.npy \
    --sampler fps \
    --num-out-points 64 \
    --batch-size 256 \
    --base-task cls_exp \
    --pretrained facescape/log/as_on/best/train_model_best.pth \
    --sampler-model facescape/log/as_on/best/train_sampler_best.pth \
    --attacker-task cls_gender \
    --attacker facescape/log/as_on/best/finetune/train_model_best.pth \
    --learn-noise \
    --adv-weight 0.01 \
    --pointwise-dist \
    --attacker-discrim
    