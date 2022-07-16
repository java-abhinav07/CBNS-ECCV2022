# entropy.py
# https://github.com/human-analysis/MaxEnt-ARL/blob/master/loss/entropy.py

import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class EntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # input is probability distribution of output classes
    def forward(self, input):
        input = F.softmax(input, dim=1)
        if (input < 0).any() or (input > 1).any():
            raise Exception("Entropy Loss takes probabilities 0<=input<=1")

        input = input + 1e-16  # for numerical stability while taking log
        H = torch.mean(torch.sum(input * torch.log(input), dim=1))

        return H
