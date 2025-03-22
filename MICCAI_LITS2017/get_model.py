import os
from time import time

import numpy as np

import torch


from .pcrl_net.ResUNet import net

# print(net.state_dict())
import net.pretrain.PCRLv2.MICCAI_LITS2017.parameter as para
import argparse


def setup_pcrlv2_model(
    weight_path: str = "/public1/cjh/workspace/AbdominalSegmentation/tensorboard_log/pretrain/pretrain_pcrlv2/pcrlv2_luna_pretask_0.8_100.pt",
):
    if weight_path is not None:
        encoder_dict = net.state_dict()
        checkpoint = torch.load(weight_path)
        state_dict = checkpoint["state_dict"]
        pretrain_dict = {
            k: v
            for k, v in state_dict.items()
            if k in encoder_dict and "down" in k and "down_tr64" not in k
        }
        print(pretrain_dict.keys())
        encoder_dict.update(pretrain_dict)
        net.load_state_dict(encoder_dict)
    return net


if __name__ == "__main__":
    model = setup_pcrlv2_model(
        "/public1/cjh/workspace/AbdominalSegmentation/tensorboard_log/pretrain/pretrain_pcrlv2/pcrlv2_luna_pretask_0.8_100.pt"
    )
    input = torch.rand(size=(2, 1, 64, 64, 64))
    output = model(input)
    print(output.shape)
    # print(net)
