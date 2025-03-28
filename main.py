import argparse
import os
import warnings

import torch.backends.cudnn
from train_2d import train_pcrlv2
from train_3d import train_pcrlv2_3d
from pcrl_data import DataGenerator

warnings.filterwarnings("ignore")


def get_dataloader(args):
    generator = DataGenerator(args)
    loader_name = args.model + "_" + args.n + "_" + args.phase
    print(loader_name)
    dataloader = getattr(generator, loader_name)()
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self Training benchmark")
    parser.add_argument(
        "--data",
        metavar="DIR",
        default="/public1/cjh/workspace/AbdominalSegmentation/dataset/PCRL_cache",
        help="path to dataset",
    )
    parser.add_argument(
        "--model", metavar="MODEL", default="pcrlv2", help="choose the model"
    )
    parser.add_argument(
        "--phase",
        default="pretask",
        type=str,
        help="pretask or finetune or train from scratch",
    )
    parser.add_argument("--b", default=8, type=int, help="batch size")
    parser.add_argument("--epochs", default=100, type=int, help="epochs to train")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument(
        "--output", default="./model_genesis_pretrain", type=str, help="output path"
    )
    parser.add_argument("--n", default="luna", type=str, help="dataset to use")
    parser.add_argument("--d", default=3, type=int, help="3d or 2d to run")
    parser.add_argument("--workers", default=4, type=int, help="num of workers")
    parser.add_argument("--gpus", default="1", type=str, help="gpu indexs")
    parser.add_argument(
        "--ratio", default=0.8, type=float, help="ratio of data used for pretraining"
    )
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--weight_decay", default=1e-4)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--amp", action="store_true", default=False)
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.backends.cudnn.benchmark = True
    data_loader = get_dataloader(args)
    if args.model == "pcrlv2" and args.phase == "pretask" and args.d == 2:
        train_pcrlv2(args, data_loader)
    elif args.model == "pcrlv2" and args.phase == "pretask" and args.d == 3:
        train_pcrlv2_3d(args, data_loader)
