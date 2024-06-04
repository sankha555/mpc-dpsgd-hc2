import argparse
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample.functorch import make_functional
from torch.func import grad, grad_and_value, vmap
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import FashionMNIST, MNIST
from tqdm import tqdm
from datetime import datetime
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from sys import modules
this_mod = modules[__name__]

hyperparameters = {
    "TRAIN_EXAMPLES": 60000,

    "DEVICE": "cpu",

    "DATA_ROOT": '../fmnist',

    "EPOCHS": 10,

    "BATCH_SIZE": 100,

    "PRINT_FREQ": 10,

    "LR": 0.1,

    "MOMENTUM": 0,

    "DISABLE_DP": True,

    "EPSILON": 4,

    "DELTA": 1e-5,

    "CLIP_FACTOR": 10,

    "SIGMA": 1.5,
}


class Mohassel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.fc1.bias.data.zero_()
        self.fc2 = nn.Linear(128, 128, bias=True)
        self.fc2.bias.data.zero_()
        self.fc3 = nn.Linear(128, 10, bias=True)
        self.fc3.bias.data.zero_()

    def forward(self, x):
        x = x.view(-1, 784)  # -> [B, 784]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = F.relu(self.fc2(x))  # -> [B, 10]
        x = self.fc3(x)
        return x
    


def LinearSecML512():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )

def LinearSecML256():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

def Linear100():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

def Linear500():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 500),
        nn.ReLU(),
        nn.Linear(500, 10)
    )

def LinearSecML():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    
def CNN1():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(16, 32, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(32, 64, 3, padding='same'),
        nn.ReLU(),

        nn.MaxPool2d(2, 2),

        nn.Flatten(),

        nn.Linear(576, 128),
        nn.ReLU(),

        nn.Linear(128, 10)
    )

def CNNTramer():
    return nn.Sequential(
        nn.Conv2d(1, 16, 8, 2, 2),
        nn.ReLU(),
        nn.MaxPool2d(2, 1),

        nn.Conv2d(16, 32, 4, 2, 0),
        nn.ReLU(),
        nn.MaxPool2d(2, 1),

        nn.Flatten(),

        nn.Linear(32*4*4, 32),
        nn.ReLU(),

        nn.Linear(32, 10)
    )

def get_model(model_name):
    # return Mohassel().to(device="cpu")
    return getattr(this_mod, model_name)()

def get_datasets():
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(
        root=hyperparameters["DATA_ROOT"], train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=hyperparameters["BATCH_SIZE"],
        generator=None,
        pin_memory=True,
        shuffle=False
    )

    test_dataset = MNIST(
        root=hyperparameters["DATA_ROOT"], train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=hyperparameters["BATCH_SIZE"],
        shuffle=False,
    )

    return train_loader, test_loader


def get_optimizer(model, train_loader, privacy_engine):
    optimizer = optim.SGD(
        model.parameters(),
        lr=hyperparameters["LR"],
        momentum=hyperparameters["MOMENTUM"],
        weight_decay=0,
    )

    if not hyperparameters["DISABLE_DP"]:
        max_grad_norm = hyperparameters["CLIP_FACTOR"]          # Each layer has the same clipping threshold. The total grad norm is still bounded by `args.max_per_sample_grad_norm`.

        clipping = "flat"
        if hyperparameters["SIGMA"] == -1:
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_delta=1e-5,
                target_epsilon=hyperparameters["EPSILON"],
                epochs=hyperparameters["EPOCHS"],
                max_grad_norm=max_grad_norm,
                clipping=clipping,
                grad_sample_mode="hooks",
            )
        else:
            model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=hyperparameters["SIGMA"],
            max_grad_norm=max_grad_norm,
            clipping=clipping,
            grad_sample_mode="hooks",
        )

    return model, optimizer, train_loader


def accuracy(preds, labels):
    return (preds == labels).mean()


def train(model, train_loader, optimizer, privacy_engine, epoch, device):
    import time
    # time.sleep(10)

    start_time = datetime.now()

    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        loss = criterion(output, target)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)
        top1_acc.append(acc1)

        # compute gradient and do SGD step
        loss.backward()

        losses.append(loss.item())
        # print(f"Epoch {epoch}.{i+1}: Loss = {loss.item()}")

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

        if i % hyperparameters["PRINT_FREQ"] == 0:
            if not hyperparameters["DISABLE_DP"]:
                epsilon = privacy_engine.accountant.get_epsilon(delta=hyperparameters["DELTA"])
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {hyperparameters['DELTA']})"
                )
            else:
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                )

    train_duration = datetime.now() - start_time
    return train_duration


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    return np.mean(top1_acc)


def main():
    train_loader, test_loader = get_datasets()

    model = get_model(hyperparameters["MODEL_NAME"])
    model = model.to(hyperparameters["DEVICE"])

    privacy_engine = None
    if not hyperparameters["DISABLE_DP"]:
        privacy_engine = PrivacyEngine(
            secure_mode=False,
            accountant="rdp"
        )

    model, optimizer, train_loader = get_optimizer(model, train_loader, privacy_engine)

    # with open("weights_secml128.txt", "w") as file:
    #     for i, p in enumerate(model.parameters()):
    #         if i % 2 == 0:
    #             file.write(p.data.tolist().__str__())

    # weights = []
    # with open("weights_secml128.txt", "r") as file:
    #     weights = eval(file.read())


    for i, p in enumerate(model.parameters()):
        if i % 2:
            p.data.fill_(0)
    #     else:
    #         p.data = Parameter(torch.Tensor(weights[i//2]))

    time_per_epoch = []
    accuracy_per_epoch = []
    for e in range(1, hyperparameters["EPOCHS"]+1):
        train_duration = train(
            model, train_loader, optimizer, privacy_engine, e, hyperparameters["DEVICE"]
        )
        top1_acc = test(model, test_loader, hyperparameters["DEVICE"])

        time_per_epoch.append(train_duration)
        accuracy_per_epoch.append(float(top1_acc))
    

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Fashion MNIST DP Training")

    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size",
        default=2000,
        type=int,
        metavar="N",
        help="approximate batch size",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0, type=float, metavar="M", help="SGD momentum"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=-1,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=10.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=4,
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
    )
    
    args = parser.parse_args()

    hyperparameters["MODEL_NAME"] = args.model_name

    hyperparameters["DATA_ROOT"] = args.data_root

    hyperparameters["EPOCHS"] = args.epochs

    hyperparameters["BATCH_SIZE"] = args.batch_size

    hyperparameters["PRINT_FREQ"] = math.ceil(hyperparameters["TRAIN_EXAMPLES"] / hyperparameters["BATCH_SIZE"])

    hyperparameters["LR"] = args.lr

    hyperparameters["MOMENTUM"] = args.momentum

    hyperparameters["DISABLE_DP"] = args.disable_dp

    hyperparameters["CLIP_FACTOR"] = args.max_per_sample_grad_norm

    hyperparameters["SIGMA"] = args.sigma

    hyperparameters["EPSILON"] = args.epsilon

    hyperparameters["DELTA"] = args.delta


if __name__ == "__main__":
    parse_args()

    DP_CONFIG = f"CLIP FACTOR = {hyperparameters['CLIP_FACTOR']} \tSIGMA = {hyperparameters['SIGMA']}"
    print(f"BATCH SIZE = {hyperparameters['BATCH_SIZE']} {'VANILLA' if hyperparameters['DISABLE_DP'] else DP_CONFIG}")
    
    main()