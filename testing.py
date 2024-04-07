import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

dev = torch.device("cpu")
print(f"Using {dev} device.")

testing_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

loader = DataLoader(
    dataset=testing_dataset,
    batch_size=128,
    shuffle=True
)

model = torch.load("./model.pt")

# checking all test images -----------------------------------------------------
all = 0
good = 0
for inputs, targets in iter(loader):
    outputs = model.forward(inputs)
    good += (torch.argmax(outputs, dim=-1) == targets).sum()
    all += targets.size(0)


print(f"Accuracy: {good / all}")