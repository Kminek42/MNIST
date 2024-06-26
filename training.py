import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

# this model is so simple that it is faster on CPU -----------------------------
dev = torch.device("cpu")
print(f"Using {dev} device.")

# prepare data ----------------------------------------------------------------- 
training_dataset = torchvision.datasets.MNIST(
    root="./MNIST",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

loader = DataLoader(
    dataset=training_dataset,
    batch_size=128,
    shuffle=True
)

# model ------------------------------------------------------------------------
input_n = 28 * 28

hidden_n = 128
output_n = 10
activation = nn.Sigmoid()

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_n, hidden_n),
    activation,
    nn.Linear(hidden_n, hidden_n),
    activation,
    nn.Linear(hidden_n, output_n)
).to(dev)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

epoch_n = 5

t0 = time.time()

# learning loop ----------------------------------------------------------------
for epoch in range(1, epoch_n + 1):
    loss_sum = 0
    for inputs, targets in iter(loader):
        inputs, targets = inputs.to(dev), targets.to(dev)
        outputs = model.forward(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss

    learning_time = time.time() - t0
    remaining_time = learning_time / epoch * (epoch_n - epoch)
    print(f"Epoch: {epoch}, mean loss: {loss_sum / len(loader)}")
    print(f"Learning time: {time.time() - t0}, Time remaining: {remaining_time}\n")

torch.save(obj=model.to('cpu'), f="model.pt")
print("Model saved.")
