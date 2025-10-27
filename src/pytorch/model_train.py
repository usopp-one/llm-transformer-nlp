import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision.transforms import ToTensor as tv_ToTensor

device = "mps" if torch.mps.is_available() else "cpu"
print(f"Using {device} device")

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 3


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_loop(dataloader, model: nn.Module, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (x, y) in enumerate(dataloader, start=1):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=-1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"test error: \n Accuracy: {100 * correct:>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    module = NeuralNetwork().to(device)
    print(module)

    training_data = tv_datasets.FashionMNIST(
        root="data", train=True, download=True, transform=tv_ToTensor()
    )
    test_data = tv_datasets.FashionMNIST(
        root="data", train=False, download=True, transform=tv_ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(module.parameters(), lr=LEARNING_RATE)

    test_loop(test_dataloader, module, loss_fn)
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}")
        print("-" * 100)
        train_loop(train_dataloader, module, loss_fn, optimizer)
        test_loop(test_dataloader, module, loss_fn)
