import torch
from torch import nn

device = "mps" if torch.mps.is_available() else "cpu"
print(f"Using {device} device")


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


if __name__ == "__main__":
    module = NeuralNetwork().to(device)
    print(module)

    x = torch.rand(4, 28, 28, device=device)
    logits = module(x)
    pred_probab = nn.Softmax(dim=1)(logits)
    print(pred_probab.size(), pred_probab.shape)
    y_pred = pred_probab.argmax(-1)
    print(f"Predicted class: {y_pred}")
