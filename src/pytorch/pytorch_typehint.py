from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn


class Model(nn.Module):
    if TYPE_CHECKING:

        def __call__(self, x: Tensor) -> Tensor: ...

    def forward(self, x: Tensor) -> Tensor:
        return x * 2 - 1


model = Model()
x = torch.rand((3, 3))
y = model(x)
