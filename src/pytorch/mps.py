import timeit

import torch

M = torch.rand(10000, 10000)
print(timeit.timeit(lambda: M.mm(M).mm(M), number=2))

N = torch.rand(10000, 10000, device="mps")
print(timeit.timeit(lambda: N.mm(N).mm(N), number=2))
