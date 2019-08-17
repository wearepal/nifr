"""Run all tests"""
import torch

x = torch.tensor([1, 2, 3, 4, 5])
inds = [2, 3]
x[inds] = x[inds][torch.randperm(len(inds))]
print(x)
