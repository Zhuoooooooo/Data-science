import torch
from torch import nn
import matplotlib.pyplot as plt 
print('Torch Version :',torch.__version__)

# Create known parameters
weight = 0.7
bias = 0.3
# Create data
start = 0
end = 1
step = 0.02
# Create features
X = torch.arange(start, end, step).unsqueeze(dim = 1)
# Create labels
Y = weight * X + bias
X
Y