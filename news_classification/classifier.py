import torch.nn as nn
import torch.optim as optim
from batched_iterator import BatchedIterator
import torch
import os
import numpy as np


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, dropout_value):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(dropout_value)

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        h = self.relu(self.hidden(X))
        h = self.drop_out(h)

        out = self.output_layer(h)
        return out
