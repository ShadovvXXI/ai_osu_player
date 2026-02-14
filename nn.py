import numpy as np
import os
import torch
from torch import nn


class OsuNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            # TODO : реализация архитектуры нейросети
        )

    def forward(self, x):
        return self.seq(x)