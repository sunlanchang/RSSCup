import torch
import numpy as np


def Accuracy(output, label):
    output = torch.argmax(output.squeeze(), dim=1).int()
    label = torch.argmax(label.squeeze(), dim=1).int()
    correctNum = (output == label).float().sum()
    labelNum = float(np.prod(list(output.shape)))
    correctPercentage = correctNum/labelNum
    return correctPercentage.item()
