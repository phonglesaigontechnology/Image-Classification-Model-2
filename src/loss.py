import torch.nn as nn

class MyLoss(nn.Module):
    """
    Defines the loss function for image classification.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        return self.criterion(inputs, targets)
