import torchvision.models as models 
import torch.nn as nn

class VGG16Model(nn.Module):
    """
    """
    def __init__(self, num_class:int=10):
        super().__init__()
        self.model = self.get_pretrained_model(num_class=num_class)

    def get_pretrained_model(self, feature_extract:bool=True, num_class:int=10):
        model = models.vgg16(pretrained=True)
        # set_parameter_requires_grad
        if feature_extract:
            for name, param in model.named_parameters():
                param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_class)
        return model

    def forward(self, x):
        x = self.model(x)
        return x