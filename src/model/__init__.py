from .cnn1_model import CNN1Model
from .cnn2_model import CNN2Model
from .vgg16_model import VGG16Model

def get_model(model_name:str="cnn1", num_class:int=10):
    """
    """
    if model_name == "cnn1":
        model = CNN1Model(num_class)
        params_to_update = model.parameters()
    elif model_name == "cnn2":
        model = CNN2Model(num_class)
        params_to_update = model.parameters()
    elif model_name == "vgg16":
        model = VGG16Model(num_class)
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        raise ValueError("Unsuported model {}, it should be `cnn1`, `cnn2`, or `vgg16`".format(model_name))
    
    return model, params_to_update