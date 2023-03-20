import torch.optim as optim

def get_optimizer(model_parameters, config):
    """
    Returns the optimizer for training the model.
    """
    optimizer_type = config.optim_type
    lr = config.learning_rate

    if optimizer_type == "sgd":
        optimizer = optim.SGD(model_parameters, lr=lr)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(model_parameters, lr=lr)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(optimizer_type))

    return optimizer
