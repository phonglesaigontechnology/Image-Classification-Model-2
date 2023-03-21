import os
import torch

def save_checkpoint(state, filename="checkpoint.pth"):
    """
    Saves the model state as a checkpoint file.
    """
    # mlflow.pytorch.save_model() 
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    """
    Loads the model state and optimizer state from a checkpoint file.
    """
    if os.path.isfile(filename):
        print("Loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint['best_accuracy']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Checkpoint loaded successfully from epoch {} with best accuracy {:.2f}%".format(
            start_epoch, best_accuracy * 100))
    else:
        print("No checkpoint found at '{}'".format(filename))
        start_epoch = 0
        best_accuracy = 0.0

    return start_epoch, best_accuracy
