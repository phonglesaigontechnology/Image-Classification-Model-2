import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter
import mlflow

from src.model import get_model
from src.loss import MyLoss
from src.optim import get_optimizer
from src.config import Configs
from src.dataset import get_data_loader
from src.util import save_checkpoint

def train(config):
    # Set up TensorBoard
    writer = SummaryWriter(os.path.join("{}_{}".format(config.tensorboard_logdir, config.model_name), time.strftime("%Y-%m-%d-%H-%M-%S")))
    
    # Set up MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment("{}_{}".format(config.mlflow_experiment_name, config.model_name))
    with mlflow.start_run():
        # Log hyperparameters
        for key, value in config.__dict__.items():
            mlflow.log_param(key, value)

        # Set up the data loaders
        train_loader = get_data_loader(config.data_dir, batch_size=config.batch_size, mode="train")
        val_loader = get_data_loader(config.data_dir, batch_size=config.batch_size, mode="val")

        # Set up the model
        model, model_parameters = get_model(model_name=config.model_name)
        model = model.to(config.device)

        # Set up the loss function
        criterion = MyLoss()

        # Set up the optimizer
        optimizer = get_optimizer(model_parameters, config)

        # Start training
        global_step = 0
        best_accuracy = 0.0
        for epoch in range(config.num_epochs):
            model.train()

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(config.device)
                labels = labels.to(config.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Log the loss to TensorBoard and MLflow
                writer.add_scalar('train/loss', loss.item(), global_step)
                mlflow.log_metric('train_loss', loss.item(), step=global_step)
                global_step += 1

            # Evaluate the model on the validation set
            accuracy = evaluate(model, val_loader, config.device)

            # Log the accuracy to TensorBoard and MLflow
            writer.add_scalar('val/accuracy', accuracy, epoch)
            mlflow.log_metric('val_accuracy', accuracy, step=epoch)

            # Save a checkpoint if the accuracy is better than the previous best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print("Saving best accuracy {:.2f}%".format(best_accuracy))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'best_accuracy': best_accuracy,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, filename=os.path.join(config.checkpoint_dir, '{}_best.pth'.format(config.model_name)))

        print("Training complete with best accuracy {:.2f}%".format(best_accuracy))

def evaluate(model, dataloader, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).float().sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='path to config file')
    args = parser.parse_args()

    config = Configs(args.config)

    train(config)
