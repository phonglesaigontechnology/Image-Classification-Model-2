import torch
import argparse
import os
from src.model import get_model
from src.config import Configs
from src.dataset import get_data_loader


def test(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    test_dataloader = get_data_loader(config.data_dir, config.test_batch_size, mode="test")

    # Load model
    model, _ = get_model(model_name=config.model_name)
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, '{}_best.pth'.format(config.model_name)))
    model.load_state_dict(checkpoint['state_dict'])

    # Set model to eval mode
    model.eval()

    # Test model on test set
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='path to config file')
    args = parser.parse_args()

    config = Configs(args.config)
    test(config)
