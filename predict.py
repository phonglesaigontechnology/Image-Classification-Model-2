import torch
import argparse
import os 
import torchvision.transforms as transforms

from PIL import Image
from src.model import get_model
from src.config import Configs
from src.dataset import get_transforms


def predict(config, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model, _ = get_model(model_name=config.model_name)
    checkpoint = torch.load(os.path.join(config.checkpoint_dir, '{}_best.pth'.format(config.model_name)))
    model.load_state_dict(checkpoint['state_dict'])

    # Set model to eval mode
    model.eval()

    # Load image and preprocess
    label_map = {i: v for i, v in enumerate(os.listdir(os.path.join(config.data_dir, "train")))}
    _, test_transforms = get_transforms()
    image = Image.open(image_path)
    image = test_transforms(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        class_index = predicted.item()
        class_name = label_map[class_index]

    print(f'Prediction: {class_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='path to config file')
    parser.add_argument('--image', type=str, required=True, help='path to input image')
    args = parser.parse_args()

    config = Configs(args.config)
    predict(config, args.image)
