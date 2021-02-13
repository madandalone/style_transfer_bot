import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def transfer(img_path, imsize=512):
    device = torch.device("cpu")
    model=torch.load("./pretrained_models/monet.pth").to(device)
    img = image_loader(img_path, imsize, device)

    for p in model.parameters():
        p.requires_grad = False

    return model(img)

def image_loader(image_name, imsize, device):
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


if __name__ == '__main__':


    pass
