import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_dropout(model):
    """Enable dropout during inference for uncertainty estimation."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

