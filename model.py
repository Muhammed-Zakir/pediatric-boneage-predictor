import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
from PIL import ImageStat

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


def is_probable_xray(image: Image.Image, threshold=0.3):
    """
    Simple heuristic: calculate ratio of pixels with brightness in
    typical X-ray range (dark + midtones) vs total pixels.
    Returns True if ratio above threshold.
    """
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    mean_brightness = stat.mean[0] / 255.0  # Normalize 0-1

    # If mean brightness is too high or too low, probably not X-ray
    if mean_brightness < 0.1 or mean_brightness > 0.6:
        return False
    return True