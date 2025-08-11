import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_(checkpoint_path):
    """
    Loads a pre-trained ResNet-50 model from a checkpoint file.

    This function initializes a ResNet-50 model with a modified final layer,
    loads the saved weights from a specified checkpoint path, and sets the model
    to evaluation mode. The model is loaded onto the CPU.

    Args:
        checkpoint_path (str): The file path to the saved model checkpoint.
                                The checkpoint is expected to be a dictionary
                                containing the model's state dictionary
                                under the key 'model_state_dict'.

    Returns:
            torch.nn.Module: The loaded ResNet-50 model ready for inference.
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    if "efficientnet" in checkpoint_path.lower():
        model = models.efficientnet_b2(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, 1)  # regression output
        )
    else:
        model = models.resnet50(weights=None)
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


def enable_dropout(model):
    """Enable dropout during inference for uncertainty estimation."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def predict_with_uncertainty(model, input_data, n_passes=25):
    """
    Perform multiple forward passes with dropout enabled to estimate uncertainty.

    Args:
        model (torch.nn.Module): The model with dropout layers.
        input_data (torch.Tensor): The input data to make predictions on.
        n_passes (int): The number of forward passes to run.

    Returns:
        tuple: A tuple containing the mean prediction and the uncertainty (variance).
    """
    model.eval()  # Set the model to evaluation mode
    enable_dropout(model)  # Enable dropout layers

    predictions = []
    with torch.no_grad():
        for _ in range(n_passes):
            # Run a forward pass
            output = model(input_data)
            predictions.append(output)

    # Stack all predictions and calculate mean and variance
    predictions = torch.stack(predictions)

    # The mean of predictions is the final prediction
    mean_prediction = torch.mean(predictions, dim=0)

    # The variance of predictions is a measure of uncertainty
    uncertainty = torch.var(predictions, dim=0)

    return mean_prediction.item(), uncertainty.item()


