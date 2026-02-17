from xml.parsers.expat import model
import torch.nn as nn
import torchvision.models as models

def get_model():

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last two blocks
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model
