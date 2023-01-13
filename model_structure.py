import os
import torch
from torchvision import models


def printmodel(model_name):
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    device = torch.device('cuda' if use_cuda else 'cpu')
    num_classes = 24
    model = getattr(models, model_name)(num_classes=num_classes).to(device)
    outdir = os.path.join(os.getcwd(), 'model_structure')
    os.makedirs(outdir, exist_ok=True)
    # with open(os.path.join(outdir, f'{model_name}.txt'), 'a') as ftxt:
    #     ftxt.write(model_name)
    print(model)


# printmodel('mobilenet_v2')
printmodel('googlenet')
