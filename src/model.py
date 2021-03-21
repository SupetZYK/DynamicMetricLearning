import torch
from torchvision.models import resnet34

class Resnet(torch.nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
        back = resnet34(pretrained=False)
        self.net = torch.nn.Sequential(
            *list(model.children())[:-2],
            torch.nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
        )
    def forward(self, x):
        feat = self.net(x)
        if self.normalize:
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat
