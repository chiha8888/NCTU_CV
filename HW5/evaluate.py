import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResNet34(nn.Module):
    def __init__(self,num_class):
        super(ResNet34,self).__init__()
        self.model = models.resnet34(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad=False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 15)

    def forward(self,X):
        out=self.model(X)
        return out


if __name__=='__main__':
    # load data
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_data = ImageFolder(os.path.join('hw5_data', 'test'), transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=4)

    # load model
    model=ResNet34(num_class=15)
    model.to(device)
    model.load_state_dict(torch.load('model_0.94.pt'))

    # test
    model.eval()
    correct = 0
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        predicts = model(images)

        # statistic
        correct += predicts.max(dim=1)[1].eq(targets).sum().item()
    print(f'Acc: {correct / len(test_loader.dataset):.2f}')