import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
    
class Encoder(nn.Module):
    def __init__(self, output_dim):
            super(Encoder, self).__init__()
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 512)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, output_dim)
            self.dropout = nn.Dropout(0.1)
            # self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=0)
        return x

class Classifier(nn.Module):
    def __init__(self):
            super(Classifier, self).__init__()
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.encoder.fc = nn.Linear(self.encoder.fc.in_features, 512)
            self.fc1 = nn.Linear(512, 256)
            self.fc2 = nn.Linear(256, 1)
            self.dropout = nn.Dropout(0.1)
            # self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        return x