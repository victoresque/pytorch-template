from base.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = None
        self.fc = None
        self.build_model()

    def build_model(self):
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(16)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return F.log_softmax(output, dim=1)
