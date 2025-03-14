import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet2(nn.Module):
    def __init__(self, config):
        super(GateNet2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2304, int(torch.prod(torch.tensor(config['output_shape']))))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=2)

        x = self.flatten(x)

        x = self.fc(x)

        return x