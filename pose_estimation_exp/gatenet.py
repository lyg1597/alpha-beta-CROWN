import torch
import torch.nn as nn
import torch.nn.functional as F

class GateNet(nn.Module):
    def __init__(self, config):
        super(GateNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(32, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn5 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True)
        self.bn6 = nn.BatchNorm2d(16, momentum=config['batch_norm_decay'], eps=config['batch_norm_epsilon'])

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, int(torch.prod(torch.tensor(config['output_shape']))))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=2)


        x = F.relu(self.bn6(self.conv6(x)))  # No pooling after conv6

        x = self.flatten(x)

        x = self.fc(x)

        return x