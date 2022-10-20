import torch.nn as nn
import torch
class SoundClassifier(nn.Module):
    def __init__(self):
        super(SoundClassifier,self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=24,kernel_size=(5,5)),
            nn.MaxPool1d(kernel_size=(4,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=24,out_channels=48,kernel_size=(5,5)),
            nn.MaxPool1d(kernel_size=(4,2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=48,out_channels=48,kernel_size=(5,5)),
            nn.ReLU(),
        )
        self.Linear = nn.Sequential(
            nn.Linear(2400,64),
            nn.ReLU(),
            nn.Linear(64,10),
            nn.Softmax()
        )
        def forward(self, input):
            out = torch.flatten(self.cnn(input))
            out = self.linear(out)
            return out

