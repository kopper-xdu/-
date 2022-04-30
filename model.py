from torch import nn


class Net(nn.Module):
    def __init__(self, in_features=22, out_features=5, dropout_p=0):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_features, 256),
                                 nn.BatchNorm1d(256),
                                 nn.Dropout(dropout_p),
                                 nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(256, 512),
                                 nn.BatchNorm1d(512),
                                 nn.Dropout(dropout_p),
                                 nn.ReLU())
        
        self.fc3 = nn.Sequential(nn.Linear(512, 128),
                                 nn.BatchNorm1d(128),
                                 nn.Dropout(dropout_p),
                                 nn.ReLU())
        
        self.linear = nn.Linear(128, out_features)
    
    def forward(self, input):
        out = self.fc3(self.fc2(self.fc1(input)))
        out = self.linear(out)
        return out
        
        