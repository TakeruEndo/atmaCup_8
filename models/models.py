import torch.nn as nn
import torch.nn.functional as F


class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.fc1 = nn.Linear(in_features, in_features * 2, bias)
        self.fc2 = nn.Linear(in_features * 2, in_features)
        self.fc3 = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.drop(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x


class MLP_Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(MLP_Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, num_features // 2))
        
        self.batch_norm2 = nn.BatchNorm1d(num_features // 2)
        self.dropout2 = nn.Dropout(0.6)
        self.dense2 = nn.utils.weight_norm(nn.Linear(num_features // 2, num_features // 4))
        
        self.batch_norm3 = nn.BatchNorm1d(num_features // 4)
        self.dropout3 = nn.Dropout(0.6)
        self.dense3 = nn.utils.weight_norm(nn.Linear(num_features // 4, num_targets))
    
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.relu(self.dense1(x))
        
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense2(x))
        
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        
        return x
