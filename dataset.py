from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch


class dataset(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()
        if train:
            data = pd.read_csv('C:\\Program1\\vscode\\数模\\A题\\训练数据集\\训练集tick数据分开\\tick1.csv')
            label = pd.read_csv('C:\\Program1\\vscode\\数模\\A题/训练数据集/label.csv')
        else:
            data = pd.read_csv('path' + '/测试数据集/tick.csv')
            # TODO

        self.data = np.array(data.iloc[:20116, 1:]) #day1
        label = np.array(label.iloc[:, 2:])
        label = np.insert(label, label.shape[1], 0, axis=1)
        self.label = np.argmax(label, axis=1)

    
    def __len__(self):
        return self.data.shape[0]
        
    
    def __getitem__(self, index):
        data = self.data[:, 1:][index]
        label = self.label[int(self.data[index][0])]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label)