import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels=[128, 128, 128], kernel_size=5):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels[0], kernel_size, padding=(kernel_size-1)//2),
            nn.GELU(),
            nn.BatchNorm1d(num_channels[0]),
            nn.Dropout(0.3),
            nn.Conv1d(num_channels[0], num_channels[1], kernel_size, padding=(kernel_size-1)//2),
            nn.GELU(),
            nn.BatchNorm1d(num_channels[1]),
            nn.Dropout(0.3),
            nn.Conv1d(num_channels[1], num_channels[2], kernel_size, padding=(kernel_size-1)//2),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, time]
        features = self.tcn(x).squeeze()
        return self.fc(features)

def train_tcn(features, targets):
    # 使用更短的序列长度
    seq_length = 30  # 15秒窗口
    X = [features[i-seq_length:i] for i in range(seq_length, len(features))]
    y = targets[seq_length:]
    
    # 添加高频数据增强
    X = [x + np.random.normal(0, 0.01*x.std(), x.shape) for x in X]
    
    # 使用更小的批处理
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 模型训练
    model = TCNModel(input_size=features.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=len(loader), epochs=10
    )
    
    for epoch in range(10):
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model 

class OnlineTCN:
    def __init__(self, input_size):
        self.model = TCNModel(input_size)
        self.buffer = []
        self.buffer_size = 10000  # 增大缓冲区
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0005)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
    def partial_fit(self, features, targets):
        # 添加数据标准化
        features = (features - np.mean(features, axis=(0,1), keepdims=True)) / \
                  (np.std(features, axis=(0,1), keepdims=True) + 1e-8)
        
        # 累积数据到缓冲区
        self.buffer.extend(zip(features, targets))
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        
        # 使用更智能的训练触发机制
        if len(self.buffer) >= min(2000, self.buffer_size//2):
            X, y = zip(*self.buffer)
            dataset = TensorDataset(torch.FloatTensor(np.array(X)), 
                                   torch.LongTensor(np.array(y)))
            loader = DataLoader(dataset, batch_size=512, shuffle=True)
            
            # 两阶段训练
            self.model.train()
            for _ in range(3):  # 少量迭代
                total_loss = 0
                for inputs, labels in loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    total_loss += loss.item()
                self.scheduler.step() 