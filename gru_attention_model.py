import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, classification_report
from loguru import logger
import psutil
import gc
from torch.cuda.amp import autocast, GradScaler
import torch.utils.data.distributed as dist

class GRUAttention(nn.Module):
    """GRU with Attention Mechanism"""
    def __init__(self, input_size, hidden_size=48, num_layers=2, num_classes=3, dropout=0.2):
        super(GRUAttention, self).__init__()
        # 修正CuDNN设置位置
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers,
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
        )  # 移除了错误的CuDNN设置
        
        # 简化注意力机制
        self.attention = nn.Linear(hidden_size*2, 1)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.to(torch.bfloat16)  # 在模型内部进行类型转换
        out, _ = self.gru(x)  # out: (batch_size, seq_len, 2*hidden_size)
        
        # Attention weights
        attn_weights = self.attention(out)  # (batch_size, seq_len, 1)
        
        # Context vector
        context = torch.sum(attn_weights * out, dim=1)  # (batch_size, 2*hidden_size)
        
        # Final prediction
        out = self.fc(context)
        return out

class MarketDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, data, sequence_length=900, start_idx=0, end_idx=None):
        self.data = data.iloc[start_idx:end_idx] if end_idx else data.iloc[start_idx:]
        self.seq_len = sequence_length
        
        # 修正特征维度加载
        numeric_cols = data.select_dtypes(include=[np.number]).columns.drop('mid_price', errors='ignore')
        num_features = len(numeric_cols)
        
        mmap_filename = self.data['scaled_features'].iloc[0]
        self.features = np.memmap(mmap_filename, dtype=np.float32, mode='r',
                                  shape=(len(self.data), num_features))  # 使用正确的特征数量
        self.targets = self.data['mid_price'].values

    def __len__(self):
        return len(self.data) - self.seq_len - 120  # 预留未来窗口
    
    def __getitem__(self, idx):
        feature_seq = self.features[idx:idx + self.seq_len].copy()
        target = self.targets[idx + self.seq_len + 120 - 1]
        
        return (
            torch.from_numpy(feature_seq).to(torch.float32),  # 修改数据类型为float32
            torch.tensor(target, dtype=torch.long)
        )

def train_gru_model(data, config):
    """训练GRU+Attention模型（优化版）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 修正特征维度计算
    numeric_cols = data.select_dtypes(include=[np.number]).columns.drop('mid_price', errors='ignore')
    input_size = len(numeric_cols)  # 直接使用数值列数量（已排除mid_price）
    
    model = GRUAttention(
        input_size=input_size,  # 这里现在使用正确的特征维度
        hidden_size=48,
        num_layers=2,
        num_classes=3,
        dropout=0.2
    ).to(device)
    
    # 使用torch.compile()加速（仅PyTorch 2.0+支持）
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # 在模型初始化后添加
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"使用 {torch.cuda.device_count()} GPUs 并行训练")
    
    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1.0, 0.5]).to(device))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    # 修改AMP配置为更节省显存的模式
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # 梯度累积步数
    accumulation_steps = config['grad_accumulation']
    
    best_f1 = 0
    chunk_size = 100000
    num_chunks = (len(data) + chunk_size - 1) // chunk_size

    # 在训练循环前添加
    torch.backends.cudnn.benchmark = True  # 启用CuDNN自动优化器
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # 每个epoch开始时清零梯度
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(data))
            
            chunk_dataset = MarketDataset(data, config['sequence_length'], start_idx, end_idx)
            chunk_loader = DataLoader(
                chunk_dataset, 
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=8,  # 根据CPU核心数调整
                pin_memory=True,
                prefetch_factor=4,  # 增加预取数量
                persistent_workers=True  # 保持worker进程
            )
            
            for i, (inputs, labels) in enumerate(chunk_loader):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).squeeze()
                
                # 修改训练循环中的AMP部分
                with autocast(enabled=True, dtype=torch.bfloat16):  # 强制使用bfloat16
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    # 使用fused优化器
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
                
                total_loss += loss.item() * accumulation_steps
                
                # 清理显存
                del inputs, labels, outputs, loss
                
            # 每个块训练完后清理显存
            gc.collect()
            torch.cuda.empty_cache()
            
            logger.info(f"Epoch {epoch+1}/{config['epochs']} - Chunk {chunk_idx+1}/{num_chunks} completed")
        
        # 在验证集上评估（使用较小的验证集）
        val_size = min(50000, len(data) - int(len(data) * 0.8))  # 限制验证集大小
        val_data = data.iloc[-val_size:]
        val_dataset = MarketDataset(val_data, config['sequence_length'])
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        val_loss, val_f1 = evaluate_model(model, val_loader, device, criterion)
        scheduler.step(val_f1)
        
        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} | "
            f"Train Loss: {total_loss/num_chunks:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}"
        )
        
        # 保存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "gru_attention_best.pth")
            logger.info("保存新最佳模型")
        
        # epoch结束后清理显存
        gc.collect()
        torch.cuda.empty_cache()
    
    return model

def evaluate_model(model, dataloader, device, criterion):
    """模型评估"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).squeeze()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss/len(dataloader), f1

def main():
    """主函数"""
    logger.info("=== 开始GRU+Attention模型训练 ===")
    
    # 统一设置CuDNN参数
    torch.backends.cudnn.enabled = True  # 正确的位置设置CuDNN
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # 设置更激进的显存清理
    torch.cuda.memory.set_per_process_memory_fraction(0.9)
    
    # 训练配置
    config = {
        'batch_size': 512,        # 增大batch size
        'sequence_length': 900,  # 进一步缩短序列长度
        'epochs': 15,
        'grad_accumulation': 2    # 减少梯度累积步数
    }
    
    # 设置较小的初始显存占用
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    # 修改数据预处理部分
    try:
        data = pd.read_feather('train.feather')
        logger.info(f"数据加载完成，样本数: {len(data):,}")
        
        # 特征预处理
        if 'scaled_features' not in data.columns:
            # 获取数值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns.drop('mid_price', errors='ignore')
            logger.info(f"特征列: {numeric_cols.tolist()}")
            
            # 对每个特征单独进行标准化
            scaled_features = np.zeros((len(data), len(numeric_cols)), dtype=np.float32)
            for i, col in enumerate(numeric_cols):
                scaler = RobustScaler()
                scaled_features[:, i] = scaler.fit_transform(data[col].values.reshape(-1, 1)).ravel()
            
            # 保存为内存映射文件
            mmap_filename = 'scaled_features.mmap'
            fp = np.memmap(mmap_filename, dtype=np.float32, mode='w+', 
                          shape=(len(data), len(numeric_cols)))
            fp[:] = scaled_features[:]
            fp.flush()
            
            # 重新打开为只读模式
            data['scaled_features'] = mmap_filename  # 仅保存文件名
            
            logger.info("特征预处理完成")
            
            # 清理临时变量
            del scaled_features
            gc.collect()
            
    except Exception as e:
        logger.error(f"数据处理失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 训练模型
    model = train_gru_model(data, config)
    
    logger.info("=== 训练完成 ===")

if __name__ == "__main__":
    main() 