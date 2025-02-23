import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix
from loguru import logger
import psutil
import gc
from datetime import timedelta
from scipy.fft import fft
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA
import shap
import wandb
import matplotlib.pyplot as plt
import seaborn as sns

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def stream_csv_chunks(filename, chunksize=100000):
    """Stream CSV file in chunks"""
    return pd.read_csv(filename, chunksize=chunksize)

def process_chunk(chunk):
    """增强版数据预处理，与 hft_pipeline 对应"""
    logger.info(f"开始处理数据块，原始数据大小：{chunk.shape}")
    chunk = chunk.copy()
    
    # 检查原始数据
    logger.info("原始数据统计:")
    logger.info(chunk.describe().T)
    
    # 处理时间列
    time_columns = ['index', 'origin_time', 'Timestamp']
    time_col = next((col for col in time_columns if col in chunk.columns), None)
    
    if time_col:
        logger.info(f"使用时间列: {time_col}")
        chunk['origin_time'] = pd.to_datetime(chunk[time_col])
        if time_col != 'origin_time':
            chunk.drop(columns=[time_col], inplace=True)
    else:
        logger.warning("未检测到时间列，创建简单时间索引")
        chunk['origin_time'] = pd.date_range(
            start='2024-01-01', 
            periods=len(chunk), 
            freq='500ms'  # 假设数据是500ms间隔
        )
    
    # 数据类型转换
    logger.info("正在优化数据类型...")
    keep_float64 = ['mid_price', 'order_book_pressure', 'weighted_price_depth']
    float_cols = [col for col in chunk.select_dtypes(include='float64').columns 
                 if col not in keep_float64]
    chunk[float_cols] = chunk[float_cols].astype(np.float32)
    
    # 修改特征保留列表，保留mid_price但不作为训练特征
    features_to_keep = [
        'mid_price',  # 保留mid_price用于后续计算
        'relative_spread', 'depth_imbalance',
        'bid_ask_slope', 'order_book_pressure', 'weighted_price_depth',
        'liquidity_imbalance', 'flow_toxicity', 'price_momentum',
        'volatility_ratio', 'ofi', 'vpin', 'pressure_change_rate',
        'orderbook_gradient', 'depth_pressure_ratio'
    ]
    
    # 确保所有保留的特征都存在
    missing_features = [f for f in features_to_keep if f not in chunk.columns]
    if missing_features:
        logger.error(f"缺失必要特征: {missing_features}")
        # 添加调试信息
        logger.info(f"当前可用列: {chunk.columns.tolist()}")
        raise ValueError(f"数据中缺失必要特征: {missing_features}")
    
    # 添加数据有效性检查
    for col in features_to_keep:
        # 检查无效值
        invalid_mask = chunk[col].isna() | np.isinf(chunk[col])
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.warning(f"{col} 包含 {invalid_count} 个无效值")
            # 使用前向填充处理无效值
            chunk[col] = chunk[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()
        
        # 检查异常值
        q1 = chunk[col].quantile(0.01)
        q3 = chunk[col].quantile(0.99)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 使用分位数进行截断
        chunk[col] = np.clip(chunk[col], lower_bound, upper_bound)
    
    chunk = chunk[features_to_keep].ffill().bfill().fillna(0)
    
    # 特征分布检查
    logger.info("特征分布检查:")
    for col in features_to_keep:
        stats = chunk[col].describe()
        logger.info(f"{col} 统计:\n{stats}")    
        zero_percent = (chunk[col] == 0).mean() * 100
        if zero_percent > 90:
            logger.warning(f"高零值警告: {col} 有 {zero_percent:.2f}% 的零值!")
    
    return chunk

def create_features(data_chunk):
    # 需要定义 base_features
    base_features = [col for col in data_chunk.columns 
                    if col not in ['target_class', 'mid_price', 'price_change_rate']]
    
    # 确保滚动窗口方向正确
    windows = [60, 300, 900, 1800]  # 单位：样本数
    
    for window in windows:
        rolled = data_chunk[base_features].rolling(
            window=window,
            min_periods=1,
            closed='left'  # 关键修正点
        )
        
        # 计算统计量时使用历史数据
        stats = rolled.agg(['mean', 'std', 'skew', 'kurt']).fillna(0)
        stats.columns = [f'{col}_{stat}' for col in base_features for stat in ['mean', 'std', 'skew', 'kurt']]
        
        data_chunk = pd.concat([data_chunk, stats.add_suffix(f'_{window}')], axis=1)
    
    # 需要确保滚动窗口只使用历史数据
    data_chunk['volatility_30s'] = data_chunk['mid_price'].rolling(
        window=60, 
        min_periods=1,
        closed='left'  # 确保不包含当前时刻
    ).std()
    
    # 检查是否存在时间列
    if 'origin_time' in data_chunk.columns:
        # 如果有origin_time列，将其转换为datetime类型
        data_chunk['origin_time'] = pd.to_datetime(data_chunk['origin_time'])
        # 添加时间序列特征
        data_chunk['time_sin'] = np.sin(2 * np.pi * data_chunk['origin_time'].dt.hour/24)
        data_chunk['time_cos'] = np.cos(2 * np.pi * data_chunk['origin_time'].dt.hour/24)
    else:
        # 如果没有时间列，创建简单的周期性特征
        data_chunk['time_sin'] = np.sin(2 * np.pi * np.arange(len(data_chunk))/len(data_chunk))
        data_chunk['time_cos'] = np.cos(2 * np.pi * np.arange(len(data_chunk))/len(data_chunk))
        logger.warning("未找到时间列，使用位置索引创建周期性特征")
    
    # 添加交互特征
    data_chunk['spread_depth_ratio'] = data_chunk['relative_spread'] / (data_chunk['depth_imbalance'] + 1e-7)
    
    # 添加订单簿动态特征
    data_chunk['order_flow_imbalance'] = (
        data_chunk['bid_ask_slope'] * data_chunk['liquidity_imbalance']
    )
    
    # 添加市场状态特征
    data_chunk['market_state'] = calculate_market_state(data_chunk)
    
    return data_chunk

def calculate_market_state(data_chunk):
    """计算市场状态（增强稳定性）"""
    try:
        volatility = data_chunk['flow_toxicity'].rolling(300, min_periods=1).mean().fillna(0)
        returns = data_chunk['price_momentum'].fillna(0)
        
        conditions = [
            (volatility < 0.1) & (returns.abs() < 0.3),
            (volatility >= 0.3),
            (returns >= 0.3),     
            (returns <= -0.3)    
        ]
        return np.select(conditions, [0, 1, 2, 3], default=0)
    except Exception as e:
        logger.error(f"市场状态计算失败: {str(e)}")
        return np.zeros(len(data_chunk))

def train_enhanced_model(data, config):
    # 新增初始化
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': config['num_leaves'],
        'learning_rate': config['learning_rate'],
        'feature_fraction': config['feature_fraction']
    }
    metrics = []
    models = []
    
    # 明确特征列
    feature_columns = [col for col in data.columns 
                      if col not in ['target_class', 'mid_price', 'price_change_rate']]
    
    # 转换为数值型数据
    X = data[feature_columns].select_dtypes(include=[np.number])
    y = data['target_class']
    
    # 时间序列分割
    tscv = TimeSeriesSplit(n_splits=3)
    
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 创建数据集
        train_set = lgb.Dataset(X_train, y_train)
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)
        
        # Define a custom learning rate schedule
        def custom_learning_rate(iteration):
            return 0.02 * (0.99 ** iteration)

        # Train the model
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            callbacks=[
                log_class_distribution,
                lgb.reset_parameter(learning_rate=custom_learning_rate),
                lgb.early_stopping(100)
            ]
        )
        
        # Evaluate the model
        y_pred = model.predict(X_val)
        current_metrics = multi_task_metric(y_pred, y_val)
        
        if current_metrics is not None:
            metrics.append(current_metrics)
            models.append(model)
            
            logger.info(f"窗口 {fold+1} 验证结果:")
            logger.info(f"Accuracy: {current_metrics[0][1]:.4f}")
            logger.info(f"Precision: {current_metrics[1][1]:.4f}")
            logger.info(f"Recall: {current_metrics[2][1]:.4f}")
            logger.info(f"F1 Score: {current_metrics[3][1]:.4f}")
        
        # 特征重要性分析
        importance = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n窗口 {fold+1} 重要特征:")
        logger.info(importance.head(10))
        
        # 实时特征重要性监控
        wandb.log({
            "top_feature_1": importance.iloc[0]['feature'],
            "top_feature_1_importance": importance.iloc[0]['importance'],
            "top_feature_2": importance.iloc[1]['feature'],
            "top_feature_2_importance": importance.iloc[1]['importance']
        })
        
        # 预测分布监控
        plt.figure(figsize=(10, 6))
        sns.histplot(y_pred, bins=50, kde=True)
        plt.title(f"Window {fold+1} Prediction Distribution")
        
        # 修改记录方式：先保存图片再记录
        plt.savefig(f'pred_dist_window_{fold+1}.png')
        wandb.log({"pred_dist": wandb.Image(f'pred_dist_window_{fold+1}.png')})
        plt.close()  # 重要：关闭图形释放内存
        
        # 特征漂移检测
        train_mean = X_train.mean(axis=0)
        val_mean = X_val.mean(axis=0)
        feature_drift = np.mean(np.abs(train_mean - val_mean))
        wandb.log({"feature_drift": feature_drift})
        
        # Log the learning rate manually
        current_lr = custom_learning_rate(model.current_iteration())
        wandb.log({
            "window": fold+1,
            "accuracy": current_metrics[0][1],
            "precision": current_metrics[1][1],
            "recall": current_metrics[2][1],
            "f1": current_metrics[3][1],
            "learning_rate": current_lr
        })
        
        # 保存每个窗口的验证预测结果
        pd.DataFrame({
            'pred': y_pred,
            'true': y_val
        }).to_csv(f'window_{fold+1}_predictions.csv', index=False)
    
    # 计算平均指标
    if metrics:
        avg_metrics = {
            'accuracy': np.mean([m[0][1] for m in metrics]),
            'precision': np.mean([m[1][1] for m in metrics]),
            'recall': np.mean([m[2][1] for m in metrics]),
            'f1': np.mean([m[3][1] for m in metrics])
        }
        
        logger.info("\n平均性能指标:")
        for metric_name, value in avg_metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
    
    return models, metrics

def compute_sample_weights(y):
    # 改进的权重计算方法
    class_counts = np.bincount(y.astype(int))
    median_freq = np.median(class_counts)
    class_weights = {i: median_freq / count for i, count in enumerate(class_counts)}
    return np.array([class_weights[int(yi)] for yi in y])

def multi_task_metric(y_pred, y_true):
    """完善的二分类评估指标"""
    # 确保输入数据的有效性
    if len(y_pred) != len(y_true):
        logger.error("预测值和真实值长度不匹配")
        return None
        
    # 转换预测概率为类别
    y_pred_class = (y_pred > 0.5).astype(int)
    
    # 计算混淆矩阵
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
        
        # 计算各项指标
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 添加类别分布信息
        class_distribution = np.bincount(y_true.astype(int)) / len(y_true)
        
        logger.info(f"类别分布: 类别0: {class_distribution[0]:.2%}, 类别1: {class_distribution[1]:.2%}")
        
        return [
            ('accuracy', accuracy, True),
            ('precision', precision, True),
            ('recall', recall, True),
            ('f1_score', f1, True)
        ]
    except Exception as e:
        logger.error(f"评估指标计算失败: {str(e)}")
        return None

def analyze_data_quality(data):
    """数据质量分析报告"""
    logger.info("\n=== 数据质量分析报告 ===")
    
    # 缺失值分析
    missing = data.isna().sum()
    logger.info(f"\n缺失值统计:\n{missing[missing > 0]}")
    
    # 无限值分析（限制为数值型数据）
    numeric_data = data.select_dtypes(include=[np.number])
    inf_counts = np.isinf(numeric_data).sum()
    logger.info(f"\n无限值统计:\n{inf_counts[inf_counts > 0]}")
    
    # 数值分布分析
    logger.info("\n数值分布统计:")
    logger.info(data.describe().T)
    
    # 零值分析
    zero_counts = (data == 0).sum()
    logger.info(f"\n零值统计:\n{zero_counts[zero_counts > 0]}")
    
    # 相关性分析（限制输出大小）
    logger.info("\n特征相关性矩阵:")
    corr_matrix = data.corr()
    # 只显示相关性大于0.7的部分
    high_corr = (corr_matrix.abs() > 0.7).any()
    logger.info(f"\n高相关性特征:\n{corr_matrix.loc[high_corr, high_corr]}")
    
    logger.info("=== 分析完成 ===")

def explain_model(model, X_val, feature_names):
    # 限制样本数量
    max_samples = min(1000, len(X_val))
    sample_indices = np.random.choice(len(X_val), max_samples, replace=False)
    X_val_sample = X_val.iloc[sample_indices]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_sample)
    
    # 生成特征重要性图
    shap.summary_plot(shap_values, X_val_sample, feature_names=feature_names)
    
    # 生成单个样本解释示例
    sample_idx = np.random.randint(0, X_val_sample.shape[0])
    shap.force_plot(explainer.expected_value, shap_values[sample_idx,:], 
                   X_val_sample.iloc[sample_idx].values, feature_names=feature_names)

def log_class_distribution(env):
    """记录每次迭代的类别分布"""
    if env.iteration % 100 == 0:  # 每100次迭代记录一次
        y_train = env.model.train_set.get_label()
        class_dist = np.bincount(y_train.astype(int)) / len(y_train)
        logger.info(f"Iteration {env.iteration}, Class distribution: "
                   f"Class 0: {class_dist[0]:.2%}, Class 1: {class_dist[1]:.2%}")

def main():
    # 添加配置定义
    config = {
        'window_size': 3600 * 6,
        'prediction_horizon': 60,
        'max_features': 500,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9
    }
    
    logger.info("=== 开始增强版模型训练 ===")
    # 初始化wandb
    wandb.init(project="hft-model", config=config)
    log_memory_usage()
    
    logger.info("加载数据...")
    try:
        full_data = pd.read_feather('train.feather')
        logger.info(f"数据加载完成，总行数：{len(full_data):,}")
        logger.info(f"可用列: {full_data.columns.tolist()}")
        
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        return
    
    # 添加特征验证
    required_features = [
        'relative_spread', 'depth_imbalance',
        'bid_ask_slope', 'order_book_pressure', 'weighted_price_depth',
        'liquidity_imbalance', 'flow_toxicity', 'price_momentum',
        'volatility_ratio', 'ofi', 'vpin', 'pressure_change_rate',
        'orderbook_gradient', 'depth_pressure_ratio'
    ]
    
    missing = [f for f in required_features if f not in full_data.columns]
    if missing:
        logger.error(f"缺失关键特征: {missing}")
        return
    
    log_memory_usage()
    
    logger.info("开始数据预处理...")
    processed_data = process_chunk(full_data)
    del full_data
    gc.collect()
    
    # 添加数据质量分析
    analyze_data_quality(processed_data)
    
    log_memory_usage()
    
    logger.info("开始特征工程...")
    feature_data = create_features(processed_data)
    del processed_data
    gc.collect()
    if feature_data is None:
        logger.error("特征工程失败，终止程序！")
        return
    
    # 在时间窗口划分后生成目标变量
    prediction_points = 120
    threshold = 0.0004
    feature_data['price_change_rate'] = (
        feature_data['mid_price'].shift(-prediction_points) - feature_data['mid_price']
    ) / feature_data['mid_price']
    # 修改目标变量生成规则：仅当收益率小于 -0.0004 时视为目标类别
    feature_data['target_class'] = np.where(
        feature_data['price_change_rate'] < -threshold, 1, 0
    )
    feature_data = feature_data.dropna(subset=['target_class'])  # 移除最后无法计算的行
    
    log_memory_usage()
    
    logger.info(f"训练配置：{config}")
    
    models, metrics = train_enhanced_model(feature_data, config)
    
    logger.info("保存模型...")
    for idx, model in enumerate(models):
        model.save_model(f'model_down_v{idx}.bin')
        logger.info(f"模型 {idx} 已保存")
    
    logger.info("=== 训练流程完成 ===")
    log_memory_usage()

    # 在模型训练后添加SHAP值分析
    feature_cols = [col for col in feature_data.columns 
                   if col not in ['target_class', 'mid_price', 'price_change_rate']]
    # 添加验证集划分
    val_size = int(len(feature_data) * 0.2)
    val_start = len(feature_data) - val_size
    val_end = len(feature_data)
    
    explain_model(models[0], feature_data[feature_cols].iloc[val_start:val_end], feature_cols)

    # 添加最终检查
    logger.info("最终特征列表: %s", feature_cols)
    logger.info("目标变量分布:\n%s", feature_data['target_class'].value_counts(normalize=True))
    
    # 检查是否存在数据泄露
    assert not np.any(feature_data['target_class'].isna()), "存在未处理的NaN目标值"
    assert 'target_class' not in feature_cols, "目标变量泄露到特征中"
    
    # 检查时间顺序
    assert feature_data.index.is_monotonic_increasing, "数据索引未按时间排序"

if __name__ == "__main__":
    main()
