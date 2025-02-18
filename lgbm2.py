import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from loguru import logger
import psutil
import gc
from datetime import timedelta
from scipy.fft import fft
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA

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
    
    # 如果没有时间列，创建一个简单的时间索引
    if not any(col in chunk.columns for col in ['index', 'origin_time', 'Timestamp']):
        logger.info("未检测到时间列，创建简单时间索引")
        chunk.index = pd.date_range(
            start='2024-01-01', 
            periods=len(chunk), 
            freq='500ms'  # 假设数据是500ms间隔
        )
        logger.info(f"创建的时间索引范围: {chunk.index[0]} 到 {chunk.index[-1]}")
    
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
    
    # 异常值处理
    for col in features_to_keep:
        if col in chunk.columns:
            q1 = chunk[col].quantile(0.01)
            q3 = chunk[col].quantile(0.99)
            chunk[col] = np.clip(chunk[col], q1, q3)
    
    # 处理缺失值
    chunk = chunk[features_to_keep].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # 特征分布检查
    logger.info("特征分布检查:")
    for col in features_to_keep:
        stats = chunk[col].describe()
        logger.info(f"{col} 统计:\n{stats}")    
        zero_percent = (chunk[col] == 0).mean() * 100
        if zero_percent > 90:
            logger.warning(f"高零值警告: {col} 有 {zero_percent:.2f}% 的零值!")
    
    return chunk

def create_features(data_chunk, window_size=1800):
    """增强版特征工程，改为五分类预测"""
    logger.info("开始特征工程...")
    data_chunk = data_chunk.copy()
    
    # 计算目标变量 - 改为五分类
    prediction_points = 120  # 60秒 = 120个500ms间隔
    if 'mid_price' not in data_chunk.columns:
        logger.error("缺少mid_price列，无法计算目标变量！")
        return None
    
    # 计算价格变化率
    data_chunk['price_change_rate'] = (data_chunk['mid_price'].shift(-prediction_points) - 
                                      data_chunk['mid_price']) / data_chunk['mid_price']
    
    # 改进分类边界定义，增加中间类别的区分度
    thresholds = [-0.0004, -0.0002, 0.0002, 0.0004]
    buffer_zone = 0.00005  # 增加缓冲区
    
    conditions = [
        (data_chunk['price_change_rate'] <= thresholds[0] - buffer_zone),
        (data_chunk['price_change_rate'] > thresholds[0] - buffer_zone) & 
        (data_chunk['price_change_rate'] <= thresholds[1] + buffer_zone),
        (data_chunk['price_change_rate'] > thresholds[1] + buffer_zone) & 
        (data_chunk['price_change_rate'] < thresholds[2] - buffer_zone),
        (data_chunk['price_change_rate'] >= thresholds[2] - buffer_zone) & 
        (data_chunk['price_change_rate'] < thresholds[3] + buffer_zone),
        (data_chunk['price_change_rate'] >= thresholds[3] + buffer_zone)
    ]
    
    # 添加目标类别赋值
    data_chunk['target_class'] = np.select(conditions, [0, 1, 2, 3, 4])
    
    # 添加动态类别合并（当某类样本过少时）
    class_counts = data_chunk['target_class'].value_counts()
    min_samples = len(data_chunk) * 0.1  # 至少10%样本
    
    if any(class_counts < min_samples):
        logger.warning("检测到类别不平衡，执行动态合并...")
        # 合并小类到相邻类别
        for cls in class_counts[class_counts < min_samples].index:
            if cls == 1:  # 小跌合并到震荡
                data_chunk['target_class'] = np.where(data_chunk['target_class'] == 1, 2, data_chunk['target_class'])
            elif cls == 3:  # 小涨合并到震荡
                data_chunk['target_class'] = np.where(data_chunk['target_class'] == 3, 2, data_chunk['target_class'])
    
    # 移除中间计算列和原始mid_price
    data_chunk = data_chunk.drop(columns=['price_change_rate', 'mid_price'])
    
    # 基础特征列表
    base_features = [
        'relative_spread', 'depth_imbalance', 'bid_ask_slope',
        'order_book_pressure', 'weighted_price_depth', 'liquidity_imbalance',
        'flow_toxicity', 'price_momentum', 'volatility_ratio',
        'ofi', 'vpin', 'pressure_change_rate',
        'orderbook_gradient', 'depth_pressure_ratio'
    ]
    
    # 根据500ms采样频率调整窗口参数
    window_multiplier = 2  # 500ms => 2 points per second
    
    # 修改窗口定义
    windows = [60 * window_multiplier,   # 60秒
               300 * window_multiplier,   # 300秒
               900 * window_multiplier,   # 900秒
               1800 * window_multiplier]  # 1800秒
    
    # 计算滚动窗口特征
    for window in windows:
        logger.info(f"处理 {window//2}秒 窗口...")
        rolled = data_chunk[base_features].rolling(window, min_periods=1)
        
        stats = pd.concat([
            rolled.mean().fillna(0).add_suffix(f'_ma{window}'),
            rolled.std().fillna(0).add_suffix(f'_std{window}'),
            rolled.skew().fillna(0).add_suffix(f'_skew{window}'),
            rolled.kurt().fillna(0).add_suffix(f'_kurt{window}')
        ], axis=1)
        
        data_chunk = pd.concat([data_chunk, stats], axis=1)
    
    # 数据验证
    if data_chunk['target_class'].isna().any():
        logger.warning("目标变量包含NaN值，进行填充")
        data_chunk['target_class'] = data_chunk['target_class'].fillna(2)
    
    # 特征重要性分析
    correlations = data_chunk.corr()['target_class'].abs().sort_values(ascending=False)
    logger.info("\n特征与目标变量的相关性:")
    logger.info(correlations.head(10))
    
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
    """增强版模型训练 - 改为五分类"""
    if data is None or data.empty:
        logger.error("输入数据为空！")
        return [], []
    
    logger.info("开始模型训练...")
    logger.info(f"数据集大小：{data.shape}")
    
    # 确保mid_price不会被用作训练特征
    training_features = [col for col in data.columns 
                        if col not in ['mid_price', 'target_class']]
    
    # 验证所有必需特征都存在
    required_features = [
        'relative_spread', 'depth_imbalance',
        'bid_ask_slope', 'order_book_pressure', 'weighted_price_depth',
        'liquidity_imbalance', 'flow_toxicity', 'price_momentum',
        'volatility_ratio', 'ofi', 'vpin', 'pressure_change_rate',
        'orderbook_gradient', 'depth_pressure_ratio'
    ]
    
    missing_features = [f for f in required_features if f not in training_features]
    if missing_features:
        logger.error(f"缺失必需特征: {missing_features}")
        return [], []
    
    logger.info("开始模型训练...")
    logger.info(f"数据集大小：{data.shape}")
    
    logger.info("创建内存映射文件...")
    try:
        # 重置索引并移除时间列
        data = data.reset_index(drop=True)
        
        # 确保只保留数值型特征
        numeric_data = data[training_features + ['target_class']].select_dtypes(include=[np.number])
        
        X = np.memmap('X.dat', dtype=np.float32, mode='w+', 
                      shape=(len(numeric_data), len(training_features)))
        y = np.memmap('y.dat', dtype=np.float32, mode='w+', shape=(len(numeric_data),))
        
        logger.info("填充训练数据...")
        X[:] = numeric_data[training_features].values
        y[:] = numeric_data['target_class'].values
    except Exception as e:
        logger.error(f"内存映射失败：{str(e)}")
        return [], []
    
    logger.info("配置训练参数...")
    # Use is_unbalance or manually adjust weights in the dataset
    params = {
        'boosting_type': 'goss',
        'objective': 'multiclass',
        'num_class': 5,
        'metric': ['multi_logloss', 'multi_error'],  # 添加多分类错误率
        'num_leaves': 63,  # 减少叶子节点防止过拟合
        'learning_rate': 0.03,  # 降低学习率
        'feature_fraction': 0.7,  # 增加特征采样比例
        'min_data_in_leaf': 200,  # 增加叶子节点最小数据量
        'max_depth': 6,  # 限制树深
        'verbosity': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'max_bin': 255,
        'num_iterations': 5000,
        'early_stopping_round': 50,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'num_threads': 0,
        'gpu_use_dp': True,
        'is_unbalance': True  # Use this for handling class imbalance
    }
    
    logger.info("开始时间序列交叉验证...")
    n_splits = 5
    fold_size = len(data) // n_splits
    
    # 添加分割有效性检查
    if fold_size == 0:
        logger.error("数据量不足以进行交叉验证！")
        return [], []
    
    metrics = []
    models = []
    
    for i in range(n_splits-1):
        logger.info(f"\n开始训练第 {i+1} 折...")
        train_start = i * fold_size
        train_end = (i+2) * fold_size
        val_start = train_end
        val_end = val_start + fold_size
        
        # 确保索引不越界
        train_end = min(train_end, len(data))
        val_start = min(val_start, len(data))
        val_end = min(val_end, len(data))
        
        # 检查训练集和验证集大小
        if train_end - train_start < 100:
            logger.warning(f"第 {i+1} 折训练数据不足，跳过")
            continue
            
        if val_end - val_start < 10:
            logger.warning(f"第 {i+1} 折验证数据不足，跳过")
            continue
        
        logger.info(f"训练集大小：{train_end-train_start}，验证集大小：{val_end-val_start}")
        
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        
        logger.info("创建训练集和验证集...")
        # Manually adjust weights in the dataset
        class_weights = data['target_class'].value_counts(normalize=True).sort_index().to_dict()
        weights = np.array([1/class_weights[c] for c in y_train])
        
        train_set = lgb.Dataset(X_train, y_train, 
                               weight=weights,  # 样本权重
                               free_raw_data=False)
        
        # Initialize val_set here
        val_set = lgb.Dataset(X_val, y_val, reference=train_set, free_raw_data=False)
                               
        # 添加自定义评估指标
        def multi_class_f1(preds, train_data):
            labels = train_data.get_label()
            preds = preds.reshape(5, -1).T.argmax(axis=1)
            f1 = f1_score(labels, preds, average='macro')
            return 'macro_f1', f1, True
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            callbacks=[
                lgb.log_evaluation(100),
                lgb.early_stopping(50),
                lgb.reset_parameter(learning_rate=lambda iter: 0.05 * (0.99 ** iter)),
            ],
            feval=multi_class_f1  # 添加F1评估
        )
        
        # 评估模型时单独计算准确率
        logger.info("评估模型性能...")
        y_pred = model.predict(X_val)
        metrics.append(multi_task_metric(y_pred, y_val))
        models.append(model)
        
        logger.info(f"第 {i+1} 折训练完成")
        logger.info(f"多分类准确率: {metrics[-1][0][1]:.6f}")
        logger.info(f"最佳迭代次数: {model.best_iteration}")
        
        # 特征重要性分析
        importance = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance()
        }).sort_values('importance', ascending=False)
        logger.info("\n前10个重要特征：")
        logger.info(importance.head(10))
        
        # 特征选择（保留前100个重要特征）
        selected_features = importance.head(100)['feature'].values
        X_train = X_train[:, [model.feature_name().index(f) for f in selected_features]]
        X_val = X_val[:, [model.feature_name().index(f) for f in selected_features]]
        
        # 修改特征选择后的处理
        logger.info("执行PCA降维...")
        try:
            pca = IncrementalPCA(n_components=50, batch_size=1000)
            
            # 确保数据没有NaN
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
            
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_val = pca.transform(X_val)
            
            # 再次检查数据有效性
            if np.isnan(X_train).any() or np.isinf(X_train).any():
                logger.error("PCA处理后训练数据仍存在无效值！")
                return [], []
            
        except Exception as e:
            logger.error(f"PCA处理失败: {str(e)}")
            return [], []
        
        del train_set, val_set
        gc.collect()
        logger.info("内存已清理")
    
    logger.info("\n训练完成！")
    logger.info(f"多分类准确率: {np.mean([m[0][1] for m in metrics]):.6f}")
    logger.info(f"最佳多分类准确率: {min([m[0][1] for m in metrics]):.6f}")
    logger.info(f"多分类准确率标准差: {np.std([m[0][1] for m in metrics]):.6f}")
    
    # 添加数据分布检查
    logger.info("目标值分布统计:")
    logger.info(data['target_class'].describe())
    
    # 添加特征相关性检查
    corr_with_target = data.corr()['target_class'].abs().sort_values(ascending=False)
    logger.info("特征相关性排名:\n" + corr_with_target.head(10).to_string())
    
    # 在train_enhanced_model函数中添加调试日志
    logger.info(f"最终训练特征列表: {data.columns.tolist()}")
    logger.info(f"mid_price存在状态: {'mid_price' in data.columns}")
    
    return models, metrics

def multi_task_metric(y_pred, y_true):
    """自定义评估指标 - 适配多分类"""
    # 转换预测概率为类别
    y_pred_class = np.argmax(y_pred.reshape(-1, 5), axis=1)
    
    # 计算准确率
    accuracy = np.mean(y_pred_class == y_true)
    
    # 计算混淆矩阵
    confusion = np.zeros((5, 5))
    for i in range(len(y_true)):
        confusion[int(y_true[i])][y_pred_class[i]] += 1
    
    # 计算每个类别的准确率
    class_acc = confusion.diagonal() / confusion.sum(axis=1)
    
    return [('accuracy', accuracy, True), 
            ('class0_acc', class_acc[0], True),
            ('class1_acc', class_acc[1], True),
            ('class2_acc', class_acc[2], True),
            ('class3_acc', class_acc[3], True),
            ('class4_acc', class_acc[4], True)]

def analyze_data_quality(data):
    """数据质量分析报告"""
    logger.info("\n=== 数据质量分析报告 ===")
    
    # 缺失值分析
    missing = data.isna().sum()
    logger.info(f"\n缺失值统计:\n{missing[missing > 0]}")
    
    # 无限值分析
    inf_counts = data.applymap(lambda x: np.isinf(x)).sum()
    logger.info(f"\n无限值统计:\n{inf_counts[inf_counts > 0]}")
    
    # 数值分布分析
    logger.info("\n数值分布统计:")
    logger.info(data.describe().T)
    
    # 零值分析
    zero_counts = (data == 0).sum()
    logger.info(f"\n零值统计:\n{zero_counts[zero_counts > 0]}")
    
    # 相关性分析
    logger.info("\n特征相关性矩阵:")
    logger.info(data.corr().style.background_gradient(cmap='coolwarm').to_string())
    
    logger.info("=== 分析完成 ===")

def main():
    logger.info("=== 开始增强版模型训练 ===")
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
    
    # 添加数据质量分析
    analyze_data_quality(processed_data)
    
    log_memory_usage()
    
    logger.info("开始特征工程...")
    feature_data = create_features(processed_data)
    if feature_data is None:
        logger.error("特征工程失败，终止程序！")
        return
    
    # 添加最终数据检查
    # logger.info("执行最终数据验证...")
    # if feature_data.isna().sum().sum() > 0:
    #     logger.error(f"最终数据包含 {feature_data.isna().sum().sum()} 个缺失值！")
    #     return
    
    # inf_check = feature_data.applymap(lambda x: np.isinf(x)).sum().sum()
    # if inf_check > 0:
    #     logger.error(f"最终数据包含 {inf_check} 个无限值！")
    #     return
    
    log_memory_usage()
    
    config = {
        'window_size': 3600 * 6,
        'prediction_horizon': 60,
        'max_features': 500
    }
    
    # 添加训练前检查
    if len(feature_data) < 1000:
        logger.error(f"数据量不足，当前样本数：{len(feature_data)}")
        return
    
    logger.info(f"训练配置：{config}")
    
    models, metrics = train_enhanced_model(feature_data, config)
    
    logger.info("保存模型...")
    for idx, model in enumerate(models):
        model.save_model(f'enhanced_model_v{idx}.bin')
        logger.info(f"模型 {idx} 已保存")
    
    logger.info("=== 训练流程完成 ===")
    log_memory_usage()

if __name__ == "__main__":
    main()
