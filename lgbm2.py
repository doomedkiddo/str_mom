import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, f1_score
from loguru import logger
import psutil
import gc
from datetime import timedelta
from scipy.fft import fft
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import IncrementalPCA
import pyarrow.feather as ft

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")

def stream_feather_chunks(filename, chunksize=100000):
    """流式读取Feather文件（修正版）"""
    table = ft.read_table(filename)
    total_rows = table.num_rows
    
    for start in range(0, total_rows, chunksize):
        end = min(start + chunksize, total_rows)
        # 使用pyarrow的slice方法分块
        yield table.slice(start, end - start).to_pandas()

def process_chunk(chunk):
    """增强的数据预处理"""
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
    
    # 修改特征保留列表，添加自动生成的特征
    features_to_keep = [
        'mid_price',  # 保留mid_price用于后续计算
        'relative_spread', 'depth_imbalance',
        'bid_ask_slope', 'order_book_pressure',
        'flow_toxicity', 'price_momentum',
        'volatility_ratio', 'ofi', 'vpin', 'pressure_change_rate',
        'orderbook_gradient', 'depth_pressure_ratio',
        'weighted_price_depth',  # 添加自动生成的特征
        'liquidity_imbalance'     # 添加自动生成的特征
    ]
    
    # 修改特征生成逻辑，增加更多错误处理和验证
    logger.info("开始生成特征...")
    
    # 生成 weighted_price_depth
    try:
        if 'weighted_price_depth' not in chunk.columns:
            logger.warning("自动生成weighted_price_depth特征")
            bid_depth = chunk.get('bid_depth', pd.Series(0, index=chunk.index))
            ask_depth = chunk.get('ask_depth', pd.Series(0, index=chunk.index))
            chunk['weighted_price_depth'] = (bid_depth + ask_depth) * 0.5
            logger.info(f"weighted_price_depth统计: \n{chunk['weighted_price_depth'].describe()}")
    except Exception as e:
        logger.error(f"生成weighted_price_depth时出错: {str(e)}")
        # 使用默认值
        chunk['weighted_price_depth'] = pd.Series(1.0, index=chunk.index)
    
    # 生成 liquidity_imbalance
    try:
        if 'liquidity_imbalance' not in chunk.columns:
            logger.warning("自动生成liquidity_imbalance特征")
            bid_volume = chunk.get('bid_volume', pd.Series(0, index=chunk.index))
            ask_volume = chunk.get('ask_volume', pd.Series(0, index=chunk.index))
            chunk['liquidity_imbalance'] = bid_volume - ask_volume
            logger.info(f"liquidity_imbalance统计: \n{chunk['liquidity_imbalance'].describe()}")
    except Exception as e:
        logger.error(f"生成liquidity_imbalance时出错: {str(e)}")
        # 使用默认值
        chunk['liquidity_imbalance'] = pd.Series(0.0, index=chunk.index)
    
    # 验证特征是否成功生成
    logger.info("验证特征生成结果...")
    for feature in ['weighted_price_depth', 'liquidity_imbalance']:
        if feature not in chunk.columns:
            logger.error(f"特征 {feature} 生成失败")
            raise ValueError(f"无法生成特征: {feature}")
        if chunk[feature].isna().any():
            logger.warning(f"特征 {feature} 包含缺失值，进行填充")
            chunk[feature] = chunk[feature].fillna(0)
    
    # 确保所有保留的特征都存在
    missing_features = [f for f in features_to_keep if f not in chunk.columns]
    if missing_features:
        logger.error(f"缺失必要特征: {missing_features}")
        logger.info(f"当前可用列: {chunk.columns.tolist()}")
        raise ValueError(f"数据中缺失必要特征: {missing_features}")
    
    # 在返回数据之前进行最后的验证
    final_features = chunk[features_to_keep].columns
    logger.info(f"最终特征列表: {final_features.tolist()}")
    
    # 确保返回的数据包含所有必要的特征
    result = chunk[features_to_keep].copy()
    logger.info(f"处理后的数据形状: {result.shape}")
    logger.info(f"处理后的特征列表: {result.columns.tolist()}")
    
    return result

def create_features(data_chunk, window_size=1800):
    """增强版特征工程"""
    logger.info("开始特征工程...")
    data_chunk = data_chunk.copy()
    
    # 保存一个mid_price的副本用于后续特征计算
    mid_price_copy = data_chunk['mid_price'].copy()
    
    # 确保索引是时间类型
    if not isinstance(data_chunk.index, pd.DatetimeIndex):
        logger.warning("索引不是时间类型，尝试转换...")
        try:
            data_chunk.index = pd.to_datetime(data_chunk.index)
        except Exception as e:
            logger.error(f"时间索引转换失败: {str(e)}，创建新时间索引")
            data_chunk.index = pd.date_range(
                start='2024-01-01', 
                periods=len(data_chunk), 
                freq='500ms'
            )
    
    # 计算目标变量 - 改为三分类
    prediction_points = 120  # 60秒 = 120个500ms间隔
    if 'mid_price' not in data_chunk.columns:
        logger.error("缺少mid_price列，无法计算目标变量！")
        return None
    
    # 计算价格变化率
    data_chunk['price_change_rate'] = (data_chunk['mid_price'].shift(-prediction_points) - 
                                      data_chunk['mid_price']) / data_chunk['mid_price']
    
    # 固定阈值定义
    LOWER_THRESHOLD = -0.0005  # -0.05%
    UPPER_THRESHOLD = 0.0005   # +0.05%

    conditions = [
        (data_chunk['price_change_rate'] < LOWER_THRESHOLD),
        (data_chunk['price_change_rate'].between(LOWER_THRESHOLD, UPPER_THRESHOLD)),
        (data_chunk['price_change_rate'] > UPPER_THRESHOLD)
    ]
    values = [0, 1, 2]
    
    # 目标类别赋值
    data_chunk['target_class'] = np.select(conditions, values, default=1)
    
    # 添加滞后特征防止未来信息泄露（在删除列之前）
    data_chunk['price_change_lag1'] = data_chunk['price_change_rate'].shift(1)
    data_chunk['order_flow_lag2'] = data_chunk['flow_toxicity'].shift(2)
    
    # 移除中间计算列和原始mid_price（现在保留price_change_rate用于创建滞后特征）
    data_chunk = data_chunk.drop(columns=['price_change_rate', 'mid_price'])
    
    # 目标类别赋值后添加分布统计
    class_dist = pd.value_counts(data_chunk['target_class'])
    logger.info("\n=== 镜像后目标分布 ===")
    logger.info(f"下跌 (0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(data_chunk):.2%})")
    logger.info(f"平稳 (1): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(data_chunk):.2%})")
    logger.info(f"上涨 (2): {class_dist.get(2, 0)} ({class_dist.get(2, 0)/len(data_chunk):.2%})")
    
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
        data_chunk['target_class'] = data_chunk['target_class'].fillna(1)
    
    # 特征重要性分析
    correlations = data_chunk.corr()['target_class'].abs().sort_values(ascending=False)
    logger.info("\n特征与目标变量的相关性:")
    logger.info(correlations.head(10))
    
    # 在目标变量创建后添加镜像样本生成
    logger.info("生成镜像样本...")
    original_samples = data_chunk.copy()
    
    # 筛选需要镜像的样本（仅下跌和上涨类别）
    mirror_mask = (data_chunk['target_class'] == 0) | (data_chunk['target_class'] == 2)
    mirror_samples = data_chunk[mirror_mask].copy()
    
    # 定义需要反转的特征（根据特征含义调整）
    inverse_features = {
        'depth_imbalance': True,
        'bid_ask_slope': True,
        'order_book_pressure': True,
        'liquidity_imbalance': True,
        'pressure_change_rate': True,
        'market_state': False  # 不反转市场状态
    }
    
    for col, should_invert in inverse_features.items():
        if should_invert and col in mirror_samples.columns:
            mirror_samples[col] = -mirror_samples[col]
    
    # 添加随机噪声增强
    noise_scale = 0.01 * mirror_samples.std()
    for col in mirror_samples.columns:
        if col not in ['target_class', 'market_state']:
            mirror_samples[col] += np.random.normal(0, noise_scale[col], size=len(mirror_samples))
    
    # 合并原始样本和镜像样本
    balanced_chunk = pd.concat([original_samples, mirror_samples], axis=0)
    
    # 打乱数据顺序但保持时间连续性
    balanced_chunk = balanced_chunk.sample(frac=1.0, random_state=42).sort_index()
    
    # 添加时间连续性检查（增强容错性）
    try:
        time_diff = balanced_chunk.index.to_series().diff().dt.total_seconds()
        if np.isclose(time_diff[1:], 0.5, atol=0.1).any():  # 允许±100ms的误差
            logger.warning("数据时间间隔异常，存在缺失或重复")
    except AttributeError as e:
        logger.error(f"时间索引异常: {str(e)}")
        logger.info(f"当前索引类型: {type(balanced_chunk.index)}")
        logger.info("尝试重建时间索引...")
        balanced_chunk.index = pd.date_range(
            start=balanced_chunk.index[0],
            periods=len(balanced_chunk),
            freq='500ms'
        )
    
    logger.info(f"镜像后数据量: {len(balanced_chunk)} (原始: {len(data_chunk)})")
    
    # 修改这部分代码，使用保存的mid_price副本
    # 添加新的价格动量特征
    data_chunk['price_momentum_1min'] = data_chunk['price_momentum'].rolling(120).mean()  # 1分钟
    data_chunk['price_momentum_5min'] = data_chunk['price_momentum'].rolling(600).mean()  # 5分钟
    
    # 添加波动率特征 - 使用保存的mid_price副本
    data_chunk['volatility_1min'] = mid_price_copy.rolling(120).std()
    data_chunk['volatility_5min'] = mid_price_copy.rolling(600).std()
    
    # 添加交易量压力特征
    data_chunk['volume_pressure'] = (data_chunk['ofi'].rolling(120).sum() / 
                                   data_chunk['vpin'].rolling(120).mean())
    
    # 添加趋势特征
    data_chunk['trend_strength'] = (data_chunk['price_momentum_5min'] / 
                                  data_chunk['volatility_5min'])
    
    # 添加市场微观结构特征
    data_chunk['spread_volatility'] = data_chunk['relative_spread'].rolling(120).std()
    data_chunk['depth_volatility'] = data_chunk['depth_imbalance'].rolling(120).std()
    
    # 添加交叉特征
    data_chunk['pressure_momentum'] = (data_chunk['order_book_pressure'] * 
                                     data_chunk['price_momentum'])
    
    # 添加新的市场微观结构特征
    data_chunk['depth_velocity'] = data_chunk['depth_imbalance'].diff() / data_chunk['depth_imbalance'].abs().rolling(60).mean()
    data_chunk['pressure_gradient'] = data_chunk['order_book_pressure'].rolling(120).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    data_chunk['spread_acceleration'] = data_chunk['relative_spread'].diff().diff()
    
    # 添加买卖压力特征
    data_chunk['bid_ask_pressure_ratio'] = (data_chunk['order_book_pressure'].rolling(60).max() / 
                                          data_chunk['order_book_pressure'].rolling(60).min())
    
    # 添加流动性特征
    data_chunk['liquidity_velocity'] = (data_chunk['liquidity_imbalance'].diff() / 
                                      data_chunk['liquidity_imbalance'].abs().rolling(60).mean())
    
    # 添加高频交易特征
    data_chunk['hft_pressure'] = (data_chunk['flow_toxicity'].rolling(30).std() * 
                                data_chunk['ofi'].rolling(30).mean())
    
    # 最后删除mid_price副本
    del mid_price_copy
    
    return balanced_chunk

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

def train_enhanced_model(data, model=None):
    """改进的模型训练"""
    params = {
        'boosting_type': 'goss',
        'objective': 'multiclass',
        'num_class': 3,
        'metric': ['multi_logloss', 'multi_error'],
        'num_leaves': 27,  # Keep your original setting
        'learning_rate': 0.003,
        'feature_fraction': 0.6,
        'min_data_in_leaf': 100,
        'max_depth': -1,  # Let LightGBM determine max_depth based on num_leaves
        'lambda_l1': 0.2,
        'lambda_l2': 0.2,
        'min_gain_to_split': 0.1,
        'path_smooth': 0.5,
        'max_bin': 127,
        'device': 'gpu'
    }
    
    # 添加动态学习率衰减
    callbacks = [
        lgb.reset_parameter(
            learning_rate=lambda epoch: params['learning_rate'] * (0.98 ** epoch)
        ),
        lgb.early_stopping(50, verbose=10),
        lgb.log_evaluation(50)
    ]
    
    # 创建存储特征重要性的列表
    feature_importances = []
    
    # 定义特征重要性追踪回调
    def save_feature_importance(env):
        importance = pd.DataFrame({
            'feature': env.model.feature_name(),
            'importance': env.model.feature_importance()
        })
        feature_importances.append(importance)
    
    callbacks.append(save_feature_importance)
    
    # 特征选择
    feature_importance = pd.DataFrame()
    if model is not None:
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance()
        })
        # 只保留重要性前80%的特征
        threshold = feature_importance['importance'].quantile(0.2)
        selected_features = feature_importance[
            feature_importance['importance'] > threshold
        ]['feature'].tolist()
        X = data[selected_features].values
    
    # 准备特征和目标变量
    excluded_columns = ['target_class', 'mid_price', 'price_change_rate']
    feature_cols = [col for col in data.columns if col not in excluded_columns]
    
    X = data[feature_cols].values
    y = data['target_class'].values
    
    if model is None:
        # 初始训练时分割训练集和验证集
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 计算样本权重
        weights_train = compute_sample_weights(y_train)
        
        train_set = lgb.Dataset(
            X_train, y_train,
            weight=weights_train,
            feature_name=feature_cols,
            free_raw_data=False
        )
        valid_set = lgb.Dataset(
            X_val, y_val,
            feature_name=feature_cols,
            free_raw_data=False
        )
        
        # 初始训练
        model = lgb.train(
            params,
            train_set,
            num_boost_round=100,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
    else:
        # 计算样本权重
        weights = compute_sample_weights(y)
        
        # 增量训练使用全部数据
        train_set = lgb.Dataset(
            X, y,
            weight=weights,
            feature_name=feature_cols,
            free_raw_data=False
        )
        
        # 继续训练现有模型
        model = lgb.train(
            params,
            train_set,
            init_model=model,
            num_boost_round=50,  # 每次增量训练的轮次
            valid_sets=[train_set],
            valid_names=['train'],
            callbacks=callbacks
        )
    
    return model

def multi_task_metric(y_pred, y_true):
    """自定义评估指标 - 适配三分类"""
    y_pred_class = np.argmax(y_pred.reshape(-1, 3), axis=1)
    
    # 计算准确率
    accuracy = np.mean(y_pred_class == y_true)
    
    # 计算混淆矩阵
    confusion = np.zeros((3, 3))
    for i in range(len(y_true)):
        confusion[int(y_true[i])][y_pred_class[i]] += 1
    
    # 计算每个类别的准确率和其他指标
    class_acc = confusion.diagonal() / confusion.sum(axis=1)
    
    # 添加更详细的评估指标
    f1_scores = f1_score(y_true, y_pred_class, average=None)
    
    return [('accuracy', accuracy, True), 
            ('class0_acc', class_acc[0], True),
            ('class1_acc', class_acc[1], True),
            ('class2_acc', class_acc[2], True),
            ('class0_f1', f1_scores[0], True),
            ('class1_f1', f1_scores[1], True),
            ('class2_f1', f1_scores[2], True)]

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
    
    # 相关性分析
    logger.info("\n特征相关性矩阵:")
    logger.info(data.corr().style.background_gradient(cmap='coolwarm').to_string())
    
    logger.info("=== 分析完成 ===")

def compute_sample_weights(y):
    """改进的权重计算方法"""
    class_counts = np.bincount(y.astype(int))
    median_freq = np.median(class_counts)
    class_weights = {i: median_freq / count for i, count in enumerate(class_counts)}
    return np.array([class_weights[int(yi)] for yi in y])

def main():
    logger.info("=== 开始增量训练 ===")
    log_memory_usage()
    
    # 添加config定义
    config = {
        'num_classes': 3,
        'max_features': 200,
        'training_params': {
            'early_stopping_rounds': 30,
            'num_boost_round': 100
        }
    }
    
    logger.info("流式加载Feather数据...")
    model = None
    feature_columns = None
    
    try:
        # 修改chunksize参数为500000（原100000的5倍）
        chunk_generator = stream_feather_chunks('merged_data.feather', chunksize=500000)
        for chunk_idx, raw_chunk in enumerate(chunk_generator):
            logger.info(f"\n处理数据块 {chunk_idx+1}...")
            
            # 数据预处理
            processed_chunk = process_chunk(raw_chunk)
            
            # 特征工程
            feature_chunk = create_features(processed_chunk)
            if feature_chunk is None:
                continue
                
            # 确保特征列一致性
            if feature_columns is None:
                feature_columns = feature_chunk.columns.tolist()
            else:
                # 对齐特征列（添加缺失列，删除多余列）
                missing_cols = set(feature_columns) - set(feature_chunk.columns)
                extra_cols = set(feature_chunk.columns) - set(feature_columns)
                
                for col in missing_cols:
                    feature_chunk[col] = 0  # 用0填充缺失特征
                feature_chunk = feature_chunk[feature_columns]
                
                if missing_cols or extra_cols:
                    logger.warning(f"特征列不一致 - 缺失: {missing_cols}, 多余: {extra_cols}")
            
            # 增量训练
            model = train_enhanced_model(feature_chunk)
            
            # 内存清理
            del raw_chunk, processed_chunk, feature_chunk
            gc.collect()
            log_memory_usage()
            
        # 最终保存模型
        if model is not None:
            model.save_model('incremental_model.bin')
            logger.info("最终模型已保存")
            
    except Exception as e:
        logger.error(f"训练过程异常终止: {str(e)}")
        if model is not None:
            model.save_model('interrupted_model.bin')
            logger.info("已保存中断时的模型")
    
    logger.info("=== 训练流程完成 ===")
    log_memory_usage()

if __name__ == "__main__":
    main()
