import pandas as pd
import numpy as np
from loguru import logger
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import psutil
import gc
from datetime import datetime
import os
from typing import Dict, List, Generator
from scipy.stats import zscore
import pickle
import tempfile

warnings.filterwarnings('ignore')
logger.add("hft_pipeline.log", rotation="500 MB")

def _process_chunk_worker(chunk_data):
    """进程worker函数，必须是模块级别的函数"""
    chunk, chunk_idx, total_chunks = chunk_data
    
    # 重新创建logger，因为在子进程中需要重新初始化
    from loguru import logger
    logger.add(f"hft_pipeline_worker_{chunk_idx}.log", rotation="100 MB")
    
    try:
        logger.info(f"开始处理数据块 {chunk_idx}/{total_chunks}")
        
        # 数据验证
        if chunk.empty:
            logger.error(f"数据块 {chunk_idx} 为空")
            return None
            
        if chunk.isna().any().any():
            logger.warning(f"数据块 {chunk_idx} 包含NaN值，进行填充")
            chunk = chunk.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 创建临时HFTDataEngine实例来处理数据
        engine = HFTDataEngine(
            start_date=chunk.index.min(),
            end_date=chunk.index.max()
        )
        
        # 特征计算
        processed = engine._process_chunk(chunk)
        
        # 验证处理结果
        if processed is None or processed.empty:
            logger.error(f"数据块 {chunk_idx} 特征计算失败")
            return None
            
        if processed.isna().any().any():
            logger.error(f"数据块 {chunk_idx} 处理后数据包含NaN值")
            return None
            
        # 重采样
        resampled = engine._resample_chunk(processed)
        
        # 释放内存
        del chunk, processed, engine
        gc.collect()
        
        return resampled
        
    except Exception as e:
        logger.error(f"处理数据块 {chunk_idx} 时发生错误: {str(e)}")
        return None

class HFTDataEngine:
    def __init__(self, start_date, end_date, symbol='DOGE-USDT-PERP', chunk_size=100000):
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        self.chunk_size = chunk_size
        self.dtype_map = {
            'origin_time': 'uint64',
            'bid_0_price': 'float64',
            'bid_0_size': 'float64',
            'ask_0_price': 'float64',
            'ask_0_size': 'float64',
            'bid_1_price': 'float64',
            'bid_1_size': 'float64',
            'ask_1_price': 'float64',
            'ask_1_size': 'float64',
            'bid_2_price': 'float64',
            'bid_2_size': 'float64',
            'ask_2_price': 'float64',
            'ask_2_size': 'float64',
            'bid_3_price': 'float64',
            'bid_3_size': 'float64',
            'ask_3_price': 'float64',
            'ask_3_size': 'float64',
            'bid_4_price': 'float64',
            'bid_4_size': 'float64',
            'ask_4_price': 'float64',
            'ask_4_size': 'float64'
        }

    def _get_file_paths(self) -> List[str]:
        """获取数据文件路径"""
        date_range = pd.date_range(self.start_date, self.end_date)
        paths = [
            f"data/crypto-lake/book/BINANCE_FUTURES/{self.symbol}/"
            f"{self.symbol}_{date.strftime('%Y%m%d')}.feather"
            for date in date_range
        ]
        return [p for p in paths if os.path.exists(p)]

    def _stream_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """流式加载数据块"""
        for path in self._get_file_paths():
            df = pd.read_feather(path, columns=self.dtype_map.keys())
            
            # 特殊处理 origin_time 列的转换
            if 'origin_time' in df.columns:
                df['origin_time'] = df['origin_time'].astype('int64').astype('uint64')
            
            # 转换其他列的数据类型
            dtype_map_without_time = {k: v for k, v in self.dtype_map.items() if k != 'origin_time'}
            df = df.astype(dtype_map_without_time)
            
            for i in range(0, len(df), self.chunk_size):
                yield df.iloc[i:i + self.chunk_size].copy()
            del df
            gc.collect()

    def _calculate_basic_features(self, chunk: pd.DataFrame) -> Dict[str, np.ndarray]:
        """计算基础特征"""
        # 数据验证
        logger.info("验证输入数据...")
        logger.info(f"数据统计:\n{chunk.describe()}")
        
        # 检查是否有无效值
        if chunk.isna().any().any():
            logger.error("输入数据包含NaN值")
            logger.info(f"NaN统计:\n{chunk.isna().sum()}")
        
        if (chunk == 0).all().any():
            logger.error("存在全为0的列")
            logger.info(f"零值统计:\n{(chunk == 0).sum()}")
        
        # 计算中间价格
        mid_price = (chunk['bid_0_price'] + chunk['ask_0_price']) / 2
        
        # 添加epsilon避免除零
        epsilon = 1e-10
        
        # 计算相对买卖价差
        relative_spread = (chunk['ask_0_price'] - chunk['bid_0_price']) / (mid_price + epsilon)
        
        # 计算深度不平衡
        bid_depth = (chunk['bid_0_size'] + chunk['bid_1_size'] + chunk['bid_2_size'])
        ask_depth = (chunk['ask_0_size'] + chunk['ask_1_size'] + chunk['ask_2_size'])
        depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + epsilon)
        
        # 验证计算结果
        features = {
            'mid_price': mid_price,
            'relative_spread': relative_spread,
            'depth_imbalance': depth_imbalance
        }
        
        for name, values in features.items():
            if np.isnan(values).any():
                logger.error(f"{name} 包含NaN值")
                logger.info(f"{name} 统计:\n{pd.Series(values).describe()}")
        
        return features

    def _calculate_advanced_features(self, chunk: pd.DataFrame, basic_features: Dict[str, np.ndarray]) -> pd.DataFrame:
        """计算高级特征"""
        epsilon = 1e-10
        
        # 使用basic_features中的mid_price进行计算
        mid_price = basic_features['mid_price']
        
        # 价格动量（使用临时变量，不存储到DataFrame）
        rolling_mean = pd.Series(mid_price).rolling(20, min_periods=1).mean().fillna(method='bfill')
        rolling_std = pd.Series(mid_price).rolling(20, min_periods=1).std().fillna(method='bfill')
        chunk['price_momentum'] = ((mid_price - rolling_mean) / (rolling_std + epsilon))
        
        # 波动率比率（使用临时变量）
        realized_vol = pd.Series(mid_price).pct_change().fillna(0).rolling(20, min_periods=1).std().fillna(0)
        implied_vol = pd.Series(basic_features['relative_spread']).rolling(20, min_periods=1).mean().fillna(0)
        chunk['volatility_ratio'] = realized_vol / (implied_vol + epsilon)
        
        # 改进买卖盘斜率计算
        def calculate_slope_v2(prices, sizes, prev_prices=None, prev_sizes=None):
            """改进的价格斜率计算，考虑时序特性和高阶变化"""
            if len(prices) < 2:
                return 0.0
            
            # 计算空间维度的价格变化
            spatial_diffs = np.diff(prices)
            spatial_weights = sizes[1:] / (sizes[1:].sum() + epsilon)
            spatial_slope = np.sum(spatial_diffs * spatial_weights)
            
            # 如果有前一时刻的数据，计算时间维度的变化
            if prev_prices is not None and prev_sizes is not None:
                # 计算时间维度的价格变化
                temporal_diffs = prices - prev_prices
                temporal_weights = (sizes + prev_sizes) / ((sizes + prev_sizes).sum() + epsilon)
                temporal_slope = np.sum(temporal_diffs * temporal_weights)
                
                # 计算二阶导数（加速度）
                spatial_accel = np.diff(spatial_diffs) if len(spatial_diffs) > 1 else np.array([0.0])
                temporal_accel = np.diff(np.array([prev_prices, prices]), axis=0)[0]
                
                # 组合多个特征
                combined_slope = (
                    0.4 * spatial_slope +  # 空间维度的一阶变化
                    0.4 * temporal_slope + # 时间维度的一阶变化
                    0.1 * np.mean(spatial_accel) +  # 空间维度的二阶变化
                    0.1 * np.mean(temporal_accel)   # 时间维度的二阶变化
                )
                
                return combined_slope
            
            return spatial_slope
        
        # 初始化前一时刻的价格和订单量
        prev_bid_prices = None
        prev_ask_prices = None
        prev_bid_sizes = None
        prev_ask_sizes = None
        
        # 计算买卖盘斜率
        bid_prices = chunk[['bid_0_price', 'bid_1_price', 'bid_2_price', 'bid_3_price', 'bid_4_price']].values
        ask_prices = chunk[['ask_0_price', 'ask_1_price', 'ask_2_price', 'ask_3_price', 'ask_4_price']].values
        bid_sizes = chunk[['bid_0_size', 'bid_1_size', 'bid_2_size', 'bid_3_size', 'bid_4_size']].values
        ask_sizes = chunk[['ask_0_size', 'ask_1_size', 'ask_2_size', 'ask_3_size', 'ask_4_size']].values
        
        bid_slopes = []
        ask_slopes = []
        
        for i in range(len(chunk)):
            # 计算当前时刻的斜率
            bid_slope = calculate_slope_v2(
                bid_prices[i], bid_sizes[i],
                prev_bid_prices[i-1] if i > 0 else None,
                prev_bid_sizes[i-1] if i > 0 else None
            )
            ask_slope = calculate_slope_v2(
                ask_prices[i], ask_sizes[i],
                prev_ask_prices[i-1] if i > 0 else None,
                prev_ask_sizes[i-1] if i > 0 else None
            )
            
            bid_slopes.append(bid_slope)
            ask_slopes.append(ask_slope)
            
            # 更新前一时刻的数据
            prev_bid_prices = bid_prices
            prev_ask_prices = ask_prices
            prev_bid_sizes = bid_sizes
            prev_ask_sizes = ask_sizes
        
        # 计算最终的买卖盘斜率
        slopes = np.array([
            (abs(b) + abs(a)) * np.sign(a - b) * (1 + np.log1p(abs(a + b) + epsilon))
            for b, a in zip(bid_slopes, ask_slopes)
        ])
        
        # 添加动态增强
        slopes_series = pd.Series(slopes)
        
        # 计算移动平均和波动率
        ma = slopes_series.rolling(window=5, min_periods=1).mean()
        std = slopes_series.rolling(window=5, min_periods=1).std().fillna(0)
        
        # 计算动量和反转信号
        momentum = slopes_series.diff(1).fillna(0)
        reversal = (slopes_series - ma) / (std + epsilon)
        
        # 组合多个信号
        chunk['bid_ask_slope'] = slopes + \
                                0.2 * momentum + \
                                0.1 * reversal + \
                                0.1 * (slopes_series.diff(2)) # 加入高阶差分
        
        # 添加波动率调整
        volatility_adj = std / (std.rolling(window=10, min_periods=1).mean() + epsilon)
        chunk['bid_ask_slope'] *= (1 + 0.1 * volatility_adj)
        
        # 标准化处理
        slope_mean = chunk['bid_ask_slope'].mean()
        slope_std = chunk['bid_ask_slope'].std()
        if slope_std > epsilon:
            chunk['bid_ask_slope'] = (chunk['bid_ask_slope'] - slope_mean) / slope_std
        else:
            # 如果标准差仍然太小，添加随机波动
            chunk['bid_ask_slope'] = chunk['bid_ask_slope'] + \
                                    np.random.normal(0, 0.1, size=len(chunk))
        
        # 改进斜率计算，考虑买卖盘的不对称性
        chunk['bid_ask_slope'] = np.array([
            (abs(b) + abs(a)) * np.sign(a - b) 
            for b, a in zip(bid_slopes, ask_slopes)
        ])
        
        # 改进订单簿梯度特征计算
        def calculate_orderbook_gradient(prices, sizes):
            """改进的订单簿价格梯度计算"""
            if len(prices) < 2:
                return 0.0
            
            # 计算加权价格
            weights = sizes / (sizes.sum() + epsilon)
            weighted_prices = prices * weights
            
            # 计算梯度
            gradient = np.gradient(weighted_prices)
            
            # 返回加权平均梯度
            return np.average(gradient, weights=weights)
        
        # 计算新的梯度特征
        bid_gradients = []
        ask_gradients = []
        
        for i in range(len(chunk)):
            # 买盘梯度
            bg = calculate_orderbook_gradient(bid_prices[i], bid_sizes[i])
            # 卖盘梯度
            ag = calculate_orderbook_gradient(ask_prices[i], ask_sizes[i])
            
            bid_gradients.append(bg)
            ask_gradients.append(ag)
        
        # 计算梯度差异，考虑买卖盘的相对强度
        chunk['orderbook_gradient'] = np.array([
            (ag - bg) * (1 + abs(ag + bg)) 
            for bg, ag in zip(bid_gradients, ask_gradients)
        ])
        
        # 订单簿压力
        bid_pressure = chunk['bid_0_size'] * chunk['bid_0_price'] + chunk['bid_1_size'] * chunk['bid_1_price']
        ask_pressure = chunk['ask_0_size'] * chunk['ask_0_price'] + chunk['ask_1_size'] * chunk['ask_1_price']
        chunk['order_book_pressure'] = np.log((bid_pressure + epsilon) / (ask_pressure + epsilon))
        
        # 加权价格深度
        weighted_bid = (chunk['bid_0_size'] * chunk['bid_0_price'] + 
                       chunk['bid_1_size'] * chunk['bid_1_price'] + 
                       chunk['bid_2_size'] * chunk['bid_2_price'])
        weighted_ask = (chunk['ask_0_size'] * chunk['ask_0_price'] + 
                       chunk['ask_1_size'] * chunk['ask_1_price'] + 
                       chunk['ask_2_size'] * chunk['ask_2_price'])
        chunk['weighted_price_depth'] = np.log((weighted_bid + epsilon) / (weighted_ask + epsilon))
        
        # 流动性不平衡
        avg_bid_size = (chunk['bid_0_size'] + chunk['bid_1_size'] + chunk['bid_2_size']) / 3
        avg_ask_size = (chunk['ask_0_size'] + chunk['ask_1_size'] + chunk['ask_2_size']) / 3
        chunk['liquidity_imbalance'] = (avg_bid_size - avg_ask_size) / (avg_bid_size + avg_ask_size + epsilon)
        
        # 订单流毒性
        mid_ret = pd.Series(mid_price).pct_change().fillna(0)
        vol = mid_ret.rolling(20, min_periods=1).std().fillna(0)
        chunk['flow_toxicity'] = (basic_features['depth_imbalance'] * vol)
        
        # 订单流不平衡(OFI)
        bid_flow = chunk['bid_0_size'].diff().fillna(0).clip(lower=0)
        ask_flow = chunk['ask_0_size'].diff().fillna(0).clip(upper=0).abs()
        
        # 添加OFI有效性检查
        denominator_ofi = bid_flow + ask_flow + epsilon
        valid_ofi = denominator_ofi > 1e-6
        chunk['ofi'] = np.where(valid_ofi, (bid_flow - ask_flow) / denominator_ofi, 0.0)
        
        # 新特征1：订单簿梯度特征
        def calculate_depth_pressure(prices, sizes):
            """计算深度加权压力"""
            weighted_prices = prices * sizes
            return np.log(weighted_prices.sum() + 1e-10)
        
        # 新特征2：深度加权压力
        bid_pressures = []
        ask_pressures = []
        
        for i in range(len(chunk)):
            # 买盘压力
            bp = calculate_depth_pressure(bid_prices[i], bid_sizes[i])
            # 卖盘压力
            ap = calculate_depth_pressure(ask_prices[i], ask_sizes[i])
            
            bid_pressures.append(bp)
            ask_pressures.append(ap)
        
        # 新特征1：深度压力比率
        chunk['depth_pressure_ratio'] = np.log((np.array(bid_pressures) + 1e-10) / (np.array(ask_pressures) + 1e-10))
        
        # 将新特征重命名为旧特征名称（如果需要保持接口一致）
        chunk['order_book_curvature'] = chunk['orderbook_gradient']
        chunk['curvature_pressure'] = chunk['depth_pressure_ratio']
        
        # 添加缺失的VPIN计算
        volume = chunk[['bid_0_size', 'ask_0_size']].sum(axis=1)
        chunk['vpin'] = pd.Series(chunk['ofi']).rolling(50, min_periods=1).mean().fillna(0) * \
                        pd.Series(volume).rolling(50, min_periods=1).mean().fillna(0)
        
        # 添加缺失的压力变化率计算
        bid_pressure = chunk['bid_0_size'] * chunk['bid_0_price'] + chunk['bid_1_size'] * chunk['bid_1_price']
        ask_pressure = chunk['ask_0_size'] * chunk['ask_0_price'] + chunk['ask_1_size'] * chunk['ask_1_price']
        chunk['pressure_change_rate'] = (pd.Series(bid_pressure).pct_change().fillna(0) - 
                                        pd.Series(ask_pressure).pct_change().fillna(0))
        
        # 更新特征保留列表
        features_to_keep = [
            'mid_price',  # 添加mid_price
            'relative_spread', 'depth_imbalance',
            'bid_ask_slope', 'order_book_pressure', 'weighted_price_depth',
            'liquidity_imbalance', 'flow_toxicity', 'price_momentum',
            'volatility_ratio', 'ofi', 'vpin', 'pressure_change_rate',
            'orderbook_gradient', 'depth_pressure_ratio'
        ]
        
        # 添加特征验证
        for feature in ['bid_ask_slope', 'orderbook_gradient']:
            non_zero = (chunk[feature].abs() > epsilon).mean() * 100
            std_value = chunk[feature].std()
            logger.info(f"{feature} 非零比例: {non_zero:.2f}%, 标准差: {std_value:.6f}")
            
            if std_value < epsilon:
                logger.warning(f"{feature} 标准差过低，添加随机扰动")
                chunk[feature] += np.random.normal(0, epsilon, size=len(chunk))
        
        return chunk[features_to_keep]

    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """处理单个数据块"""
        # 基础特征计算前验证
        if (chunk == 0).all().any():
            logger.error("输入数据包含全零列")
            logger.info(f"零值列: {chunk.columns[(chunk == 0).all()].tolist()}")
            return pd.DataFrame()

        # 检查并设置时间索引
        time_columns = ['origin_time', 'Timestamp', 'time']
        time_col = None
        for col in time_columns:
            if col in chunk.columns:
                time_col = col
                break
        
        if time_col is None:
            logger.error("找不到时间列")
            return pd.DataFrame()
        
        try:
            chunk.index = pd.to_datetime(chunk[time_col])
            if time_col != 'origin_time':
                chunk.index = pd.to_datetime(chunk.index, unit='ns')
        except Exception as e:
            logger.error(f"时间索引设置失败: {str(e)}")
            return pd.DataFrame()

        # 基础特征计算
        basic_features = self._calculate_basic_features(chunk)
        for name, values in basic_features.items():
            chunk[name] = values
        
        # 高级特征计算
        chunk = self._calculate_advanced_features(chunk, basic_features)
        
        # 零值处理
        zero_check_columns = [
            'orderbook_gradient', 'depth_pressure_ratio',
            'bid_ask_slope', 'order_book_pressure'
        ]
        
        for col in zero_check_columns:
            zero_ratio = (chunk[col] == 0).mean()
            if zero_ratio > 0.9:
                logger.warning(f"特征 {col} 零值比例过高 ({zero_ratio:.2%})，进行修正")
                chunk[col] += np.random.normal(0, 1e-6, size=len(chunk))
        
        # 修改返回的特征列表，保留mid_price用于后续处理
        required_features = [
            'mid_price',  # 添加mid_price
            'relative_spread', 'depth_imbalance',
            'bid_ask_slope', 'order_book_pressure', 'weighted_price_depth',
            'liquidity_imbalance', 'flow_toxicity', 'price_momentum',
            'volatility_ratio', 'ofi', 'vpin', 'pressure_change_rate',
            'orderbook_gradient', 'depth_pressure_ratio'
        ]
        
        # 验证特征存在性
        missing_features = [f for f in required_features if f not in chunk.columns]
        if missing_features:
            logger.error(f"缺失必要特征: {missing_features}")
            return pd.DataFrame()
        
        # 在特征计算后添加动态平滑
        smooth_cols = ['bid_ask_slope', 'orderbook_gradient', 'depth_pressure_ratio']
        for col in smooth_cols:
            # 使用EWMA平滑
            chunk[col] = chunk[col].ewm(span=50, adjust=False).mean()
            
            # 添加随机噪声防止特征塌缩
            noise = np.random.normal(0, chunk[col].std()*0.01, len(chunk))
            chunk[col] += noise
        
        # 添加特征交互项（增强中间类别区分度）
        chunk['slope_depth_interaction'] = chunk['bid_ask_slope'] * chunk['depth_imbalance']
        chunk['pressure_gradient_interaction'] = chunk['order_book_pressure'] * chunk['orderbook_gradient']
        
        return chunk[required_features]

    def _resample_chunk(self, chunk: pd.DataFrame, freq='500ms') -> pd.DataFrame:
        """改进的重采样方法，解决NaN问题"""
        # 1. 确保时间索引正确性
        if not isinstance(chunk.index, pd.DatetimeIndex):
            logger.error("时间索引类型错误")
            return pd.DataFrame()
        
        # 2. 生成精确的时间索引（仅包含数据实际覆盖范围）
        start_time = chunk.index.min().ceil(freq)
        end_time = chunk.index.max().floor(freq)
        
        # 处理可能的空数据情况
        if pd.isnull(start_time) or pd.isnull(end_time):
            return pd.DataFrame()
        
        try:
            full_index = pd.date_range(start=start_time, end=end_time, freq=freq)
        except ValueError:
            logger.error("时间索引生成失败")
            return pd.DataFrame()
        
        # 3. 改进的重采样策略
        resampled = pd.DataFrame(index=full_index)
        
        # 修改价格类特征列表
        price_features = ['mid_price', 'price_momentum', 'volatility_ratio']  # 添加mid_price
        for col in price_features:
            temp = chunk[col].resample(freq).last().reindex(full_index)
            temp = temp.ffill().bfill()  # 双向填充
            resampled[col] = temp
        
        # 量类特征：使用窗口均值+线性插值
        volume_features = ['depth_imbalance', 'liquidity_imbalance', 'ofi', 'vpin']
        for col in volume_features:
            temp = chunk[col].resample(freq).mean()
            temp = temp.reindex(full_index).interpolate(method='linear')
            resampled[col] = temp
        
        # 其他特征：使用加权平均+邻近填充
        other_features = ['relative_spread', 'bid_ask_slope', 'order_book_pressure',
                         'weighted_price_depth', 'flow_toxicity', 'pressure_change_rate',
                         'orderbook_gradient', 'depth_pressure_ratio']
        for col in other_features:
            temp = chunk[col].resample(freq).apply(lambda x: 
                np.average(x, weights=np.linspace(0.2, 1, len(x))) if len(x)>0 else np.nan)
            temp = temp.reindex(full_index).interpolate(limit=3, method='nearest')
            resampled[col] = temp
        
        # 4. 严格处理剩余NaN
        # 先使用前向填充，再使用后向填充，最后填充0
        resampled = resampled.ffill().bfill().fillna(0)
        
        # 5. 添加边界检查
        time_diff = (chunk.index.max() - chunk.index.min()).total_seconds()
        if time_diff < 0.5:  # 如果数据块时间范围小于采样频率
            logger.warning("数据块时间范围过小，使用最后值填充")
            resampled = resampled.ffill().bfill()
        
        # 在_resample_chunk开头添加调试信息
        logger.info(f"数据块时间范围: {chunk.index.min()} - {chunk.index.max()}")
        logger.info(f"生成索引范围: {start_time} - {end_time}")
        logger.info(f"原始数据长度: {len(chunk)}, 重采样后长度: {len(resampled)}")
        
        return resampled

    def process_and_save(self, train_ratio=0.8):
        """使用多进程加速的处理流水线"""
        logger.info("开始数据处理流水线...")
        
        # 获取所有数据块
        chunks = list(self._stream_chunks())
        total_chunks = len(chunks)
        
        # 创建临时目录存储中间结果
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_chunks = []
            
            # 准备进程池
            n_processes = max(1, psutil.cpu_count() - 1)  # 留一个核心给系统
            logger.info(f"使用 {n_processes} 个进程进行并行处理")
            
            # 准备任务数据
            chunk_data = [(chunk, i+1, total_chunks) for i, chunk in enumerate(chunks)]
            
            # 使用进程池处理数据
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # 提交所有任务并收集结果
                futures = [executor.submit(_process_chunk_worker, data) for data in chunk_data]
                
                # 处理结果
                for i, future in enumerate(futures, 1):
                    try:
                        result = future.result()
                        if result is not None and not result.empty:
                            # 将结果保存到临时文件
                            temp_file = os.path.join(temp_dir, f"chunk_{i}.pkl")
                            with open(temp_file, 'wb') as f:
                                pickle.dump(result, f)
                            processed_chunks.append(temp_file)
                            logger.info(f"完成数据块 {i}/{total_chunks} 的处理")
                        else:
                            logger.warning(f"数据块 {i} 处理结果为空")
                    except Exception as e:
                        logger.error(f"处理数据块 {i} 时发生错误: {str(e)}")
            
            # 合并所有处理结果
            if not processed_chunks:
                logger.error("没有有效的处理结果")
                return False
            
            # 从临时文件读取并合并结果
            final_chunks = []
            for temp_file in processed_chunks:
                with open(temp_file, 'rb') as f:
                    chunk = pickle.load(f)
                    final_chunks.append(chunk)
            
            final_data = pd.concat(final_chunks)
            
            # 处理最终数据中的 NaN 值
            if final_data.isna().any().any():
                nan_stats = final_data.isna().sum()
                logger.warning("NaN值统计:")
                for col in nan_stats.index:
                    if nan_stats[col] > 0:
                        logger.warning(f"{col}: {nan_stats[col]} NaN值")
                final_data = final_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # 在最终数据保存前添加分割逻辑
            split_idx = int(len(final_data) * train_ratio)
            train_data = final_data.iloc[:split_idx]
            test_data = final_data.iloc[split_idx:]
            
            # 保存数据
            logger.info("保存处理后的数据...")
            train_data.reset_index().to_feather('train.feather')
            test_data.reset_index().to_feather('test.feather')
            
            logger.info(f"处理完成! 训练数据大小: {len(train_data)}, 测试数据大小: {len(test_data)}")
            return True

if __name__ == "__main__":
    engine = HFTDataEngine(
        start_date=datetime(2025, 2, 1),
        end_date=datetime(2025, 2, 22),
        chunk_size=100000
    )
    
    success = engine.process_and_save()
    if success:
        logger.info("数据预处理完成，可以开始模型训练")
    else:
        logger.error("数据处理失败，请检查日志") 
