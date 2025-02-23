from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import gc
import os
from tqdm import tqdm
import matplotlib.dates as mdates
from scipy import stats
from datetime import datetime, timedelta
from sklearn.metrics import classification_report
import wandb

class DualModelEvaluator:
    def __init__(self, model_up_paths, model_down_paths, test_data_path='test.feather', 
                 threshold=0.0004, trade_threshold=0.7, project="hft-evaluation"):
        # 添加wandb初始化
        wandb.init(project=project, config={
            "threshold": threshold,
            "trade_threshold": trade_threshold,
            "model_up": model_up_paths,
            "model_down": model_down_paths
        })
        self.models_up = [lgb.Booster(model_file=path) for path in model_up_paths]
        self.models_down = [lgb.Booster(model_file=path) for path in model_down_paths]
        self.test_data_path = test_data_path
        self.threshold = threshold
        self.trade_threshold = trade_threshold
        self.wandb = wandb  # 保存wandb引用
        
    def _load_and_preprocess(self):
        """加载并预处理测试数据"""
        logger.info("加载测试数据...")
        test_data = pd.read_feather(self.test_data_path)
        
        # 确保与训练时相同的预处理
        required_features = [
            'relative_spread', 'depth_imbalance', 'bid_ask_slope',
            'order_book_pressure', 'weighted_price_depth', 'liquidity_imbalance',
            'flow_toxicity', 'price_momentum', 'volatility_ratio',
            'ofi', 'vpin', 'pressure_change_rate',
            'orderbook_gradient', 'depth_pressure_ratio'
        ]
        
        # 检查是否存在时间列并设置为索引
        time_columns = ['index', 'time', 'timestamp', 'datetime']
        time_col = next((col for col in time_columns if col in test_data.columns), None)
        
        if time_col:
            test_data[time_col] = pd.to_datetime(test_data[time_col])
            test_data.set_index(time_col, inplace=True)
        
        # 创建目标变量
        prediction_points = 120
        test_data['price_change'] = test_data['mid_price'].shift(-prediction_points) - test_data['mid_price']
        test_data['target_up'] = (test_data['price_change'] >= self.threshold).astype(int)
        test_data['target_down'] = (test_data['price_change'] <= -self.threshold).astype(int)
        
        # 删除中间列
        test_data = test_data.dropna(subset=['price_change'])
        return test_data

    def _create_rolling_features(self, data, window_sizes=[60, 300, 900, 1800]):
        """滚动特征生成（与训练时一致）"""
        logger.info("创建滚动特征...")
        
        base_features = [
            'relative_spread', 'depth_imbalance', 'bid_ask_slope',
            'order_book_pressure', 'weighted_price_depth', 'liquidity_imbalance',
            'flow_toxicity', 'price_momentum', 'volatility_ratio',
            'ofi', 'vpin', 'pressure_change_rate',
            'orderbook_gradient', 'depth_pressure_ratio'
        ]
        
        # 添加时间特征
        data['time_sin'] = np.sin(2 * np.pi * pd.to_datetime(data.index).hour/24)
        data['time_cos'] = np.cos(2 * np.pi * pd.to_datetime(data.index).hour/24)
        
        # 添加交互特征
        data['spread_depth_ratio'] = data['relative_spread'] / (data['depth_imbalance'] + 1e-7)
        data['order_flow_imbalance'] = data['bid_ask_slope'] * data['liquidity_imbalance']
        
        # 添加市场状态
        volatility = data['flow_toxicity'].rolling(300, min_periods=1).mean().fillna(0)
        returns = data['price_momentum'].fillna(0)
        conditions = [
            (volatility < 0.1) & (returns.abs() < 0.3),
            (volatility >= 0.3),
            (returns >= 0.3),     
            (returns <= -0.3)    
        ]
        data['market_state'] = np.select(conditions, [0, 1, 2, 3], default=0)
        
        # 计算波动率
        data['volatility_30s'] = data['mid_price'].pct_change().rolling(60, min_periods=1).std()
        
        # 生成滚动特征
        for window in window_sizes:
            rolled = data[base_features].rolling(
                window=window,
                min_periods=1,
                closed='left'  # 确保不使用当前值
            )
            
            # 计算统计量
            stats = pd.concat([
                rolled.mean().add_suffix(f'_mean_{window}'),
                rolled.std().add_suffix(f'_std_{window}'),
                rolled.skew().add_suffix(f'_skew_{window}'),
                rolled.kurt().add_suffix(f'_kurt_{window}')
            ], axis=1)
            
            data = pd.concat([data, stats], axis=1)
        
        # 修改后的填充方法
        data = data.ffill().fillna(0)
        
        return data

    def _get_model_features(self, model):
        """获取模型特征并添加调试信息"""
        features = model.feature_name()
        if not features:
            logger.warning("模型特征名为空，尝试从树结构中提取")
            try:
                features = [f'f{i}' for i in range(model.num_feature())]
            except:
                logger.error("无法获取模型特征列表")
                return None
        return features

    def _predict_ensemble(self, models, data):
        """集成模型预测"""
        preds = []
        for model in models:
            try:
                pred = model.predict(data)
                preds.append(pred)
            except Exception as e:
                logger.error(f"预测失败: {str(e)}")
                continue
        return np.mean(preds, axis=0) if preds else None

    def _log_metrics(self, metrics, step=None):
        """统一指标记录方法"""
        log_data = {k: v for k, v in metrics.items() if isinstance(v, (int, float, np.number))}
        if step:
            log_data['step'] = step
        self.wandb.log(log_data)

    def _generate_enhanced_plots(self, signals, results):
        """修改后的可视化方法，使用新的信号条件"""
        # 1. 信号分布热力图（使用交易信号）
        plt.figure(figsize=(10, 8))
        signal_counts = signals['trade_signal'].value_counts().to_frame().T
        sns.heatmap(signal_counts, annot=True, fmt='d', cmap='YlGnBu')
        plt.title('Trade Signal Distribution')
        self.wandb.log({"signal_heatmap": wandb.Image(plt)})
        plt.close()

        # 2. 预测概率分布
        plt.figure(figsize=(12, 6))
        sns.kdeplot(signals['up_prob'], label='Up Probability')
        sns.kdeplot(signals['down_prob'], label='Down Probability')
        plt.axvline(self.trade_threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        self.wandb.log({"probability_dist": wandb.Image(plt)})
        plt.close()

        # 3. 累计收益曲线
        signals['cum_return'] = signals['price_change'].cumsum()
        signals['strategy_return'] = (signals['trade_signal'].shift(1) * signals['price_change']).cumsum()
        
        plt.figure(figsize=(12, 6))
        signals[['cum_return', 'strategy_return']].plot()
        plt.title('Cumulative Returns Comparison')
        plt.ylabel('Returns')
        self.wandb.log({"cumulative_returns": wandb.Image(plt)})
        plt.close()

        # 4. 交易信号与价格变化（采样显示）
        plt.figure(figsize=(14, 8))
        sample_df = signals.iloc[::10]  # 每10个点采样显示
        plt.plot(sample_df['timestamp'], sample_df['price_change'].cumsum(), label='Price Change')
        plt.scatter(sample_df[sample_df['trade_signal'] == 1]['timestamp'], 
                   sample_df[sample_df['trade_signal'] == 1]['price_change'].cumsum(),
                   color='g', label='Long', marker='^')
        plt.scatter(sample_df[sample_df['trade_signal'] == -1]['timestamp'], 
                   sample_df[sample_df['trade_signal'] == -1]['price_change'].cumsum(),
                   color='r', label='Short', marker='v')
        plt.title('Trading Signals on Price Series')
        plt.legend()
        plt.gcf().autofmt_xdate()
        self.wandb.log({"trading_signals": wandb.Image(plt)})
        plt.close()

    def _analyze_trades(self, signals):
        """修改后的交易分析，添加固定持仓时间"""
        trades = []
        current_position = 0
        entry_index = 0
        entry_time = None  # 新增入场时间记录
        max_drawdown = 0
        max_return = 0
        cumulative = 0
        
        for i in tqdm(range(len(signals)), desc="分析交易"):
            current_time = signals.index[i]
            
            # 平仓条件：1.持仓时间达到1分钟 或 2.信号方向改变
            if current_position != 0:
                time_in_trade = (current_time - entry_time).total_seconds() / 60  # 分钟数
                
                # 条件1：持仓时间超过1分钟
                time_exit = time_in_trade >= 1.0
                
                if time_exit:
                    exit_return = signals['price_change'].iloc[i] * current_position
                    trade_duration = i - entry_index
                    
                    # 计算最大回撤
                    cumulative += exit_return
                    if cumulative > max_return:
                        max_return = cumulative
                    else:
                        drawdown = max_return - cumulative
                        if drawdown > max_drawdown:
                            max_drawdown = drawdown
                    
                    trades.append({
                        'duration': trade_duration,
                        'time_duration': time_in_trade,  # 记录实际时间长度
                        'return': exit_return,
                        'direction': current_position,
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'exit_reason': 'time' if time_exit else 'signal'
                    })
                    current_position = 0
                    entry_time = None
                
            # 开仓逻辑
            if current_position == 0 and signals['trade_signal'].iloc[i] != 0:
                current_position = signals['trade_signal'].iloc[i]
                entry_index = i
                entry_time = current_time  # 记录入场时间
        
        # 记录回撤指标
        self.wandb.log({
            "max_drawdown": max_drawdown,
            "max_return": max_return,
            "drawdown_ratio": max_drawdown / (max_return + 1e-7)
        })
        
        trade_df = pd.DataFrame(trades)
        wins = trade_df['return'] > 0
        losses = trade_df['return'] < 0
        
        trade_results = {
            'win_rate': wins.mean(),
            'avg_win': trade_df[wins]['return'].mean(),
            'avg_loss': trade_df[losses]['return'].mean() if any(losses) else 0,
            'profit_factor': abs(trade_df[wins]['return'].sum() / trade_df[losses]['return'].sum()) if any(losses) else np.inf,
            'trade_count': len(trades),
            'long_win_rate': trade_df[trade_df['direction'] == 1]['return'].gt(0).mean(),
            'short_win_rate': trade_df[trade_df['direction'] == -1]['return'].gt(0).mean()
        }
        
        return trade_results

    def evaluate(self):
        """执行完整评估流程"""
        test_data = self._load_and_preprocess()
        test_data = self._create_rolling_features(test_data)
        
        # 获取模型特征
        up_features = self._get_model_features(self.models_up[0])
        down_features = self._get_model_features(self.models_down[0])
        
        # 确保所有特征都存在
        missing_features = set(up_features) - set(test_data.columns)
        if missing_features:
            logger.warning(f"缺失特征: {missing_features}")
            # 为缺失特征添加零列
            for feat in missing_features:
                test_data[feat] = 0
            
        missing_features = set(down_features) - set(test_data.columns)
        if missing_features:
            logger.warning(f"缺失特征: {missing_features}")
            # 为缺失特征添加零列
            for feat in missing_features:
                test_data[feat] = 0
        
        # 预测
        up_pred = self._predict_ensemble(self.models_up, test_data[up_features])
        down_pred = self._predict_ensemble(self.models_down, test_data[down_features])
        
        if up_pred is None or down_pred is None:
            logger.error("预测失败，无法继续评估")
            return None
        
        # 生成交易信号
        signals = pd.DataFrame({
            'up_prob': up_pred,
            'down_prob': down_pred,
            'price_change': test_data['price_change'],
            'timestamp': test_data.index
        })
        
        # 修改交易信号生成逻辑
        signals['up_condition'] = signals['up_prob'] > self.trade_threshold
        signals['down_condition'] = signals['down_prob'] > self.trade_threshold
        signals['trade_signal'] = 0
        signals.loc[signals['up_condition'], 'trade_signal'] = 1
        signals.loc[signals['down_condition'], 'trade_signal'] = -1
        
        # 计算评估指标
        results = self._calculate_metrics(signals)
        
        # 生成可视化
        self._generate_enhanced_plots(signals, results)
        
        # 记录样本数据
        sample_data = test_data.sample(min(1000, len(test_data)))
        self.wandb.log({"feature_distribution": wandb.Table(dataframe=sample_data[up_features])})
        
        # 在预测后记录预测分布
        self.wandb.log({
            "up_prob_dist": wandb.Histogram(signals['up_prob']),
            "down_prob_dist": wandb.Histogram(signals['down_prob'])
        })
        
        # 记录交易信号表
        signal_table = wandb.Table(dataframe=signals.sample(1000))
        self.wandb.log({"trade_signals": signal_table})
        
        return results

    def _calculate_metrics(self, signals):
        """计算综合评估指标"""
        metrics = {}
        
        # 基础分类指标
        metrics['up_auc'] = roc_auc_score(signals['price_change'] >= self.threshold, signals['up_prob'])
        metrics['down_auc'] = roc_auc_score(signals['price_change'] <= -self.threshold, signals['down_prob'])
        
        # 信号冲突分析（使用新的条件列）
        conflict_mask = signals['up_condition'] & signals['down_condition']
        metrics['conflict_rate'] = conflict_mask.mean()
        
        # 空仓准确率（当两个模型都预测不交易时的市场平静程度）
        no_trade_mask = (signals['trade_signal'] == 0)
        metrics['neutral_accuracy'] = ((signals['price_change'].abs() < self.threshold) & no_trade_mask).mean()
        
        # 交易指标
        trade_results = self._analyze_trades(signals)
        
        # 在_calculate_metrics中添加
        conflict_samples = signals[conflict_mask].sample(min(5, len(signals[conflict_mask])))
        if not conflict_samples.empty:
            logger.warning("信号冲突样本示例:\n%s", conflict_samples[['up_prob', 'down_prob']])
        
        return {**metrics, **trade_results}

if __name__ == "__main__":
    # 示例用法
    model_up_paths = [f'model_up_v{i}.bin' for i in range(3)]  # 0,1,2 三个模型
    model_down_paths = [f'model_down_v{i}.bin' for i in range(3)]  # 0,1,2 三个模型
    
    evaluator = DualModelEvaluator(
        model_up_paths=model_up_paths,
        model_down_paths=model_down_paths,
        threshold=0.0004,
        trade_threshold=0.6
    )
    
    results = evaluator.evaluate()
    
    logger.info("\n=== 双模型评估结果 ===")
    logger.info(f"上涨模型AUC: {results['up_auc']:.4f}")
    logger.info(f"下跌模型AUC: {results['down_auc']:.4f}")
    logger.info(f"信号冲突率: {results['conflict_rate']:.2%}")
    logger.info(f"空仓准确率: {results['neutral_accuracy']:.2%}")
    
    logger.info("\n=== 交易表现 ===")
    logger.info(f"总交易次数: {results['trade_count']}")
    logger.info(f"胜率: {results['win_rate']:.2%}")
    logger.info(f"多头胜率: {results['long_win_rate']:.2%}")
    logger.info(f"空头胜率: {results['short_win_rate']:.2%}")
    logger.info(f"平均盈利: {results['avg_win']:.6f}")
    logger.info(f"平均亏损: {results['avg_loss']:.6f}")
    logger.info(f"盈亏比: {results['profit_factor']:.2f}")
