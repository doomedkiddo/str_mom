from sklearn.preprocessing import RobustScaler
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

class ModelEvaluator:
    def __init__(self, model_paths, test_data_path='test.feather', thresholds=[-0.0004, -0.0002, 0.0002, 0.0004]):
        self.models = [lgb.Booster(model_file=path) for path in model_paths]
        self.test_data_path = test_data_path
        self.scaler = RobustScaler()
        self.thresholds = thresholds  # 分类阈值
        
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
        time_col = None
        for col in time_columns:
            if col in test_data.columns:
                time_col = col
                break
        
        if time_col:
            test_data.set_index(time_col, inplace=True)
        
        # 创建log_return（与训练时相同逻辑）
        prediction_points = 120
        test_data['log_return'] = np.log(test_data['mid_price']).diff(prediction_points).shift(-prediction_points)
        
        # 确保所有特征都是数值类型
        for col in test_data.columns:
            if test_data[col].dtype == 'object' or test_data[col].dtype.name == 'category':
                logger.warning(f"删除非数值列: {col}")
                test_data = test_data.drop(columns=[col])
        
        # 修改特征验证列表（移除mid_price）
        required_features = [
            'relative_spread', 'depth_imbalance',
            'bid_ask_slope', 'order_book_pressure', 'weighted_price_depth',
            'liquidity_imbalance', 'flow_toxicity', 'price_momentum',
            'volatility_ratio', 'ofi', 'vpin', 'pressure_change_rate',
            'orderbook_gradient', 'depth_pressure_ratio'
        ]
        
        # 添加调试信息
        logger.info(f"可用特征列表: {test_data.columns.tolist()}")
        
        # 删除训练时没有的特征（根据错误日志，训练模型有238个特征）
        # 移除以下可能多余的特征生成
        if 'volume' in test_data.columns:
            test_data = test_data.drop(columns=['volume'])
        if 'hour' in test_data.columns:
            test_data = test_data.drop(columns=['hour'])
        
        return test_data.dropna(subset=['log_return'])

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
        
        for window in window_sizes:
            rolled = data[base_features].rolling(window, min_periods=1)
            
            stats = pd.concat([
                rolled.mean().fillna(0).add_suffix(f'_ma{window}'),
                rolled.std().fillna(0).add_suffix(f'_std{window}'),
                rolled.skew().fillna(0).add_suffix(f'_skew{window}'),
                rolled.kurt().fillna(0).add_suffix(f'_kurt{window}')
            ], axis=1)
            
            data = pd.concat([data, stats], axis=1)
        
        return data

    def _analyze_large_moves(self, y_true, y_pred):
        """分析大幅波动的预测准确性"""
        # 将预测概率转换为类别
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # 识别大幅波动（类别0和4）
        large_moves = (y_true == 0) | (y_true == 4)
        
        if not any(large_moves):
            return {
                'large_move_accuracy': 0,
                'large_move_count': 0,
                'direction_accuracy': 0
            }
        
        # 计算大幅波动的预测准确性
        correct_predictions = y_pred_class[large_moves] == y_true[large_moves]
        
        # 关注大变动的预测准确率
        large_move_accuracy = correct_predictions.mean()
        
        results = {
            'large_move_accuracy': large_move_accuracy,
            'large_move_count': large_moves.sum(),
            'direction_accuracy': correct_predictions.sum() / large_moves.sum(),
            'true_moves': y_true[large_moves],
            'pred_moves': y_pred_class[large_moves]
        }
        
        return results

    def _calculate_trading_metrics(self, y_true, y_pred):
        """计算交易相关指标"""
        # 将预测概率转换为类别
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # 定义交易信号
        # 0: 大跌, 1: 小跌, 2: 震荡, 3: 小涨, 4: 大涨
        position_changes = y_pred_class[1:] != y_pred_class[:-1]
        trade_points = np.where(position_changes)[0]
        
        trades = []
        current_position = 0
        current_entry = 0
        
        for i in range(len(y_true)):
            if i in trade_points:
                if current_position != 0:
                    # 平仓
                    trade_return = (y_true[i] - y_true[current_entry]) * current_position
                    trades.append(trade_return)
                
                # 根据预测类别开新仓，且收益率绝对值大于万分之4
                pred_class = y_pred_class[i]
                if np.abs(y_pred[i, pred_class]) > 0.0004:
                    if pred_class in [0, 1]:  # 预测下跌
                        current_position = -1
                    elif pred_class in [3, 4]:  # 预测上涨
                        current_position = 1
                    else:  # 震荡时不开仓
                        current_position = 0
                    current_entry = i
        
        trades = np.array(trades)
        
        return {
            'win_rate': (trades > 0).mean() if len(trades) > 0 else 0,
            'avg_win': trades[trades > 0].mean() if any(trades > 0) else 0,
            'avg_loss': trades[trades < 0].mean() if any(trades < 0) else 0,
            'profit_factor': abs(trades[trades > 0].sum() / trades[trades < 0].sum()) if any(trades < 0) else np.inf,
            'trade_count': len(trades)
        }

    def evaluate(self):
        """执行完整评估流程"""
        test_data = self._load_and_preprocess()
        test_data = self._create_rolling_features(test_data)
        
        # 创建目标变量（五分类）
        prediction_points = 120
        price_change_rate = (test_data['mid_price'].shift(-prediction_points) - 
                            test_data['mid_price']) / test_data['mid_price']
        
        conditions = [
            price_change_rate <= self.thresholds[0],  # 大跌
            (price_change_rate > self.thresholds[0]) & (price_change_rate <= self.thresholds[1]),  # 小跌
            (price_change_rate > self.thresholds[1]) & (price_change_rate < self.thresholds[2]),   # 震荡
            (price_change_rate >= self.thresholds[2]) & (price_change_rate < self.thresholds[3]),  # 小涨
            price_change_rate >= self.thresholds[3]    # 大涨
        ]
        values = [0, 1, 2, 3, 4]
        test_data['target_class'] = np.select(conditions, values, default=2)
        
        # 获取训练模型的特征列表
        train_features = self.models[0].feature_name()
        logger.info(f"训练模型特征数: {len(train_features)}")
        logger.info(f"训练模型特征: {train_features}")
        
        # 准备特征
        excluded_columns = ['target_class', 'mid_price', 'log_return']
        feature_cols = [col for col in test_data.columns 
                       if col not in excluded_columns
                       and test_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        # 重命名特征列以匹配训练模型的特征名称
        feature_mapping = {feature: f'Column_{i}' for i, feature in enumerate(feature_cols)}
        test_data_renamed = test_data[feature_cols].rename(columns=feature_mapping)
        
        logger.info(f"测试数据特征数: {len(feature_cols)}")
        logger.info(f"测试数据特征: {feature_cols[:10]}...")  # 只显示前10个特征
        
        # 特征对齐检查
        logger.info(f"训练模型特征数: {len(train_features)}")
        logger.info(f"测试数据特征数: {len(feature_cols)}")
        
        # 特征对齐检查
        missing_in_test = set(train_features) - set(test_data_renamed.columns)
        missing_in_train = set(test_data_renamed.columns) - set(train_features)
        
        if missing_in_test:
            logger.error(f"测试数据缺少训练时的特征: {missing_in_test}")
        if missing_in_train:
            logger.error(f"测试数据包含训练时没有的特征: {missing_in_train}")
        
        # 确保特征顺序一致
        test_data_renamed = test_data_renamed[train_features]
        
        # 使用重命名后的特征进行预测
        X_test = test_data_renamed.values
        y_test = test_data['target_class'].values
        
        # 确保数据类型正确
        X_test = X_test.astype(np.float32)
        y_test = y_test.astype(np.int32)
        
        # 模型预测
        predictions = np.zeros((len(X_test), 5))  # 5个类别的概率
        for model in self.models:
            predictions += model.predict(X_test)
        predictions /= len(self.models)
        
        # 计算混淆矩阵  
        y_pred_class = np.argmax(predictions, axis=1)
        
        # 确保生成5x5的混淆矩阵
        y_test_cat = pd.Categorical(y_test, categories=[0, 1, 2, 3, 4])
        y_pred_cat = pd.Categorical(y_pred_class, categories=[0, 1, 2, 3, 4])
        
        confusion_matrix = pd.crosstab(
            y_test_cat, 
            y_pred_cat,
            rownames=['Actual'], 
            colnames=['Predicted'],
            dropna=False  # 确保所有类别都被包含
        )
        
        # 处理零除错误
        confusion_np = confusion_matrix.values.astype(float)
        row_sums = confusion_np.sum(axis=1)
        row_sums[row_sums == 0] = 1e-9  # 避免除以零
        
        class_accuracies = pd.Series(
            np.diag(confusion_np) / row_sums,
            index=confusion_matrix.index
        )
        
        # 分析大幅波动预测
        large_move_analysis = self._analyze_large_moves(y_test, predictions)
        
        # 计算交易指标
        trading_metrics = self._calculate_trading_metrics(test_data['mid_price'].values, predictions)
        
        # 添加详细的类别分布分析
        class_distribution = pd.Series(y_test).value_counts(normalize=True)
        logger.info("\n类别分布分析:")
        logger.info(class_distribution)
        
        # 添加每个类别的F1-score
        logger.info("\nDetailed Classification Report:")
        logger.info(classification_report(y_test, y_pred_class, 
                  target_names=['Sharp Drop', 'Mild Drop', 'Neutral', 'Mild Rise', 'Sharp Rise']))
        
        # 添加特征与目标的相关性分析
        corr_matrix = test_data.corr()[['target_class']]
        logger.info("\n特征与目标相关性:")
        logger.info(corr_matrix.sort_values('target_class', ascending=False))
        
        # 汇总结果
        results = {
            'ClassAccuracies': class_accuracies.to_dict(),
            'OverallAccuracy': (y_pred_class == y_test).mean(),
            'ConfusionMatrix': confusion_matrix,
            'LargeMoveAccuracy': large_move_analysis['large_move_accuracy'],
            'LargeMoveCount': large_move_analysis['large_move_count'],
            'DirectionAccuracy': large_move_analysis['direction_accuracy'],
            'WinRate': trading_metrics['win_rate'],
            'AvgWin': trading_metrics['avg_win'],
            'AvgLoss': trading_metrics['avg_loss'],
            'ProfitFactor': trading_metrics['profit_factor'],
            'TradeCount': trading_metrics['trade_count'],
            'Predictions': predictions,
            'Actuals': y_test
        }
        
        # 生成可视化
        self._generate_enhanced_plots(test_data, results, large_move_analysis)
        
        return results

    def _generate_enhanced_plots(self, data, results, large_move_analysis):
        """Generate enhanced evaluation plots"""
        # Create output directory
        output_dir = "evaluation_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Hourly extreme class accuracy
        plt.figure(figsize=(10, 6))
        # ... [plotting code for hourly accuracy] ...
        plt.savefig(f"{output_dir}/01_hourly_extreme_accuracy.png")
        plt.close()

        # 2. Cumulative returns
        plt.figure(figsize=(10, 6))
        # ... [plotting code for cumulative returns] ...
        plt.savefig(f"{output_dir}/02_cumulative_returns.png")
        plt.close()

        # 3. Prediction error distribution
        plt.figure(figsize=(10, 6))
        # ... [plotting code for error distribution] ...
        plt.savefig(f"{output_dir}/03_error_distribution.png")
        plt.close()

        # 4. Trading session accuracy
        plt.figure(figsize=(10, 6))
        # ... [plotting code for session accuracy] ...
        plt.savefig(f"{output_dir}/04_session_accuracy.png")
        plt.close()

        # 5. Rolling Sharpe ratio
        plt.figure(figsize=(10, 6))
        # ... [plotting code for Sharpe ratio] ...
        plt.savefig(f"{output_dir}/05_rolling_sharpe.png")
        plt.close()

        # 6. Feature importance
        plt.figure(figsize=(10, 6))
        # ... [plotting code for feature importance] ...
        plt.savefig(f"{output_dir}/06_feature_importance.png")
        plt.close()

        # 7. Direction confusion matrix
        plt.figure(figsize=(10, 6))
        # ... [plotting code for confusion matrix] ...
        plt.savefig(f"{output_dir}/07_direction_confusion.png")
        plt.close()

        # 8. Drawdown analysis
        plt.figure(figsize=(10, 6))
        # ... [plotting code for drawdown] ...
        plt.savefig(f"{output_dir}/08_drawdown_analysis.png")
        plt.close()

        # 9. Returns distribution
        plt.figure(figsize=(10, 6))
        # ... [plotting code for returns distribution] ...
        plt.savefig(f"{output_dir}/09_returns_distribution.png")
        plt.close()

        logger.info(f"All evaluation plots saved to {output_dir}/ directory")

    def _generate_large_move_analysis(self, results):
        """Generate large move analysis plots"""
        output_dir = "evaluation_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Large moves timeline
        plt.figure(figsize=(12, 6))
        # ... [plotting code] ...
        plt.savefig(f"{output_dir}/10_large_moves_timeline.png")
        plt.close()

        # 2. Accuracy by magnitude
        plt.figure(figsize=(12, 6))
        # ... [plotting code] ...
        plt.savefig(f"{output_dir}/11_accuracy_by_magnitude.png")
        plt.close()

        logger.info(f"Large moves analysis saved to {output_dir}/ directory")

if __name__ == "__main__":
    # 示例用法
    model_paths = [f'enhanced_model_v{i}.bin' for i in range(3)]
    
    evaluator = ModelEvaluator(model_paths, thresholds=[-0.0004, -0.0002, 0.0002, 0.0004])  # 设置分类阈值
    results = evaluator.evaluate()
    
    logger.info("\n=== 最终评估结果 ===")
    logger.info(f"分类准确率:")
    logger.info(f"ClassAccuracies: {results['ClassAccuracies']}")
    logger.info(f"OverallAccuracy: {results['OverallAccuracy']:.4f}")
    
    logger.info(f"\n交易统计:")
    logger.info(f"胜率: {results['WinRate']:.4f}")
    logger.info(f"平均盈利: {results['AvgWin']:.6f}")
    logger.info(f"平均亏损: {results['AvgLoss']:.6f}")
    logger.info(f"盈亏比: {results['ProfitFactor']:.2f}")
    logger.info(f"总交易次数: {results['TradeCount']}")
