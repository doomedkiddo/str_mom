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
from sklearn.metrics import classification_report, f1_score
import argparse

class ModelEvaluator:
    def __init__(self, model_paths, test_data_path='test.feather', 
                 lower_threshold=-0.0004, upper_threshold=0.0004,
                 window_multiplier=2):
        self.models = [lgb.Booster(model_file=path) for path in model_paths]
        self.test_data_path = test_data_path
        self.scaler = RobustScaler()
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.window_multiplier = window_multiplier
        
    def _load_and_preprocess(self):
        """加载并预处理测试数据"""
        logger.info("加载测试数据...")
        try:
            test_data = pd.read_feather(self.test_data_path)
        except Exception as e:
            logger.error(f"无法加载测试数据: {str(e)}")
            raise ValueError(f"无法加载测试数据: {str(e)}")
        
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
            try:
                test_data.set_index(time_col, inplace=True)
                logger.info(f"成功设置时间索引: {time_col}")
            except Exception as e:
                logger.warning(f"时间索引设置失败: {str(e)}")
        else:
            logger.warning("未找到时间列，将使用默认索引")
        
        # 创建log_return（与训练时相同逻辑）
        prediction_points = 120
        if 'mid_price' in test_data.columns:
            test_data['log_return'] = np.log(test_data['mid_price']).diff(prediction_points).shift(-prediction_points)
            logger.info("成功创建log_return特征")
        else:
            logger.error("缺少mid_price列，无法创建log_return特征")
            raise ValueError("缺少mid_price列，无法创建log_return特征")
        
        # 确保所有特征都是数值类型
        for col in test_data.columns:
            if test_data[col].dtype == 'object' or test_data[col].dtype.name == 'category':
                logger.warning(f"删除非数值列: {col}")
                test_data = test_data.drop(columns=[col])
        
        # 在删除列之前添加滞后特征
        if 'log_return' in test_data.columns:
            test_data['price_change_lag1'] = test_data['log_return'].shift(1)
        
        if 'flow_toxicity' in test_data.columns:
            test_data['order_flow_lag2'] = test_data['flow_toxicity'].shift(2)
        
        # 添加特征验证
        missing_features = [f for f in required_features if f not in test_data.columns]
        if missing_features:
            logger.warning(f"测试数据缺少以下特征: {missing_features}")
            for feature in missing_features:
                logger.info(f"创建空白特征: {feature}")
                test_data[feature] = 0.0
        
        return test_data.dropna(subset=['log_return'])

    def _create_rolling_features(self, data):
        """滚动特征生成（与训练时严格一致）"""
        logger.info("创建滚动特征...")
        
        # 使用与训练代码完全相同的窗口定义
        base_window_sizes = [60, 300, 900, 1800]  # 单位：秒
        adjusted_windows = [w * self.window_multiplier for w in base_window_sizes]
        
        base_features = [
            'relative_spread', 'depth_imbalance', 'bid_ask_slope',
            'order_book_pressure', 'weighted_price_depth', 'liquidity_imbalance',
            'flow_toxicity', 'price_momentum', 'volatility_ratio',
            'ofi', 'vpin', 'pressure_change_rate',
            'orderbook_gradient', 'depth_pressure_ratio'
        ]
        
        # 检查并填充缺失的基础特征
        for feature in base_features:
            if feature not in data.columns:
                logger.warning(f"缺失基础特征: {feature}，使用零值填充")
                data[feature] = 0.0
        
        # 在创建新特征前记录现有列数
        original_column_count = len(data.columns)
        
        try:
            for window in adjusted_windows:  # 使用调整后的窗口
                logger.info(f"处理窗口大小: {window}")
                rolled = data[base_features].rolling(window, min_periods=max(1, window//10))
                
                stats = pd.concat([
                    rolled.mean().fillna(0).add_suffix(f'_ma{window}'),
                    rolled.std().fillna(0).add_suffix(f'_std{window}'),
                    rolled.skew().fillna(0).add_suffix(f'_skew{window}'),
                    rolled.kurt().fillna(0).add_suffix(f'_kurt{window}')
                ], axis=1)
                
                data = pd.concat([data, stats], axis=1)
                
            # 验证特征生成是否成功
            new_column_count = len(data.columns)
            expected_new_columns = len(base_features) * 4 * len(adjusted_windows)
            logger.info(f"已创建 {new_column_count - original_column_count} 个新特征 (期望: {expected_new_columns})")
            
            if new_column_count - original_column_count < expected_new_columns:
                logger.warning("部分滚动特征可能创建失败")
            
            return data
            
        except Exception as e:
            logger.error(f"滚动特征创建失败: {str(e)}")
            # 如果特征创建失败，返回原始数据以确保评估过程不中断
            return data

    def _analyze_large_moves(self, y_true, y_pred):
        """分析大幅波动的预测准确性"""
        # 将预测概率转换为类别
        y_pred_class = np.argmax(y_pred, axis=1)
        
        # 识别大幅波动（类别0和2）
        large_moves = (y_true == 0) | (y_true == 2)
        
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

    def _calculate_trading_metrics(self, prices, predictions):
        """Enhanced trading strategy analysis without confidence threshold"""
        if prices is None or len(prices) == 0:
            logger.warning("未提供价格数据，无法计算交易指标")
            return {
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'trade_count': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # 将预测概率转换为类别
        y_pred_class = np.argmax(predictions, axis=1)
        
        # 计算预测的信心水平 (仅用于日志记录和分析)
        confidence = np.max(predictions, axis=1)
        
        # 设置初始资金
        initial_capital = 10000
        capital = initial_capital
        position_size = initial_capital * 0.1  # 固定使用10%资金
        
        # 跟踪每日资本变化
        equity_curve = [initial_capital]
        trades = []
        current_position = 0
        entry_idx = 0
        entry_price = 0.0  # Store the actual price value, not the index
        
        # 每N个点检查一次交易信号（控制交易频率）
        sample_interval = 10
        
        # 强制在每个采样时间点进行交易决策
        for i in range(1, len(prices)):
            price_change = (prices[i] / prices[i-1]) - 1  # 价格变动百分比
            
            # 应用当前仓位的盈亏
            if current_position != 0:
                # 仓位价值变动
                position_pnl = position_size * price_change * current_position
                capital += position_pnl
            
            # 按指定间隔检查交易信号
            if i % sample_interval == 0:
                pred_class = y_pred_class[i]
                
                # 如果已有仓位，检查是否需要平仓
                if current_position != 0:
                    # 如果预测转为中性或反向，平仓
                    if (current_position > 0 and pred_class != 2) or (current_position < 0 and pred_class != 0):
                        # 平仓 - 使用当前价格和入场价格计算收益，而不是索引
                        current_price = prices[i]
                        trade_return = (current_price / entry_price - 1) * current_position
                        trade_info = {
                            'entry': entry_idx,
                            'exit': i,
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'return': trade_return,
                            'duration': i - entry_idx,
                            'type': 'long' if current_position > 0 else 'short',
                            'confidence': confidence[entry_idx]
                        }
                        trades.append(trade_info)
                        logger.debug(f"平仓: 时间点={i}, 收益={trade_return:.4f}")
                        current_position = 0
                
                # 只在没有仓位时开新仓（不考虑信心水平）
                if current_position == 0:
                    if pred_class == 0:  # 预测下跌
                        current_position = -1
                        entry_idx = i
                        entry_price = prices[i]  # Store the actual price
                        logger.debug(f"开空仓: 时间点={i}, 价格={entry_price}")
                    elif pred_class == 2:  # 预测上涨
                        current_position = 1
                        entry_idx = i
                        entry_price = prices[i]  # Store the actual price
                        logger.debug(f"开多仓: 时间点={i}, 价格={entry_price}")
            
            # 更新权益曲线
            equity_curve.append(capital)
        
        # 如果最后还有开仓，添加到交易列表
        if current_position != 0 and len(prices) > 0:
            final_price = prices[-1]
            trade_return = (final_price / entry_price - 1) * current_position
            trade_info = {
                'entry': entry_idx,
                'exit': len(prices) - 1,
                'entry_price': entry_price,
                'exit_price': final_price,
                'return': trade_return,
                'duration': len(prices) - 1 - entry_idx,
                'type': 'long' if current_position > 0 else 'short',
                'confidence': confidence[entry_idx]
            }
            trades.append(trade_info)
        
        # 输出交易统计信息
        logger.info(f"总计生成交易数: {len(trades)}")
        
        # 计算交易统计
        if len(trades) == 0:
            logger.warning("没有生成任何交易，请检查预测类别分布")
            return {
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'trade_count': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'equity_curve': np.array(equity_curve)
            }
        
        # 计算交易统计
        trade_returns = [t['return'] for t in trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r <= 0]
        
        win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # 计算盈亏比
        total_win = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        
        # 计算最大回撤
        equity_curve = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = drawdown.max()
        
        # 计算夏普比率
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # 保存交易日志
        try:
            trade_df = pd.DataFrame(trades)
            os.makedirs('evaluation_plots', exist_ok=True)
            trade_df.to_csv('evaluation_plots/trade_log.csv', index=False)
            
            # 绘制权益曲线
            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve)
            plt.title('Equity Curve')
            plt.xlabel('Trade Days')
            plt.ylabel('Capital')
            plt.grid(True)
            plt.savefig('evaluation_plots/equity_curve.png')
            plt.close()
        except Exception as e:
            logger.warning(f"保存交易日志出错: {str(e)}")
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trade_count': len(trades),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'equity_curve': equity_curve,
            'total_return': (equity_curve[-1] / equity_curve[0]) - 1
        }

    def evaluate(self):
        """执行完整评估流程"""
        try:
            # 递增式数据处理以节省内存
            logger.info("开始数据预处理...")
            test_data = self._load_and_preprocess()
            
            # 监控内存使用情况
            process = psutil.Process()
            logger.info(f"预处理后内存使用: {process.memory_info().rss / (1024 * 1024):.2f} MB")
            
            # 创建滚动特征
            test_data = self._create_rolling_features(test_data)
            logger.info(f"特征工程后内存使用: {process.memory_info().rss / (1024 * 1024):.2f} MB")
            
            # 创建目标变量（三分类）
            prediction_points = 120
            price_change_rate = (test_data['mid_price'].shift(-prediction_points) - 
                                test_data['mid_price']) / test_data['mid_price']
            
            conditions = [
                (price_change_rate < self.lower_threshold),  # 下跌
                price_change_rate.between(self.lower_threshold, self.upper_threshold),  # 平稳
                (price_change_rate > self.upper_threshold)  # 上涨
            ]
            values = [0, 1, 2]
            test_data['target_class'] = np.select(conditions, values, default=1)
            
            # 获取训练模型的特征列表
            train_features = self.models[0].feature_name()
            logger.info(f"训练模型特征数: {len(train_features)}")
            
            # 准备特征
            excluded_columns = ['target_class', 'mid_price', 'log_return']
            feature_cols = [col for col in test_data.columns 
                           if col not in excluded_columns
                           and test_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
            
            # 更智能的特征映射
            test_features_mapped = {}
            feature_mapping = {}
            
            # 1. 首先匹配完全相同的特征名
            exact_matches = set(train_features).intersection(set(test_data.columns))
            for feature in exact_matches:
                feature_mapping[feature] = feature
                test_features_mapped[feature] = test_data[feature]
            
            logger.info(f"精确匹配特征数: {len(exact_matches)}/{len(train_features)}")
            
            # 2. 尝试模糊匹配剩余特征
            remaining_train_features = set(train_features) - exact_matches
            remaining_test_features = set(feature_cols) - exact_matches
            
            for train_feat in remaining_train_features:
                best_match = None
                best_score = 0
                
                # 简单的字符串相似度比较
                for test_feat in remaining_test_features:
                    # 移除后缀数字进行比较
                    train_base = ''.join([c for c in train_feat if not c.isdigit()])
                    test_base = ''.join([c for c in test_feat if not c.isdigit()])
                    
                    if train_base == test_base:
                        if best_match is None:
                            best_match = test_feat
                            best_score = 1.0
                        break
                
                if best_match:
                    feature_mapping[train_feat] = best_match
                    test_features_mapped[train_feat] = test_data[best_match]
                    remaining_test_features.remove(best_match)
                    logger.info(f"模糊匹配: {train_feat} -> {best_match}")
            
            # 3. 对于仍然未匹配的特征，使用零填充
            for feat in set(train_features) - set(test_features_mapped.keys()):
                logger.warning(f"未匹配特征 '{feat}' 使用零值填充")
                test_features_mapped[feat] = pd.Series(0, index=test_data.index)
            
            # 创建最终特征DataFrame，确保顺序与训练模型匹配
            X_test_aligned = pd.DataFrame({feat: test_features_mapped.get(feat, pd.Series(0, index=test_data.index)) 
                                     for feat in train_features}, index=test_data.index)
            
            # 确保数据类型正确
            X_test = X_test_aligned.values.astype(np.float32)
            y_test = test_data['target_class'].values.astype(np.int32)
            
            # 保存mid_price以备后用
            mid_price_values = test_data['mid_price'].values.copy() if 'mid_price' in test_data.columns else None
            
            # 内存优化 - 保留必要的数据
            del X_test_aligned, test_features_mapped
            gc.collect()
            logger.info(f"特征准备完成后内存使用: {process.memory_info().rss / (1024 * 1024):.2f} MB")
            
            # 模型预测
            predictions = np.zeros((len(X_test), 3))  # 3个类别的概率
            for model in self.models:
                predictions += model.predict(X_test)
            predictions /= len(self.models)
            
            # 计算混淆矩阵  
            y_pred_class = np.argmax(predictions, axis=1)
            
            # 生成3x3的混淆矩阵
            y_test_cat = pd.Categorical(y_test, categories=[0, 1, 2])
            y_pred_cat = pd.Categorical(y_pred_class, categories=[0, 1, 2])
            
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
            
            # 计算交易指标 - 使用保存的mid_price_values而不是从test_data中获取
            trading_metrics = self._calculate_trading_metrics(mid_price_values, predictions)
            
            # 添加详细的类别分布分析
            class_distribution = pd.Series(y_test).value_counts(normalize=True)
            logger.info("\n类别分布分析:")
            logger.info(class_distribution)
            
            # 添加每个类别的F1-score
            logger.info("\nDetailed Classification Report:")
            logger.info(classification_report(y_test, y_pred_class, 
                      target_names=['Down', 'Neutral', 'Up']))
            
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
            
            # 添加更多评估指标
            results.update({
                'MacroF1': f1_score(y_test, y_pred_class, average='macro'),
                'WeightedF1': f1_score(y_test, y_pred_class, average='weighted'),
                'ClassDistribution': class_distribution.to_dict(),
                'ConfidenceStats': {
                    'mean': np.mean(np.max(predictions, axis=1)),
                    'std': np.std(np.max(predictions, axis=1)),
                    'min': np.min(np.max(predictions, axis=1)),
                    'max': np.max(np.max(predictions, axis=1))
                }
            })
            
            # 添加高级交易指标
            if 'max_drawdown' in trading_metrics:
                results['max_drawdown'] = trading_metrics['max_drawdown']
            if 'sharpe_ratio' in trading_metrics:
                results['sharpe_ratio'] = trading_metrics['sharpe_ratio']
            if 'total_return' in trading_metrics:
                results['total_return'] = trading_metrics['total_return']
            
            # 生成可视化
            output_dir = "evaluation_plots"
            os.makedirs(output_dir, exist_ok=True)
            self._generate_enhanced_plots(results, large_move_analysis)
            
            return results
            
        except Exception as e:
            logger.error(f"评估过程发生错误: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _generate_enhanced_plots(self, results, large_move_analysis):
        """Generate enhanced evaluation plots"""
        output_dir = "evaluation_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Confusion Matrix Heatmap with normalized values
        plt.figure(figsize=(10, 8))
        conf_matrix = results['ConfusionMatrix'].values
        
        # Calculate normalized confusion matrix
        row_sums = conf_matrix.sum(axis=1)
        norm_conf_matrix = conf_matrix / row_sums[:, np.newaxis]
        
        # Create heatmap with two values in each cell (count and percentage)
        sns.heatmap(norm_conf_matrix, annot=False, cmap='YlOrRd',
                    xticklabels=['Down', 'Neutral', 'Up'],
                    yticklabels=['Down', 'Neutral', 'Up'])
        
        # Add text annotations with count and percentage
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                text = f"{conf_matrix[i, j]}\n({norm_conf_matrix[i, j]:.1%})"
                plt.text(j+0.5, i+0.5, text, ha='center', va='center')
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_confusion_matrix.png", dpi=300)
        plt.close()

        # 2. Class Distribution with enhanced styling
        plt.figure(figsize=(10, 6))
        class_counts = pd.Series(results['Actuals']).value_counts().sort_index()
        
        # Create DataFrame for seaborn
        df = pd.DataFrame({'Class': class_counts.index, 'Count': class_counts.values})
        ax = sns.barplot(x='Class', y='Count', data=df, palette='viridis', hue='Class', legend=False)
        
        # Calculate percentages
        total = class_counts.sum()
        percentages = class_counts / total * 100
        
        # Add percentage labels
        for i, (count, percentage) in enumerate(zip(class_counts, percentages)):
            ax.text(i, count/2, f"{percentage:.1f}%", ha='center', fontsize=12, color='white', fontweight='bold')
        
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(range(3), ['Down', 'Neutral', 'Up'], fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/02_class_distribution.png", dpi=300)
        plt.close()

        # 3. Prediction Confidence Distribution with KDE
        plt.figure(figsize=(12, 7))
        max_probs = np.max(results['Predictions'], axis=1)
        
        # Create histogram with KDE
        ax = sns.histplot(max_probs, bins=50, kde=True, color='darkblue')
        
        # Add vertical lines for quartiles
        quartiles = np.percentile(max_probs, [25, 50, 75])
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        labels = ['25th', '50th', '75th']
        
        for q, c, l in zip(quartiles, colors, labels):
            plt.axvline(x=q, color=c, linestyle='--', linewidth=2, label=f"{l}: {q:.3f}")
        
        # Add confidence threshold lines
        high_conf_threshold = 0.8
        plt.axvline(x=high_conf_threshold, color='red', linestyle='-', linewidth=2, 
                   label=f"High Conf: {(max_probs >= high_conf_threshold).mean():.1%}")
        
        # Add mean line
        plt.axvline(x=max_probs.mean(), color='green', linestyle='-', linewidth=2,
                   label=f"Mean: {max_probs.mean():.3f}")
        
        plt.title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Maximum Probability', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_prediction_confidence.png", dpi=300)
        plt.close()

        # 4. Class Accuracies with error bars
        plt.figure(figsize=(10, 6))
        accuracies = pd.Series(results['ClassAccuracies'])
        
        # Create DataFrame for seaborn
        acc_df = pd.DataFrame({'Class': accuracies.index, 'Accuracy': accuracies.values})
        ax = sns.barplot(x='Class', y='Accuracy', data=acc_df, palette='coolwarm', hue='Class', legend=False)
        
        # Calculate confidence intervals (assuming binomial distribution)
        n_samples = len(results['Actuals'])
        class_counts = pd.Series(results['Actuals']).value_counts()
        
        # Calculate standard errors
        std_errors = {}
        for cls in accuracies.index:
            p = accuracies[cls]
            n = class_counts.get(cls, 0)
            if n > 0:
                std_errors[cls] = np.sqrt(p * (1-p) / n)
            else:
                std_errors[cls] = 0
        
        std_errors = pd.Series(std_errors)
        
        # Add error bars
        for i, (idx, acc) in enumerate(accuracies.items()):
            yerr = std_errors[idx] * 1.96  # 95% confidence interval
            plt.errorbar(i, acc, yerr=yerr, fmt='none', color='black', capsize=5)
        
        # Add text labels
        for i, acc in enumerate(accuracies):
            ax.text(i, acc/2, f"{acc:.1%}", ha='center', fontsize=12, color='white', fontweight='bold')
        
        plt.title('Accuracy by Class', fontsize=14, fontweight='bold')
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xticks(range(3), ['Down', 'Neutral', 'Up'], fontsize=10)
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_class_accuracies.png", dpi=300)
        plt.close()

        # 5. Confidence vs. Accuracy Analysis
        plt.figure(figsize=(12, 8))
        
        # Calculate confidence bins and their accuracies
        bin_edges = np.linspace(0.33, 1.0, 15)  # From random (0.33) to certain (1.0)
        bin_accuracies = []
        bin_samples = []
        bin_centers = []
        
        predictions = results['Predictions']
        y_true = results['Actuals']
        
        for i in range(len(bin_edges)-1):
            lower, upper = bin_edges[i], bin_edges[i+1]
            bin_center = (lower + upper) / 2
            
            # Get max probability for each prediction
            max_probs = np.max(predictions, axis=1)
            
            # Find samples in this confidence bin
            bin_mask = (max_probs >= lower) & (max_probs < upper)
            if not any(bin_mask):
                continue
            
            # Get predictions for these samples
            bin_preds = np.argmax(predictions[bin_mask], axis=1)
            bin_true = y_true[bin_mask]
            
            # Calculate accuracy
            bin_acc = (bin_preds == bin_true).mean()
            bin_count = bin_mask.sum()
            
            bin_accuracies.append(bin_acc)
            bin_samples.append(bin_count)
            bin_centers.append(bin_center)
        
        # Create scatter plot with size proportional to sample count
        sizes = [max(20, min(1000, count / 10)) for count in bin_samples]
        
        plt.scatter(bin_centers, bin_accuracies, s=sizes, alpha=0.7, c=bin_centers, 
                   cmap='viridis', edgecolor='black')
        
        # Add perfect calibration line
        plt.plot([0.33, 1], [0.33, 1], 'r--', label='Perfect Calibration')
        
        # Add labels with sample counts
        for i, (x, y, count) in enumerate(zip(bin_centers, bin_accuracies, bin_samples)):
            plt.annotate(f"n={count}", (x, y), xytext=(5, 5), textcoords='offset points')
        
        plt.title('Confidence vs. Accuracy Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Confidence', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlim(0.33, 1.0)
        plt.ylim(0.0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_confidence_calibration.png", dpi=300)
        plt.close()

        # 6. Trading Performance
        if 'WinRate' in results and 'ProfitFactor' in results:
            plt.figure(figsize=(12, 8))
            
            metrics = ['WinRate', 'ProfitFactor', 'TradeCount']
            values = [results['WinRate'], min(5, results['ProfitFactor']), 
                     results['TradeCount'] / max(1, results['TradeCount'])]
            colors = ['#66b3ff', '#99ff99', '#ff9999']
            
            # Create primary bar plot for metrics
            ax1 = plt.subplot(111)
            bars = ax1.bar(metrics[:2], values[:2], color=colors[:2], alpha=0.7)
            
            # Add value labels
            for bar, val, metric in zip(bars, [results['WinRate'], results['ProfitFactor']], metrics[:2]):
                if metric == 'WinRate':
                    label = f"{val:.1%}"
                else:
                    label = f"{val:.2f}"
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        label, ha='center', fontsize=12)
            
            # Add secondary axis for trade count
            ax2 = ax1.twinx()
            ax2.bar(metrics[2], 1, color=colors[2], alpha=0.7)
            ax2.set_ylabel('Trade Count', fontsize=12)
            ax2.set_ylim(0, 1.2)
            ax2.text(2, 0.5, f"{results['TradeCount']}", ha='center', fontsize=12, fontweight='bold')
            
            # Add average win/loss info
            plt.figtext(0.15, 0.02, f"Avg Win: {results['AvgWin']:.6f}", ha='left', fontsize=10)
            plt.figtext(0.45, 0.02, f"Avg Loss: {results['AvgLoss']:.6f}", ha='left', fontsize=10)
            
            plt.title('Trading Performance Metrics', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Metric Value', fontsize=12)
            ax1.set_ylim(0, max(2, values[1]) * 1.2)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(f"{output_dir}/06_trading_performance.png", dpi=300)
            plt.close()
        
        logger.info(f"Enhanced evaluation plots saved to {output_dir}/ directory")

    def _generate_volatility_analysis(self, results, output_dir):
        """分析不同波动率水平下的预测表现"""
        # 计算滚动波动率
        volatility = pd.Series(results['Actuals']).pct_change().rolling(window=120).std()
        volatility_quantiles = pd.qcut(volatility, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # 计算每个波动率区间的预测准确率
        y_pred = np.argmax(results['Predictions'], axis=1)
        accuracies = []
        for label in volatility_quantiles.unique():
            mask = volatility_quantiles == label
            acc = (y_pred[mask] == results['Actuals'][mask]).mean()
            accuracies.append({'Volatility': label, 'Accuracy': acc})
        
        # 绘制波动率分析图
        plt.figure(figsize=(12, 6))
        acc_df = pd.DataFrame(accuracies)
        sns.barplot(data=acc_df, x='Volatility', y='Accuracy')
        plt.title('Prediction Accuracy by Volatility Level')
        plt.xlabel('Volatility Level')
        plt.ylabel('Accuracy')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_volatility_analysis.png")
        plt.close()
        
        # 波动率与预测置信度的关系
        plt.figure(figsize=(12, 6))
        confidence = np.max(results['Predictions'], axis=1)
        plt.scatter(volatility, confidence, alpha=0.5)
        plt.title('Prediction Confidence vs Volatility')
        plt.xlabel('Volatility')
        plt.ylabel('Prediction Confidence')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/06_volatility_confidence.png")
        plt.close()

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
    # 修改模型路径处理逻辑
    parser = argparse.ArgumentParser(description='评估LightGBM模型性能')
    parser.add_argument('--models', nargs='+', help='模型文件路径列表')
    parser.add_argument('--test_data', default='test.feather', help='测试数据路径')
    parser.add_argument('--lower', type=float, default=-0.0004, help='下跌阈值')
    parser.add_argument('--upper', type=float, default=0.0004, help='上涨阈值')
    
    args = parser.parse_args()
    
    # 确定模型路径
    if args.models:
        model_paths = args.models
    else:
        # 尝试查找默认模型文件
        possible_models = [
            'incremental_model.bin',  # 先尝试使用增量模型
            'model.bin',
            'enhanced_model.bin',
            'lgbm_model.bin'
        ]
        
        model_paths = []
        for model_name in possible_models:
            if os.path.exists(model_name):
                model_paths.append(model_name)
                logger.info(f"找到模型文件: {model_name}")
        
        if not model_paths:
            # 如果没有找到任何模型，给出明确的错误信息
            logger.error("未找到有效的模型文件！请使用--models参数指定模型路径。")
            exit(1)
    
    # 验证所有模型文件是否存在
    for path in model_paths:
        if not os.path.exists(path):
            logger.error(f"模型文件不存在: {path}")
            exit(1)
    
    logger.info(f"使用模型: {model_paths}")
    
    # 创建评估器
    evaluator = ModelEvaluator(
        model_paths, 
        test_data_path=args.test_data,
        lower_threshold=args.lower, 
        upper_threshold=args.upper
    )
    
    # 执行评估
    results = evaluator.evaluate()
    
    if results:
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
        
        # 添加增强的评估指标输出
        logger.info(f"\n增强评估指标:")
        logger.info(f"Macro F1: {results['MacroF1']:.4f}")
        logger.info(f"Weighted F1: {results['WeightedF1']:.4f}")
        
        if 'max_drawdown' in results:
            logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
            
        if 'sharpe_ratio' in results:
            logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
            
        if 'total_return' in results:
            logger.info(f"总收益率: {results['total_return']:.2%}")
    else:
        logger.error("评估失败，请检查错误日志。")
