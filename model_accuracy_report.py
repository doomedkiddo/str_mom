from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
import pandas as pd
import lightgbm as lgb
from loguru import logger
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ModelAccuracyReporter:
    def __init__(self, model_paths, test_data_path='test.feather', model_type='up'):
        """
        :param model_type: 'up'代表上涨模型，'down'代表下跌模型
        """
        self.models = [lgb.Booster(model_file=path) for path in model_paths]
        self.test_data_path = test_data_path
        self.model_type = model_type
        self.wandb = wandb
        
    def _load_data(self):
        """加载并预处理数据（添加缺失特征计算）"""
        data = pd.read_feather(self.test_data_path)
        
        # 添加缺失特征计算
        if 'weighted_price_depth' not in data.columns:
            # 计算加权价格深度（示例计算，根据实际业务调整）
            data['weighted_price_depth'] = (data['bid_price1'] * data['bid_volume1'] + 
                                          data['ask_price1'] * data['ask_volume1']) / 2
        
        if 'liquidity_imbalance' not in data.columns:
            # 计算流动性失衡指标
            data['liquidity_imbalance'] = (data['bid_volume1'] - data['ask_volume1']) / (
                data['bid_volume1'] + data['ask_volume1'] + 1e-6)
        
        # 创建目标变量
        prediction_points = 120
        data['price_change'] = data['mid_price'].shift(-prediction_points) - data['mid_price']
        
        if self.model_type == 'up':
            data['target'] = (data['price_change'] >= 0.0004).astype(int)
        else:
            data['target'] = (data['price_change'] <= -0.0004).astype(int)
            
        return data.dropna(subset=['price_change'])

    def _get_features(self):
        """获取模型特征（复用原有逻辑）"""
        model = self.models[0]
        features = model.feature_name()
        if not features:
            features = [f'f{i}' for i in range(model.num_feature())]
        return features

    def _predict(self, data):
        """集成预测"""
        features = self._get_features()
        
        # 处理缺失特征 - 使用更高效的方式
        missing = set(features) - set(data.columns)
        if missing:
            # 一次性添加所有缺失特征列
            missing_features = list(missing)
            zero_data = pd.DataFrame(0, index=data.index, columns=missing_features)
            data = pd.concat([data, zero_data], axis=1)
            # 创建去碎片化的副本
            data = data.copy()
            
        preds = []
        for model in self.models:
            try:
                preds.append(model.predict(data[features]))
            except Exception as e:
                logger.error(f"预测失败: {str(e)}")
                
        return np.mean(preds, axis=0) if preds else None

    def generate_report(self, output_file='model_accuracy.txt'):
        """生成简化版评估报告"""
        data = self._load_data()
        predictions = self._predict(data)
        
        if predictions is None:
            logger.error("无法生成预测结果")
            return

        # 只计算核心指标
        y_pred = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(data['target'], y_pred)
        confusion = confusion_matrix(data['target'], y_pred)
        
        # 生成核心报告
        report = {
            'accuracy': accuracy,
            'true_positive': confusion[1, 1],  # 正确预测大涨/大跌的数量
            'true_negative': confusion[0, 0],  # 正确预测非大涨/大跌的数量
            'total_samples': len(data)
        }

        # 生成核心图表
        self._generate_simple_plots(data['target'], predictions, report)
        
        # 写入简化报告
        with open(output_file, 'w') as f:
            f.write(f"=== {self.model_type.upper()}模型核心指标 ===\n\n")
            f.write(f"正确率: {report['accuracy']:.4f}\n")
            f.write(f"正确预测目标事件次数: {report['true_positive']}/{report['total_samples']}\n")
            f.write(f"正确预测非目标事件次数: {report['true_negative']}/{report['total_samples']}\n")
        
        # 记录到wandb
        wandb.log({
            f"{self.model_type}_accuracy": report['accuracy'],
            f"{self.model_type}_true_positive": report['true_positive'],
            f"{self.model_type}_true_negative": report['true_negative']
        })
        
        logger.success(f"简化报告已生成: {output_file}")
        return report

    def _generate_simple_plots(self, y_true, y_pred_prob, report):
        """生成核心可视化图表"""
        plt.figure(figsize=(12, 5))
        
        # 正确率饼图
        plt.subplot(1, 2, 1)
        plt.pie([report['accuracy'], 1-report['accuracy']], 
               labels=['正确预测', '错误预测'],
               autopct='%1.1f%%')
        plt.title('整体正确率分布')
        
        # 事件预测分布柱状图
        plt.subplot(1, 2, 2)
        sns.barplot(x=['目标事件', '非目标事件'], 
                  y=[report['true_positive'], report['true_negative']])
        plt.title('正确预测事件分布')
        plt.ylabel('数量')
        
        plt.tight_layout()
        plt.savefig(f'{self.model_type}_core_metrics.png')
        plt.close()
        
        # 上传图表到wandb
        wandb.log({f"{self.model_type}_core_plots": wandb.Image(f'{self.model_type}_core_metrics.png')})

if __name__ == "__main__":
    # 示例用法 - 分别运行两次
    # 上涨模型评估
    wandb.init(project="hft-accuracy-report", config={"model_type": "up"})
    up_reporter = ModelAccuracyReporter(
        model_paths=['model_up_v0.bin', 'model_up_v1.bin', 'model_up_v2.bin'],
        model_type='up'
    )
    up_report = up_reporter.generate_report('up_model_accuracy.txt')
    
    # 下跌模型评估 
    wandb.init(project="hft-accuracy-report", config={"model_type": "down"})
    down_reporter = ModelAccuracyReporter(
        model_paths=['model_down_v0.bin', 'model_down_v1.bin', 'model_down_v2.bin'],
        model_type='down'
    )
    down_report = down_reporter.generate_report('down_model_accuracy.txt') 