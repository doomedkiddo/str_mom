import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from deap import base, creator, tools, algorithms
import shap
from loguru import logger
import random
import ast
from sklearn.metrics import f1_score
import time
import wandb
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import matplotlib.pyplot as plt
import tempfile
import os

os.environ['OMP_NUM_THREADS'] = '1'  # 避免LightGBM的并行冲突
os.environ['WANDB_SILENT'] = 'true'  # 减少wandb的线程操作

logger.add("genetic_factor.log", rotation="100 MB")

# 在类外部定义并行映射函数
def _parallel_map_wrapper(func, iterable):
    """独立的并行映射函数"""
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        return list(executor.map(func, iterable))

def evaluate_individual_wrapper(args):
    """独立的评估函数"""
    individual, data_dict, target_array, base_features, use_gpu = args
    try:
        # 评估逻辑移到这里，避免依赖类实例
        factor_values = evaluate_expression(individual, data_dict)
        factor_values = np.asarray(factor_values, dtype=np.float64)
        
        if np.isnan(factor_values).any() or np.isinf(factor_values).any():
            return (0,)
        
        factor_values = (factor_values - factor_values.mean()) / (factor_values.std() + 1e-10)
        
        X = np.column_stack([
            factor_values,
            data_dict[base_features[0]],
            data_dict[base_features[1]]
        ])
        
        n_samples = len(X)
        n_splits = 3
        scores = []
        
        for i in range(n_splits):
            split_point = int(n_samples * (i + 1) / (n_splits + 1))
            X_train = X[:split_point]
            X_test = X[split_point:int(split_point * 1.5)]
            y_train = target_array[:split_point]
            y_test = target_array[split_point:int(split_point * 1.5)]
            
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                device='gpu' if use_gpu else 'cpu',
                gpu_platform_id=0,
                gpu_device_id=0
            )
            model.fit(X_train, y_train)
            
            preds = model.predict(X_test)
            score = f1_score(y_test, preds, average='weighted')
            scores.append(score)
        
        avg_score = np.mean(scores)
        complexity = len(str(individual))
        fitness = avg_score * (1 - 0.1 * np.log1p(complexity))
        
        return (fitness,)
    except Exception as e:
        logger.error(f"个体评估失败: {str(e)}")
        return (0,)

def evaluate_expression(expr, data_dict):
    """独立的表达式评估函数"""
    try:
        if isinstance(expr, list):
            expr = tuple(expr)
        
        if isinstance(expr, str):
            return data_dict[expr]
        
        op = expr[0]
        
        if op in ['add', 'sub', 'mul', 'div']:
            if len(expr) != 3:
                return np.zeros(len(next(iter(data_dict.values()))))
            a = evaluate_expression(expr[1], data_dict)
            b = evaluate_expression(expr[2], data_dict)
            
            if op == 'add':
                result = a + b
            elif op == 'sub':
                result = a - b
            elif op == 'mul':
                result = a * b
            elif op == 'div':
                result = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
                
        elif op in ['sqrt', 'log', 'zscore']:
            if len(expr) != 2:
                return np.zeros(len(next(iter(data_dict.values()))))
            x = evaluate_expression(expr[1], data_dict)
            
            if op == 'sqrt':
                result = np.sqrt(np.abs(x))
            elif op == 'log':
                result = np.log(np.abs(x) + 1e-10)
            elif op == 'zscore':
                result = (x - x.mean()) / (x.std() + 1e-10)
        else:
            return np.zeros(len(next(iter(data_dict.values()))))
        
        return result.astype(np.float64)
            
    except Exception as e:
        logger.error(f"表达式执行失败: {expr}, 错误: {str(e)}")
        return np.zeros(len(next(iter(data_dict.values()))), dtype=np.float64)

class GeneticFactorMiner:
    def __init__(self, data, target, population_size=50, generations=10, 
                 cx_prob=0.7, mut_prob=0.3, elite_size=10, 
                 wandb_project="genetic-factor-miner", wandb_config=None,
                 base_features=['price_momentum', 'volatility_ratio'],
                 use_gpu=True):
        """
        初始化遗传算法参数
        :param data: 输入数据 (DataFrame)
        :param target: 目标变量 (Series)
        :param population_size: 种群大小
        :param generations: 迭代代数
        :param cx_prob: 交叉概率
        :param mut_prob: 变异概率
        :param elite_size: 精英保留数量
        :param wandb_project: wandb项目名称
        :param wandb_config: wandb配置字典
        :param base_features: 基础特征列表
        :param use_gpu: 是否使用GPU
        """
        self.data = data
        self.target = target
        self.pop_size = population_size
        self.generations = generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.elite_size = elite_size
        
        # 定义遗传算法基本元素
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self._setup_genetic_operators()
        
        # SHAP解释器初始化
        self.explainer = None
        self.shap_values = None
        
        # 初始化wandb
        wandb.init(
            project=wandb_project,
            config=wandb_config or {
                "population_size": population_size,
                "generations": generations,
                "crossover_prob": cx_prob,
                "mutation_prob": mut_prob,
                "elite_size": elite_size,
                "use_gpu": use_gpu
            }
        )
        self.wandb = wandb
        
        # 添加基础特征配置
        self.base_features = base_features
        if len(self.base_features) < 2:
            raise ValueError("需要至少两个基础特征")
        
        self.use_gpu = use_gpu
        self.n_jobs = multiprocessing.cpu_count()  # 自动检测CPU核心数
        
        # 将数据转换为numpy数组，避免DataFrame序列化问题
        self.data_array = {col: data[col].values for col in data.columns}
        self.target_array = target.values
        self.column_names = data.columns.tolist()
    
    def _setup_genetic_operators(self):
        """配置遗传算法操作符"""
        # 定义原子操作
        self.toolbox.register("expr", self._generate_random_expression, min_depth=2, max_depth=4)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册遗传操作
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._cx_expression)
        self.toolbox.register("mutate", self._mut_expression)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _generate_random_expression(self, min_depth=2, max_depth=4):
        """
        生成随机因子表达式(使用表达式树)
        示例因子形式: (feature1 + (feature2 * feature3)) / (feature4 - feature5)
        """
        # 基础特征池 - 使用实际可用的特征
        base_features = [
            'price_momentum', 'volatility_ratio', 'depth_imbalance',
            'liquidity_imbalance', 'ofi', 'vpin', 'relative_spread',
            'bid_ask_slope', 'order_book_pressure', 'weighted_price_depth',
            'flow_toxicity', 'pressure_change_rate', 'orderbook_gradient',
            'depth_pressure_ratio'
        ]
        
        # 运算符集合
        operators = [
            ('add', lambda a, b: a + b),
            ('sub', lambda a, b: a - b),
            ('mul', lambda a, b: a * b),
            ('div', lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b!=0)),
            ('sqrt', lambda a: np.sqrt(np.abs(a))),
            ('log', lambda a: np.log(np.abs(a) + 1e-10)),
            ('zscore', lambda a: (a - a.mean()) / (a.std() + 1e-10))
        ]
        
        def build_expr(depth=0):
            if depth >= max_depth or (depth >= min_depth and random.random() < 0.5):
                return random.choice(base_features)
            
            op = random.choice(operators)
            if op[0] in ['sqrt', 'log', 'zscore']:  # 一元运算符
                return (op[0], build_expr(depth+1))
            else:  # 二元运算符
                return (op[0], build_expr(depth+1), build_expr(depth+1))
        
        def validate_expression(expr):
            """验证生成的表达式结构"""
            if isinstance(expr, str):
                return True
            op = expr[0]
            if op in ['add', 'sub', 'mul', 'div']:
                return len(expr) == 3 and validate_expression(expr[1]) and validate_expression(expr[2])
            elif op in ['sqrt', 'log', 'zscore']:
                return len(expr) == 2 and validate_expression(expr[1])
            return False

        expr = None
        while not expr or not validate_expression(expr):
            expr = build_expr()
        
        return expr
    
    def _evaluate_individual(self, individual):
        """评估个体适应度的包装方法"""
        return evaluate_individual_wrapper((
            individual,
            self.data_array,
            self.target_array,
            self.base_features,
            self.use_gpu
        ))
    
    def _cx_expression(self, ind1, ind2):
        """交叉操作：子树交叉"""
        try:
            new_ind1, new_ind2 = tools.cxOnePoint(ind1, ind2)
            del ind1.fitness.values
            del ind2.fitness.values
            return new_ind1, new_ind2
        except Exception as e:
            logger.warning(f"交叉操作失败: {str(e)}")
            return ind1, ind2
    
    def _mut_expression(self, individual):
        """变异操作：随机替换子树"""
        try:
            index = random.randrange(len(individual))
            individual[index] = self._generate_random_expression(min_depth=1, max_depth=2)
            del individual.fitness.values
            return individual,
        except Exception as e:
            logger.warning(f"变异操作失败: {str(e)}")
            return individual,
    
    def _shap_analysis(self, best_factors):
        """执行SHAP分析"""
        # 生成最佳因子集
        X = pd.DataFrame()
        for i, factor in enumerate(best_factors):
            X[f'factor_{i}'] = self._evaluate_expression(factor)
        
        # 训练最终模型
        model = lgb.LGBMClassifier(
            n_estimators=200, 
            num_leaves=63, 
            learning_rate=0.05,
            device='gpu' if self.use_gpu else 'cpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            n_jobs=self.n_jobs
        )
        model.fit(X, self.target)
        
        # 计算SHAP值
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X)
        
        # 可视化分析
        shap.summary_plot(self.shap_values, X, plot_type="bar")
        shap.summary_plot(self.shap_values[1], X)
        
        # 返回重要因子索引
        return np.argsort(np.abs(self.shap_values[1]).mean(axis=0))[::-1]
    
    def _validate_data(self):
        """验证输入数据的有效性"""
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError("输入数据必须是pandas DataFrame类型")
        
        if not isinstance(self.target, pd.Series):
            raise TypeError("目标变量必须是pandas Series类型")
        
        if len(self.data) != len(self.target):
            raise ValueError("输入数据和目标变量长度不匹配")
        
        if self.data.isnull().any().any():
            logger.warning("输入数据包含缺失值")
    
    def _log_shap_analysis(self, important_idx):
        """记录SHAP分析结果到wandb"""
        # 创建重要性表格
        importance_table = self.wandb.Table(columns=["Rank", "Factor_Index", "SHAP_Value"])
        
        for rank, idx in enumerate(important_idx):
            importance_table.add_data(rank+1, idx, 
                                     np.abs(self.shap_values[1]).mean(axis=0)[idx])
        
        # 创建SHAP图并保存
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            shap.summary_plot(self.shap_values[1], X, show=False)
            plt.savefig(tmp.name)
            plt.close()
            
            # 记录表格和图像
            self.wandb.log({
                "SHAP Importance Ranking": importance_table,
                "SHAP Summary Plot": self.wandb.Image(tmp.name)
            })
            
            # 删除临时文件
            os.unlink(tmp.name)
    
    def run(self):
        """执行遗传算法优化"""
        try:
            pop = self.toolbox.population(n=self.pop_size)
            hof = tools.HallOfFame(self.elite_size)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # 使用map函数直接处理
            self.toolbox.register("map", map)  # 使用内置map函数
            
            if self.n_jobs > 1:
                # 创建进程池
                with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                    def parallel_evaluate(individuals):
                        args_list = [(ind, self.data_array, self.target_array, 
                                    self.base_features, self.use_gpu) for ind in individuals]
                        return list(executor.map(evaluate_individual_wrapper, args_list))
                    
                    # 替换评估函数
                    original_evaluate = self.toolbox.evaluate
                    self.toolbox.evaluate = lambda ind: evaluate_individual_wrapper((
                        ind, self.data_array, self.target_array, self.base_features, self.use_gpu
                    ))
                    
                    # 执行进化
                    best_factors, important_idx, results = self._run_evolution(pop, hof, stats)
                    
                    # 恢复原始评估函数
                    self.toolbox.evaluate = original_evaluate
            else:
                best_factors, important_idx, results = self._run_evolution(pop, hof, stats)
            
            return best_factors, important_idx, results
            
        except Exception as e:
            logger.error(f"遗传算法运行失败: {str(e)}")
            raise

    def _run_evolution(self, pop, hof, stats):
        """执行进化过程"""
        # 记录开始时间
        start_time = time.time()
        
        # 进化循环
        for gen in range(self.generations):
            pop, log = algorithms.eaSimple(
                pop, self.toolbox, cxpb=self.cx_prob, mutpb=self.mut_prob,
                ngen=1, stats=stats, halloffame=hof, verbose=True
            )
            
            # 记录到wandb
            self.wandb.log({
                "generation": gen,
                "avg_fitness": log[-1]['avg'],
                "max_fitness": log[-1]['max'],
                "min_fitness": log[-1]['min']
            })
        
        # 记录运行时间
        run_time = time.time() - start_time
        logger.info(f"遗传算法运行完成，耗时: {run_time:.2f}秒")
        
        # 提取最佳因子
        best_factors = [ind for ind in hof]
        logger.info(f"找到最佳因子: {best_factors}")
        
        # SHAP分析
        important_idx = self._shap_analysis(best_factors)
        logger.info(f"SHAP分析重要因子排序: {important_idx}")
        
        # 保存运行日志和统计信息
        results = {
            'best_factors': best_factors,
            'important_indices': important_idx,
            'run_time': run_time,
            'stats_log': log
        }
        
        # 记录最终结果
        self.wandb.log({
            "final_max_fitness": results['stats_log'][-1]['max'],
            "run_time": results['run_time'],
            "num_factors": len(results['best_factors'])
        })
        
        # 记录最佳因子
        factors_table = self.wandb.Table(columns=["Rank", "Factor"])
        for i, factor in enumerate(results['best_factors']):
            factors_table.add_data(i+1, str(factor))
        self.wandb.log({"Best Factors": factors_table})
        
        # 记录SHAP分析
        self._log_shap_analysis(results['important_indices'])
        
        return best_factors, important_idx, results

# 使用示例
if __name__ == "__main__":
    try:
        # 加载预处理好的高频数据
        logger.info("正在加载数据...")
        data = pd.read_feather('train.feather')
        
        # 检查数据列
        logger.info(f"数据列: {data.columns.tolist()}")
        
        # 构造目标变量：基于未来价格变动的方向
        logger.info("构造目标变量...")
        future_returns = data['mid_price'].shift(-1) / data['mid_price'] - 1
        target = pd.Series(0, index=data.index)  # 默认为0（持平）
        target[future_returns > 0.0001] = 1      # 上涨
        target[future_returns < -0.0001] = -1    # 下跌
        
        # 删除最后一行（因为无法计算未来收益）
        target = target[:-1]
        features = data[:-1].copy()
        
        # 移除不需要的列
        features = features.drop(columns=['index', 'mid_price'])
            
        logger.info(f"数据准备完成。特征数量: {features.shape[1]}, 样本数量: {features.shape[0]}")
        logger.info(f"目标变量分布:\n{target.value_counts(normalize=True)}")
        
        # 初始化wandb配置
        wandb_config = {
            "data_source": "train.feather",
            "target_type": "三分类(-1,0,1)",
            "feature_columns": features.columns.tolist(),
            "n_samples": len(features),
            "threshold": 0.0001
        }
        
        # 初始化因子挖掘器
        logger.info("初始化因子挖掘器...")
        miner = GeneticFactorMiner(
            data=features,
            target=target,
            population_size=30,
            generations=15,
            cx_prob=0.6,
            mut_prob=0.4,
            wandb_project="高频因子挖掘",
            wandb_config=wandb_config,
            use_gpu=True
        )
        
        # 运行挖掘
        logger.info("开始因子挖掘...")
        best_factors, shap_rank, results = miner.run()
        
        # 保存最佳因子
        output_file = 'best_factors.txt'
        logger.info(f"保存最佳因子到 {output_file}")
        with open(output_file, 'w') as f:
            f.write("目标变量构造方法：\n")
            f.write("1: 未来收益率 > 0.0001\n")
            f.write("0: -0.0001 <= 未来收益率 <= 0.0001\n")
            f.write("-1: 未来收益率 < -0.0001\n\n")
            f.write("最佳因子：\n")
            for i, factor in enumerate(best_factors):
                f.write(f"Factor {i+1}: {str(factor)}\n")
        
        logger.info("因子挖掘完成！")
        
    except FileNotFoundError:
        logger.error("未找到数据文件 train.feather")
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
    finally:
        wandb.finish() 