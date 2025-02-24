import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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
import matplotlib.pyplot as plt
import tempfile
import os
from functools import lru_cache
import ray
import sys
import ctypes

os.environ['OMP_NUM_THREADS'] = '1'  # 避免LightGBM的并行冲突
os.environ['WANDB_SILENT'] = 'true'  # 减少wandb的线程操作

logger.add("genetic_factor.debug.log", rotation="100 MB", level="DEBUG")  # 保存所有日志
logger.add("genetic_factor.info.log", rotation="100 MB", level="INFO")    # 仅保存INFO及以上
logger.add(sys.stdout, level="INFO")  # 控制台只输出INFO及以上
logger.add("genetic_factor.error.log", rotation="100 MB", level="ERROR")  # 单独记录错误日志

ray.init()

@ray.remote
def evaluate_remote(args):
    return evaluate_individual_wrapper(args)

def _parallel_map_wrapper(func, iterable):
    futures = [evaluate_remote.remote(arg) for arg in iterable]
    return ray.get(futures)

def evaluate_individual_wrapper(args):
    """独立的评估函数"""
    individual, data_dict, target_array, base_features, use_gpu = args
    try:
        logger.debug(f"开始评估因子: {str(individual)[:50]}...")
        
        # 将Individual对象转换为元组
        if hasattr(individual, '__iter__'):
            expr = tuple(individual)
        else:
            expr = individual
        
        # 修改数据哈希方式
        hashable_data = []
        try:
            for k, v in data_dict.items():
                if isinstance(v, np.ndarray):
                    # 使用数组的内存视图和属性创建可哈希的元组
                    hashable_data.append((str(k), v.tobytes(), v.shape, v.dtype.str))
            
            # 转换为元组确保可哈希
            data_tuple = tuple(sorted(hashable_data))
            
            # 评估因子
            factor_values = evaluate_expression(expr, data_tuple)
            
        except Exception as e:
            logger.error(f"数据处理失败: {str(e)}")
            return (0, 0, 0, 0)
            
        if factor_values is None:
            return (0, 0, 0, 0)
            
        factor_values = np.asarray(factor_values, dtype=np.float64)
        
        # 异常值处理
        if np.isnan(factor_values).any() or np.isinf(factor_values).any():
            logger.warning(f"⚠️ 因子值异常: 包含NaN/Inf - {str(individual)[:50]}...")
            return (0, 0, 0, 0)
            
        # 向量化信号生成
        lookback = 60  # 使用最近60分钟的滚动窗口
        series = pd.Series(factor_values)
        rolling_high = series.rolling(lookback, min_periods=1).quantile(0.9)
        rolling_low = series.rolling(lookback, min_periods=1).quantile(0.1)
        
        # 向量化信号生成
        signal = np.select(
            [factor_values > rolling_high, factor_values < rolling_low],
            [1, -1],
            default=0
        )
        
        # 向量化收益计算 - 移除手续费计算
        returns = data_dict.get('return_1min', np.zeros_like(signal)) * signal
        net_returns = returns  # 直接使用returns，不再减去手续费
        
        # 简化指标计算
        total_return = np.exp(np.sum(np.log1p(net_returns))) - 1  # 更稳定的累计收益计算
        risk_free_rate = 0.02  # 年化2%
        excess_returns = net_returns - risk_free_rate/(252*24*60)
        valid_returns = net_returns[np.isfinite(net_returns)]  # 过滤无效值

        if len(valid_returns) < 2:  # 最少需要2个样本计算波动率
            sharpe_ratio = 0.0
        else:
            std_dev = np.std(valid_returns)
            if std_dev < 1e-8:
                std_dev = 1e-8
            sharpe_ratio = np.mean(excess_returns) / std_dev * np.sqrt(252*24*60)
        win_rate = np.mean(net_returns > 0)          # 胜率
        max_drawdown = (np.maximum.accumulate(1 + net_returns) - (1 + net_returns)).max() # 最大回撤

        # 打印关键指标
        logger.info(
            f"\n📈 因子表现 [{str(individual)[:30]}...]\n"
            f"▸ 夏普比率: {sharpe_ratio:.2f}\n"
            f"▸ 累计收益: {total_return:.2%}\n"
            f"▸ 胜率: {win_rate:.2%}\n"
            f"▸ 最大回撤: {max_drawdown:.2%}"
        )
        
        # 在返回前添加最终清理
        factor_values = np.nan_to_num(factor_values, nan=0.0, posinf=0.0, neginf=0.0)
        factor_values = np.clip(factor_values, -1e10, 1e10)  # 防止极端值
        
        return (sharpe_ratio, total_return, win_rate, max_drawdown)
        
    except Exception as e:
        logger.error(f"因子评估失败: {str(e)}")
        return (0, 0, 0, 0)

@lru_cache(maxsize=1000)
def evaluate_expression(expr, data_dict_hash):
    """增强型表达式评估函数"""
    try:
        # 重建数据字典
        data_dict = {}
        if isinstance(data_dict_hash, (list, tuple)):  # 添加类型检查
            for k, data_bytes, shape, dtype_str in data_dict_hash:
                # 从字节重建数组
                data = np.frombuffer(data_bytes, dtype=np.dtype(dtype_str)).reshape(shape)
                data_dict[k] = data
        else:
            logger.error(f"无效的数据字典哈希类型: {type(data_dict_hash)}")
            return None
            
        # 处理基本特征直接返回
        if isinstance(expr, str):
            if expr in data_dict:
                return data_dict[expr]
            logger.warning(f"特征名 {expr} 不在数据字典中")
            return None
            
        # 确保expr是元组
        if not isinstance(expr, (list, tuple)):
            logger.warning(f"无效的表达式类型: {type(expr)}")
            return None
            
        # 将list转换为tuple以确保可哈希
        if isinstance(expr, list):
            expr = tuple(expr)
            
        op = expr[0]
        
        # 处理滚动窗口操作（增加参数验证和嵌套处理）
        if op == 'rolling':
            if len(expr) != 4:
                return None
            _, operation, feature_expr, window = expr
            
            # 先评估特征表达式
            feature_values = evaluate_expression(feature_expr, data_dict_hash)
            if feature_values is None:
                return None
                
            # 验证window参数
            if isinstance(window, tuple):  # 如果window是表达式
                window_value = evaluate_expression(window, data_dict_hash)
                if window_value is None:
                    return None
                # 取window表达式结果的均值作为窗口大小
                window = int(np.mean(window_value))
            
            # 确保window是有效的正整数
            try:
                window = int(window)
                if window <= 0:
                    logger.warning(f"无效的窗口大小: {window}，使用默认值5")
                    window = 5
            except (TypeError, ValueError):
                logger.warning(f"无效的窗口参数类型: {type(window)}，使用默认值5")
                window = 5
            
            # 验证operation参数
            valid_operations = ['mean', 'std', 'max', 'min']
            if operation not in valid_operations:
                logger.warning(f"无效的滚动操作: {operation}，使用mean")
                operation = 'mean'
            
            try:
                # 使用pandas的rolling操作
                series = pd.Series(feature_values)
                result = series.rolling(window, min_periods=1).agg(operation)
                # 填充开始的NaN值
                result = result.fillna(method='bfill').fillna(method='ffill')
                return result.values
            except Exception as e:
                logger.error(f"滚动计算失败: {operation} window={window}, 错误: {str(e)}")
                return None

        # 处理二元运算（增加空值检查和长度对齐）
        elif op in ['add', 'sub', 'mul', 'div', 'corr', 'cov', 'ratio', 'residual']:
            if len(expr) != 3:
                return None
            a = evaluate_expression(expr[1], data_dict_hash)
            b = evaluate_expression(expr[2], data_dict_hash)
            
            # 新增空值检查
            if a is None or b is None:
                return None
                
            # 统一数组长度（取最小值）
            min_length = min(len(a), len(b))
            a = a[:min_length]
            b = b[:min_length]
            
            # 执行运算
            if op == 'add':
                return a + b
            elif op == 'sub':
                return a - b
            elif op == 'mul':
                return a * b
            elif op == 'div':
                return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
            # ... [其他二元运算处理保持不变]

        # 处理一元运算（增加空值检查和长度处理）
        elif op in ['sqrt', 'log', 'zscore', 'delta', 'lag']:
            if len(expr) != 2:
                return None
            x = evaluate_expression(expr[1], data_dict_hash)
            
            # 新增空值检查
            if x is None:
                return None
                
            x_length = len(x)
            if x_length == 0:
                return None
                
            try:
                if op == 'sqrt':
                    return np.sqrt(np.abs(x))  # 防止负值
                elif op == 'log':
                    return np.log(np.abs(x) + 1e-8)  # 防止零值
                elif op == 'zscore':
                    return (x - np.mean(x)) / (np.std(x) + 1e-8)
                elif op == 'delta':
                    return np.diff(x, prepend=x[0])
                elif op == 'lag':
                    return np.roll(x, shift=1)  # 改为向量化操作
            except Exception as e:
                logger.error(f"{op}运算失败: {str(e)}")
                return None

    except Exception as e:
        logger.error(f"表达式执行失败: {expr}, 错误: {str(e)}")
        return None

def visualize_expression(expr, level=0):
    """可视化表达式树结构"""
    if isinstance(expr, str):
        return "  " * level + expr + "\n"
    if isinstance(expr, tuple):
        result = "  " * level + expr[0] + "\n"
        for e in expr[1:]:
            result += visualize_expression(e, level+1)
        return result
    return "  " * level + "Invalid Node\n"

class GeneticFactorMiner:
    def __init__(self, data, target, population_size=50, generations=10, 
                 cx_prob=0.7, mut_prob=0.3, elite_size=10, 
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
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self._setup_genetic_operators()
        
        # SHAP解释器初始化
        self.explainer = None
        self.shap_values = None
        
        self.use_gpu = use_gpu
        self.n_jobs = multiprocessing.cpu_count()  # 自动检测CPU核心数
        
        # 将数据转换为numpy数组，避免DataFrame序列化问题
        self.data_array = {col: data[col].values for col in data.columns}
        self.target_array = target.values
        self.column_names = data.columns.tolist()
        
        # 添加这一行来保存base_features参数
        self.base_features = base_features  # 新增行
        
        # 验证数据包含所有基础特征
        missing = [f for f in self.base_features if f not in data.columns]
        if missing:
            raise ValueError(f"数据缺失必需的基础特征: {missing}")
        
        # 验证数据格式
        for col, values in self.data_array.items():
            if not isinstance(values, np.ndarray):
                raise TypeError(f"列 {col} 的数据类型应为numpy数组，实际为 {type(values)}")
            if len(values) != len(target):
                raise ValueError(f"列 {col} 的长度与目标变量不匹配")
    
    def _setup_genetic_operators(self):
        """配置遗传算法操作符"""
        # 修改个体生成方式，确保直接返回元组
        self.toolbox.register("expr", self._generate_random_expression, min_depth=2, max_depth=4)
        # 不再将表达式包装为元组，因为_generate_random_expression已经返回元组
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # 注册遗传操作
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._cx_expression)
        self.toolbox.register("mutate", self._mut_expression)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def _build_expr(self, min_depth, max_depth, depth=0):
        """构建随机因子表达式（确保返回元组，限制最大嵌套深度为7）"""
        try:
            # 当达到最大深度或超过7层时直接返回特征字符串
            if depth >= max_depth or depth >= 7:  # 添加深度限制
                return random.choice(self.base_features)
            
            # 根据当前深度调整操作符的选择概率
            if depth >= 5:  # 当深度较大时，增加返回基础特征的概率
                if random.random() < 0.6:  # 60%概率直接返回基础特征
                    return random.choice(self.base_features)
            
            op = random.choice([
                'add', 'sub', 'mul', 'div', 'sqrt', 'log',
                'zscore', 'delta', 'lag', 'corr', 'cov',
                'ratio', 'residual', 'rolling'
            ])
            
            # 确保操作符结构正确并返回元组
            if op == 'rolling':
                return tuple([op, 
                        random.choice(['mean', 'std', 'max', 'min']),
                        random.choice(self.base_features),  # rolling操作直接使用基础特征
                        random.randint(5, 60)])
            elif op in ['add', 'sub', 'mul', 'div', 'corr', 'cov', 'ratio', 'residual']:
                return tuple([op, 
                        self._build_expr(min_depth, max_depth, depth+1),
                        self._build_expr(min_depth, max_depth, depth+1)])
            elif op in ['sqrt', 'log', 'zscore', 'delta', 'lag']:
                return tuple([op, 
                        self._build_expr(min_depth, max_depth, depth+1)])
            else:
                return random.choice(self.base_features)
            
        except Exception as e:
            logger.error(f"构建表达式失败: {str(e)}")
            return random.choice(self.base_features)

    def _generate_random_expression(self, min_depth=2, max_depth=3):
        """生成随机因子表达式（确保不超过7层）"""
        attempt = 0
        max_attempts = 5
        max_depth = min(max_depth, 7)  # 确保max_depth不超过7
        
        while attempt < max_attempts:
            attempt += 1
            try:
                expr = self._build_expr(min_depth, max_depth)
                if self._validate_expression(expr) and self._get_depth(expr) <= 7:
                    return expr
                elif attempt == max_attempts:
                    logger.warning("生成的表达式深度超过7层，返回简单表达式")
                    return random.choice(self.base_features)
            except RecursionError:
                logger.warning("递归深度过大，尝试生成更简单表达式")
                return random.choice(self.base_features)
            except Exception as e:
                logger.warning(f"生成表达式出错: {str(e)}")
        
        base = random.choice(self.base_features)
        logger.info(f"达到最大尝试次数，返回基础特征: {base}")
        return base

    def _validate_expression(self, expr):
        """验证表达式结构（防止特征名被拆分）"""
        if isinstance(expr, str):
            return expr in self.base_features
        
        if not isinstance(expr, tuple):
            return False
        
        # 检查操作符结构
        if expr[0] == 'rolling':
            return (len(expr) == 4 and 
                   expr[1] in ['mean', 'std', 'max', 'min'] and
                   isinstance(expr[2], str) and  # 特征名必须是字符串
                   expr[2] in self.base_features)
        elif expr[0] in ['add', 'sub', 'mul', 'div', 'corr', 'cov', 'ratio', 'residual']:
            return len(expr) == 3 and all(self._validate_expression(e) for e in expr[1:])
        elif expr[0] in ['sqrt', 'log', 'zscore', 'delta', 'lag']:
            return len(expr) == 2 and self._validate_expression(expr[1])
        return False

    def _get_depth(self, expr):
        """计算表达式深度"""
        if isinstance(expr, str):
            return 1
        return 1 + max(self._get_depth(e) for e in expr[1:])
    
    def _evaluate_individual(self, individual):
        """评估前增加结构校验"""
        # 将Individual对象转换为普通元组
        if hasattr(individual, '__iter__'):
            expr = tuple(individual)
        else:
            expr = individual
        
        if not self._validate_expression(expr):
            logger.warning(f"非法个体结构: {expr}")
            return (0, 0, 0, 0)
        return evaluate_individual_wrapper((
            expr,  # 传递转换后的元组
            self.data_array,
            self.target_array,
            self.base_features,
            self.use_gpu
        ))
    
    def _cx_expression(self, ind1, ind2):
        """增强型交叉操作"""
        try:
            # 增加类型检查
            if not (isinstance(ind1, (list, tuple)) and isinstance(ind2, (list, tuple))):
                return ind1, ind2
            
            # 限制交叉深度
            max_depth = 3
            if self._get_depth(ind1) > max_depth or self._get_depth(ind2) > max_depth:
                return ind1, ind2
            
            # 寻找有效交叉点
            cx_points = []
            for i, e in enumerate(ind1):
                if isinstance(e, (list, tuple)) and len(e) > 1:
                    cx_points.append(i)
            if not cx_points:
                return ind1, ind2
            
            # 执行交叉
            cx_point = random.choice(cx_points)
            ind1[cx_point], ind2[cx_point] = ind2[cx_point], ind1[cx_point]
            
            # 交叉后验证
            if not self._validate_expression(ind1):
                ind1[:] = self._generate_random_expression()
            if not self._validate_expression(ind2):
                ind2[:] = self._generate_random_expression()
            
            return ind1, ind2
        except Exception as e:
            logger.warning(f"交叉操作失败: {str(e)}")
            return ind1, ind2
    
    def _mut_expression(self, individual):
        """变异操作：增加结构校验"""
        try:
            if not isinstance(individual, (list, tuple)):
                return individual,
            
            index = random.randrange(len(individual))
            original_subtree = individual[index]
            
            # 生成新子树直到合法
            for _ in range(5):
                new_subtree = self._generate_random_expression(min_depth=1, max_depth=2)
                if self._validate_expression(new_subtree):
                    individual[index] = new_subtree
                    del individual.fitness.values
                    logger.debug(f"成功替换子树: {original_subtree} -> {new_subtree}")
                    return individual,
            
            logger.warning("无法生成合法子树，保持原状")
            return individual,
        except Exception as e:
            logger.warning(f"变异操作失败: {str(e)}")
            return individual,
    
    def _shap_analysis(self, best_factors):
        """执行SHAP分析"""
        # 生成最佳因子集
        X = pd.DataFrame()
        for i, factor in enumerate(best_factors):
            X[f'factor_{i}'] = evaluate_expression(factor, self.data_array)
        
        # 调整模型参数
        model = lgb.LGBMClassifier(
            n_estimators=100,          # 减少树的数量
            num_leaves=31,             # 减少叶子节点数量
            max_depth=5,               # 添加最大深度限制
            min_child_samples=50,      # 增加叶子节点最小样本数
            learning_rate=0.1,         # 提高学习率
            reg_alpha=0.1,             # 添加L1正则化
            reg_lambda=0.1,            # 添加L2正则化
            class_weight='balanced',   # 处理类别不平衡
            device='gpu' if self.use_gpu else 'cpu',
            n_jobs=self.n_jobs
        )
        
        # 添加数据平衡校验
        logger.info(f"目标变量分布:\n{self.target.value_counts(normalize=True)}")
        
        # 使用交叉验证
        tscv = TimeSeriesSplit(n_splits=3)
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = self.target.iloc[train_idx], self.target.iloc[val_idx]
            
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=20,
                      verbose=10)
        
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
        logger.info("🚀 开始因子进化...")
        start_time = time.time()
        performance_history = []  # 新增性能跟踪
        
        try:
            for gen in range(self.generations):
                gen_start = time.time()
                logger.info(f"\n🌀 第 {gen+1}/{self.generations} 代")
                
                pop, log = algorithms.eaSimple(
                    pop, self.toolbox, cxpb=self.cx_prob, mutpb=self.mut_prob,
                    ngen=1, stats=stats, halloffame=hof, verbose=True
                )
                
                # 记录精英个体
                logger.info("🏆 当前最优因子:")
                for idx, ind in enumerate(hof[:3]):  # 只显示前三
                    sharpe, ret, win, dd = ind.fitness.values
                    logger.info(
                        f"{idx+1}. {str(ind)[:50]}...\n"
                        f"  夏普: {sharpe:.2f} | 收益: {ret:.2%} | "
                        f"胜率: {win:.2%} | 回撤: {dd:.2%}"
                    )
                
                # 新增性能监控
                current_best = hof[0].fitness.values
                performance_history.append({
                    'generation': gen,
                    'sharpe': current_best[0],
                    'return': current_best[1],
                    'max_drawdown': current_best[3],
                    'diversity': len(set(map(str, pop))) / len(pop)  # 种群多样性
                })
                
                # 早停机制
                if gen > 5 and np.std([p['sharpe'] for p in performance_history[-5:]]) < 0.1:
                    logger.info("🛑 夏普比率变化小于0.1，提前终止进化")
                    break
            
            # 最终报告
            run_time = time.time() - start_time
            logger.success(
                f"\n✅ 进化完成! 耗时 {run_time//60:.0f}分{run_time%60:.0f}秒\n"
                f"🏁 最佳因子指标:\n"
                f"▸ 夏普比率: {hof[0].fitness.values[0]:.2f}\n"
                f"▸ 累计收益: {hof[0].fitness.values[1]:.2%}\n"
                f"▸ 平均胜率: {hof[0].fitness.values[2]:.2%}"
            )
            
            # 收集最佳因子
            best_factors = [ind[0] for ind in hof]
            # 执行SHAP分析
            important_idx = self._shap_analysis(best_factors)
            
            return best_factors, important_idx, log
            
        except Exception as e:
            logger.critical(f"⚠️ 进化中断! 错误: {str(e)}")
            raise

# 使用示例
if __name__ == "__main__":
    try:
        # 加载预处理好的高频数据
        logger.info("正在加载数据...")
        data = pd.read_feather('train.feather')
        
        # 检查数据列
        logger.info(f"数据列: {data.columns.tolist()}")
        
        # 构造目标变量：基于未来1分钟（120个500ms）的收益率
        logger.info("构造目标变量...")
        future_window = 120  # 500ms * 120 = 60秒
        future_returns = data['mid_price'].shift(-future_window) / data['mid_price'] - 1
        
        # 设置分类阈值（万分之四=0.0004）
        up_threshold = 0.0004
        down_threshold = -0.0004
        
        target = pd.Series(0, index=data.index)  # 默认为0（持平）
        target[future_returns > up_threshold] = 1    # 上涨
        target[future_returns < down_threshold] = -1 # 下跌
        
        # 删除最后future_window行（因为无法计算未来收益）
        target = target[:-future_window]
        data['return_1min'] = future_returns  # 新增行
        features = data[:-future_window].copy()
        
        # 移除不需要的列
        features = features.drop(columns=['index', 'mid_price'])
            
        logger.info(f"数据准备完成。特征数量: {features.shape[1]}, 样本数量: {features.shape[0]}")
        logger.info(f"目标变量分布:\n{target.value_counts(normalize=True)}")
        
        # 初始化因子挖掘器（移除wandb相关参数）
        logger.info("初始化因子挖掘器...")
        miner = GeneticFactorMiner(
            data=features,
            target=target,
            population_size=50,  # 增大种群规模
            generations=20,      # 增加迭代次数
            cx_prob=0.5,         # 降低交叉概率
            mut_prob=0.4,
            use_gpu=True
        )
        
        # 运行挖掘
        logger.info("开始因子挖掘...")
        best_factors, shap_rank, results = miner.run()
        
        # 保存最佳因子
        output_file = 'best_factors.txt'
        logger.info(f"保存最佳因子到 {output_file}")
        with open(output_file, 'w') as f:
            f.write(f"目标变量构造方法（未来1分钟收益率）：\n")
            f.write(f"1: 未来收益率 > {up_threshold}\n")
            f.write(f"0: {down_threshold} <= 未来收益率 <= {up_threshold}\n")
            f.write(f"-1: 未来收益率 < {down_threshold}\n\n")
            f.write("最佳因子：\n")
            for i, factor in enumerate(best_factors):
                f.write(f"Factor {i+1}: {str(factor)}\n")
        
        logger.info("因子挖掘完成！")
        
    except FileNotFoundError:
        logger.error("未找到数据文件 train.feather")
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
    finally:
        logger.info("程序运行结束") 