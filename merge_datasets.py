import pandas as pd
import os
from loguru import logger

# 配置日志
logger.add("merge_datasets.log", rotation="100 MB")

def merge_train_test_datasets(train_path="train.feather", test_path="test.feather", output_path="merged_data.feather"):
    """
    合并训练集和测试集为一个完整数据集
    
    参数:
        train_path: 训练集文件路径
        test_path: 测试集文件路径
        output_path: 输出文件路径
    """
    logger.info("开始合并数据集...")
    
    # 检查文件是否存在
    if not os.path.exists(train_path):
        logger.error(f"训练集文件不存在: {train_path}")
        return False
    
    if not os.path.exists(test_path):
        logger.error(f"测试集文件不存在: {test_path}")
        return False
    
    try:
        # 读取训练集和测试集
        logger.info(f"读取训练集: {train_path}")
        train_data = pd.read_feather(train_path)
        
        logger.info(f"读取测试集: {test_path}")
        test_data = pd.read_feather(test_path)
        
        # 检查数据
        logger.info(f"训练集形状: {train_data.shape}")
        logger.info(f"测试集形状: {test_data.shape}")
        
        # 检查并设置时间索引（如果需要）
        time_columns = ['index', 'time', 'Timestamp', 'origin_time']
        time_col = None
        
        for col in time_columns:
            if col in train_data.columns:
                time_col = col
                break
        
        if time_col:
            logger.info(f"使用 '{time_col}' 列作为时间索引")
            if time_col == 'index':
                train_data.set_index('index', inplace=True)
                test_data.set_index('index', inplace=True)
        
        # 合并数据集
        logger.info("合并数据集...")
        merged_data = pd.concat([train_data, test_data], axis=0)
        
        # 确保数据按时间排序
        if time_col and time_col != 'index':
            merged_data.sort_values(by=time_col, inplace=True)
            
        # 重置索引，确保索引连续
        merged_data.reset_index(drop=True, inplace=True)
        
        # 保存合并后的数据集
        logger.info(f"保存合并后的数据集到: {output_path}")
        merged_data.to_feather(output_path)
        
        logger.info(f"合并完成! 合并后数据集形状: {merged_data.shape}")
        return True
        
    except Exception as e:
        logger.error(f"合并数据集时发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = merge_train_test_datasets()
    if success:
        logger.info("数据集已成功合并")
    else:
        logger.error("数据集合并失败，请检查日志") 