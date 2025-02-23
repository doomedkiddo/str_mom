from time import time
from config import POSITIVE_FACTORS, STYLE_FACTOR, NEGATIVE, end_date, WINDOW, trade, STYLE_FACTOR
from combine_factor import get_train_factor_df, init_models
from util import simulate_trading
from tqdm import tqdm
import shap
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def filter_feature(shap_values, correlation_matrix, high_corr_threshold=0.8):
    # 阈值为 0.8
    to_remove = set()

    # 按照特征重要性降序排列

    for i in range(len(shap_values)):
        if shap_values.index[i] in to_remove:
            continue
        for j in range(i + 1, len(shap_values)):
            if correlation_matrix.loc[shap_values.index[i], shap_values.index[j]] > high_corr_threshold:
                to_remove.add(shap_values.index[j])

    features = sorted(set(shap_values.index) - set(to_remove))
    return features


def update_model(data):
    model_list = init_models()
    new_model_list = []
    # factor_names = POSITIVE_FACTORS + STYLE_FACTOR
    for model in model_list:
        new_model = deepcopy(model)
        # data[factor_names] = data.groupby("date")[factor_names].transform(lambda x: x.rank(pct=True))
        # data[factor_names] = data.groupby("date")[factor_names].transform(lambda x: (x - x.mean()) / x.std())
        data["ret"] = data.groupby("date")["ret"].transform(lambda x: x.rank(pct=True))
        model.fit(data[POSITIVE_FACTORS + STYLE_FACTOR], data["ret"])
        # 使用 TreeExplainer 计算 SHAP 值
        explainer = shap.Explainer(model)
        shap_values = explainer(data[POSITIVE_FACTORS + STYLE_FACTOR])
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': POSITIVE_FACTORS + STYLE_FACTOR,
            'importance': feature_importance
        })
        feature_importance_df.set_index('feature', inplace=True)
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        # 特征相关性
        correlation_matrix = data[POSITIVE_FACTORS + STYLE_FACTOR].corr()
        plt.figure(figsize=(80, 60))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)

        # 添加标题
        plt.title("Factor Correlation Heatmap", fontsize=16)
        plt.show()
        filter_features = filter_feature(feature_importance_df, correlation_matrix.abs())
        # 根据特征重要性排序
        # feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        # top_20_percent = feature_importance_df.head(int(len(feature_importance_df)*0.2)).feature
        # 可视化特征重要性
        shap.summary_plot(shap_values, data[POSITIVE_FACTORS + STYLE_FACTOR],
                          feature_names=POSITIVE_FACTORS + STYLE_FACTOR)
        plt.savefig('summary_plot.png', dpi=300, bbox_inches='tight')
        new_model.fit(data[filter_features], data["ret"])
        new_model_list.append([filter_features, new_model])
    return new_model_list


def refactor_alpha(train_data):
    # 对样本进行筛选
    train_data["ret"] = train_data.groupby("date")["ret"].transform(lambda x: x.fillna(x.mean()))
    train_data[POSITIVE_FACTORS] = train_data.groupby("date")[POSITIVE_FACTORS].transform(lambda x: x.fillna(x.mean()))
    train_data[POSITIVE_FACTORS] = train_data[POSITIVE_FACTORS].fillna(0)
    factor = train_data.dropna()
    factor = factor[
        ~((factor.is_sub_new == True) | (factor.is_new == True) | (factor.is_st == True)
          | (factor.paused == True))]
    # 获取交易日期
    trade_date = sorted(factor.reset_index()["date"].unique())

    # 获取因子预测数据
    factor_names = POSITIVE_FACTORS + STYLE_FACTOR
    tmp_date_list = []
    tmp_factor_list = []
    ret_dict = {}
    alpha_ret_dict = {}
    for date in tqdm(trade_date):
        tmp_date_list.append(date)
        if len(tmp_date_list) <= WINDOW:
            continue
        available_dates = tmp_date_list[-WINDOW - 1:-1]
        tmp_factor = factor.query("date in @available_dates")

        # 获取最新的数据
        latest_factor = factor.query("date==@date")

        models = update_model(tmp_factor)
        predict_data = [model.predict(latest_factor[feature]) for (feature, model) in models]

        # 获取最新的数据
        stock_num = int(len(latest_factor) / 10)
        latest_factor["predict"] = sum(predict_data)
        latest_factor["alpha"] = latest_factor[POSITIVE_FACTORS].mean(axis=1)
        latest_factor.sort_values(by="predict", inplace=True)
        ret_dict[date] = latest_factor.tail(stock_num)['ret'].mean()
        latest_factor.sort_values(by="alpha", inplace=True)
        alpha_ret_dict[date] = latest_factor.tail(stock_num)['ret'].mean()
        print(f"{date}  ***predict_ret:  {ret_dict[date]}")
        print(f"{date}  ***alpha_ret:  {alpha_ret_dict[date]}")
        tmp_factor_list.append(latest_factor)

    ret = (pd.DataFrame(ret_dict, index=["ret"]).T + 1).cumprod()
    alpha_ret = (pd.DataFrame(alpha_ret_dict, index=["ret"]).T + 1).cumprod()
    print(ret)
    print("---------------")
    print(alpha_ret)

    data = pd.concat(tmp_factor_list, axis=0)

    data[NEGATIVE] = data.groupby("date")[NEGATIVE].transform(lambda x: x.rank(pct=True))
    # data["new_predict"] = data.groupby("date")["predict"].transform(lambda x: x.rank(pct=True))
    for na_factor in NEGATIVE:
        data.loc[
            (data.groupby("date")[na_factor].transform(lambda x: (x <= x.quantile(0.1)))) & \
            (data.groupby("date")["predict"].transform(lambda x: (x <= x.quantile(0.5))))
            , "alpha"] = \
            -1e1

    # data.loc[(data.groupby("date")[NEGATIVE].transform(lambda x: (x <= x.quantile(0.1)))), NEGATIVE] = 0

    return data