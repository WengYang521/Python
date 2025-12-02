import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# 路径配置
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
output_dir = os.path.join(script_dir, '../output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 模型参数配置（精简有效取值，提升训练速度）
LR_PARAMS = {
    'C': [0.1, 1, 5],
    'penalty': ['l2'],
    'solver': ['liblinear']
}

RF_PARAMS = {
    'n_estimators': [100, 200],
    'max_depth': [8, 15],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}


def load_data():
    """加载并预处理订单数据"""
    data_path = os.path.join(data_dir, 'erp_order.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError("请先运行数据生成脚本，确保data目录下存在erp_order.csv")

    df = pd.read_csv(data_path, encoding='utf-8-sig')

    # 时间字段转换
    df['order_time'] = pd.to_datetime(df['order_time'])

    # 提取品牌和商品类型
    if 'brand' not in df.columns:
        df['brand'] = df['product_name'].str.split(' ').str[0]
    df['product_type'] = df['product_name'].str.split(' ').str[1].str.replace('（.*', '', regex=True)

    print(f"原始订单数据量：{len(df)} 条")
    print(f"唯一用户数：{df['full_channel_user_id'].nunique()} 个")
    return df


def build_user_features(df):
    """构建用户特征与复购目标变量"""
    end_date = datetime(2024, 12, 31)

    # 预处理折扣率（避免循环内重复计算）
    df['discount_rate'] = (df['original_price'] - df['paid_amount']) / df['original_price']
    df['discount_rate'] = df['discount_rate'].fillna(0).clip(0, 1)  # 处理异常值

    # 按用户聚合基础特征
    user_agg = df.groupby('full_channel_user_id').agg({
        'order_time': ['count', lambda x: (end_date - x.max()).days],
        'paid_amount': 'sum',
        'product_type': 'nunique',
        'brand': 'nunique',
        'unit_price': lambda x: (x >= 1000).sum() / len(x),
        'discount_rate': 'mean',
        'quantity': 'sum',
        'platform': 'nunique'
    }).reset_index()

    # 重命名列
    user_agg.columns = [
        'full_channel_user_id', 'purchase_count', 'last_purchase_days',
        'total_paid', 'product_type_count', 'brand_count', 'high_price_ratio',
        'discount_sensitivity', 'total_quantity', 'platform_count'
    ]

    # 计算派生特征
    user_agg['avg_consumption'] = user_agg['total_paid'] / user_agg['purchase_count']
    user_agg['avg_purchase_quantity'] = user_agg['total_quantity'] / user_agg['purchase_count']
    user_agg['is_repurchase'] = (user_agg['purchase_count'] >= 2).astype(int)

    # 计算平均购买间隔
    repeat_users = user_agg[user_agg['is_repurchase'] == 1]['full_channel_user_id'].tolist()
    interval_list = []
    for user_id in repeat_users:
        order_times = df[df['full_channel_user_id'] == user_id].sort_values('order_time')['order_time'].tolist()
        intervals = [(order_times[i] - order_times[i - 1]).days for i in range(1, len(order_times))]
        interval_list.extend(intervals)
    median_interval = np.median(interval_list) if interval_list else 30

    # 填充间隔值
    def calc_interval(user_id):
        if user_id not in repeat_users:
            return median_interval
        order_times = df[df['full_channel_user_id'] == user_id].sort_values('order_time')['order_time'].tolist()
        intervals = [(order_times[i] - order_times[i - 1]).days for i in range(1, len(order_times))]
        return np.mean(intervals) if intervals else median_interval

    user_agg['avg_purchase_interval'] = user_agg['full_channel_user_id'].apply(calc_interval)

    # 筛选特征列
    feature_cols = [
        'full_channel_user_id', 'is_repurchase', 'purchase_count', 'avg_consumption',
        'product_type_count', 'brand_count', 'discount_sensitivity', 'last_purchase_days',
        'avg_purchase_interval', 'avg_purchase_quantity', 'platform_count', 'high_price_ratio'
    ]
    feature_df = user_agg[feature_cols].copy()

    print(f"特征构建完成，样本数：{len(feature_df)}")
    print(f"复购用户占比：{feature_df['is_repurchase'].mean():.2%}")
    return feature_df


def prepare_train_data(feature_df):
    """数据划分、平衡与特征缩放"""
    # 分离特征和目标变量
    X = feature_df.drop(['full_channel_user_id', 'is_repurchase'], axis=1)
    y = feature_df['is_repurchase']
    feature_names = X.columns.tolist()

    # 划分训练集和测试集（7:3分层抽样）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # SMOTE过采样平衡训练集
    smote_k = min(5, sum(y_train) - 1) if sum(y_train) > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=smote_k)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"训练集平衡前复购占比：{y_train.mean():.2%}")
    print(f"训练集平衡后复购占比：{y_train_balanced.mean():.2%}")

    # 特征缩放：标准化+归一化
    standard_cols = ['discount_sensitivity', 'high_price_ratio', 'avg_consumption']
    minmax_cols = ['last_purchase_days', 'avg_purchase_interval', 'purchase_count']
    other_cols = [col for col in feature_names if col not in standard_cols + minmax_cols]

    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()

    # 训练集缩放
    X_train_std = scaler_standard.fit_transform(X_train_balanced[standard_cols])
    X_train_mm = scaler_minmax.fit_transform(X_train_balanced[minmax_cols])
    X_train_other = X_train_balanced[other_cols].values
    X_train_proc = np.hstack([X_train_std, X_train_mm, X_train_other])

    # 测试集缩放（复用训练集缩放器）
    X_test_std = scaler_standard.transform(X_test[standard_cols])
    X_test_mm = scaler_minmax.transform(X_test[minmax_cols])
    X_test_other = X_test[other_cols].values
    X_test_proc = np.hstack([X_test_std, X_test_mm, X_test_other])

    processed_feature_names = standard_cols + minmax_cols + other_cols
    return (X_train_proc, X_test_proc, y_train_balanced, y_test,
            processed_feature_names, scaler_standard, scaler_minmax)


def train_models(X_train, y_train, X_test, y_test, feature_names):
    """训练逻辑回归和随机森林模型"""
    # 逻辑回归训练
    lr = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
    lr_grid = GridSearchCV(lr, LR_PARAMS, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    lr_grid.fit(X_train, y_train)
    lr_best = lr_grid.best_estimator_

    # 逻辑回归评估
    y_prob_lr = lr_best.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, y_prob_lr)
    lr_acc = accuracy_score(y_test, lr_best.predict(X_test))

    # 特征权重
    feature_weights = pd.DataFrame({
        'feature': feature_names,
        'weight': lr_best.coef_[0]
    }).sort_values('weight', ascending=False)

    # 随机森林训练（n_jobs在模型初始化时设置，不在网格搜索参数中）
    rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
    rf_grid = GridSearchCV(rf, RF_PARAMS, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    # 随机森林评估
    y_prob_rf = rf_best.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, y_prob_rf)
    rf_acc = accuracy_score(y_test, rf_best.predict(X_test))

    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_best.feature_importances_
    }).sort_values('importance', ascending=False)

    # 输出模型对比结果
    print("\n" + "=" * 50)
    print("模型性能对比")
    print("=" * 50)
    print(f"逻辑回归 - AUC: {lr_auc:.4f}, 准确率: {lr_acc:.4f}")
    print(f"随机森林 - AUC: {rf_auc:.4f}, 准确率: {rf_acc:.4f}")
    print("\n逻辑回归Top3特征权重：")
    print(feature_weights.head(3))
    print("\n随机森林Top3特征重要性：")
    print(feature_importance.head(3))

    return {
        'lr_model': lr_best, 'rf_model': rf_best,
        'lr_metrics': {'auc': lr_auc, 'accuracy': lr_acc},
        'rf_metrics': {'auc': rf_auc, 'accuracy': rf_acc},
        'feature_weights': feature_weights, 'feature_importance': feature_importance
    }


def predict_repurchase(model, feature_df, scaler_std, scaler_mm, feature_names):
    """预测用户复购概率并输出结果"""
    X = feature_df.drop(['full_channel_user_id', 'is_repurchase'], axis=1)

    # 特征缩放
    standard_cols = ['discount_sensitivity', 'high_price_ratio', 'avg_consumption']
    minmax_cols = ['last_purchase_days', 'avg_purchase_interval', 'purchase_count']
    other_cols = [col for col in feature_names if col not in standard_cols + minmax_cols]

    X_std = scaler_std.transform(X[standard_cols])
    X_mm = scaler_mm.transform(X[minmax_cols])
    X_other = X[other_cols].values
    X_proc = np.hstack([X_std, X_mm, X_other])

    # 预测复购概率
    repurchase_prob = model.predict_proba(X_proc)[:, 1]

    # 构建结果表
    result_df = feature_df[['full_channel_user_id', 'is_repurchase']].copy()
    result_df['repurchase_probability'] = repurchase_prob

    # 划分用户层级
    def get_user_level(prob):
        if prob >= 0.6:
            return '高复购概率'
        elif 0.3 <= prob < 0.6:
            return '中复购概率'
        else:
            return '低复购概率'

    result_df['user_level'] = result_df['repurchase_probability'].apply(get_user_level)

    # 统计结果
    level_stats = result_df['user_level'].value_counts()
    print("\n" + "=" * 50)
    print("用户复购概率层级分布")
    print("=" * 50)
    print(level_stats)
    print(f"\n各层级占比：")
    for level, count in level_stats.items():
        print(f"{level}: {count / len(result_df):.2%}")

    # 保存结果
    result_path = os.path.join(output_dir, 'repurchase_prediction_result.csv')
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"\n复购预测结果已保存至：{result_path}")

    return result_df


def main():
    print("=" * 50)
    print("复购预测模型训练")
    print("=" * 50)

    # 1. 加载数据
    print("\n1. 加载订单数据...")
    df = load_data()

    # 2. 构建特征
    print("\n2. 构建用户特征...")
    feature_df = build_user_features(df)

    # 3. 准备训练数据
    print("\n3. 数据预处理...")
    (X_train, X_test, y_train, y_test,
     feature_names, scaler_std, scaler_mm) = prepare_train_data(feature_df)

    # 4. 训练模型
    print("\n4. 模型训练...")
    model_results = train_models(X_train, y_train, X_test, y_test, feature_names)

    # 5. 复购预测
    print("\n5. 复购概率预测...")
    result_df = predict_repurchase(
        model=model_results['rf_model'],
        feature_df=feature_df,
        scaler_std=scaler_std,
        scaler_mm=scaler_mm,
        feature_names=feature_names
    )

    print("\n" + "=" * 50)
    print("模型训练完成！")
    print("=" * 50)



if __name__ == "__main__":
    main()