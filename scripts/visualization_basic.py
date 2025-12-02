import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# -------------------------- 基础配置 --------------------------
# 设置中文字体（兼容Windows/Mac）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录（绝对路径，避免报错）
script_dir = os.path.dirname(os.path.abspath(__file__))
figures_dir = os.path.join(script_dir, '../figures')
os.makedirs(figures_dir, exist_ok=True)

# 加载数据（仅erp_order单表）
data_path = os.path.join(script_dir, '../data/erp_order.csv')
erp_df = pd.read_csv(data_path, encoding='utf-8-sig')

# -------------------------- 数据预处理（挖掘多维度特征）--------------------------
# 1. 时间维度扩展
erp_df['order_time'] = pd.to_datetime(erp_df['order_time'])
erp_df['payment_date'] = pd.to_datetime(erp_df['payment_date'])
erp_df['shipping_date'] = pd.to_datetime(erp_df['shipping_date'])
erp_df['month'] = erp_df['order_time'].dt.strftime('%Y-%m')  # 年月
erp_df['week'] = erp_df['order_time'].dt.strftime('%Y-W%U')  # 年周
erp_df['day'] = erp_df['order_time'].dt.day  # 日
erp_df['hour'] = erp_df['order_time'].dt.hour  # 小时
erp_df['quarter'] = erp_df['order_time'].dt.quarter  # 季度
erp_df['is_weekend'] = erp_df['order_time'].dt.weekday >= 5  # 是否周末

# 2. 商品维度拆分
erp_df['brand'] = erp_df['product_name'].str.split(' ').str[0]  # 品牌（耐克/阿迪等）
erp_df['product_type'] = erp_df['product_name'].str.split(' ').str[1].str.strip('（')  # 商品类型（T恤/外套等）
erp_df['color'] = erp_df['color_and_spec'].str.split('/').str[0]  # 颜色
erp_df['spec'] = erp_df['color_and_spec'].str.split('/').str[1]  # 规格（S/M/L等）

# 3. 价格维度分组
erp_df['price_range'] = pd.cut(
    erp_df['unit_price'],
    bins=[0, 299, 599, 999, 1999],
    labels=['低价(0-299元)', '中低价(300-599元)', '中高价(600-999元)', '高价(1000元+)']
)

# 4. 业务指标计算
erp_df['payment_duration'] = (erp_df['payment_date'] - erp_df['order_time']).dt.total_seconds() / 3600  # 付款时长（小时）
erp_df['shipping_duration'] = (erp_df['shipping_date'] - erp_df['payment_date']).dt.total_seconds() / 86400  # 发货时长（天）
erp_df['discount_rate'] = (1 - erp_df['product_amount'] / erp_df['original_price']).round(2) * 100  # 折扣率（%）

# 5. 区域维度映射
region_mapping = {
    '北京市': '华北', '上海市': '华东',
    '广东省': '华南', '江苏省': '华东', '浙江省': '华东',
    '山东省': '华东', '四川省': '西南', '湖北省': '华中',
    '湖南省': '华中', '河南省': '华中'
}
erp_df['region'] = erp_df['province'].map(region_mapping)

# 6. 用户复购标识
user_purchase_count = erp_df.groupby('full_channel_user_id', observed=True).size().reset_index(
    name='purchase_count')  # 修复警告：添加observed=True
erp_df = pd.merge(erp_df, user_purchase_count, on='full_channel_user_id', how='left')
erp_df['is_repurchase'] = erp_df['purchase_count'] >= 2  # 是否复购用户

# 统一配色（贴合商业图表风格）
colors_mpl = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# 过滤异常值（确保图表可读性）
erp_df = erp_df[erp_df['payment_duration'] <= 24]  # 过滤付款时长>24小时的异常值
erp_df = erp_df[erp_df['shipping_duration'] >= 0]  # 过滤发货时长为负的异常值


# ------------------------------
# 一、趋势类图表（1-5张：时间维度分析）
# ------------------------------
# 01-月度销售额+订单量双轴趋势图
def plot_01_monthly_sales_order_trend():
    monthly_data = erp_df.groupby('month').agg({
        'product_amount': 'sum',
        'id': 'count'
    }).reset_index()
    monthly_data['month'] = pd.to_datetime(monthly_data['month'])
    monthly_data = monthly_data.sort_values('month')
    monthly_data['month_str'] = monthly_data['month'].dt.strftime('%Y-%m')

    fig, ax1 = plt.subplots(figsize=(14, 7))
    # 销售额折线
    ax1.plot(monthly_data['month_str'], monthly_data['product_amount'],
             color='#1f77b4', linewidth=3, marker='o', markersize=6, label='销售额')
    ax1.set_xlabel('月份', fontsize=12)
    ax1.set_ylabel('销售额（元）', fontsize=12, color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xticks(range(0, len(monthly_data['month_str']), 1))
    ax1.set_xticklabels(monthly_data['month_str'], rotation=45)
    ax1.grid(True, alpha=0.3)

    # 订单量柱状图
    ax2 = ax1.twinx()
    ax2.bar(monthly_data['month_str'], monthly_data['id'],
            color='#ff7f0e', alpha=0.6, width=0.5, label='订单量')
    ax2.set_ylabel('订单量（笔）', fontsize=12, color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('2024年月度销售额与订单量趋势', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '01-月度销售额订单量双轴图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 02-季度销售额堆叠柱状图
def plot_02_quarterly_brand_sales():
    quarterly_brand = erp_df.groupby(['quarter', 'brand'], observed=True)[
        'product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    quarterly_pivot = quarterly_brand.pivot(index='quarter', columns='brand', values='product_amount').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 7))
    quarterly_pivot.plot(kind='bar', stacked=True, color=colors_mpl[:len(quarterly_pivot.columns)],
                         alpha=0.8, ax=ax)
    ax.set_xlabel('季度', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.set_title('2024年各季度品牌销售额堆叠趋势', fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='品牌', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '02-季度品牌销售额堆叠图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 03-周度销售额趋势图（替换Pyecharts为Matplotlib）
def plot_03_weekly_sales_trend():
    weekly_sales = erp_df.groupby('week', observed=True)['product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    weekly_sales['week'] = pd.to_datetime(weekly_sales['week'] + '-1', format='%Y-W%U-%w')
    weekly_sales = weekly_sales.sort_values('week')
    weekly_sales['week_str'] = weekly_sales['week'].dt.strftime('%Y-W%U')

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(weekly_sales['week_str'], weekly_sales['product_amount'],
            color='#1f77b4', linewidth=2, marker='o', markersize=4)
    # 添加最大值、最小值、平均值标注
    max_idx = weekly_sales['product_amount'].idxmax()
    min_idx = weekly_sales['product_amount'].idxmin()
    avg_val = weekly_sales['product_amount'].mean()
    ax.axhline(y=avg_val, color='red', linestyle='--', alpha=0.7, label=f'平均值：{avg_val:.0f}元')
    ax.scatter(weekly_sales.loc[max_idx, 'week_str'], weekly_sales.loc[max_idx, 'product_amount'],
               color='red', s=100, zorder=5)
    ax.scatter(weekly_sales.loc[min_idx, 'week_str'], weekly_sales.loc[min_idx, 'product_amount'],
               color='green', s=100, zorder=5)
    ax.annotate(f'最大值：{weekly_sales.loc[max_idx, "product_amount"]:.0f}元',
                xy=(weekly_sales.loc[max_idx, 'week_str'], weekly_sales.loc[max_idx, 'product_amount']),
                xytext=(10, 10), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    ax.annotate(f'最小值：{weekly_sales.loc[min_idx, "product_amount"]:.0f}元',
                xy=(weekly_sales.loc[min_idx, 'week_str'], weekly_sales.loc[min_idx, 'product_amount']),
                xytext=(10, -20), textcoords='offset points', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    ax.set_xlabel('周度', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.set_title('2024年周度销售额趋势', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(0, len(weekly_sales['week_str']), 2))  # 每2周显示一个刻度
    ax.set_xticklabels(weekly_sales['week_str'][::2], rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '03-周度销售额趋势图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 04-日内订单量分布折线图
def plot_04_hourly_order_distribution():
    hourly_order = erp_df.groupby('hour', observed=True).size().reset_index(name='order_count')  # 修复警告：添加observed=True

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly_order['hour'], hourly_order['order_count'],
            color='#2ca02c', linewidth=3, marker='s', markersize=5)
    ax.fill_between(hourly_order['hour'], hourly_order['order_count'], alpha=0.3, color='#2ca02c')
    # 标注峰值
    peak_hour = hourly_order.loc[hourly_order['order_count'].idxmax(), 'hour']
    peak_count = hourly_order['order_count'].max()
    ax.annotate(f'峰值：{peak_hour}时({peak_count}笔)',
                xy=(peak_hour, peak_count), xytext=(peak_hour + 1, peak_count + 5),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=11, fontweight='bold')

    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('订单量（笔）', fontsize=12)
    ax.set_title('日内订单量分布（24小时）', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '04-日内订单量分布.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 05-周末vs工作日销售额对比图
def plot_05_weekend_weekday_comparison():
    weekend_data = erp_df.groupby('is_weekend', observed=True).agg({
        'product_amount': 'sum',
        'id': 'count'
    }).reset_index()
    weekend_data['is_weekend'] = weekend_data['is_weekend'].map({True: '周末', False: '工作日'})

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 销售额柱状图
    bars = ax1.bar(weekend_data['is_weekend'], weekend_data['product_amount'],
                   color=['#ff7f0e', '#1f77b4'], alpha=0.8, width=0.5)
    ax1.set_xlabel('日期类型', fontsize=12)
    ax1.set_ylabel('销售额（元）', fontsize=12, color='black')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{int(height)}', ha='center', va='bottom', fontsize=11)

    # 订单量折线
    ax2 = ax1.twinx()
    # 核心调整：限制订单量y轴的最大值，避免数值超出范围
    ax2.set_ylim(0, weekend_data['id'].max() * 1.2)  # 留出1.2倍的空间放标注
    line = ax2.plot(weekend_data['is_weekend'], weekend_data['id'],
             color='#d62728', linewidth=3, marker='o', markersize=8)
    ax2.set_ylabel('订单量（笔）', fontsize=12, color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    # 标注订单量数值（调整位置：从“v+5”改为“v * 0.95”，避免超出顶部）
    for i, v in enumerate(weekend_data['id']):
        ax2.text(i, v * 0.95, f'{v}笔', ha='center', va='top', fontsize=11, color='#d62728')

    plt.title('周末vs工作日销售额与订单量对比', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '05-周末工作日对比图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ------------------------------
# 二、商品分析类图表（6-12张：品牌/类型/属性分析）
# ------------------------------
# 06-品牌销售额TOP10柱状图（替换Pyecharts为Matplotlib）
def plot_06_brand_sales_top10():
    brand_sales = erp_df.groupby('brand', observed=True)['product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    brand_sales_top10 = brand_sales.sort_values('product_amount', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(brand_sales_top10['brand'], brand_sales_top10['product_amount'],
                  color=colors_mpl[:10], alpha=0.8)
    ax.set_xlabel('品牌', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.set_title('品牌销售额TOP10', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标签和排名
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                f'第{i + 1}名\n{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '06-品牌销售额TOP10.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 07-商品类型销量饼图
def plot_07_product_type_sales_pie():
    type_sales = erp_df.groupby('product_type', observed=True)['quantity'].sum().reset_index()  # 修复警告：添加observed=True
    type_sales = type_sales.sort_values('quantity', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(
        type_sales['quantity'],
        labels=type_sales['product_type'],
        autopct='%1.1f%%',
        colors=colors_mpl[:len(type_sales)],
        startangle=90,
        wedgeprops=dict(width=0.6, edgecolor='white'),
        textprops=dict(fontsize=11)
    )
    # 美化文本
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')

    ax.set_title('各商品类型销量占比', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '07-商品类型销量饼图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 08-颜色销量分布柱状图
def plot_08_color_sales_distribution():
    color_sales = erp_df.groupby('color', observed=True)['quantity'].sum().reset_index()  # 修复警告：添加observed=True
    color_sales = color_sales.sort_values('quantity', ascending=False)

    # 颜色映射（真实颜色）
    color_map = {
        '黑色': '#000000', '白色': '#ffffff', '灰色': '#808080', '蓝色': '#0000ff',
        '红色': '#ff0000', '绿色': '#00ff00', '黄色': '#ffff00', '紫色': '#9932cc',
        '粉色': '#ff69b4', '卡其色': '#d2b48c'
    }
    # 处理未匹配的颜色
    color_map = {k: v for k, v in color_map.items() if k in color_sales['color'].values}

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(color_sales['color'], color_sales['quantity'],
                  color=[color_map.get(c, '#cccccc') for c in color_sales['color']],
                  alpha=0.8, edgecolor='black', linewidth=1)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 2,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('商品颜色', fontsize=12)
    ax.set_ylabel('销量（件）', fontsize=12)
    ax.set_title('各颜色服装销量分布', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '08-颜色销量分布.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 09-规格销量热力图
def plot_09_spec_sales_heatmap():
    spec_type_sales = erp_df.groupby(['product_type', 'spec'], observed=True)[
        'quantity'].sum().reset_index()  # 修复警告：添加observed=True
    spec_pivot = spec_type_sales.pivot(index='product_type', columns='spec', values='quantity').fillna(0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        spec_pivot,
        annot=True,
        fmt='.0f',
        cmap='YlOrRd',
        linewidths=0.5,
        cbar_kws={'label': '销量（件）'},
        ax=ax
    )
    ax.set_xlabel('商品规格', fontsize=12)
    ax.set_ylabel('商品类型', fontsize=12)
    ax.set_title('各类型商品规格销量热力图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '09-规格销量热力图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 10-价格区间销售额占比图（替换Pyecharts环形图为Matplotlib环形图）
def plot_10_price_range_pie():
    price_sales = erp_df.groupby('price_range', observed=True)[
        'product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    price_sales['percentage'] = (price_sales['product_amount'] / price_sales['product_amount'].sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        price_sales['percentage'],
        labels=price_sales['price_range'],
        autopct='%1.1f%%',
        colors=colors_mpl[:len(price_sales)],
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='white'),  # 环形效果
        textprops=dict(fontsize=11)
    )
    # 美化文本
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')

    ax.set_title('各价格区间销售额占比', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '10-价格区间占比图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 11-品牌折扣率对比箱线图（修复警告：添加hue和legend=False）
def plot_11_brand_discount_boxplot():
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=erp_df,
        x='brand',
        y='discount_rate',
        hue='brand',  # 添加hue参数，与x一致
        palette=colors_mpl[:len(erp_df['brand'].unique())],
        legend=False,  # 隐藏图例（避免重复）
        ax=ax
    )
    # 添加平均值线
    for i, brand in enumerate(erp_df['brand'].unique()):
        brand_avg = erp_df[erp_df['brand'] == brand]['discount_rate'].mean()
        ax.axhline(y=brand_avg, xmin=i / len(erp_df['brand'].unique()),
                   xmax=(i + 1) / len(erp_df['brand'].unique()), color='red', linestyle='--', alpha=0.7)
        ax.text(i, brand_avg + 1, f'均值：{brand_avg:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('品牌', fontsize=12)
    ax.set_ylabel('折扣率（%）', fontsize=12)
    ax.set_title('各品牌折扣率分布对比', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '11-品牌折扣率箱线图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 12-商品类型销售额对比图（替换Pyecharts TreeMap为Matplotlib柱状图）
def plot_12_product_type_sales_bar():
    type_sales = erp_df.groupby('product_type', observed=True)[
        'product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    type_sales = type_sales.sort_values('product_amount', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(type_sales['product_type'], type_sales['product_amount'],
                  color=colors_mpl[:len(type_sales)], alpha=0.8)
    ax.set_xlabel('商品类型', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.set_title('各商品类型销售额对比', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标签和占比
    total_sales = type_sales['product_amount'].sum()
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_sales) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '12-商品类型销售额对比图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ------------------------------
# 三、用户行为类图表（13-18张：复购/消费/支付分析）
# ------------------------------
# 13-用户复购率饼图（修复autopct格式错误）
def plot_13_user_repurchase_pie():
    repurchase_count = erp_df.groupby('is_repurchase', observed=True).size().reset_index(
        name='count')  # 修复警告：添加observed=True
    repurchase_count['is_repurchase'] = repurchase_count['is_repurchase'].map({True: '复购用户', False: '新用户'})
    repurchase_rate = (repurchase_count[repurchase_count['is_repurchase'] == '复购用户']['count'].values[0] /
                       repurchase_count['count'].sum()) * 100

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        repurchase_count['count'],
        labels=repurchase_count['is_repurchase'],
        autopct='%1.1f%%',  # 仅保留百分比格式，复购率通过标题展示
        colors=['#1f77b4', '#ff7f0e'],
        startangle=90,
        explode=(0.05, 0),
        wedgeprops=dict(edgecolor='white')
    )
    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    # 标题中展示复购率
    ax.set_title(f'用户复购率分布（复购率：{repurchase_rate:.1f}%）', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '13-用户复购率饼图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 14-用户购买次数分布直方图
def plot_14_user_purchase_count_hist():
    user_purchase = erp_df.groupby('full_channel_user_id', observed=True).size().reset_index(
        name='purchase_count')  # 修复警告：添加observed=True
    max_count = min(user_purchase['purchase_count'].max(), 10)  # 限制最大显示10次
    user_purchase_filtered = user_purchase[user_purchase['purchase_count'] <= max_count]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=user_purchase_filtered,
        x='purchase_count',
        bins=range(1, max_count + 2),
        color='#2ca02c',
        alpha=0.7,
        discrete=True,
        ax=ax
    )
    ax.set_xlabel('购买次数', fontsize=12)
    ax.set_ylabel('用户数量（人）', fontsize=12)
    ax.set_title('用户购买次数分布', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(range(1, max_count + 1))
    # 添加数值标签
    for i in range(1, max_count + 1):
        count = len(user_purchase_filtered[user_purchase_filtered['purchase_count'] == i])
        ax.text(i, count + 2, f'{count}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '14-用户购买次数直方图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 15-支付时长分布小提琴图（修复警告：添加hue和legend=False）
def plot_15_payment_duration_violin():
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(
        data=erp_df,
        x='platform',
        y='payment_duration',
        hue='platform',  # 添加hue参数，与x一致
        palette=colors_mpl[:len(erp_df['platform'].unique())],
        inner='quartile',
        legend=False,  # 隐藏图例
        ax=ax
    )
    # 添加平均值点
    for i, platform in enumerate(erp_df['platform'].unique()):
        platform_avg = erp_df[erp_df['platform'] == platform]['payment_duration'].mean()
        ax.scatter(i, platform_avg, color='red', s=50, zorder=5)
        ax.text(i, platform_avg + 0.5, f'均值：{platform_avg:.1f}小时', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('销售平台', fontsize=12)
    ax.set_ylabel('支付时长（小时）', fontsize=12)
    ax.set_title('各平台用户支付时长分布', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '15-支付时长小提琴图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 16-购买数量与金额散点图
def plot_16_quantity_amount_scatter():
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        erp_df['quantity'],
        erp_df['product_amount'],
        c=erp_df['unit_price'],
        cmap='viridis',
        alpha=0.7,
        s=60,
        edgecolor='black',
        linewidth=0.5
    )
    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('商品单价（元）', fontsize=12)
    # 添加趋势线
    z = np.polyfit(erp_df['quantity'], erp_df['product_amount'], 1)
    p = np.poly1d(z)
    ax.plot(erp_df['quantity'], p(erp_df['quantity']), "r--", alpha=0.8, linewidth=2,
            label=f'趋势线：y={z[0]:.1f}x+{z[1]:.1f}')

    ax.set_xlabel('购买数量（件）', fontsize=12)
    ax.set_ylabel('订单金额（元）', fontsize=12)
    ax.set_title('购买数量与订单金额关系（颜色=单价）', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '16-购买数量金额散点图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 17-复购用户vs新用户消费对比
def plot_17_repurchase_vs_new_consume():
    user_consume = erp_df.groupby(['full_channel_user_id', 'is_repurchase'], observed=True).agg(
        {  # 修复警告：添加observed=True
            'product_amount': 'sum',
            'quantity': 'sum'
        }).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # 总消费额对比
    repurchase_consume = user_consume.groupby('is_repurchase', observed=True)[
        'product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    repurchase_consume['is_repurchase'] = repurchase_consume['is_repurchase'].map({True: '复购用户', False: '新用户'})
    bars1 = ax1.bar(repurchase_consume['is_repurchase'], repurchase_consume['product_amount'],
                    color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    ax1.set_xlabel('用户类型', fontsize=12)
    ax1.set_ylabel('总消费额（元）', fontsize=12)
    ax1.set_title('复购用户vs新用户总消费额对比', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for i, v in enumerate(repurchase_consume['product_amount']):
        ax1.text(i, v + v * 0.01, f'{int(v)}', ha='center', va='bottom', fontsize=11)

    # 平均消费额对比
    avg_consume = user_consume.groupby('is_repurchase', observed=True)[
        'product_amount'].mean().reset_index()  # 修复警告：添加observed=True
    avg_consume['is_repurchase'] = avg_consume['is_repurchase'].map({True: '复购用户', False: '新用户'})
    bars2 = ax2.bar(avg_consume['is_repurchase'], avg_consume['product_amount'],
                    color=['#2ca02c', '#d62728'], alpha=0.8)
    ax2.set_xlabel('用户类型', fontsize=12)
    ax2.set_ylabel('平均消费额（元）', fontsize=12)
    ax2.set_title('复购用户vs新用户平均消费额对比', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(avg_consume['product_amount']):
        ax2.text(i, v + v * 0.01, f'{int(v)}', ha='center', va='bottom', fontsize=11)

    plt.suptitle('复购用户与新用户消费行为对比', fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '17-复购新用户消费对比.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 18-用户购买频次行为对比图（替换Pyecharts雷达图为Matplotlib多子图）
def plot_18_user_purchase_freq_comparison():
    # 按购买次数分组统计
    purchase_freq = erp_df.groupby('purchase_count', observed=True).agg({  # 修复警告：添加observed=True
        'product_amount': 'mean',
        'quantity': 'mean',
        'discount_rate': 'mean'
    }).reset_index()
    purchase_freq['purchase_count'] = purchase_freq['purchase_count'].astype(str) + '次购买'
    # 限制显示前5组
    purchase_freq = purchase_freq.head(5)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # 平均消费额
    axes[0].bar(purchase_freq['purchase_count'], purchase_freq['product_amount'], color='#1f77b4', alpha=0.8)
    axes[0].set_xlabel('购买频次', fontsize=11)
    axes[0].set_ylabel('平均消费额（元）', fontsize=11)
    axes[0].set_title('不同频次平均消费额', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(purchase_freq['product_amount']):
        axes[0].text(i, v + 5, f'{int(v)}', ha='center', va='bottom', fontsize=9)

    # 平均购买量
    axes[1].bar(purchase_freq['purchase_count'], purchase_freq['quantity'], color='#ff7f0e', alpha=0.8)
    axes[1].set_xlabel('购买频次', fontsize=11)
    axes[1].set_ylabel('平均购买量（件）', fontsize=11)
    axes[1].set_title('不同频次平均购买量', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(purchase_freq['quantity']):
        axes[1].text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    # 平均折扣率
    axes[2].bar(purchase_freq['purchase_count'], purchase_freq['discount_rate'], color='#2ca02c', alpha=0.8)
    axes[2].set_xlabel('购买频次', fontsize=11)
    axes[2].set_ylabel('平均折扣率（%）', fontsize=11)
    axes[2].set_title('不同频次平均折扣率', fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(purchase_freq['discount_rate']):
        axes[2].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.suptitle('不同购买频次用户行为对比', fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '18-用户购买频次行为对比图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ------------------------------
# 四、区域与渠道类图表（19-24张：区域/平台/门店分析）
# ------------------------------
# 19-省份销售额柱状图（替换Pyecharts地图为Matplotlib柱状图）
def plot_19_province_sales_bar():
    province_sales = erp_df.groupby('province', observed=True)[
        'product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    province_sales = province_sales.sort_values('product_amount', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(province_sales['province'], province_sales['product_amount'],
                  color=colors_mpl[:len(province_sales)], alpha=0.8)
    ax.set_xlabel('省份', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.set_title('各省份销售额分布', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '19-省份销售额分布.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 20-区域销售额柱状图
def plot_20_region_sales_bar():
    region_sales = erp_df.groupby('region', observed=True)['product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    region_sales = region_sales.sort_values('product_amount', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(region_sales['region'], region_sales['product_amount'],
                  color=colors_mpl[:len(region_sales)], alpha=0.8)
    ax.set_xlabel('区域', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.set_title('各区域销售额对比', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标签和占比
    total_sales = region_sales['product_amount'].sum()
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_sales) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '20-区域销售额柱状图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 21-城市销售额TOP15横向柱状图
def plot_21_city_sales_top15():
    city_sales = erp_df.groupby('city', observed=True)['product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    city_sales_top15 = city_sales.sort_values('product_amount', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        city_sales_top15['city'][::-1],
        city_sales_top15['product_amount'][::-1],
        color='#1f77b4', alpha=0.8
    )
    ax.set_xlabel('销售额（元）', fontsize=12)
    ax.set_ylabel('城市', fontsize=12)
    ax.set_title('城市销售额TOP15', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2.,
                f'{int(width)}', ha='left', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '21-城市销售额TOP15.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 22-销售平台月度销售额趋势图（替换Pyecharts为Matplotlib多折线图）
def plot_22_platform_monthly_sales():
    platform_monthly = erp_df.groupby(['month', 'platform'], observed=True)[
        'product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    platform_monthly['month'] = pd.to_datetime(platform_monthly['month'])
    platform_monthly = platform_monthly.sort_values('month')
    months = platform_monthly['month'].dt.strftime('%Y-%m').unique()

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, platform in enumerate(platform_monthly['platform'].unique()):
        platform_data = platform_monthly[platform_monthly['platform'] == platform]
        # 补全缺失月份的销售额（填充0）
        platform_sales = []
        for month in months:
            sales = platform_data[platform_data['month'].dt.strftime('%Y-%m') == month]['product_amount'].values
            platform_sales.append(sales[0] if len(sales) > 0 else 0)

        ax.plot(months, platform_sales, color=colors_mpl[i], linewidth=2, marker='o', markersize=4, label=platform)

    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.set_title('各销售平台月度销售额趋势', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '22-平台月度销售额趋势.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 23-门店销售额TOP10柱状图（替换Pyecharts漏斗图为Matplotlib柱状图）
def plot_23_store_sales_top10():
    store_sales = erp_df.groupby('store_name', observed=True)[
        'product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    store_sales_top10 = store_sales.sort_values('product_amount', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(store_sales_top10['store_name'], store_sales_top10['product_amount'],
                  color=colors_mpl[:10], alpha=0.8)
    ax.set_xlabel('门店名称', fontsize=12)
    ax.set_ylabel('销售额（元）', fontsize=12)
    ax.set_title('门店销售额TOP10', fontsize=14, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '23-门店销售额TOP10.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 24-平台订单状态分布堆叠柱状图
def plot_24_platform_status_stack_bar():
    platform_status = erp_df.groupby(['platform', 'status'], observed=True).size().reset_index(
        name='count')  # 修复警告：添加observed=True
    platform_status_pivot = platform_status.pivot(index='platform', columns='status', values='count').fillna(0)

    fig, ax = plt.subplots(figsize=(14, 7))
    platform_status_pivot.plot(kind='bar', stacked=True, color=colors_mpl[:len(platform_status_pivot.columns)],
                               alpha=0.8, ax=ax)
    ax.set_xlabel('销售平台', fontsize=12)
    ax.set_ylabel('订单数量（笔）', fontsize=12)
    ax.set_title('各平台订单状态分布', fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='订单状态', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '24-平台订单状态堆叠图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ------------------------------
# 五、业务状态类图表（25-30张：订单/退款/发货分析）
# ------------------------------
# 25-订单状态分布饼图
def plot_25_order_status_pie():
    status_count = erp_df.groupby('status', observed=True).size().reset_index(name='count')  # 修复警告：添加observed=True
    status_count = status_count.sort_values('count', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        status_count['count'],
        labels=status_count['status'],
        autopct='%1.1f%%',
        colors=colors_mpl[:len(status_count)],
        startangle=90,
        wedgeprops=dict(edgecolor='white')
    )
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(11)
    ax.set_title('订单状态分布', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '25-订单状态饼图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 26-退款状态分布柱状图
def plot_26_refund_status_bar():
    refund_count = erp_df.groupby('refund_status', observed=True).size().reset_index(
        name='count')  # 修复警告：添加observed=True
    refund_count = refund_count.sort_values('count', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(refund_count['refund_status'], refund_count['count'],
                  color=['#2ca02c', '#ff7f0e', '#d62728', '#9467bd'], alpha=0.8)
    ax.set_xlabel('退款状态', fontsize=12)
    ax.set_ylabel('订单数量（笔）', fontsize=12)
    ax.set_title('退款状态分布', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    # 添加数值标签和占比
    total_count = refund_count['count'].sum()
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_count) * 100
        ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                f'{int(height)}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '26-退款状态柱状图.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 27-发货时长分布直方图
def plot_27_shipping_duration_hist():
    # 过滤发货时长>10天的异常值
    shipping_data = erp_df[erp_df['shipping_duration'] <= 10]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data=shipping_data,
        x='shipping_duration',
        bins=range(1, int(shipping_data['shipping_duration'].max()) + 2),
        color='#d62728',
        alpha=0.7,
        kde=True,
        ax=ax
    )
    # 添加平均值线
    avg_duration = shipping_data['shipping_duration'].mean()
    ax.axvline(x=avg_duration, color='blue', linestyle='--', alpha=0.8, label=f'平均：{avg_duration:.1f}天')
    ax.legend()

    ax.set_xlabel('发货时长（天）', fontsize=12)
    ax.set_ylabel('订单数量（笔）', fontsize=12)
    ax.set_title('发货时长分布（付款到发货）', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(range(1, int(shipping_data['shipping_duration'].max()) + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '27-发货时长直方图.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_28_gift_vs_non_gift():
    # 按is_gift（字符串“是”/“否”）分组统计
    gift_data = erp_df.groupby('is_gift', observed=True).agg({
        'quantity': 'sum',
        'product_amount': 'sum'
    }).reset_index()

    # 映射中文标签
    gift_data['category'] = gift_data['is_gift'].map({'是': '赠品', '否': '非赠品'})
    # 补全缺失类别（确保有“赠品”和“非赠品”）
    full_categories = pd.DataFrame({'category': ['赠品', '非赠品']})
    gift_data = pd.merge(full_categories, gift_data, on='category', how='left').fillna(0)

    # 宽表转长表，用于分组柱状图
    gift_long = gift_data.melt(
        id_vars='category',
        value_vars=['quantity', 'product_amount'],
        var_name='指标',
        value_name='数值'
    )
    # 重命名指标并区分量级
    gift_long['指标'] = gift_long['指标'].map({
        'quantity': '销量（件）',
        'product_amount': '销售额（元）'
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    # 绘制分组柱状图（不同指标用不同颜色）
    sns.barplot(
        data=gift_long,
        x='category',
        y='数值',
        hue='指标',
        palette=['#1f77b4', '#2ca02c'],
        ax=ax
    )

    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + height * 0.02,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=11
            )

    # 设置图表样式
    ax.set_xlabel('是否赠品', fontsize=12)
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('赠品vs非赠品 销量与销售额对比', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='指标', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 保存图表
    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, '28-赠品非赠品对比.png'),
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close()

# 29-月度退款率趋势图
def plot_29_monthly_refund_rate():
    monthly_refund = erp_df.groupby('month', observed=True).agg({  # 修复警告：添加observed=True
        'id': 'count',  # 总订单数
        'refund_status': lambda x: (x != '无退款').sum()  # 退款订单数
    }).reset_index()
    monthly_refund['refund_rate'] = (monthly_refund['refund_status'] / monthly_refund['id'] * 100).round(2)
    monthly_refund['month'] = pd.to_datetime(monthly_refund['month'])
    monthly_refund = monthly_refund.sort_values('month')
    monthly_refund['month_str'] = monthly_refund['month'].dt.strftime('%Y-%m')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly_refund['month_str'], monthly_refund['refund_rate'],
            color='#d62728', linewidth=3, marker='o', markersize=6)
    ax.fill_between(monthly_refund['month_str'], monthly_refund['refund_rate'], alpha=0.3, color='#d62728')
    # 添加平均值线
    avg_rate = monthly_refund['refund_rate'].mean()
    ax.axhline(y=avg_rate, color='blue', linestyle='--', alpha=0.8, label=f'平均退款率：{avg_rate:.1f}%')
    ax.legend()

    ax.set_xlabel('月份', fontsize=12)
    ax.set_ylabel('退款率（%）', fontsize=12)
    ax.set_title('2024年月度退款率趋势', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(0, len(monthly_refund['month_str']), 1))
    ax.set_xticklabels(monthly_refund['month_str'], rotation=45)
    ax.grid(True, alpha=0.3)
    # 添加数值标签
    for i, v in enumerate(monthly_refund['refund_rate']):
        ax.text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '29-月度退款率趋势.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 30-综合业务指标看板（替换Pyecharts大屏为Matplotlib多子图看板）
def plot_30_business_dashboard():
    # 计算核心指标
    total_sales = erp_df['product_amount'].sum()
    total_orders = len(erp_df)
    total_users = erp_df['full_channel_user_id'].nunique()

    # 修复1：处理total_users=0的情况，避免除以0
    if total_users == 0:
        repurchase_rate = 0.0
    else:
        repurchase_count = erp_df[erp_df['is_repurchase']]['full_channel_user_id'].nunique()
        repurchase_rate = round(repurchase_count / total_users * 100, 2)  # 用Python内置round()

    # 修复2：avg_order_amount用round()
    avg_order_amount = round(erp_df['product_amount'].mean(), 2) if total_orders > 0 else 0.0

    # 修复3：refund_rate用round()
    refund_count = erp_df[erp_df['refund_status'] != '无退款'].shape[0]
    refund_rate = round(refund_count / total_orders * 100, 2) if total_orders > 0 else 0.0


    # 准备子图数据
    monthly_sales = erp_df.groupby('month', observed=True)['product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    monthly_sales['month'] = pd.to_datetime(monthly_sales['month'])
    monthly_sales = monthly_sales.sort_values('month')
    brand_top5 = erp_df.groupby('brand', observed=True)['product_amount'].sum().nlargest(
        5).reset_index()  # 修复警告：添加observed=True
    region_sales = erp_df.groupby('region', observed=True)['product_amount'].sum().reset_index()  # 修复警告：添加observed=True

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('服装订单业务综合指标看板', fontsize=18, fontweight='bold', y=0.95)

    # 1. 核心指标卡片（第一行）
    metrics = [
        ('总销售额', f'{total_sales:,.0f} 元', '#1f77b4'),
        ('总订单数', f'{total_orders} 笔', '#ff7f0e'),
        ('总用户数', f'{total_users} 人', '#2ca02c'),
        ('复购率', f'{repurchase_rate}%', '#d62728'),
        ('平均客单价', f'{avg_order_amount:.0f} 元', '#9467bd'),
        ('退款率', f'{refund_rate:.1f}%', '#8c564b')
    ]
    for i, (title, value, color) in enumerate(metrics):
        ax = fig.add_subplot(3, 6, i + 1)
        ax.text(0.5, 0.7, title, ha='center', va='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.3, value, ha='center', va='center', fontsize=14, fontweight='bold', color=color,
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=True, color=color, alpha=0.1, transform=ax.transAxes))

    # 2. 月度销售额趋势（第二行左）
    ax1 = fig.add_subplot(3, 3, 4)
    ax1.plot(monthly_sales['month'].dt.strftime('%Y-%m'), monthly_sales['product_amount'],
             color='#1f77b4', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('月份', fontsize=10)
    ax1.set_ylabel('销售额（元）', fontsize=10)
    ax1.set_title('月度销售额趋势', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 3. 品牌销售额TOP5（第二行中）
    ax2 = fig.add_subplot(3, 3, 5)
    ax2.bar(brand_top5['brand'], brand_top5['product_amount'], color=colors_mpl[:5], alpha=0.8)
    ax2.set_xlabel('品牌', fontsize=10)
    ax2.set_ylabel('销售额（元）', fontsize=10)
    ax2.set_title('品牌销售额TOP5', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # 4. 区域销售额分布（第二行右）
    ax3 = fig.add_subplot(3, 3, 6)
    ax3.pie(region_sales['product_amount'], labels=region_sales['region'], autopct='%1.1f%%',
            colors=colors_mpl[:len(region_sales)], startangle=90)
    ax3.set_title('区域销售额分布', fontsize=12, fontweight='bold')

    # 5. 订单状态分布（第三行左）
    status_count = erp_df.groupby('status', observed=True).size().reset_index(name='count')  # 修复警告：添加observed=True
    ax4 = fig.add_subplot(3, 3, 7)
    ax4.pie(status_count['count'], labels=status_count['status'], autopct='%1.1f%%',
            colors=colors_mpl[:len(status_count)], startangle=90)
    ax4.set_title('订单状态分布', fontsize=12, fontweight='bold')

    # 6. 价格区间销售额占比（第三行中）
    price_sales = erp_df.groupby('price_range', observed=True)[
        'product_amount'].sum().reset_index()  # 修复警告：添加observed=True
    ax5 = fig.add_subplot(3, 3, 8)
    ax5.pie(price_sales['product_amount'], labels=price_sales['price_range'], autopct='%1.1f%%',
            colors=colors_mpl[:len(price_sales)], startangle=90)
    ax5.set_title('价格区间销售额占比', fontsize=12, fontweight='bold')

    # 7. 日内订单量分布（第三行右）
    hourly_order = erp_df.groupby('hour', observed=True).size().reset_index(name='order_count')  # 修复警告：添加observed=True
    ax6 = fig.add_subplot(3, 3, 9)
    ax6.plot(hourly_order['hour'], hourly_order['order_count'], color='#2ca02c', linewidth=2)
    ax6.set_xlabel('小时', fontsize=10)
    ax6.set_ylabel('订单量（笔）', fontsize=10)
    ax6.set_title('日内订单量分布', fontsize=12, fontweight='bold')
    ax6.set_xticks(range(0, 24, 4))
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, '30-业务综合指标看板.png'), dpi=300, bbox_inches='tight')
    plt.close()



# ------------------------------
# 执行所有图表生成
# ------------------------------
if __name__ == "__main__":
    print("开始生成30张服装订单业务可视化图表 ...")

    # 一、趋势类图表（1-5）
    plot_01_monthly_sales_order_trend()
    plot_02_quarterly_brand_sales()
    plot_03_weekly_sales_trend()
    plot_04_hourly_order_distribution()
    plot_05_weekend_weekday_comparison()

    # 二、商品分析类图表（6-12）
    plot_06_brand_sales_top10()
    plot_07_product_type_sales_pie()
    plot_08_color_sales_distribution()
    plot_09_spec_sales_heatmap()
    plot_10_price_range_pie()
    plot_11_brand_discount_boxplot()
    plot_12_product_type_sales_bar()

    # 三、用户行为类图表（13-18）
    plot_13_user_repurchase_pie()
    plot_14_user_purchase_count_hist()
    plot_15_payment_duration_violin()
    plot_16_quantity_amount_scatter()
    plot_17_repurchase_vs_new_consume()
    plot_18_user_purchase_freq_comparison()

    # 四、区域与渠道类图表（19-24）
    plot_19_province_sales_bar()
    plot_20_region_sales_bar()
    plot_21_city_sales_top15()
    plot_22_platform_monthly_sales()
    plot_23_store_sales_top10()
    plot_24_platform_status_stack_bar()

    # 五、业务状态类图表（25-30）
    plot_25_order_status_pie()
    plot_26_refund_status_bar()
    plot_27_shipping_duration_hist()
    plot_28_gift_vs_non_gift()
    plot_29_monthly_refund_rate()
    plot_30_business_dashboard()

    print(f"30张PNG静态图表已全部生成完成！保存路径：{figures_dir}")