# 零售数据可视化分析项目

Retail Data Visualization & Business Analysis

## 项目简介

本项目基于Python技术栈，完成零售行业模拟数据生成、30种经典可视化图表复现与交互式数据大屏构建，旨在为商业数据分析提供可复用的技术方案与决策参考。

## 项目结构



retail-data-visualization/

├── data/                  # 生成的数据文件（CSV + SQLite）

├── scripts/               # 核心代码脚本

│   ├── data_generator.py  # 模拟数据生成

│   ├── Mode.py            # 模型训练

│   └── visualization_basic.py  # 30张静态图表（Matplotlib/Seaborn）

├── figures/               # 可视化输出图表（PNG + HTML）

├── requirements.txt       # 依赖包清单

└── README.md              # 项目说明文档

## 核心功能

### 1. 模拟数据生成

- 基于Faker库生成3000条逻辑自洽的零售数据
- 支持CSV和SQLite两种格式导出，字段符合零售业务逻辑
- 数据包含客户、商品、订单、库存4大核心表，支持外键关联

### 3. 数据大屏

- 基于Pyecharts构建交互式数据大屏，整合KPI指标、趋势图、分布图等核心组件
- 支持浏览器直接打开，适配PC端展示

## 环境配置

### 依赖包安装

```bash
# 推荐使用Python 3.8+
pip install -r requirements.txt
```

### 环境要求

- Python版本：3.8+
- 操作系统：Windows/macOS/Linux（无特殊限制）
- 网络要求：安装依赖包时需联网，生成数据与可视化无需联网

## 快速使用

### 1. 生成模拟数据

```bash
cd scripts
python data_generator.py
```

- 数据将保存至 `../data` 目录，包含4个CSV文件和1个SQLite数据库

### 2. 生成图表

```bash
python visualization_basic.py
```

- 30张PNG格式图表将保存至 `../figures` 目录

### 3. 逻辑回归模型训练

```bash
python Model.py
```

