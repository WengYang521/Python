# 零售数据可视化分析项目
Retail Data Visualization & Business Analysis

## 项目简介
本项目基于Python技术栈，完成零售行业模拟数据生成、30种经典可视化图表复现与交互式数据大屏构建，旨在为商业数据分析提供可复用的技术方案与决策参考。

## 项目结构

retail-data-visualization/
├── data/                  # 生成的数据文件（CSV + SQLite）
├── scripts/               # 核心代码脚本
│   ├── data_generator.py  # 模拟数据生成（客户/商品/订单/库存）
│   └── visualization_basic.py  # 30张静态图表（Matplotlib/Seaborn）
├── figures/               # 可视化输出图表（PNG + HTML）
├── requirements.txt       # 依赖包清单
└── README.md              # 项目说明文档

## 核心功能
### 1. 模拟数据生成
- 基于Faker库生成1200条逻辑自洽的零售数据（客户300人、商品100种、订单700笔、库存100条）
- 支持CSV和SQLite两种格式导出，字段符合零售业务逻辑（如毛利率15%-40%、消费人群18-65岁）
- 数据包含客户、商品、订单、库存4大核心表，支持外键关联

### 2. 可视化图表复现（共30张）
| 类型       | 数量 | 技术工具             | 代表图表                                     |
| ---------- | ---- | -------------------- | -------------------------------------------- |
| 基础静态图 | 15   | Matplotlib + Seaborn | 月度销售额折线图、商品类别饼图、库存热力图等 |
| 交互式图表 | 15   | Plotly               | 3D客户RFM分析、区域销售地图、多维度联动图等  |

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

### 2. 生成基础静态图表

```bash
python visualization_basic.py
```

- 15张PNG格式图表将保存至 `../figures` 目录

### 3. 生成交互式图表

```bash
python visualization_interactive.py
```

- 15张HTML格式交互式图表（支持hover、缩放、筛选）将保存至 `../figures` 目录

