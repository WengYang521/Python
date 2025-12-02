import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')
os.makedirs(data_dir, exist_ok=True)
fake = Faker('zh_CN')
# 配置常量
ORDER_COUNT = 5000
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)

# -------------------------- 业务数据字典 --------------------------
BRANDS = ['耐克', '阿迪达斯', '李宁', '安踏', '优衣库', 'ZARA', 'H&M', '森马', '以纯', '美特斯邦威']
PRODUCT_TYPES = ['T恤', '衬衫', '卫衣', '外套', '牛仔裤', '休闲裤', '运动裤', '连衣裙', '毛衣', '风衣']
COLORS = ['黑色', '白色', '灰色', '蓝色', '红色', '绿色', '黄色', '紫色', '粉色', '卡其色']
SPECS = ['S', 'M', 'L', 'XL', 'XXL', 'XXXL', '均码']
SKU_PREFIXES = ['NK', 'AD', 'LN', 'AN', 'UN', 'ZR', 'HM', 'SM', 'YC', 'MT']
STORES = ['天猫旗舰店', '京东自营店', '抖音直播间', '拼多多专卖店', '线下直营店', '唯品会特卖店']
PLATFORMS = ['天猫', '京东', '抖音电商', '拼多多', '线下', '唯品会']
ORDER_STATUSES = ['已完成', '待发货', '已发货', '已取消', '退款中', '已退款']
REFUND_STATUSES = ['无退款', '待审核', '已退款', '退款失败']
IS_GIFT = ['否', '是']
PROVINCE_CITY_MAP = {
    '北京市': ['北京市'],
    '上海市': ['上海市'],
    '广东省': ['广州市', '深圳市', '佛山市', '东莞市'],
    '江苏省': ['南京市', '苏州市', '无锡市', '常州市'],
    '浙江省': ['杭州市', '宁波市', '温州市', '嘉兴市'],
    '山东省': ['济南市', '青岛市', '烟台市', '潍坊市'],
    '四川省': ['成都市', '绵阳市', '德阳市', '宜宾市'],
    '湖北省': ['武汉市', '宜昌市', '襄阳市', '荆州市'],
    '湖南省': ['长沙市', '株洲市', '湘潭市', '衡阳市'],
    '河南省': ['郑州市', '洛阳市', '开封市', '新乡市']
}


def generate_erp_order_data():
    """生成更贴合实际的服装订单数据"""
    data = []
    provinces = list(PROVINCE_CITY_MAP.keys())
    # 1. 月份权重（强化Q3月份权重）
    month_weights = {
        1: 0.7, 2: 0.8, 3: 0.9,  # Q1
        4: 1.2, 5: 1.5, 6: 1.6,  # Q2（本土品牌增长）
        7: 1.8, 8: 2.0, 9: 2.2,  # Q3（总销售额最高）
        10: 1.5, 11: 1.9, 12: 1.3  # Q4
    }

    # 2. 日内权重
    hour_weights = {
        0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.2,
        6: 0.3, 7: 0.5, 8: 1.2, 9: 1.5, 10: 1.2,
        11: 0.8, 12: 1.0, 13: 0.9, 14: 0.8,
        15: 0.7, 16: 0.6, 17: 0.8,
        18: 1.8, 19: 2.0, 20: 2.2, 21: 1.8, 22: 1.2,
        23: 0.5
    }

    # 3. 工作日/周末权重
    weekday_weights = {
        0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.3,  # 周一至周五
        5: 3.0, 6: 3.5  # 周六、周日
    }

    # 4. 季节商品匹配
    season_product_map = {
        '春': ['卫衣', '衬衫', '休闲裤'],
        '夏': ['T恤', '连衣裙', '运动裤'],
        '秋': ['牛仔裤', '风衣', '毛衣'],
        '冬': ['外套', '毛衣', '羽绒服']
    }

    # 5. 大促月金额加成
    promotion_month = [5, 6, 11]

    # 6. 品牌-季度权重映射（核心优化）
    brand_quarter_weight = {
        # 头部品牌（Q3权重最高，确保占比超60%）
        '耐克': {1: 1.5, 2: 1.8, 3: 2.5, 4: 2.2},
        '阿迪达斯': {1: 1.4, 2: 1.7, 3: 2.4, 4: 2.1},
        '李宁': {1: 1.3, 2: 1.6, 3: 2.3, 4: 2.0},
        # 本土品牌（Q2权重提升）
        '森马': {1: 0.9, 2: 1.8, 3: 1.2, 4: 1.1},
        '以纯': {1: 0.8, 2: 1.7, 3: 1.1, 4: 1.0},
        # 其他品牌（权重降低）
        '安踏': {1: 0.7, 2: 0.8, 3: 0.9, 4: 0.8},
        '优衣库': {1: 0.6, 2: 0.7, 3: 0.8, 4: 0.7},
        'ZARA': {1: 0.5, 2: 0.6, 3: 0.7, 4: 0.6},
        'H&M': {1: 0.4, 2: 0.5, 3: 0.6, 4: 0.5},
        '美特斯邦威': {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.4}
    }

    for id in range(1, ORDER_COUNT + 1):
        # 1. 时间信息
        # 选月份
        month = random.choices(
            list(month_weights.keys()),
            weights=list(month_weights.values())
        )[0]
        quarter = (month - 1) // 3 + 1  # 计算季度（1-4）

        # 选日期
        while True:
            if month in [1, 3, 5, 7, 8, 10, 12]:
                day = random.randint(1, 31)
            elif month == 2:
                day = random.randint(1, 28)
            else:
                day = random.randint(1, 30)

            temp_date = datetime(2024, month, day)
            weekday = temp_date.weekday()
            if random.random() < weekday_weights[weekday]:
                break

        # 选小时
        hour = random.choices(
            list(hour_weights.keys()),
            weights=list(hour_weights.values())
        )[0]

        order_time = datetime(2024, month, day, hour, random.randint(0, 59))
        payment_date = order_time + timedelta(hours=random.randint(0, 24))
        shipping_date = payment_date + timedelta(days=random.randint(1, 3))

        # 2. 商品信息
        if month in [3, 4, 5]:
            season = '春'
        elif month in [6, 7, 8]:
            season = '夏'
        elif month in [9, 10, 11]:
            season = '秋'
        else:
            season = '冬'

        product_type = random.choice(season_product_map[season] + PRODUCT_TYPES[:3])

        # 按品牌-季度权重选择品牌
        brand = random.choices(
            BRANDS,
            weights=[brand_quarter_weight[b][quarter] for b in BRANDS]
        )[0]

        color = random.choice(COLORS)
        spec = random.choice(SPECS)
        product_name = f'{brand} {product_type} ({color}/{spec})'
        color_and_spec = f'{color}/{spec}'
        spu = f'SPU-{brand[:2].upper()}-{product_type[:2].upper()}'
        sku = f'{random.choice(SKU_PREFIXES)}-{random.randint(1000, 9999)}'

        # 3. 金额信息
        # 3月低价商品逻辑
        if month == 3:
            unit_price = round(random.uniform(59.00, 199.00), 2)
        elif month in promotion_month:
            unit_price = round(random.uniform(199.00, 2499.00), 2)
        else:
            unit_price = round(random.uniform(99.00, 1999.00), 2)

        # 头部品牌单价更高
        if brand in ['耐克', '阿迪达斯', '李宁']:
            unit_price *= random.uniform(1.2, 1.5)
        unit_price = round(unit_price, 2)

        quantity = random.randint(1, 5)
        product_amount = round(unit_price * quantity, 2)

        if month in promotion_month:
            original_price = round(product_amount * random.uniform(1.5, 2.0), 2)
        else:
            original_price = round(product_amount * random.uniform(1.05, 1.3), 2)

        payable_amount = original_price

        if month in promotion_month:
            paid_amount = payable_amount if random.random() > 0.05 else round(
                random.uniform(product_amount * 0.5, payable_amount), 2)
        else:
            paid_amount = payable_amount if random.random() > 0.1 else round(
                random.uniform(product_amount * 0.5, payable_amount), 2)

        # 4.其他信息
        internal_order_number = f'ORD-{random.randint(10000000, 99999999)}'
        online_order_number = f'ONL-{random.randint(1000000000, 9999999999)}'
        store_name = random.choice(STORES)
        platform = random.choice(PLATFORMS)
        full_channel_user_id = f'USER-{random.randint(10000, 99999)}'

        if paid_amount == payable_amount:
            status = random.choice(['已完成', '已发货', '待发货']) if month in promotion_month else random.choice(
                ['已完成', '已发货', '待发货', '已取消'])
        else:
            status = random.choice(['退款中', '已取消'])

        consignee = fake.name()
        province = random.choice(provinces)
        city = random.choice(PROVINCE_CITY_MAP[province])
        sub_order_number = f'SUB-{internal_order_number.split("-")[1]}-{random.randint(100, 999)}'
        online_sub_order_number = f'SONL-{online_order_number.split("-")[1]}-{random.randint(100, 999)}'
        original_online_order_number = online_order_number
        refund_status = random.choice(REFUND_STATUSES)
        registered_quantity = quantity
        actual_refund_quantity = quantity if refund_status == '已退款' else 0
        sub_order_status = status
        is_gift = random.choice(IS_GIFT) if random.random() > 0.9 else '否'

        # 组装数据
        data.append([
            id, internal_order_number, online_order_number, store_name, full_channel_user_id,
            shipping_date.strftime('%Y-%m-%d %H:%M:%S'), payment_date.strftime('%Y-%m-%d %H:%M:%S'),
            payable_amount, paid_amount, status, consignee, spu, order_time.strftime('%Y-%m-%d %H:%M:%S'),
            province, city, platform, sub_order_number, online_sub_order_number, original_online_order_number,
            sku, quantity, unit_price, product_name, color_and_spec, product_amount, original_price,
            is_gift, sub_order_status, refund_status, registered_quantity, actual_refund_quantity
        ])

    columns = [
        'id', 'internal_order_number', 'online_order_number', 'store_name', 'full_channel_user_id',
        'shipping_date', 'payment_date', 'payable_amount', 'paid_amount', 'status',
        'consignee', 'spu', 'order_time', 'province', 'city', 'platform',
        'sub_order_number', 'online_sub_order_number', 'original_online_order_number',
        'sku', 'quantity', 'unit_price', 'product_name', 'color_and_spec',
        'product_amount', 'original_price', 'is_gift', 'sub_order_status',
        'refund_status', 'registered_quantity', 'actual_refund_quantity'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df


def save_data(df):
    csv_path = os.path.join(data_dir, 'erp_order.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV文件已保存至: {csv_path}")
    excel_path = os.path.join(data_dir, 'erp_order.xlsx')
    try:
        df.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Excel文件已保存至: {excel_path}")
    except ImportError:
        print(f"\n提示：未安装 openpyxl，跳过 Excel 保存。")




def main():
    print("开始生成erp_order服装订单数据...")
    print(f"生成数量：{ORDER_COUNT} 条")
    erp_order_df = generate_erp_order_data()
    erp_order_df['brand'] = erp_order_df['product_name'].str.split(' ').str[0]
    save_data(erp_order_df)


if __name__ == "__main__":
    main()