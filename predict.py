import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

print("🔮 SoupAIDD 预测系统启动！")
print("=" * 50)

# ===== 第1步：加载训练好的AI模型 =====
current_folder = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(current_folder, 'soup_predictor_model.pkl'))
    label_encoder = joblib.load(os.path.join(current_folder, 'label_encoder.pkl'))
    feature_columns = joblib.load(os.path.join(current_folder, 'feature_columns.pkl'))
    print("✅ AI模型加载成功！")
    print(f"   已学习汤品：{', '.join(label_encoder.classes_)}")
except FileNotFoundError:
    print("❌ 错误：找不到模型文件！请先运行 train_model.py 训练模型")
    exit()

# ===== 第2步：获取用户输入（明天的信息） =====
print("\n📅 请输入明天的信息：")

# 日期
date_input = input("日期（回车默认明天，或输入如2024-02-01）：").strip()
if not date_input:
    tomorrow = datetime.now() + timedelta(days=1)
    date_input = tomorrow.strftime("%Y-%m-%d")
    print(f"   使用默认：{date_input}")

# 星期
weekday = input("星期几（1=周一，7=周日，回车自动计算）：").strip()
if not weekday:
    weekday = str(datetime.strptime(date_input, "%Y-%m-%d").isoweekday())
    print(f"   自动判断：星期{weekday}")

# 天气
weather_map = {'晴': 3, '多云': 2, '阴': 1, '雨': 0, '小雨': 0, '潮湿': 1, '干燥': 3}
print("\n天气选项：晴(3), 多云(2), 阴(1), 雨(0)")
weather_str = input("明天天气：").strip()
weather_code = weather_map.get(weather_str, 2)  # 默认多云
print(f"   编码：{weather_code}")

# 温度
temp = input("明天温度（摄氏度，如25）：").strip()
if not temp:
    temp = "20"
temp = float(temp)
print(f"   温度：{temp}°C")

# 是否周末
is_weekend = 1 if weekday in ['6', '7'] else 0
print(f"   是否周末：{'是' if is_weekend else '否'}")

# 月份和季节
month = datetime.strptime(date_input, "%Y-%m-%d").month
season = 1 if month in [3,4,5] else 2 if month in [6,7,8] else 3 if month in [9,10,11] else 4
print(f"   月份：{month}月，季节编码：{season}")

# ===== 第3步：获取冰箱库存 =====
print("\n🥬 冰箱现在有什么食材？（输入学过的食材，用逗号分隔）")
print(f"   可选食材：{', '.join([col.replace('食材_', '') for col in feature_columns if col.startswith('食材_')])}")

inventory_input = input("库存食材：").strip()
inventory_list = [i.strip() for i in inventory_input.replace('，', ',').split(',') if i.strip()]

# ===== 第4步：构建预测数据 =====
print("\n🔧 分析中...")

# 创建数据框
tomorrow_data = pd.DataFrame([{
    '温度': temp,
    '天气编码': weather_code,
    '月份': month,
    '季节编码': season,
    '是否周末': is_weekend,
    '反馈分数': 75  # 默认中等期待
}])

# 添加食材特征（有就在冰箱里标1，没有标0）
for col in feature_columns:
    if col.startswith('食材_'):
        ingredient_name = col.replace('食材_', '')
        tomorrow_data[col] = 1 if ingredient_name in inventory_list else 0

# 确保列顺序和训练时一致
tomorrow_features = tomorrow_data[feature_columns]

# ===== 第5步：AI预测 =====
# 预测概率（看所有汤的可能性）
probabilities = model.predict_proba(tomorrow_features)[0]

# 获取排名前3的汤
top3_indices = np.argsort(probabilities)[-3:][::-1]  # 从大到小
top3_soups = label_encoder.inverse_transform(top3_indices)
top3_probs = probabilities[top3_indices]

# ===== 第6步：输出结果 =====
print("\n" + "=" * 50)
print("🍲 明天汤品预测结果：")
print("=" * 50)

print(f"\n🏆 第1推荐：【{top3_soups[0]}】")
print(f"   置信度：{top3_probs[0]*100:.1f}%")
print(f"   理由：", end="")

# 生成理由（可解释性）
reasons = []
if is_weekend:
    reasons.append("周末时间充裕，适合煲老火汤")
if weather_code <= 1:
    reasons.append("天气阴/雨，适合暖身汤品")
if temp < 15:
    reasons.append("气温较低，需要温补")
elif temp > 28:
    reasons.append("天气炎热，适合清淡解腻")

# 检查是否有相克食材（简单规则）
print(f"\n   冰箱库存：{', '.join(inventory_list) if inventory_list else '无特定食材'}")

# 检查缺什么食材（对比第一推荐的配方）
print(f"\n📋 如果要煲【{top3_soups[0]}】：")

# 从训练数据中找到这个汤的标准配方
history_df = pd.read_csv(os.path.join(current_folder, 'history_cleaned.csv'), encoding='utf-8')
soup_history = history_df[history_df['汤名'] == top3_soups[0]]

if not soup_history.empty:
    # 找出这种汤通常用什么食材
    typical_ingredients = []
    for col in feature_columns:
        if col.startswith('食材_') and soup_history[col].mean() > 0.5:
            typical_ingredients.append(col.replace('食材_', ''))
    
    print(f"   通常需要：{', '.join(typical_ingredients)}")
    
    # 检查缺什么
    missing = [ing for ing in typical_ingredients if ing not in inventory_list]
    if missing:
        print(f"   ⚠️  缺少食材：{', '.join(missing)}（建议购买）")
    else:
        print(f"   ✅ 食材齐全，可以开煲！")
else:
    print(f"   建议查看历史记录了解配方")

# 显示备选方案
if len(top3_soups) > 1:
    print(f"\n🥈 备选方案：")
    for i in range(1, len(top3_soups)):
        if top3_probs[i] > 0.05:  # 只显示概率>5%的
            print(f"   {i+1}. {top3_soups[i]} (概率{top3_probs[i]*100:.1f}%)")

print("\n" + "=" * 50)
print("💡 提示：预测基于历史数据，妈妈实际选择可能受心情影响！")
print("=" * 50)

# 保存预测记录（方便以后对比AI猜得准不准）
save_record = input("\n是否保存这次预测到记录？(y/n)：").strip().lower()
if save_record == 'y':
    record_file = os.path.join(current_folder, 'predictions_log.txt')
    with open(record_file, 'a', encoding='utf-8') as f:
        f.write(f"{date_input} | 预测：{top3_soups[0]} | 概率：{top3_probs[0]*100:.1f}% | 天气：{weather_str} {temp}°C\n")
    print("✅ 已保存到 predictions_log.txt")