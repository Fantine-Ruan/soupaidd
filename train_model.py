import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

print("🤖 开始训练汤品预测AI...")

# ===== 第1步：加载清洗好的数据 =====
current_folder = os.path.dirname(os.path.abspath(__file__))
history_df = pd.read_csv(os.path.join(current_folder, 'history_cleaned.csv'), encoding='utf-8')

print(f"📊 加载数据：{len(history_df)}条历史记录")

# ===== 第2步：准备训练数据（告诉AI：看什么特征→预测什么）=====
print("\n🔧 准备训练数据...")

# 特征X（AI要观察的"线索"）：天气、温度、季节、是否周末、反馈分数等
feature_columns = [
    '温度', '天气编码', '月份', '季节编码', '是否周末', '反馈分数'
]

# 自动添加食材特征（所有以"食材_"开头的列）
ingredient_cols = [col for col in history_df.columns if col.startswith('食材_')]
feature_columns.extend(ingredient_cols)

print(f"   使用特征：{len(feature_columns)}个")
print(f"   包括：天气、温度、季节、是否周末、反馈分数、{len(ingredient_cols)}种食材")

# 构建X（特征矩阵）
X = history_df[feature_columns].fillna(0)  # 如果有空值填0

# 构建y（标签：要预测的目标——汤名）
y = history_df['汤名']

# 把汤名转成数字（AI只认识数字，不认识中文）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"   预测目标：{len(label_encoder.classes_)}种汤")
print(f"   包括：{', '.join(label_encoder.classes_)}")

# ===== 第3步：划分训练集和测试集 =====
# 因为数据少，我们用80%训练，20%测试（如果只有10条，就8条训练，2条测试）
if len(history_df) >= 15:  # 提高到15条才做测试
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"   训练集：{len(X_train)}条，测试集：{len(X_test)}条")
    except ValueError as e:
        # 如果某种汤只出现1次，分层抽样会失败，改成随机划分
        print(f"   注意：某些汤品记录太少（只喝过1次），无法分层测试")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42  # 去掉 stratify
        )
        print(f"   改用随机划分：训练集{len(X_train)}条，测试集{len(X_test)}条")
else:
    # 数据太少，全部训练，不测准确率（反正学了总比不学好）
    X_train, y_train = X, y_encoded
    X_test, y_test = None, None
    print(f"   数据较少（{len(history_df)}条），全部用于学习（暂不测试准确率）")

# ===== 第4步：训练模型（核心！）=====
print("\n🎯 开始训练随机森林模型...")

# 创建模型（随机森林：由100棵决策树投票决定，不容易错）
model = RandomForestClassifier(
    n_estimators=100,      # 100棵树投票
    max_depth=5,           # 树不要太深（防止死记硬背）
    min_samples_split=2,   # 最少2个样本才分叉
    random_state=42,       # 固定随机种子（每次结果一样）
    class_weight='balanced' # 如果某汤出现少，也公平对待
)

# 开始训练（拟合）
model.fit(X_train, y_train)
print("   ✅ 模型训练完成！")

# ===== 第5步：评估模型（看看学得怎么样）=====
if X_test is not None:
    print("\n📈 模型评估：")
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   预测准确率：{accuracy*100:.1f}%")
    
    # 详细报告（看每种汤预测得准不准）
    print("\n   详细报告：")
    target_names = label_encoder.inverse_transform(np.unique(y_test))
    print(classification_report(y_test, y_pred, target_names=target_names))
else:
    print("\n⚠️  数据量小，跳过评估（建议积累20条以上数据再评估）")

# ===== 第6步：看看AI最看重什么特征（可解释性）=====
print("\n🔍 AI决策依据（特征重要性）：")

# 获取特征重要性
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    '特征': feature_columns,
    '重要性': importances
}).sort_values('重要性', ascending=False)

# 显示前5个重要特征
print("   最重要的5个因素：")
for idx, row in feature_importance_df.head(5).iterrows():
    print(f"   {row['特征']}: {row['重要性']*100:.1f}%")

# ===== 第7步：保存模型（下次直接加载用）=====
print("\n💾 保存模型文件...")

# 保存模型
model_path = os.path.join(current_folder, 'soup_predictor_model.pkl')
joblib.dump(model, model_path)

# 保存标签编码器（把数字变回汤名用）
encoder_path = os.path.join(current_folder, 'label_encoder.pkl')
joblib.dump(label_encoder, encoder_path)

# 保存特征列表（预测时要知道有哪些特征）
feature_path = os.path.join(current_folder, 'feature_columns.pkl')
joblib.dump(feature_columns, feature_path)

print(f"   ✅ 模型已保存：soup_predictor_model.pkl")
print(f"   ✅ 标签映射已保存：label_encoder.pkl")
print(f"\n🎉 训练完成！你的AI已经学会了{len(label_encoder.classes_)}种汤的配方！")

# ===== 彩蛋：测试预测明天 =====
print("\n🔮 做个小测试：假设明天是情况，AI会推荐什么？")

# 假设明天：周六，晴天，20度，春天，有排骨和玉米
tomorrow = pd.DataFrame([{
    '温度': 20,
    '天气编码': 3,  # 晴
    '月份': 3,
    '季节编码': 1,  # 春
    '是否周末': 1,  # 是周末
    '反馈分数': 80
}])

# 补齐食材列（没有的就填0）
for col in ingredient_cols:
    if col not in tomorrow.columns:
        tomorrow[col] = 0

# 假设冰箱里有排骨和玉米（设为1）
if '食材_排骨' in tomorrow.columns:
    tomorrow['食材_排骨'] = 1
if '食材_玉米' in tomorrow.columns:
    tomorrow['食材_玉米'] = 1
