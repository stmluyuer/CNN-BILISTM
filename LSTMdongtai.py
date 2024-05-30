import math
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib
import pandas as pd
import numpy as np

# 设置 Matplotlib 后端和字体
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
df = pd.read_csv('data.csv')

# 假设最后一列是目标变量
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 生成多项式特征和交叉特征
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# 将多项式特征转换为 DataFrame，并保留原始特征名
poly_feature_names = poly.get_feature_names_out(X.columns)
X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)

# 使用随机森林进行特征选择
forest = RandomForestRegressor(
    n_estimators=200,            
    criterion='squared_error',   
    max_depth=None,              
    min_samples_split=2,         
    min_samples_leaf=1,          
    max_features='sqrt',       
    bootstrap=True,               
    oob_score=True,               
    random_state=42,              
    n_jobs=-1                     
)
forest.fit(X_poly_df, y)

# 输出每个特征的重要性
feature_importances = forest.feature_importances_
for feature, importance in zip(X_poly_df.columns, feature_importances):
    print(f"Feature: {feature}, Importance: {importance}")
    
# 使用特征重要性进行选择
model = SelectFromModel(forest, threshold='mean', prefit=True)
X_selected = model.transform(X_poly_df)

# 输出选择后的特征
selected_features = poly_feature_names[(model.get_support())]
print(f"Selected features: {selected_features}")

# 使用选择的重要特征创建新的 DataFrame
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# 将选定的特征和目标变量合并为新的 DataFrame
df_selected = pd.concat([X_selected_df, y], axis=1)

# 划分训练和测试集
train_data, test_data = train_test_split(df_selected.values, test_size=0.5, random_state=42)

# 查看划分后的数据大小
print(f"训练集大小: {len(train_data)}")
print(f"测试集大小: {len(test_data)}")

def print_data_stats(data, label):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    print(f"{label} - Mean: {mean}, Std: {std}")

print_data_stats(train_data, "Train Data")
print_data_stats(test_data, "Test Data")

# 准备输入和输出数据
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# 将输入数据重塑为LSTM层的格式
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 归一化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

print(f"特征数量：{X_test.shape[1]}")
# 构建模型
model_lstm = Sequential()
model_lstm.add(LSTM(128, return_sequences=True, input_shape=(X_test.shape[1], 1)))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(64, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(32))
model_lstm.add(Dense(16))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(32))
model_lstm.add(Dense(1))


from keras.optimizers import SGD

# 使用SGD优化器
optimizer = SGD(learning_rate=0.01, momentum=0.9)

from keras.optimizers import RMSprop

# 使用RMSprop优化器
optimizer = RMSprop(learning_rate=0.01)

# 自定义 Adam 优化器参数
custom_adam = Adam(learning_rate=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)


# 编译模型
model_lstm.compile(optimizer=custom_adam, loss='mse')

# 设置早停和学习率调度器
early_stopping = EarlyStopping(monitor='val_loss', patience=15)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6)

# 训练模型
history = model_lstm.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, 
                         callbacks=[early_stopping,reduce_lr])
model_lstm.summary()

# 可视化损失函数的训练曲线
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.figure(figsize=(20, 8))
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 预测
y_pred = model_lstm.predict(X_test)

# 可视化预测结果
plt.figure(figsize=(20, 8))
plt.plot(y_test, color='b', label='Actual')
plt.plot(y_pred, color='r', label='Predicted')
plt.xlabel('Timestamps')
plt.ylabel('Values')
plt.legend()

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MSE: {:.8f}'.format(mse))
print('RMSE: {:.4f}'.format(rmse))
print('MAE: {:.4f}'.format(mae))
print('R^2: {:.4f}'.format(r2))

# 计算残差
residuals = y_test - y_pred.flatten()

# 绘制残差图
plt.figure(figsize=(20, 8))
plt.plot(residuals, 'b-')
plt.xlabel('检测数据')
plt.ylabel('残差')
plt.title('残差图')

# 确定 bin 的数量和范围
bins = 80
range_min = np.min(residuals)
range_max = np.max(residuals)

# 绘制残差直方图和拟合的正态分布曲线
plt.figure(figsize=(20, 8))
n, bins, patches = plt.hist(residuals, bins=bins, range=(range_min, range_max), density=True, alpha=0.6, color='b', edgecolor='black')
mu, std = norm.fit(residuals)  # 拟合正态分布参数
x = np.linspace(range_min, range_max, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r-', linewidth=2)
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差直方图和正态分布拟合曲线')

# 最后一次性显示所有图像
# plt.show()
