import math
from keras.layers import Conv1D, MaxPooling1D, Flatten, Bidirectional, LSTM, Dense, Dropout
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib

# 设置 Matplotlib 后端和字体
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
df = pd.read_csv('data.csv')
data = df.values

# 划分训练和测试集
train_size = 1190
train_data = data[:train_size]
test_data = data[train_size:]

window_size = 7
X_train, y_train = train_data[:, :window_size], train_data[:, window_size]
X_test, y_test = test_data[:, :window_size], test_data[:, window_size]

# 将输入数据重塑为CNN层的格式
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 归一化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, window_size)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, window_size)).reshape(X_test.shape)

# 构建模型
model_cnn_bilstm = Sequential()
model_cnn_bilstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, 1)))
model_cnn_bilstm.add(MaxPooling1D(pool_size=2))
model_cnn_bilstm.add(Dropout(0.1)) 
model_cnn_bilstm.add(Bidirectional(LSTM(64, activation='relu')))
model_cnn_bilstm.add(Dense(64, activation='relu'))
model_cnn_bilstm.add(Dropout(0.1))  
model_cnn_bilstm.add(Dense(32, activation='relu'))
model_cnn_bilstm.add(Dense(1))

# 自定义 Adam 优化器参数
custom_adam = Adam(learning_rate=0.008, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

def scheduler(epoch, lr):
    if epoch == 12:
        return lr * 0.5
    return lr

def lr_scheduler(epoch, lr):
    if epoch > 0 and epoch % 20 == 0:
        return lr * 0.9
    return lr

# 设置早停和学习率调度器
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=2, min_lr=1e-6)
# reduce_lr = LearningRateScheduler(lr_scheduler)

# 编译模型
model_cnn_bilstm.compile(optimizer=custom_adam, loss='mse')

# 训练模型
history = model_cnn_bilstm.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.2)
model_cnn_bilstm.summary()

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
y_pred = model_cnn_bilstm.predict(X_test)

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
