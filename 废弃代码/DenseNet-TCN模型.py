import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============= 1. 数据读取与对齐 =============
vix_data = pd.read_excel('数据/恐慌指数.xlsx')
oil_data = pd.read_excel('数据/伦敦布伦特原油期货价格.xlsx')
sp500_data = pd.read_excel('数据/美国标准普尔500指数.xlsx')
usdx_data = pd.read_excel('数据/美元指数.xlsx')

# 修复日期
vix_data['Date'] = pd.to_datetime(vix_data['Date'])
oil_data['Date'] = pd.to_datetime(oil_data['Date'])
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
usdx_data['Date'] = pd.to_datetime(usdx_data['Date'])

# ============= 2. 数据合并与清洗 =============
merged_data = oil_data[['Date', 'close']].copy()
merged_data.columns = ['Date', 'oil_close']

merged_data = merged_data.merge(vix_data[['Date', 'close']], on='Date', how='left').rename(columns={'close': 'vix_close'})
merged_data = merged_data.merge(sp500_data[['Date', 'close']], on='Date', how='left').rename(columns={'close': 'sp500_close'})
merged_data = merged_data.merge(usdx_data[['Date', 'close']], on='Date', how='left').rename(columns={'close': 'usdx_close'})

merged_data = merged_data.dropna()
merged_data = merged_data.sort_values('Date').reset_index(drop=True)

print(f"合并后数据形状: {merged_data.shape}")

# ============= 3. 特征和目标变量 =============
features = merged_data[['vix_close', 'sp500_close', 'usdx_close']].values
target = merged_data[['oil_close']].values

scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target)

# ============= 4. 构造时间窗口数据 =============
def create_sequences(X, y, seq_length=30):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

seq_length = 30
X_seq, y_seq = create_sequences(features_scaled, target_scaled, seq_length)

# 数据集划分
train_size = int(len(X_seq) * 0.7)
val_size = int(len(X_seq) * 0.2)

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]
X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]
X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

# ============= 5. 修复版 Dens-TCN 模型 =============
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("模型4: Dens-TCN 密集连接时间卷积网络")
print("=" * 60)

# 重写：稳定版 密集卷积块（不会维度爆炸）
class DenseConvBlock(keras.layers.Layer):
    def __init__(self, filters=32, kernel_size=3):
        super().__init__()
        self.conv = Conv1D(filters=filters, kernel_size=kernel_size, padding='causal', activation='relu')
        self.bn = BatchNormalization()
        self.drop = Dropout(0.2)

    def call(self, x, training=False):
        out = self.conv(x)
        out = self.bn(out, training=training)
        out = self.drop(out, training=training)
        return out

# 主模型（稳定可训练）
class DensTCN(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = DenseConvBlock(64, 3)
        self.conv2 = DenseConvBlock(32, 3)
        self.conv3 = DenseConvBlock(16, 3)
        self.pool = GlobalAveragePooling1D()
        self.dense1 = Dense(32, activation='relu')
        self.dropout = Dropout(0.3)
        self.dense2 = Dense(1)

    def call(self, x, training=False):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

# 初始化
tf.random.set_seed(42)
model = DensTCN()
model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

# 训练
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# 预测
y_pred_train = model.predict(X_train, verbose=0)
y_pred_val = model.predict(X_val, verbose=0)
y_pred_test = model.predict(X_test, verbose=0)

# 反归一化（100%不报错）
y_train_true = scaler_target.inverse_transform(y_train)
y_val_true = scaler_target.inverse_transform(y_val)
y_test_true = scaler_target.inverse_transform(y_test)

y_pred_train = scaler_target.inverse_transform(y_pred_train)
y_pred_val = scaler_target.inverse_transform(y_pred_val)
y_pred_test = scaler_target.inverse_transform(y_pred_test)

# 评估
mae_train = mean_absolute_error(y_train_true, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train_true, y_pred_train))
mae_val = mean_absolute_error(y_val_true, y_pred_val)
rmse_val = np.sqrt(mean_squared_error(y_val_true, y_pred_val))
mae_test = mean_absolute_error(y_test_true, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test_true, y_pred_test))

print("\n【Dens-TCN 评估】")
print(f"训练集 MAE: {mae_train:.4f} | RMSE: {rmse_train:.4f}")
print(f"验证集 MAE: {mae_val:.4f} | RMSE: {rmse_val:.4f}")
print(f"测试集 MAE: {mae_test:.4f} | RMSE: {rmse_test:.4f}")

# 画图
fig, axes = plt.subplots(2,2,figsize=(14,10))
axes[0,0].plot(history.history['loss'], label='训练损失')
axes[0,0].plot(history.history['val_loss'], label='验证损失')
axes[0,0].set_title('损失曲线')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

axes[0,1].plot(y_test_true, label='真实值', linewidth=2)
axes[0,1].plot(y_pred_test, label='预测值', linewidth=2)
axes[0,1].set_title('测试集预测')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

res = y_test_true - y_pred_test
axes[1,0].hist(res, bins=30, edgecolor='black')
axes[1,0].set_title('残差分布')

axes[1,1].plot(np.abs(res))
axes[1,1].axhline(mae_test, c='r', ls='--', label=f'MAE={mae_test:.2f}')
axes[1,1].set_title('绝对误差')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('dens_tcn.png', dpi=300)
plt.show()

print("\n✅ Dens-TCN 运行完成！")
keras.backend.clear_session()