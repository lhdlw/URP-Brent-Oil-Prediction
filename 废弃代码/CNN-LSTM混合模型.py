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

# ✅ 修复：日期正确转换（删除错误函数）
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
print(f"时间范围: {merged_data['Date'].min()} 到 {merged_data['Date'].max()}")

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

print(f"\n训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

# ============= 5. CNN-LSTM 模型 =============
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

print("=" * 60)
print("模型3: CNN-LSTM 混合模型")
print("=" * 60)

tf.random.set_seed(42)

cnn_lstm_model = Sequential([
    # CNN 特征提取
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # LSTM 时序学习
    LSTM(units=64, activation='relu', return_sequences=True),
    Dropout(0.2),
    LSTM(units=32, activation='relu'),
    Dropout(0.2),

    # 全连接输出
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

cnn_lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print("\nCNN-LSTM模型结构:")
cnn_lstm_model.summary()

# 训练
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
cnn_lstm_history = cnn_lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# 预测
y_train_pred_cnn_lstm = cnn_lstm_model.predict(X_train, verbose=0)
y_val_pred_cnn_lstm = cnn_lstm_model.predict(X_val, verbose=0)
y_test_pred_cnn_lstm = cnn_lstm_model.predict(X_test, verbose=0)

# ===================== ✅ 终极修复：反归一化（绝对不报错） =====================
y_train_actual = scaler_target.inverse_transform(y_train)
y_val_actual = scaler_target.inverse_transform(y_val)
y_test_actual = scaler_target.inverse_transform(y_test)

y_train_pred_cnn_lstm = scaler_target.inverse_transform(y_train_pred_cnn_lstm)
y_val_pred_cnn_lstm = scaler_target.inverse_transform(y_val_pred_cnn_lstm)
y_test_pred_cnn_lstm = scaler_target.inverse_transform(y_test_pred_cnn_lstm)
# ===========================================================================

# 计算指标
mae_train_cnn_lstm = mean_absolute_error(y_train_actual, y_train_pred_cnn_lstm)
rmse_train_cnn_lstm = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_cnn_lstm))
mae_val_cnn_lstm = mean_absolute_error(y_val_actual, y_val_pred_cnn_lstm)
rmse_val_cnn_lstm = np.sqrt(mean_squared_error(y_val_actual, y_val_pred_cnn_lstm))
mae_test_cnn_lstm = mean_absolute_error(y_test_actual, y_test_pred_cnn_lstm)
rmse_test_cnn_lstm = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_cnn_lstm))

print(f"\n【CNN-LSTM 模型性能评估】")
print(f"训练集 - MAE: {mae_train_cnn_lstm:.4f}, RMSE: {rmse_train_cnn_lstm:.4f}")
print(f"验证集 - MAE: {mae_val_cnn_lstm:.4f}, RMSE: {rmse_val_cnn_lstm:.4f}")
print(f"测试集 - MAE: {mae_test_cnn_lstm:.4f}, RMSE: {rmse_test_cnn_lstm:.4f}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(cnn_lstm_history.history['loss'], label='Training Loss')
axes[0, 0].plot(cnn_lstm_history.history['val_loss'], label='Validation Loss')
axes[0, 0].set_title('CNN-LSTM - Loss Curve', fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(y_test_actual, label='实际价格', linewidth=2)
axes[0, 1].plot(y_test_pred_cnn_lstm, label='预测价格', linewidth=2, alpha=0.8)
axes[0, 1].set_title('CNN-LSTM 测试集预测结果', fontweight='bold')
axes[0, 1].set_ylabel('油价 ($/桶)')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

residuals_cnn_lstm = y_test_actual - y_test_pred_cnn_lstm
axes[1, 0].hist(residuals_cnn_lstm, bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('残差分布', fontweight='bold')

axes[1, 1].plot(np.abs(residuals_cnn_lstm), label='绝对误差', linewidth=1.5)
axes[1, 1].axhline(y=mae_test_cnn_lstm, color='r', linestyle='--', label=f'MAE={mae_test_cnn_lstm:.2f}')
axes[1, 1].set_title('预测绝对误差', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_lstm_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ CNN-LSTM 模型运行完成！图表已保存")
keras.backend.clear_session()