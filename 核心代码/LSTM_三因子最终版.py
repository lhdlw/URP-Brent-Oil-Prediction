import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ===================== 中文与绘图配置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 读取数据 + 日期对齐插值 =====================
vix_data = pd.read_excel('../数据/恐慌指数.xlsx')
oil_data = pd.read_excel('../数据/伦敦布伦特原油期货价格.xlsx')
sp500_data = pd.read_excel('../数据/美国标准普尔500指数.xlsx')
usdx_data = pd.read_excel('../数据/美元指数.xlsx')

# 日期转换
vix_data['Date'] = pd.to_datetime(vix_data['Date'])
oil_data['Date'] = pd.to_datetime(oil_data['Date'])
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])
usdx_data['Date'] = pd.to_datetime(usdx_data['Date'])

# 合并
merged_data = oil_data[['Date', 'close']].copy()
merged_data.columns = ['Date', 'oil_close']
merged_data = merged_data.merge(vix_data[['Date','close']],on='Date',how='left').rename(columns={'close':'vix_close'})
merged_data = merged_data.merge(sp500_data[['Date','close']],on='Date',how='left').rename(columns={'close':'sp500_close'})
merged_data = merged_data.merge(usdx_data[['Date','close']],on='Date',how='left').rename(columns={'close':'usdx_close'})

# 关键优化1：全日历重采样 + 时间插值补缺失
merged_data = merged_data.set_index('Date')
merged_data = merged_data.resample('D').interpolate(method='time')
merged_data = merged_data.dropna()

# ===================== 2. 金融时序平稳化：对数+一阶差分 =====================
def log_diff(series):
    return np.log(series).diff()

cols = ['oil_close','vix_close','sp500_close','usdx_close']
for c in cols:
    merged_data[f'{c}_ld'] = log_diff(merged_data[c])

merged_data = merged_data.dropna()
# 建模使用平稳化后的差分序列
use_cols = ['vix_close_ld','sp500_close_ld','usdx_close_ld']
target_col = 'oil_close_ld'

features = merged_data[use_cols].values
target = merged_data[[target_col]].values

# ===================== 3. 归一化 =====================
scaler_features = MinMaxScaler(feature_range=(0,1))
scaler_target = MinMaxScaler(feature_range=(0,1))

features_scaled = scaler_features.fit_transform(features)
target_scaled = scaler_target.fit_transform(target)

# ===================== 4. 构造时序样本（窗口改为60） =====================
def create_sequences(X, y, seq_length=60):
    X_seq, y_seq = [], []
    for i in range(len(X)-seq_length):
        X_seq.append(X[i:i+seq_length,:])
        y_seq.append(y[i+seq_length,:])
    return np.array(X_seq), np.array(y_seq)

seq_length = 60
X_seq, y_seq = create_sequences(features_scaled, target_scaled, seq_length)

# 严格时间划分 7:1.5:1.5 无打乱
total_len = len(X_seq)
train_size = int(total_len * 0.7)
val_size = int(total_len * 0.15)

X_train, y_train = X_seq[:train_size], y_seq[:train_size]
X_val, y_val     = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
X_test, y_test   = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

print(f"训练集:{X_train.shape}, 验证集:{X_val.shape}, 测试集:{X_test.shape}")

# ===================== 5. 优化版LSTM模型 =====================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(42)

model = Sequential([
    LSTM(64, return_sequences=True, activation='tanh',
         input_shape=(X_train.shape[1], X_train.shape[2]),
         recurrent_dropout=0.2),
    Dropout(0.3),
    LSTM(32, return_sequences=False, activation='tanh', recurrent_dropout=0.2),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.0003), loss='mse', metrics=['mae'])

# 早停
early_stop = EarlyStopping(monitor='val_loss', patience=25,
                           restore_best_weights=True, mode='min')

# 时序必须 shuffle=False
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# ===================== 6. 预测 + 反归一化 =====================
y_train_pred = model.predict(X_train, verbose=0)
y_val_pred   = model.predict(X_val, verbose=0)
y_test_pred  = model.predict(X_test, verbose=0)

# 反归一化（差分收益率空间）
y_train_true = scaler_target.inverse_transform(y_train)
y_val_true   = scaler_target.inverse_transform(y_val)
y_test_true  = scaler_target.inverse_transform(y_test)

y_train_pred = scaler_target.inverse_transform(y_train_pred)
y_val_pred   = scaler_target.inverse_transform(y_val_pred)
y_test_pred  = scaler_target.inverse_transform(y_test_pred)

# ===================== 7. 计算评价指标 =====================
def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

mae_train, rmse_train = calc_metrics(y_train_true, y_train_pred)
mae_val, rmse_val     = calc_metrics(y_val_true, y_val_pred)
mae_test, rmse_test   = calc_metrics(y_test_true, y_test_pred)

print("\n===== LSTM优化后 模型评价(差分序列) =====")
print(f"训练集  MAE:{mae_train:.6f}, RMSE:{rmse_train:.6f}")
print(f"验证集  MAE:{mae_val:.6f}, RMSE:{rmse_val:.6f}")
print(f"测试集  MAE:{mae_test:.6f}, RMSE:{rmse_test:.6f}")

# ===================== 8. 绘图 =====================
fig, axes = plt.subplots(2,2,figsize=(14,10))

# 损失曲线
axes[0,0].plot(history.history['loss'], label='训练损失')
axes[0,0].plot(history.history['val_loss'], label='验证损失')
axes[0,0].set_title('损失曲线')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# 测试集预测
axes[0,1].plot(y_test_true, label='真实差分序列', linewidth=1.2)
axes[0,1].plot(y_test_pred, label='预测差分序列', linewidth=1.2)
axes[0,1].set_title('测试集预测结果(对数差分)')
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# 残差
res = y_test_true - y_test_pred
axes[1,0].hist(res, bins=30, edgecolor='black')
axes[1,0].set_title('残差分布')

axes[1,1].plot(np.abs(res))
axes[1,1].axhline(mae_test, color='r', linestyle='--', label=f'MAE={mae_test:.4f}')
axes[1,1].set_title('绝对误差')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('lstm_optimized_result.png', dpi=300)
plt.show()

model.save('lstm_optimized.h5')
print("\n✅ 优化版模型运行完成，已保存模型与图片")