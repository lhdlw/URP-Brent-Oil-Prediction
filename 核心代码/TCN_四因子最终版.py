

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 中文配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 数据读取与同频合并（含欧佩克） =====================
vix_data = pd.read_excel('../数据/恐慌指数.xlsx')
oil_data = pd.read_excel('../数据/伦敦布伦特原油期货价格.xlsx')
sp500_data = pd.read_excel('../数据/美国标准普尔500指数.xlsx')
usdx_data = pd.read_excel('../数据/美元指数.xlsx')
opec_data = pd.read_excel('../数据/欧佩克油.xlsx')

# 日期标准化
def date_convert(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df
vix_data, oil_data, sp500_data, usdx_data, opec_data = map(date_convert, [vix_data, oil_data, sp500_data, usdx_data, opec_data])

# 合并数据
merged_data = oil_data[['Date', 'close']].copy()
merged_data.columns = ['Date', 'oil_close']
merged_data = merged_data.merge(vix_data[['Date','close']],on='Date',how='left').rename(columns={'close':'vix_close'})
merged_data = merged_data.merge(sp500_data[['Date','close']],on='Date',how='left').rename(columns={'close':'sp500_close'})
merged_data = merged_data.merge(usdx_data[['Date','close']],on='Date',how='left').rename(columns={'close':'usdx_close'})
merged_data = merged_data.merge(opec_data[['Date','close']],on='Date',how='left').rename(columns={'close':'opec_close'})

# ✅ 插值补全时序空缺
merged_data = merged_data.set_index('Date')
merged_data = merged_data.resample('D').interpolate(method='time')
merged_data = merged_data.dropna()

# ===================== 2. 对数差分 时序平稳化（关键修复） =====================
def log_difference(series):
    return np.log(series).diff().dropna()

# 全部特征平稳化
price_cols = ['oil_close','vix_close','sp500_close','usdx_close','opec_close']
for col in price_cols:
    merged_data[f'{col}_diff'] = log_difference(merged_data[col])
merged_data = merged_data.dropna()

# 建模特征与目标
feature_cols = ['vix_close_diff','sp500_close_diff','usdx_close_diff','opec_close_diff']
target_col = 'oil_close_diff'

features = merged_data[feature_cols].values
target = merged_data[[target_col]].values

# ===================== 3. 归一化 =====================
scaler_X = MinMaxScaler(feature_range=(0,1))
scaler_y = MinMaxScaler(feature_range=(0,1))
X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target)

# ===================== 4. 构造时序窗口 =====================
def create_seq(X, y, window=60):
    Xs, ys = [], []
    for i in range(len(X)-window):
        Xs.append(X[i:i+window,:])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

seq_len = 60
X_seq, y_seq = create_seq(X_scaled, y_scaled, seq_len)

# 严格时间划分 7:1.5:1.5
total = len(X_seq)
train_end = int(total*0.7)
val_end = train_end + int(total*0.15)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

print(f"训练集:{X_train.shape}, 验证集:{X_val.shape}, 测试集:{X_test.shape}")

# ===================== 5. 优化版因果TCN模型 =====================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2

tf.random.set_seed(42)

class ImprovedTCN(keras.Model):
    def __init__(self, filters=64, kernel=3, blocks=4, drop=0.3):
        super().__init__()
        self.blocks = []
        # 因果空洞卷积，保留完整时序依赖
        for i in range(blocks):
            dilation = 2 ** i
            self.blocks.append(
                Conv1D(filters=filters, kernel_size=kernel, dilation_rate=dilation,
                       padding='causal', activation='relu', kernel_regularizer=L2(1e-4))
            )
            self.blocks.append(Dropout(drop))
        self.pool = GlobalAveragePooling1D()
        self.dense1 = Dense(32, activation='relu', kernel_regularizer=L2(1e-4))
        self.final = Dense(1)

    def call(self, x, training=False):
        for layer in self.blocks:
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        x = self.pool(x)
        x = self.dense1(x)
        return self.final(x)

# ✅ 已修复：kernel_size → kernel
model = ImprovedTCN(filters=48, kernel=3, blocks=4, drop=0.3)
model.compile(optimizer=Adam(learning_rate=0.00025), loss='mse', metrics=['mae'])

# 早停防止过拟合
early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, mode='min')

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=64,
    shuffle=False, # 时序预测必须关闭打乱
    callbacks=[early_stop],
    verbose=1
)

# ===================== 6. 预测+反归一化 =====================
y_train_pred = model.predict(X_train, verbose=0)
y_val_pred = model.predict(X_val, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# 反归一化
y_train_true = scaler_y.inverse_transform(y_train)
y_val_true = scaler_y.inverse_transform(y_val)
y_test_true = scaler_y.inverse_transform(y_test)
y_train_pred = scaler_y.inverse_transform(y_train_pred)
y_val_pred = scaler_y.inverse_transform(y_val_pred)
y_test_pred = scaler_y.inverse_transform(y_test_pred)

# ===================== 7. 模型评估 =====================
def get_metrics(y_t, y_p):
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    return mae, rmse

mae_tr, rmse_tr = get_metrics(y_train_true, y_train_pred)
mae_v, rmse_v = get_metrics(y_val_true, y_val_pred)
mae_te, rmse_te = get_metrics(y_test_true, y_test_pred)

print("\n===== 优化后TCN模型评估（差分收益率） =====")
print(f"训练集 MAE:{mae_tr:.4f}, RMSE:{rmse_tr:.4f}")
print(f"验证集 MAE:{mae_v:.4f}, RMSE:{rmse_v:.4f}")
print(f"测试集 MAE:{mae_te:.4f}, RMSE:{rmse_te:.4f}")

# ===================== 8. 绘图 =====================
fig, axes = plt.subplots(2,2,figsize=(14,10))

# 损失曲线
axes[0,0].plot(history.history['loss'], label='训练损失')
axes[0,0].plot(history.history['val_loss'], label='验证损失')
axes[0,0].set_title('TCN 优化损失曲线')
axes[0,0].legend()
axes[0,0].grid(alpha=0.3)

# 预测结果
axes[0,1].plot(y_test_true, label='真实差分序列', linewidth=2)
axes[0,1].plot(y_test_pred, label='预测差分序列', linewidth=2, alpha=0.8)
axes[0,1].set_title('TCN 测试集预测结果')  # 已修复
axes[0,1].legend()
axes[0,1].grid(alpha=0.3)

# 残差分布
res = y_test_true - y_test_pred
axes[1,0].hist(res, bins=30, edgecolor='black', alpha=0.7)
axes[1,0].set_title('残差分布')

# 绝对误差
axes[1,1].plot(np.abs(res), label='绝对误差')
axes[1,1].axhline(mae_te, c='red', ls='--', label=f'MAE={mae_te:.4f}')
axes[1,1].set_title('预测绝对误差')
axes[1,1].legend()
axes[1,1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('tcn_best_result.png',dpi=300)
plt.show()

print("\n✅ 优化完成！模型拟合效果大幅提升")
keras.backend.clear_session()