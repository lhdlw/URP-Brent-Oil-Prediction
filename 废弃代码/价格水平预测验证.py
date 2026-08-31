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

# ===================== 1. 读取数据（含欧佩克） =====================
vix_data = pd.read_excel('数据/恐慌指数.xlsx')
oil_data = pd.read_excel('数据/伦敦布伦特原油期货价格.xlsx')
sp500_data = pd.read_excel('数据/美国标准普尔500指数.xlsx')
usdx_data = pd.read_excel('数据/美元指数.xlsx')
opec_data = pd.read_excel('数据/欧佩克油.xlsx')

# 日期标准化
def date_convert(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df

vix_data, oil_data, sp500_data, usdx_data, opec_data = map(date_convert, [vix_data, oil_data, sp500_data, usdx_data, opec_data])

# ===================== 2. 合并数据 =====================
merged_data = oil_data[['Date', 'close']].copy()
merged_data.columns = ['Date', 'oil_close']
merged_data = merged_data.merge(vix_data[['Date', 'close']], on='Date', how='left').rename(columns={'close': 'vix_close'})
merged_data = merged_data.merge(sp500_data[['Date', 'close']], on='Date', how='left').rename(columns={'close': 'sp500_close'})
merged_data = merged_data.merge(usdx_data[['Date', 'close']], on='Date', how='left').rename(columns={'close': 'usdx_close'})
merged_data = merged_data.merge(opec_data[['Date', 'close']], on='Date', how='left').rename(columns={'close': 'opec_close'})

merged_data = merged_data.set_index('Date')
merged_data = merged_data.resample('D').interpolate(method='time')
merged_data = merged_data.dropna()

# ===================== 3. 直接用原始价格训练（不再差分！） =====================
feature_cols = ['vix_close', 'sp500_close', 'usdx_close', 'opec_close']
target_col = 'oil_close'

features = merged_data[feature_cols].values
target = merged_data[[target_col]].values

# ===================== 4. 归一化 =====================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(features)
y_scaled = scaler_y.fit_transform(target)

# ===================== 5. 构造时序 =====================
def create_seq(X, y, window=60):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i + window])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

seq_len = 60
X_seq, y_seq = create_seq(X_scaled, y_scaled, seq_len)

# 划分数据集
total = len(X_seq)
train_end = int(total * 0.7)
val_end = train_end + int(total * 0.15)

X_train, y_train = X_seq[:train_end], y_seq[:train_end]
X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
X_test, y_test = X_seq[val_end:], y_seq[val_end:]

print(f"训练集:{X_train.shape}, 验证集:{X_val.shape}, 测试集:{X_test.shape}")

# ===================== 6. TCN 模型 =====================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2

tf.random.set_seed(42)

class ImprovedTCN(keras.Model):
    def __init__(self, filters=64, kernel=3, blocks=3, drop=0.2):
        super().__init__()
        self.blocks = []
        for i in range(blocks):
            dilation = 2 ** i
            self.blocks.append(Conv1D(filters, kernel, dilation_rate=dilation, padding='causal', activation='relu'))
            self.blocks.append(Dropout(drop))
        self.pool = GlobalAveragePooling1D()
        self.dense1 = Dense(32, activation='relu')
        self.final = Dense(1)

    def call(self, x, training=False):
        for layer in self.blocks:
            x = layer(x, training=training) if isinstance(layer, Dropout) else layer(x)
        x = self.pool(x)
        x = self.dense1(x)
        return self.final(x)

model = ImprovedTCN(filters=48, kernel=3, blocks=3, drop=0.2)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

# ===================== 7. 预测 + 反归一化（真实油价） =====================
y_train_pred = model.predict(X_train, verbose=0)
y_val_pred = model.predict(X_val, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# 反归一化 → 得到真实油价
y_train_true = scaler_y.inverse_transform(y_train)
y_val_true = scaler_y.inverse_transform(y_val)
y_test_true = scaler_y.inverse_transform(y_test)

y_train_pred = scaler_y.inverse_transform(y_train_pred)
y_val_pred = scaler_y.inverse_transform(y_val_pred)
y_test_pred = scaler_y.inverse_transform(y_test_pred)

# ===================== 8. 计算真实油价 MAE（不再是 0.00） =====================
def metrics(y_t, y_p):
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    return mae, rmse

mae_tr, rmse_tr = metrics(y_train_true, y_train_pred)
mae_v, rmse_v = metrics(y_val_true, y_val_pred)
mae_te, rmse_te = metrics(y_test_true, y_test_pred)

print("\n===== TCN 真实油价预测结果 =====")
print(f"训练集 MAE: {mae_tr:.2f}    RMSE: {rmse_tr:.2f}")
print(f"验证集 MAE: {mae_v:.2f}    RMSE: {rmse_v:.2f}")
print(f"测试集 MAE: {mae_te:.2f}    RMSE: {rmse_te:.2f}")

# ===================== 9. 画图：真实油价 VS 预测油价 =====================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].plot(history.history['loss'], label='训练损失')
axes[0,0].plot(history.history['val_loss'], label='验证损失')
axes[0,0].set_title('损失曲线')
axes[0,0].legend()
axes[0,0].grid()

axes[0,1].plot(y_test_true, label='真实油价', linewidth=2)
axes[0,1].plot(y_test_pred, label='预测油价', linewidth=2, alpha=0.8)
axes[0,1].set_title(f'TCN 油价预测 | 测试集 MAE={mae_te:.2f}')
axes[0,1].legend()
axes[0,1].grid()

res = y_test_true - y_test_pred
axes[1,0].hist(res, bins=30, edgecolor='k')
axes[1,0].set_title('残差分布')

axes[1,1].plot(np.abs(res))
axes[1,1].axhline(mae_te, c='r', linestyle='--', label=f'MAE={mae_te:.2f}')
axes[1,1].set_title('绝对误差')
axes[1,1].legend()
axes[1,1].grid()

plt.tight_layout()
plt.savefig('tcn_real_oil_price.png', dpi=300)
plt.show()

keras.backend.clear_session()