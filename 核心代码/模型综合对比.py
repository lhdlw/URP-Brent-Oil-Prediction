import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 读取数据 =====================
vix_data = pd.read_excel('../数据/恐慌指数.xlsx')
oil_data = pd.read_excel('../数据/伦敦布伦特原油期货价格.xlsx')
sp500_data = pd.read_excel('../数据/美国标准普尔500指数.xlsx')
usdx_data = pd.read_excel('../数据/美元指数.xlsx')
opec_data = pd.read_excel('../数据/欧佩克油.xlsx')

def date_convert(df):
    df['Date'] = pd.to_datetime(df['Date'])
    return df

vix_data, oil_data, sp500_data, usdx_data, opec_data = map(date_convert,
                                                          [vix_data, oil_data, sp500_data, usdx_data, opec_data])

merged_data = oil_data[['Date', 'close']].copy()
merged_data.columns = ['Date', 'oil_close']
merged_data = merged_data.merge(vix_data[['Date','close']],on='Date',how='left').rename(columns={'close':'vix_close'})
merged_data = merged_data.merge(sp500_data[['Date','close']],on='Date',how='left').rename(columns={'close':'sp500_close'})
merged_data = merged_data.merge(usdx_data[['Date','close']],on='Date',how='left').rename(columns={'close':'usdx_close'})
merged_data = merged_data.merge(opec_data[['Date','close']],on='Date',how='left').rename(columns={'close':'opec_close'})

merged_data = merged_data.set_index('Date')
merged_data = merged_data.resample('D').interpolate(method='time')
merged_data = merged_data.dropna()

def log_diff(series):
    return np.log(series).diff()

cols = ['oil_close','vix_close','sp500_close','usdx_close','opec_close']
for c in cols:
    merged_data[f'{c}_ld'] = log_diff(merged_data[c])
merged_data = merged_data.dropna()

target_col = 'oil_close_ld'
feats_no_opec = ['vix_close_ld','sp500_close_ld','usdx_close_ld']  # 无OPEC特征
feats_with_opec = ['vix_close_ld','sp500_close_ld','usdx_close_ld','opec_close_ld'] # 有OPEC

# ===================== 构建数据集 =====================
def build_data(features):
    X = merged_data[features].values
    y = merged_data[[target_col]].values
    sc_X = MinMaxScaler()
    sc_y = MinMaxScaler()
    Xs = sc_X.fit_transform(X)
    ys = sc_y.fit_transform(y)
    seq_len = 60
    X_seq, y_seq = [], []
    for i in range(len(Xs)-seq_len):
        X_seq.append(Xs[i:i+seq_len])
        y_seq.append(ys[i+seq_len])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    total = len(X_seq)
    train = int(total*0.7)
    val = int(total*0.15)
    return X_seq, y_seq, sc_y, X_seq[:train], y_seq[:train], X_seq[train:train+val], y_seq[train:train+val], X_seq[train+val:], y_seq[train+val:]

# 两个数据集：无OPEC / 有OPEC
X_no, y_no, scy_no, Xn_tr, yn_tr, Xn_val, yn_val, Xn_te, yn_te = build_data(feats_no_opec)
X_wi, y_wi, scy_wi, Xw_tr, yw_tr, Xw_val, yw_val, Xw_te, yw_te = build_data(feats_with_opec)
y_true = scy_wi.inverse_transform(yw_te)

# ===================== 模型构建 =====================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import L2
tf.random.set_seed(42)

# LSTM 模型
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, activation='tanh', input_shape=input_shape, recurrent_dropout=0.2),
        Dropout(0.3),
        LSTM(32, return_sequences=False, activation='tanh', recurrent_dropout=0.2),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.0003), loss='mse', metrics=['mae'])
    return model

# TCN 模型
class ImprovedTCN(keras.Model):
    def __init__(self, filters=48, kernel=3, blocks=4, drop=0.3):
        super().__init__()
        self.blocks = []
        for i in range(blocks):
            dilation = 2**i
            self.blocks.append(Conv1D(filters, kernel, dilation_rate=dilation, padding='causal', activation='relu', kernel_regularizer=L2(1e-4)))
            self.blocks.append(Dropout(drop))
        self.pool = GlobalAveragePooling1D()
        self.dense1 = Dense(32, activation='relu', kernel_regularizer=L2(1e-4))
        self.final = Dense(1)
    def call(self, x, training=False):
        for layer in self.blocks:
            x = layer(x, training=training) if isinstance(layer, Dropout) else layer(x)
        return self.final(self.dense1(self.pool(x)))

def create_early_stopping():
    # 每次训练使用独立回调，避免上一模型的最佳损失影响当前模型。
    return EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# ===================== 训练 4 个模型 =====================
print("\n【1/4】训练 LSTM 无OPEC")
m1 = build_lstm((Xn_tr.shape[1], Xn_tr.shape[2]))
m1.fit(Xn_tr,yn_tr,validation_data=(Xn_val,yn_val),epochs=100,batch_size=32,shuffle=False,callbacks=[create_early_stopping()],verbose=1)

print("\n【2/4】训练 LSTM +OPEC")
m2 = build_lstm((Xw_tr.shape[1], Xw_tr.shape[2]))
m2.fit(Xw_tr,yw_tr,validation_data=(Xw_val,yw_val),epochs=100,batch_size=32,shuffle=False,callbacks=[create_early_stopping()],verbose=1)

print("\n【3/4】训练 TCN 无OPEC")
m3 = ImprovedTCN()
m3.compile(optimizer=Adam(0.00025), loss='mse', metrics=['mae'])
m3.fit(Xn_tr,yn_tr,validation_data=(Xn_val,yn_val),epochs=100,batch_size=64,shuffle=False,callbacks=[create_early_stopping()],verbose=1)

print("\n【4/4】训练 TCN +OPEC")
m4 = ImprovedTCN()
m4.compile(optimizer=Adam(0.00025), loss='mse', metrics=['mae'])
m4.fit(Xw_tr,yw_tr,validation_data=(Xw_val,yw_val),epochs=100,batch_size=64,shuffle=False,callbacks=[create_early_stopping()],verbose=1)

# ===================== 预测 =====================
p1 = scy_no.inverse_transform(m1.predict(Xn_te,verbose=0))
p2 = scy_wi.inverse_transform(m2.predict(Xw_te,verbose=0))
p3 = scy_no.inverse_transform(m3.predict(Xn_te,verbose=0))
p4 = scy_wi.inverse_transform(m4.predict(Xw_te,verbose=0))

# ===================== 指标 =====================
def metrics(y,p):
    mae = mean_absolute_error(y,p)
    rmse = np.sqrt(mean_squared_error(y,p))
    return round(mae,6), round(rmse,6)

mae1,rmse1=metrics(y_true,p1)
mae2,rmse2=metrics(y_true,p2)
mae3,rmse3=metrics(y_true,p3)
mae4,rmse4=metrics(y_true,p4)

# ===================== 输出对比表 =====================
print("="*65)
print("                    四模型最终对比（测试集）")
print("="*65)
print(f"① LSTM 无OPEC    | MAE={mae1:.6f} | RMSE={rmse1:.6f}")
print(f"② LSTM +OPEC     | MAE={mae2:.6f} | RMSE={rmse2:.6f}")
print(f"③ TCN  无OPEC    | MAE={mae3:.6f} | RMSE={rmse3:.6f}")
print(f"④ TCN  +OPEC     | MAE={mae4:.6f} | RMSE={rmse4:.6f}")
print("="*65)
print("✅ 数值越小，拟合效果越好")

# ===================== 统一对比图 =====================
plt.figure(figsize=(16,8))
plt.plot(y_true, label='真实值', lw=2.5, c='black')
plt.plot(p1, label=f'LSTM无OPEC  MAE={mae1:.4f}', alpha=0.6)
plt.plot(p2, label=f'LSTM+OPEC   MAE={mae2:.4f}', alpha=0.7)
plt.plot(p3, label=f'TCN无OPEC   MAE={mae3:.4f}', alpha=0.8)
plt.plot(p4, label=f'TCN+OPEC    MAE={mae4:.4f}', lw=2.5, alpha=0.9)
plt.title('四模型预测对比（原油价格差分预测）', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('四模型预测对比图.png',dpi=300)
plt.show()

# ===================== 指标柱状图 =====================
names = ['LSTM\n无OPEC','LSTM\n+OPEC','TCN\n无OPEC','TCN\n+OPEC']
maes = [mae1, mae2, mae3, mae4]
rmses = [rmse1, rmse2, rmse3, rmse4]

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
plt.bar(names, maes, color=['orange','dodgerblue','purple','green'])
plt.title('MAE 对比（越小越好）', fontsize=12)

plt.subplot(1,2,2)
plt.bar(names, rmses, color=['orange','dodgerblue','purple','green'])
plt.title('RMSE 对比（越小越好）', fontsize=12)
plt.tight_layout()
plt.savefig('四模型指标对比图.png',dpi=300)
plt.show()
