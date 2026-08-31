import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------------------
# 1. 读取你的4个数据
# ------------------------------------------------------------------------------
vix = pd.read_excel("数据/恐慌指数.xlsx", engine="openpyxl")
oil = pd.read_excel("数据/伦敦布伦特原油期货价格.xlsx", engine="openpyxl")
spx = pd.read_excel("数据/美国标准普尔500指数.xlsx", engine="openpyxl")
dxy = pd.read_excel("数据/美元指数.xlsx", engine="openpyxl")

def proc(df, name):
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[["Date", "close"]].rename(columns={"close": name})
    return df

vix = proc(vix, "vix")
oil = proc(oil, "oil")
spx = proc(spx, "spx")
dxy = proc(dxy, "dxy")

# ------------------------------------------------------------------------------
# 2. 合并 + 【关键：升频到日度 + 线性插值】
# ------------------------------------------------------------------------------
df = vix.merge(oil, on="Date", how="outer") \
       .merge(spx, on="Date", how="outer") \
       .merge(dxy, on="Date", how="outer")

df = df.sort_values("Date").set_index("Date")

# ---------------- 你要的：插值成日度数据 ----------------
df = df.asfreq("D")               # 转为每日
df = df.interpolate(method="linear")  # 线性插值填满
df = df.dropna()
# --------------------------------------------------------

# 保存原始价格用于还原
oil_original = df["oil"].values

# ------------------------------------------------------------------------------
# 3. 对原油做差分（论文标准）
# ------------------------------------------------------------------------------
df["oil_diff"] = df["oil"].diff()
df = df.dropna()

# ------------------------------------------------------------------------------
# 4. 30日滑动窗口
# ------------------------------------------------------------------------------
WINDOW = 30
X, y = [], []

features = df[["vix", "spx", "dxy", "oil"]].values
target = df["oil_diff"].values

for i in range(WINDOW, len(target)):
    X.append(features[i-WINDOW:i, :])
    y.append(target[i])

X, y = np.array(X), np.array(y)

# ------------------------------------------------------------------------------
# 5. 划分训练集 / 测试集
# ------------------------------------------------------------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------------------------------------------------------------------
# 6. 归一化
# ------------------------------------------------------------------------------
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, 4)).reshape(-1, WINDOW, 4)
X_test = scaler.transform(X_test.reshape(-1, 4)).reshape(-1, WINDOW, 4)

# ------------------------------------------------------------------------------
# 7. CNN-LSTM 论文模型
# ------------------------------------------------------------------------------
model = Sequential([
    Conv1D(64, 3, activation="relu", input_shape=(WINDOW, 4)),
    MaxPooling1D(2),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
early = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=120, batch_size=16,
    callbacks=[early], verbose=1
)

# ------------------------------------------------------------------------------
# 8. 预测
# ------------------------------------------------------------------------------
y_pred_diff = model.predict(X_test, verbose=0).flatten()

# ------------------------------------------------------------------------------
# 9. 还原真实油价（不会飘！）
# ------------------------------------------------------------------------------
test_start = split + WINDOW
y_true = oil_original[test_start : test_start + len(y_test)]
y_pred = np.zeros_like(y_true)
y_pred[0] = oil_original[test_start]

for i in range(1, len(y_pred)):
    y_pred[i] = y_pred[i-1] + y_pred_diff[i-1]

# ------------------------------------------------------------------------------
# 10. 论文指标（RMSE、MAE、R²）
# ------------------------------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n" + "="*60)
print("📊 【日频插值版】布伦特原油期货预测结果")
print(f"RMSE 均方根误差：{rmse:.4f}")
print(f"MAE  平均绝对误差：{mae:.4f}")
print(f"R²   决定系数：{r2:.4f}")
print("="*60)

# ------------------------------------------------------------------------------
# 11. 画图
# ------------------------------------------------------------------------------
plt.figure(figsize=(15, 6), dpi=120)
plt.plot(y_true, label="真实原油价格", linewidth=2.2)
plt.plot(y_pred, label="CNN-LSTM 预测价格", linewidth=2.2, linestyle="--")
plt.title("布伦特原油期货预测（日度插值+论文模型）", fontsize=15)
plt.legend()
plt.grid(alpha=0.3)
plt.show()