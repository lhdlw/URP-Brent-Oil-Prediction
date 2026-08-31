import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eig
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class OHLCAnalyzer:
    """OHLC数据分析器（简化版）"""

    def __init__(self):
        self.data = {}

    def load_data(self, file_path, name):
        """加载Excel数据"""
        df = pd.read_excel(file_path)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close']
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # 数据清洗：移除异常值
        df = df[(df['High'] >= df['Low']) & (df['Low'] > 0)]
        df = df[(df['Open'] >= df['Low']) & (df['Open'] <= df['High'])]
        df = df[(df['Close'] >= df['Low']) & (df['Close'] <= df['High'])]

        self.data[name] = df
        print(f"✓ 已加载 {name} 数据: {len(df)} 条记录")
        return df

    def transform(self, ohlc_data):
        """OHLC数据无约束转换（增加鲁棒性）"""
        O = ohlc_data['Open'].values
        H = ohlc_data['High'].values
        L = ohlc_data['Low'].values
        C = ohlc_data['Close'].values

        # 添加小的epsilon避免除零
        epsilon = 1e-8

        # 确保 High > Low
        range_hl = H - L
        range_hl = np.maximum(range_hl, epsilon)

        # 计算lambda，限制在(epsilon, 1-epsilon)区间
        lambda_o = np.clip((O - L) / range_hl, epsilon, 1 - epsilon)
        lambda_c = np.clip((C - L) / range_hl, epsilon, 1 - epsilon)

        # 确保 Low > 0
        L = np.maximum(L, epsilon)

        # 转换
        y1 = np.log(L)
        y2 = np.log(range_hl)
        y3 = np.log(lambda_o / (1 - lambda_o))
        y4 = np.log(lambda_c / (1 - lambda_c))

        # 检查并处理无穷大和NaN
        result = np.column_stack([y1, y2, y3, y4])

        # 替换无穷大值为有限值
        result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result


class RobustPLS:
    """鲁棒PLS回归模型（修复维度问题）"""

    def __init__(self, max_iter=100, tol=0.0005, c_param=4):
        self.max_iter = max_iter
        self.tol = tol
        self.c_param = c_param
        self.vip_scores = None
        self.explained_variance = {}

    def fair_weight(self, distance):
        """Fair权重函数"""
        return 1 / (1 + (np.abs(distance) / self.c_param) ** 2)

    def calc_weights(self, X, Y):
        """计算初始权重"""
        X_median = np.median(X, axis=0)
        Y_median = np.median(Y, axis=0)

        X_dist = np.linalg.norm(X - X_median, axis=1)
        Y_dist = np.linalg.norm(Y - Y_median, axis=1)

        med_X = np.median(X_dist)
        med_Y = np.median(Y_dist)

        g_i = X_dist / (med_X + 1e-10)
        h_i = Y_dist / (med_Y + 1e-10)

        w_l = self.fair_weight(g_i)
        w_r = self.fair_weight(h_i)

        return np.sqrt(w_l * w_r)

    def extract_component(self, X_w, Y_w):
        """提取主成分（修复版）"""
        # X_w: (n_samples, n_features_X)
        # Y_w: (n_samples, n_features_Y)

        n_samples = X_w.shape[0]
        n_features_X = X_w.shape[1]
        n_features_Y = Y_w.shape[1]

        # 计算协方差矩阵
        # XY: (n_features_X, n_features_Y)
        XY = X_w.T @ Y_w
        # YX: (n_features_Y, n_features_X)
        YX = Y_w.T @ X_w

        # M_x: (n_features_X, n_features_X)
        M_x = XY @ YX
        # M_y: (n_features_Y, n_features_Y)
        M_y = YX @ XY

        # 提取特征向量
        evals_x, evecs_x = eig(M_x)
        evals_y, evecs_y = eig(M_y)

        idx_x = np.argmax(np.real(evals_x))
        idx_y = np.argmax(np.real(evals_y))

        # m: (n_features_X,)
        m = np.real(evecs_x[:, idx_x])
        # c: (n_features_Y,)
        c = np.real(evecs_y[:, idx_y])

        # 归一化
        m = m / (np.linalg.norm(m) + 1e-10)
        c = c / (np.linalg.norm(c) + 1e-10)

        # 计算主成分得分
        # t: (n_samples,)
        t = X_w @ m
        # u: (n_samples,)
        u = Y_w @ c

        # 计算回归系数
        # p: (n_features_X,)
        p = (X_w.T @ t) / (t.T @ t + 1e-10)
        # r: (n_features_Y,)
        r = (Y_w.T @ t) / (t.T @ t + 1e-10)

        return t, u, m, c, p, r

    def fit(self, X, Y):
        """拟合模型"""
        # 检查输入数据
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            print("警告：输入数据包含无穷大或NaN，尝试清理...")
            X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
            Y = np.nan_to_num(Y, nan=0.0, posinf=10.0, neginf=-10.0)

        print(f"输入维度 - X: {X.shape}, Y: {Y.shape}")

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        try:
            X_std = self.scaler_X.fit_transform(X)
            Y_std = self.scaler_Y.fit_transform(Y)
        except Exception as e:
            print(f"标准化失败: {e}")
            # 使用简单的归一化
            X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-10)
            Y_std = (Y - np.mean(Y, axis=0)) / (np.std(Y, axis=0) + 1e-10)

        X_res = X_std.copy()
        Y_res = Y_std.copy()

        self.weights_X = []
        self.scores = []
        self.loadings_X = []
        self.loadings_Y = []

        prev_coef = np.zeros((X.shape[1], Y.shape[1]))

        for iteration in range(self.max_iter):
            try:
                # 计算权重
                weights = self.calc_weights(X_res, Y_res)

                # 加权
                W = np.diag(weights)
                X_w = W @ X_res
                Y_w = W @ Y_res

                # 提取主成分
                t, u, m, c, p, r = self.extract_component(X_w, Y_w)

                self.weights_X.append(m)
                self.scores.append(t)
                self.loadings_X.append(p)
                self.loadings_Y.append(r)

                # 计算回归系数
                curr_coef = np.outer(m, r)

                # 更新残差
                X_res = X_res - np.outer(t, p)
                Y_res = Y_res - np.outer(t, r)

                # 检查收敛
                rel_change = np.mean(((curr_coef - prev_coef) / (np.abs(prev_coef) + 1e-10)) ** 2)

                if rel_change < self.tol:
                    print(f"✓ 第 {iteration + 1} 次迭代后收敛")
                    break

                prev_coef = curr_coef

            except Exception as e:
                print(f"迭代 {iteration + 1} 出错: {e}")
                import traceback
                traceback.print_exc()
                break

        self._calc_metrics(X_std, Y_std)
        return self

    def _calc_metrics(self, X, Y):
        """计算VIP等指标"""
        n_features = X.shape[1]

        for i, (t, m) in enumerate(zip(self.scores, self.weights_X)):
            try:
                exp_X = np.mean([np.corrcoef(X[:, j], t)[0, 1] ** 2
                                 for j in range(X.shape[1]) if np.isfinite(np.corrcoef(X[:, j], t)[0, 1])])
                exp_Y = np.mean([np.corrcoef(Y[:, j], t)[0, 1] ** 2
                                 for j in range(Y.shape[1]) if np.isfinite(np.corrcoef(Y[:, j], t)[0, 1])])

                self.explained_variance[f't{i + 1}'] = {
                    'X': exp_X if np.isfinite(exp_X) else 0.0,
                    'Y': exp_Y if np.isfinite(exp_Y) else 0.0
                }
            except:
                self.explained_variance[f't{i + 1}'] = {'X': 0.0, 'Y': 0.0}

        # 计算VIP - 每个变量有4个特征，现在共3个变量
        self.vip_scores = np.zeros(3)
        total_exp_Y = sum([v['Y'] for v in self.explained_variance.values()])

        if total_exp_Y > 0:
            for var_idx in range(3):
                vip_sum = 0
                for m, exp in zip(self.weights_X, self.explained_variance.values()):
                    feat_indices = range(var_idx * 4, (var_idx + 1) * 4)
                    weight_sum = sum([m[j] ** 2 for j in feat_indices])
                    vip_sum += exp['Y'] * weight_sum

                self.vip_scores[var_idx] = np.sqrt(3 * vip_sum / total_exp_Y)


def analyze(sp500_file, dollar_file, vix_file, brent_file):
    """
    主分析函数：3个影响因子 → 伦敦布伦特原油
    因子：标普500、美元指数、恐慌指数(VIX)
    """

    print("=" * 60)
    print("伦敦布伦特原油期货价格影响因素分析（3因子）")
    print("=" * 60)

    # 1. 加载数据（已删除欧佩克）
    analyzer = OHLCAnalyzer()
    print("\n步骤1: 加载数据")
    print("-" * 60)
    sp500 = analyzer.load_data(sp500_file, '美国标准普尔500指数')
    dollar = analyzer.load_data(dollar_file, '美元指数')
    vix = analyzer.load_data(vix_file, '恐怖指数(VIX)')
    brent = analyzer.load_data(brent_file, '伦敦布伦特原油期货')

    # 2. 数据对齐
    print("\n步骤2: 数据对齐")
    print("-" * 60)
    common_dates = set(sp500['Date']) \
                    & set(dollar['Date']) \
                    & set(vix['Date']) \
                    & set(brent['Date'])

    common_dates = sorted(list(common_dates))
    print(f"✓ 共同交易日: {len(common_dates)} 天")

    sp500 = sp500[sp500['Date'].isin(common_dates)].reset_index(drop=True)
    dollar = dollar[dollar['Date'].isin(common_dates)].reset_index(drop=True)
    vix = vix[vix['Date'].isin(common_dates)].reset_index(drop=True)
    brent = brent[brent['Date'].isin(common_dates)].reset_index(drop=True)

    # 3. 转换
    print("\n步骤3: OHLC转换")
    print("-" * 60)
    sp500_t = analyzer.transform(sp500)
    dollar_t = analyzer.transform(dollar)
    vix_t = analyzer.transform(vix)
    brent_t = analyzer.transform(brent)

    print(f"标普500: {sp500_t.shape}")
    print(f"美元指数: {dollar_t.shape}")
    print(f"恐怖指数: {vix_t.shape}")
    print(f"布伦特原油: {brent_t.shape}")

    # X = 3个因子
    X = np.hstack([sp500_t, dollar_t, vix_t])
    Y = brent_t

    print(f"最终 X: {X.shape}, Y: {Y.shape}")
    print("✓ 转换完成")

    # 4. 逐年分析
    print("\n步骤4: 逐年分析")
    print("=" * 60)

    years = sorted(set([d.year for d in common_dates]))
    results = {}

    for year in years:
        print(f"\n{'*' * 60}")
        print(f"分析年份: {year}")
        print(f"{'*' * 60}")

        mask = [d.year == year for d in common_dates]
        X_year = X[mask]
        Y_year = Y[mask]

        print(f"样本数: {len(X_year)}")

        model = RobustPLS()

        try:
            model.fit(X_year, Y_year)

            results[year] = {
                'vip_sp500': model.vip_scores[0],
                'vip_dollar': model.vip_scores[1],
                'vip_vix': model.vip_scores[2],
                'explained_Y': model.explained_variance['t1']['Y']
            }

            print(f"\nVIP评分:")
            print(f"  标普500: {model.vip_scores[0]:.3f} {'***' if model.vip_scores[0] > 1 else ''}")
            print(f"  美元指数: {model.vip_scores[1]:.3f} {'***' if model.vip_scores[1] > 1 else ''}")
            print(f"  恐怖指数: {model.vip_scores[2]:.3f} {'***' if model.vip_scores[2] > 1 else ''}")

        except Exception as e:
            print(f"年份 {year} 分析失败: {e}")
            results[year] = {
                'vip_sp500': 0.0,
                'vip_dollar': 0.0,
                'vip_vix': 0.0,
                'explained_Y': 0.0
            }

    # 5. 可视化
    plot_results(results, years)

    # 6. 结论
    print_conclusion(results, years)

    return results


def plot_results(results, years):
    """3因子图表"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    vip_sp500 = [results[y]['vip_sp500'] for y in years]
    vip_dollar = [results[y]['vip_dollar'] for y in years]
    vip_vix = [results[y]['vip_vix'] for y in years]
    explained = [results[y]['explained_Y'] for y in years]

    # 图1: 3条线
    axes[0, 0].plot(years, vip_sp500, 's-', label='标普500', lw=2, ms=6, c='#3498DB')
    axes[0, 0].plot(years, vip_dollar, '^-', label='美元指数', lw=2, ms=6, c='#F39C12')
    axes[0, 0].plot(years, vip_vix, 'd-', label='恐怖指数', lw=2, ms=6, c='#9B59B6')
    axes[0, 0].axhline(1, c='gray', ls='--', label='VIP=1 重要阈值')
    axes[0, 0].set_title('VIP得分时间序列', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # 图2: 分组柱状图
    x_pos = np.arange(len(years))
    w = 0.25
    axes[0, 1].bar(x_pos - w, vip_sp500, w, label='标普500', color='#3498DB', alpha=0.8)
    axes[0, 1].bar(x_pos, vip_dollar, w, label='美元指数', color='#F39C12', alpha=0.8)
    axes[0, 1].bar(x_pos + w, vip_vix, w, label='恐怖指数', color='#9B59B6', alpha=0.8)
    axes[0, 1].axhline(1, c='gray', ls='--')
    axes[0, 1].set_title('3因子VIP对比', fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(years)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3, axis='y')

    # 图3: 解释能力
    axes[1, 0].bar(years, explained, color='#2ECC71', alpha=0.8)
    axes[1, 0].set_title('模型对布伦特原油的解释能力', fontweight='bold')
    for i, v in enumerate(explained):
        axes[1, 0].text(years[i], v + 0.01, f'{v:.2f}', ha='center')
    axes[1, 0].grid(alpha=0.3, axis='y')

    # 图4: 热图
    mat = np.array([vip_sp500, vip_dollar, vip_vix]).T
    im = axes[1,1].imshow(mat, cmap='RdYlGn', aspect='auto', vmin=0, vmax=2)
    axes[1,1].set_xticks([0,1,2])
    axes[1,1].set_xticklabels(['标普500','美元指数','恐怖指数'])
    axes[1,1].set_yticks(range(len(years)))
    axes[1,1].set_yticklabels(years)
    axes[1,1].set_title('VIP热图', fontweight='bold')
    for i in range(len(years)):
        for j in range(3):
            axes[1,1].text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center')

    plt.colorbar(im, ax=axes[1,1])
    plt.tight_layout()
    plt.savefig('布伦特原油_3因子分析.png', dpi=300, bbox_inches='tight')
    print("\n✓ 图表已保存")
    plt.show()


def print_conclusion(results, years):
    """3因子结论"""
    vip_sp500 = [results[y]['vip_sp500'] for y in years]
    vip_dollar = [results[y]['vip_dollar'] for y in years]
    vip_vix = [results[y]['vip_vix'] for y in years]

    print("\n\n" + "=" * 60)
    print("伦敦布伦特原油 3 大影响因子 —— 最终结论")
    print("=" * 60)

    factors = {
        "标普500": np.mean(vip_sp500),
        "美元指数": np.mean(vip_dollar),
        "恐怖指数": np.mean(vip_vix)
    }

    for name, score in factors.items():
        print(f"\n{name}:")
        print(f"  平均VIP: {score:.3f}")
        sig_years = []
        for y in years:
            if name == "标普500" and results[y]['vip_sp500']>1:
                sig_years.append(y)
            elif name == "美元指数" and results[y]['vip_dollar']>1:
                sig_years.append(y)
            elif name == "恐怖指数" and results[y]['vip_vix']>1:
                sig_years.append(y)
        if sig_years:
            print(f"  显著年份: {sig_years}")
            print(f"  ✅ 重要影响因子 (VIP>1)")
        else:
            print(f"  ❌ 影响较弱")

    top1 = max(factors, key=factors.get)
    print(f"\n🏆 综合最重要影响因子：【{top1}】")


# ------------------- 运行入口（已删除欧佩克）-------------------
if __name__ == "__main__":
    results = analyze(
        sp500_file='../数据/美国标准普尔500指数.xlsx',   # 1
        dollar_file='../数据/美元指数.xlsx',            # 2
        vix_file='../数据/恐慌指数.xlsx',               # 3
        brent_file='../数据/伦敦布伦特原油期货价格.xlsx'     # 目标
    )