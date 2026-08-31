"""
布伦特原油期货价格预测 — 三组模型×两种架构 = 6个模型对比
纯NumPy实现LSTM与CNN（优化速度版）
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings, time, json, sys
warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh_act(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def relu_d(x):
    return (x > 0).astype(float)

class Adam:
    def __init__(self, lr=0.001):
        self.lr, self.b1, self.b2, self.eps = lr, 0.9, 0.999, 1e-8
        self.m, self.v, self.t = {}, {}, 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            if k not in self.m:
                self.m[k] = np.zeros_like(params[k])
                self.v[k] = np.zeros_like(params[k])
            g = np.clip(grads[k], -5, 5)
            self.m[k] = self.b1*self.m[k] + (1-self.b1)*g
            self.v[k] = self.b2*self.v[k] + (1-self.b2)*g**2
            mh = self.m[k]/(1-self.b1**self.t)
            vh = self.v[k]/(1-self.b2**self.t)
            params[k] -= self.lr * mh/(np.sqrt(vh)+self.eps)

class LSTMModel:
    def __init__(self, in_dim, hid=16, lr=0.002):
        self.hid = hid
        s = 0.08
        d = in_dim + hid
        self.p = {
            'Wf': np.random.randn(hid,d)*s, 'bf': np.zeros((hid,1)),
            'Wi': np.random.randn(hid,d)*s, 'bi': np.zeros((hid,1)),
            'Wg': np.random.randn(hid,d)*s, 'bg': np.zeros((hid,1)),
            'Wo': np.random.randn(hid,d)*s, 'bo': np.zeros((hid,1)),
            'Wy': np.random.randn(1,hid)*s, 'by': np.zeros((1,1)),
        }
        self.opt = Adam(lr)

    def fwd(self, x):
        T = x.shape[0]
        h = np.zeros((self.hid,1))
        c = np.zeros((self.hid,1))
        self.cache = {'h':[h],'c':[c],'f':[],'i':[],'g':[],'o':[],'z':[]}
        for t in range(T):
            xt = x[t].reshape(-1,1)
            z = np.vstack([h, xt])
            f = sigmoid(self.p['Wf']@z + self.p['bf'])
            i = sigmoid(self.p['Wi']@z + self.p['bi'])
            g = tanh_act(self.p['Wg']@z + self.p['bg'])
            o = sigmoid(self.p['Wo']@z + self.p['bo'])
            c = f*c + i*g
            h = o*tanh_act(c)
            self.cache['h'].append(h); self.cache['c'].append(c)
            self.cache['f'].append(f); self.cache['i'].append(i)
            self.cache['g'].append(g); self.cache['o'].append(o)
            self.cache['z'].append(z)
        return (self.p['Wy']@h + self.p['by']).flatten()

    def bwd(self, x, yt, yp):
        T = x.shape[0]
        gr = {k: np.zeros_like(v) for k,v in self.p.items()}
        dy = (yp.reshape(-1,1) - yt.reshape(-1,1))
        gr['Wy'] = dy @ self.cache['h'][-1].T
        gr['by'] = dy
        dh = self.p['Wy'].T @ dy
        dc = np.zeros((self.hid,1))
        for t in reversed(range(T)):
            c_prev = self.cache['c'][t]
            c_cur = self.cache['c'][t+1]
            f,i,g,o = self.cache['f'][t],self.cache['i'][t],self.cache['g'][t],self.cache['o'][t]
            z = self.cache['z'][t]
            do = dh*tanh_act(c_cur)*o*(1-o)
            dc2 = dh*o*(1-tanh_act(c_cur)**2) + dc
            df = dc2*c_prev*f*(1-f)
            di = dc2*g*i*(1-i)
            dg = dc2*i*(1-g**2)
            for gate,dgate in [('Wf',df),('Wi',di),('Wg',dg),('Wo',do)]:
                gr[gate] += dgate@z.T
                gr['b'+gate[1]] += dgate
            dz = self.p['Wf'].T@df + self.p['Wi'].T@di + self.p['Wg'].T@dg + self.p['Wo'].T@do
            dh = dz[:self.hid]
            dc = dc2*f
        self.opt.step(self.p, gr)

    def train(self, X, Y, epochs=15):
        n = len(X)
        losses = []
        for ep in range(epochs):
            idx = np.random.permutation(n)
            el = 0
            for j in idx:
                yp = self.fwd(X[j])
                el += np.mean((yp-Y[j])**2)
                self.bwd(X[j], Y[j], yp)
            losses.append(el/n)
            sys.stdout.write(f'\r    Epoch {ep+1}/{epochs} loss={el/n:.6f}')
            sys.stdout.flush()
        print()
        return losses

    def predict(self, X):
        return np.array([self.fwd(x) for x in X])


class CNN1DModel:
    def __init__(self, in_dim, seq_len, nf=12, ks=3, lr=0.002):
        self.in_dim, self.nf, self.ks = in_dim, nf, ks
        self.col = seq_len - ks + 1
        self.pl = max(self.col//2, 1)
        fc_in = nf * self.pl
        s = 0.08
        self.p = {
            'Wc': np.random.randn(nf,ks,in_dim)*s, 'bc': np.zeros((nf,1)),
            'W1': np.random.randn(24,fc_in)*s, 'b1': np.zeros((24,1)),
            'W2': np.random.randn(1,24)*s, 'b2': np.zeros((1,1)),
        }
        self.opt = Adam(lr)

    def fwd(self, x):
        # Conv
        self.x = x
        conv = np.zeros((self.nf, self.col))
        for f in range(self.nf):
            for t in range(self.col):
                conv[f,t] = np.sum(self.p['Wc'][f]*x[t:t+self.ks]) + self.p['bc'][f,0]
        self.conv_raw = conv
        self.conv_out = relu(conv)
        # Pool
        pl = self.pl
        self.pool = np.zeros((self.nf, pl))
        self.pidx = np.zeros((self.nf, pl), dtype=int)
        for f in range(self.nf):
            for i in range(pl):
                seg = self.conv_out[f, i*2:i*2+2] if i*2+2<=self.col else self.conv_out[f, i*2:i*2+1]
                idx = np.argmax(seg)
                self.pool[f,i] = seg[idx]
                self.pidx[f,i] = i*2+idx
        # FC
        self.flat = self.pool.flatten().reshape(-1,1)
        self.z1 = self.p['W1']@self.flat + self.p['b1']
        self.a1 = relu(self.z1)
        return (self.p['W2']@self.a1 + self.p['b2']).flatten()

    def bwd(self, x, yt, yp):
        gr = {k: np.zeros_like(v) for k,v in self.p.items()}
        dy = (yp.reshape(-1,1)-yt.reshape(-1,1))
        gr['W2'] = dy@self.a1.T; gr['b2'] = dy
        d1 = self.p['W2'].T@dy * relu_d(self.z1)
        gr['W1'] = d1@self.flat.T; gr['b1'] = d1
        df = self.p['W1'].T@d1
        dp = df.reshape(self.pool.shape)
        dc = np.zeros_like(self.conv_out)
        for f in range(self.nf):
            for i in range(self.pl):
                dc[f, self.pidx[f,i]] = dp[f,i]
        dc *= relu_d(self.conv_raw)
        for f in range(self.nf):
            for t in range(self.col):
                gr['Wc'][f] += dc[f,t]*x[t:t+self.ks]
            gr['bc'][f,0] += np.sum(dc[f])
        self.opt.step(self.p, gr)

    def train(self, X, Y, epochs=15):
        n = len(X); losses = []
        for ep in range(epochs):
            idx = np.random.permutation(n); el = 0
            for j in idx:
                yp = self.fwd(X[j])
                el += np.mean((yp-Y[j])**2)
                self.bwd(X[j], Y[j], yp)
            losses.append(el/n)
            sys.stdout.write(f'\r    Epoch {ep+1}/{epochs} loss={el/n:.6f}')
            sys.stdout.flush()
        print()
        return losses

    def predict(self, X):
        return np.array([self.fwd(x) for x in X])


def prepare_data(brent_close, features=None, lb=10, tr=0.8):
    sy = MinMaxScaler()
    ys = sy.fit_transform(brent_close.reshape(-1,1)).flatten()
    if features is not None:
        sx = MinMaxScaler()
        xs = sx.fit_transform(features)
        data = np.column_stack([ys, xs])
    else:
        data = ys.reshape(-1,1)
    X, Y = [], []
    for i in range(lb, len(data)):
        X.append(data[i-lb:i]); Y.append(ys[i:i+1])
    X, Y = np.array(X), np.array(Y)
    sp = int(len(X)*tr)
    return X[:sp], Y[:sp], X[sp:], Y[sp:], sy


def evaluate(yt, yp, sc):
    yti = sc.inverse_transform(yt.reshape(-1,1)).flatten()
    ypi = sc.inverse_transform(yp.reshape(-1,1)).flatten()
    mse = mean_squared_error(yti, ypi)
    return {
        'MSE': mse, 'RMSE': np.sqrt(mse),
        'MAE': mean_absolute_error(yti, ypi),
        'R2': r2_score(yti, ypi),
        'MAPE': np.mean(np.abs((yti-ypi)/(yti+1e-8)))*100
    }, yti, ypi


def main():
    print("="*70)
    print("第二阶段：三组模型 × 两种架构 = 6个预测模型对比")
    print("="*70)

    # Load & align
    dfs = {}
    for fn, nm in [('数据/伦敦布伦特原油期货价格.xlsx','brent'),('数据/美国标准普尔500指数.xlsx','sp500'),
                    ('数据/美元指数.xlsx','dollar'),('数据/恐慌指数.xlsx','vix'),('数据/欧佩克油.xlsx','opec')]:
        df = pd.read_excel(fn)
        df.columns = ['Date','Open','High','Low','Close']
        df['Date'] = pd.to_datetime(df['Date'])
        dfs[nm] = df.sort_values('Date').reset_index(drop=True)

    common = sorted(set.intersection(*[set(d['Date']) for d in dfs.values()]))
    for k in dfs:
        dfs[k] = dfs[k][dfs[k]['Date'].isin(common)].reset_index(drop=True)

    bc = dfs['brent']['Close'].values.astype(float)
    feat_all = np.column_stack([dfs[k]['Close'].values for k in ['sp500','dollar','vix','opec']]).astype(float)
    feat_vip = np.column_stack([dfs[k]['Close'].values for k in ['sp500','dollar','vix']]).astype(float)
    print(f"样本数: {len(bc)}")

    LB, EP = 10, 15
    configs = [
        ('M1_单变量', None, '仅历史价格序列'),
        ('M2_全因子', feat_all, 'SP500+USD+VIX+OPEC'),
        ('M3_VIP筛选', feat_vip, 'SP500+USD+VIX (VIP>1)'),
    ]

    results = {}
    preds = {}

    for mname, feat, desc in configs:
        Xtr,Ytr,Xte,Yte,sc = prepare_data(bc, feat, lb=LB)
        indim = Xtr.shape[2]
        print(f"\n{'='*70}")
        print(f"  {mname}: {desc}  (输入维度={indim}, 训练={len(Xtr)}, 测试={len(Xte)})")

        # LSTM
        print(f"\n  >>> LSTM <<<")
        np.random.seed(42)
        t0 = time.time()
        lstm = LSTMModel(indim, hid=16, lr=0.002)
        ll = lstm.train(Xtr, Ytr, epochs=EP)
        lp = lstm.predict(Xte)
        lm, yt, lpi = evaluate(Yte, lp, sc)
        print(f"    时间: {time.time()-t0:.1f}s | RMSE={lm['RMSE']:.3f} MAE={lm['MAE']:.3f} R²={lm['R2']:.4f} MAPE={lm['MAPE']:.2f}%")
        results[f'{mname}_LSTM'] = lm
        preds[f'{mname}_LSTM'] = {'true':yt, 'pred':lpi, 'loss':ll}

        # CNN
        print(f"\n  >>> CNN <<<")
        np.random.seed(42)
        t0 = time.time()
        cnn = CNN1DModel(indim, LB, nf=12, ks=3, lr=0.002)
        cl = cnn.train(Xtr, Ytr, epochs=EP)
        cp = cnn.predict(Xte)
        cm, _, cpi = evaluate(Yte, cp, sc)
        print(f"    时间: {time.time()-t0:.1f}s | RMSE={cm['RMSE']:.3f} MAE={cm['MAE']:.3f} R²={cm['R2']:.4f} MAPE={cm['MAPE']:.2f}%")
        results[f'{mname}_CNN'] = cm
        preds[f'{mname}_CNN'] = {'true':yt, 'pred':cpi, 'loss':cl}

    # Summary
    print("\n\n" + "="*90)
    print("                    六大模型性能对比总表")
    print("="*90)
    print(f"{'模型':>22} | {'MSE':>10} | {'RMSE':>8} | {'MAE':>8} | {'R²':>8} | {'MAPE%':>8}")
    print("-"*78)
    for name, m in results.items():
        print(f"{name:>22} | {m['MSE']:>10.2f} | {m['RMSE']:>8.3f} | {m['MAE']:>8.3f} | {m['R2']:>8.4f} | {m['MAPE']:>8.2f}")

    # Find best
    best = min(results.items(), key=lambda x: x[1]['RMSE'])
    print(f"\n  最优模型: {best[0]} (RMSE={best[1]['RMSE']:.3f}, R²={best[1]['R2']:.4f})")

    # Save
    with open('model_results.json','w') as f:
        json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in results.items()}, f, indent=2, ensure_ascii=False)

    np.savez('predictions.npz',
        **{f'{k}_true':v['true'] for k,v in preds.items()},
        **{f'{k}_pred':v['pred'] for k,v in preds.items()},
        **{f'{k}_loss':np.array(v['loss']) for k,v in preds.items()})

    return results, preds

if __name__ == '__main__':
    main()