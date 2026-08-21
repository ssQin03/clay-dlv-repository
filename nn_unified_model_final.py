"""
=============================================================================
Unified Neural Network Model for Double-Layer Mid-Plane Potential
=============================================================================
Input:  [D, boundary_value, is_cc]
  D:      dimensionless separation (0.002 ~ 10.5)
  val:    surface potential z for CP (0.2~12) or charge p for CC (0.2~5000)
  is_cc:  0=CP, 1=CC
Output: u: dimensionless mid-plane potential

Data source:
  matlab/poten_mid_full.mat     → CP: 50 z values × 500 D values
  matlab/poten_mid_ce_full.mat  → CC: 78 p values × 500 D values

Usage: python nn_unified_model_final.py
=============================================================================
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import scipy.io as sio
import warnings, os, time, json
warnings.filterwarnings('ignore')

# ── Plot Style ──
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.2,
})

OUT_DIR = 'nn_results_final'
os.makedirs(OUT_DIR, exist_ok=True)

# ===================================================================
# 1. DATA LOADING
# ===================================================================
def load_data():
    """Load MATLAB data, return X (N,3) and y (N,)"""
    # ── CP ──
    cp = sio.loadmat('matlab/poten_mid_full.mat')['poten_mid']  # (500, 50) = (D, z)
    z_vals = np.arange(0.2, 10.01, 0.2)          # 50 values
    D_vals = np.arange(0.02, 10.01, 0.02)        # 500 values

    X_cp, y_cp = [], []
    for j, z in enumerate(z_vals):
        for i, D in enumerate(D_vals):
            X_cp.append([D, z, 0.0])
            y_cp.append(float(cp[i, j]))

    # ── CC ──
    cc = sio.loadmat('matlab/poten_mid_ce_full.mat')['poten_mid']  # (78, 500) = (p, D)
    p1 = np.arange(0.2, 10.1, 0.2)
    p2 = np.arange(20, 101, 10)
    p3 = np.arange(150, 301, 50)
    p4 = np.arange(400, 1001, 100)
    p5 = np.arange(1500, 5001, 500)
    p_vals = np.concatenate([p1, p2, p3, p4, p5])  # 78 values

    X_cc, y_cc = [], []
    for i, p in enumerate(p_vals):
        for j, D in enumerate(D_vals):
            X_cc.append([D, p, 1.0])
            y_cc.append(float(cc[i, j]))

    X = np.array(X_cp + X_cc)
    y = np.array(y_cp + y_cc)
    meta = {
        'n_cp': len(y_cp), 'n_cc': len(y_cc), 'total': len(y),
        'z_range': [float(z_vals.min()), float(z_vals.max())],
        'p_range': [float(p_vals.min()), float(p_vals.max())],
        'D_range': [float(D_vals.min()), float(D_vals.max())],
    }
    return X, y, meta


# ===================================================================
# 2. PREPROCESSING
# ===================================================================
class Normalizer:
    """Log-transform D and boundary_value, z-score u in log-space."""
    def fit(self, X, y):
        D_log = np.log10(X[:, 0] + 1e-10)
        self.D_mean, self.D_std = D_log.mean(), D_log.std()

        # boundary_value: log for CC, linear for CP
        self.val_mean_cc = np.log10(X[X[:, 2]==1, 1] + 1e-10).mean()
        self.val_std_cc = np.log10(X[X[:, 2]==1, 1] + 1e-10).std()
        self.val_mean_cp = X[X[:, 2]==0, 1].mean()
        self.val_std_cp = X[X[:, 2]==0, 1].std()

        log_y = np.log(y)
        self.y_mean, self.y_std = log_y.mean(), log_y.std()

    def transform_X(self, X):
        Xn = np.zeros_like(X)
        D_log = np.log10(X[:, 0] + 1e-10)
        Xn[:, 0] = (D_log - self.D_mean) / self.D_std

        is_cc = X[:, 2] == 1
        val = X[:, 1].copy()
        val[is_cc] = (np.log10(val[is_cc] + 1e-10) - self.val_mean_cc) / self.val_std_cc
        val[~is_cc] = (val[~is_cc] - self.val_mean_cp) / self.val_std_cp
        Xn[:, 1] = val
        Xn[:, 2] = X[:, 2]
        return Xn.astype(np.float32)

    def transform_y(self, y):
        return ((np.log(y) - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, y_norm):
        return np.exp(y_norm * self.y_std + self.y_mean)


# ===================================================================
# 3. NEURAL NETWORK
# ===================================================================
class MidplaneNN(nn.Module):
    def __init__(self, widths=[64, 32, 16]):
        super().__init__()
        layers = []
        prev = 3
        for w in widths:
            layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.SiLU()]
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(model, train_loader, val_loader, epochs=500):
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    loss_fn = nn.MSELoss()

    best_state, best_loss = None, float('inf')
    hist = {'train': [], 'val': []}
    stall = 0

    for ep in range(epochs):
        # Train
        model.train()
        tl = 0.0
        for Xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * Xb.size(0)
        tl /= len(train_loader.dataset)
        hist['train'].append(tl)

        # Val
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                vl += loss_fn(model(Xb), yb).item() * Xb.size(0)
        vl /= len(val_loader.dataset)
        hist['val'].append(vl)

        sched.step()

        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1

        if (ep+1) % 100 == 0:
            print(f'  [{ep+1:>4d}/{epochs}] train={tl:.3e} val={vl:.3e} lr={sched.get_last_lr()[0]:.2e}')

        if stall >= 80:
            print(f'  Early stop at epoch {ep+1}')
            break

    model.load_state_dict(best_state)
    return model, hist


# ===================================================================
# 4. ANALYTICAL FORMULAS (from Appendix I & II)
# ===================================================================
def analytical_cp(z, D):
    # Appendix I coefficients (log-quadratic form in ln z)
    lz = np.log(z)
    if z <= 1:
        lna = -0.00003880*lz**2 + 0.99989319*lz + 0.03109701
        b = 0.00812285*lz**2 + 0.02208481*lz + 0.18901893
        c = -0.02269477*lz**2 - 0.06239428*lz + 1.42394752
    elif z <= 3:
        lna = -0.21817637*lz**2 + 1.27655407*lz - 0.02329595
        b = 0.08015650*lz**2 + 0.01400527*lz + 0.19127175
        c = -0.10392482*lz**2 - 0.11071209*lz + 1.42975126
    elif z <= 8:
        lna = -0.17086119*lz**2 + 1.54337158*lz - 0.41164313
        b = 0.21514507*lz**2 - 0.17508924*lz + 0.22020251
        c = 0.07953524*lz**2 - 0.74978804*lz + 1.94595267
    else:
        lna = -0.60623545*lz**2 + 3.31869898*lz - 2.22143938
        b = -0.38301428*lz**2 + 2.25391138*lz - 2.24557015
        c = 0.34039426*lz**2 - 1.79623008*lz + 2.99498977
    return np.exp(lna) * np.exp(-b * D**c)
def analytical_cc(p, D):
    # Appendix II coefficients (log-quadratic form in ln p)
    lz = np.log(p)
    if p <= 1:
        lna = 0.08537273*lz**2 - 0.03560028*lz + 1.76372871
        b = 0.18642722*lz**2 - 0.65314192*lz + 1.48048619
        c = 0.00730216*lz**2 + 0.11663244*lz + 0.53018798
    elif p <= 10:
        lna = 0.02866336*lz**2 + 0.02682373*lz + 1.70625653
        b = 0.12615301*lz**2 - 0.56561330*lz + 1.43622644
        c = -0.04036393*lz**2 + 0.19368240*lz + 0.48535577
    elif p <= 1000:
        lna = -0.01796863*lz**2 + 0.27140114*lz + 1.37183006
        b = -0.00809105*lz**2 + 0.15494659*lz + 0.41013065
        c = 0.00443583*lz**2 - 0.07700676*lz + 0.90515582
    else:
        lna = -0.00542249*lz**2 + 0.09771339*lz + 1.97471926
        b = -0.00548169*lz**2 + 0.09887974*lz + 0.66895504
        c = 0.00216282*lz**2 - 0.03895442*lz + 0.75205041
    return np.exp(lna) * np.exp(-b * D**c)
def predict_analytical(X):
    y = np.zeros(len(X))
    for i, (D, v, t) in enumerate(X):
        y[i] = analytical_cc(v, D) if t > 0.5 else analytical_cp(v, D)
    return y


# ===================================================================
# 5. METRICS
# ===================================================================
def metrics(y_t, y_p, label=''):
    mae = float(np.mean(np.abs(y_p - y_t)))
    rmse = float(np.sqrt(np.mean((y_p - y_t)**2)))
    r2 = float(1 - np.sum((y_t - y_p)**2) / np.sum((y_t - y_t.mean())**2))
    # Relative error only where u >= 0.01 (physically meaningful)
    mask = y_t >= 0.01
    re = np.abs((y_p[mask] - y_t[mask]) / y_t[mask]) * 100
    if len(re):
        mape, p95, mx = float(re.mean()), float(np.percentile(re, 95)), float(re.max())
    else:
        mape = p95 = mx = 0.0
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE(%)': mape, 'P95(%)': p95, 'Max(%)': mx}


# ===================================================================
# 6. FIGURES
# ===================================================================
def fig_u_curves(model, norm, X_cp, y_cp, X_cc, y_cc):
    """Fig 1: u vs D curves comparing NN | Analytical | Numerical"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, 6))
    D_plot = np.linspace(0.05, 10, 200)

    # ── CP panel ──
    ax = axes[0]; z_list = [0.5, 1, 2, 4, 7, 11]
    for k, z in enumerate(z_list):
        # NN
        x_nn = norm.transform_X(np.array([[D, z, 0] for D in D_plot]))
        with torch.no_grad():
            u_nn = norm.inverse_y(model(torch.FloatTensor(x_nn)).numpy())
        ax.plot(D_plot, u_nn, '-', color=colors[k], lw=2, label=f'z={z}')
        # Analytical
        u_ana = np.array([analytical_cp(z, d) for d in D_plot])
        ax.plot(D_plot, u_ana, '--', color=colors[k], lw=1, alpha=0.5)
    ax.set_xlabel('D', fontsize=13); ax.set_ylabel('u', fontsize=13)
    ax.set_title('Constant Potential (CP)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    # ── CC panel ──
    ax = axes[1]; p_list = [1, 5, 20, 100, 500, 3000]
    for k, p in enumerate(p_list):
        x_nn = norm.transform_X(np.array([[D, p, 1] for D in D_plot]))
        with torch.no_grad():
            u_nn = norm.inverse_y(model(torch.FloatTensor(x_nn)).numpy())
        ax.plot(D_plot, u_nn, '-', color=colors[k], lw=2, label=f'p={p}')
        u_ana = np.array([analytical_cc(p, d) for d in D_plot])
        ax.plot(D_plot, u_ana, '--', color=colors[k], lw=1, alpha=0.5)
    ax.set_xlabel('D', fontsize=13); ax.set_ylabel('u', fontsize=13)
    ax.set_title('Constant Charge (CC)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)

    fig.savefig(f'{OUT_DIR}/fig1_u_curves.png'); plt.close()
    print(f'  [fig] {OUT_DIR}/fig1_u_curves.png')


def fig_error_analysis(model, norm, X_test, y_test):
    """Fig 2: Error analysis with clean layout."""
    from matplotlib.gridspec import GridSpec
    model.eval()
    x_norm = norm.transform_X(X_test)
    with torch.no_grad():
        y_pred = norm.inverse_y(model(torch.FloatTensor(x_norm)).numpy())
    y_true = y_test

    is_cp = X_test[:, 2] == 0
    re_cp = np.abs(y_pred[is_cp] - y_true[is_cp]) / np.maximum(y_true[is_cp], 1e-6) * 100
    re_cc = np.abs(y_pred[~is_cp] - y_true[~is_cp]) / np.maximum(y_true[~is_cp], 1e-6) * 100

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig)

    # (a) CP scatter
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(y_true[is_cp], y_pred[is_cp], c=np.clip(re_cp, 0, 5),
                    s=2, cmap='jet', alpha=0.5)
    ax.plot([0, 12], [0, 12], 'r--', lw=1)
    ax.set_xlabel('Numerical u', fontsize=12); ax.set_ylabel('NN u', fontsize=12)
    ax.set_title('CP Predictions', fontsize=13, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='|RE| (%)')

    # (b) CC scatter
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(y_true[~is_cp], y_pred[~is_cp], c=np.clip(re_cc, 0, 5),
                    s=2, cmap='jet', alpha=0.5)
    ax.plot([0, 12], [0, 12], 'r--', lw=1)
    ax.set_xlabel('Numerical u', fontsize=12); ax.set_ylabel('NN u', fontsize=12)
    ax.set_title('CC Predictions', fontsize=13, fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label='|RE| (%)')

    # (c) Cumulative distribution
    ax = fig.add_subplot(gs[0, 2])
    for data, lbl, c in [(re_cp, 'CP', '#2196F3'), (re_cc, 'CC', '#FF5722')]:
        s = np.sort(data); cd = np.arange(1, len(s)+1)/len(s)*100
        ax.plot(s, cd, lw=2, label=lbl, color=c)
    ax.set_xscale('log'); ax.set_xlim([1e-4, 100])
    ax.set_xlabel('Relative Error (%)', fontsize=12)
    ax.set_ylabel('Cumulative (%)', fontsize=12)
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.axhline(95, ls=':', c='gray', lw=0.8)

    # (d) Error vs D (CP)
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(X_test[is_cp, 0], re_cp, s=1, c='#2196F3', alpha=0.3)
    ax.set_yscale('log'); ax.set_ylim([1e-4, 100])
    ax.set_xlabel('D', fontsize=12); ax.set_ylabel('|RE| (%)', fontsize=12)
    ax.set_title('CP: Error vs D', fontsize=13, fontweight='bold'); ax.grid(True, alpha=0.3)

    # (e) Error vs D (CC)
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(X_test[~is_cp, 0], re_cc, s=1, c='#FF5722', alpha=0.3)
    ax.set_yscale('log'); ax.set_ylim([1e-4, 100])
    ax.set_xlabel('D', fontsize=12); ax.set_ylabel('|RE| (%)', fontsize=12)
    ax.set_title('CC: Error vs D', fontsize=13, fontweight='bold'); ax.grid(True, alpha=0.3)

    # (f) Bar chart: NN vs Analytical
    ax = fig.add_subplot(gs[1, 2])
    m_all = {}
    for name, X_p, y_p in [('NN_CP', X_test[is_cp], y_pred[is_cp]),
                             ('NN_CC', X_test[~is_cp], y_pred[~is_cp])]:
        m_all[name] = metrics(y_p, y_p)
    y_ana = predict_analytical(X_test)
    for name, X_p, y_p in [('Ana_CP', X_test[is_cp], y_ana[is_cp]),
                            ('Ana_CC', X_test[~is_cp], y_ana[~is_cp])]:
        m_all[name] = metrics(y_p, y_p)

    keys = ['NN_CP', 'Ana_CP', 'NN_CC', 'Ana_CC']
    colors_k = ['#2196F3', '#90CAF9', '#FF5722', '#FFAB91']
    x_pos = np.arange(len(keys))
    maes = [m_all[k]['MAE'] for k in keys]
    rmses = [m_all[k]['RMSE'] for k in keys]
    w = 0.3
    ax.bar(x_pos - w/2, maes, w, label='MAE', color='#1565C0')
    ax.bar(x_pos + w/2, rmses, w, label='RMSE', color='#E65100')
    ax.set_xticks(x_pos); ax.set_xticklabels(keys, fontsize=9)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('MAE / RMSE Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig2_error_analysis.png'); plt.close()
    print(f'  [fig] {OUT_DIR}/fig2_error_analysis.png')


def fig_error_heatmap(model, norm):
    """Fig 3: Error heatmaps over parameter space."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model.eval()

    # CP heatmap
    ax = axes[0]
    z_r = np.linspace(0.3, 11.5, 60); D_r = np.logspace(-1.7, 1.02, 60)
    Z, DD = np.meshgrid(z_r, D_r)
    err = np.zeros_like(Z)
    for i in range(len(D_r)):
        for j in range(len(z_r)):
            u_ref = analytical_cp(z_r[j], D_r[i])
            xn = norm.transform_X(np.array([[D_r[i], z_r[j], 0]]))
            with torch.no_grad():
                u_nn = norm.inverse_y(model(torch.FloatTensor(xn)).numpy())[0]
            err[i, j] = abs(u_nn - u_ref) / max(u_ref, 1e-6) * 100
    im = ax.pcolormesh(DD, Z, np.clip(err, 0, 3), cmap='jet', shading='auto')
    ax.set_xscale('log'); ax.set_xlabel('D', fontsize=12)
    ax.set_ylabel('z', fontsize=12)
    ax.set_title('CP Relative Error (%)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='|RE| (%)')

    # CC heatmap
    ax = axes[1]
    p_r = np.logspace(np.log10(0.3), np.log10(4500), 60)
    PP, DD2 = np.meshgrid(p_r, D_r)
    err2 = np.zeros_like(PP)
    for i in range(len(D_r)):
        for j in range(len(p_r)):
            u_ref = analytical_cc(p_r[j], D_r[i])
            xn = norm.transform_X(np.array([[D_r[i], p_r[j], 1]]))
            with torch.no_grad():
                u_nn = norm.inverse_y(model(torch.FloatTensor(xn)).numpy())[0]
            err2[i, j] = abs(u_nn - u_ref) / max(u_ref, 1e-6) * 100
    im = ax.pcolormesh(DD2, PP, np.clip(err2, 0, 3), cmap='jet', shading='auto')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('D', fontsize=12); ax.set_ylabel('p', fontsize=12)
    ax.set_title('CC Relative Error (%)', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='|RE| (%)')

    fig.tight_layout()
    fig.savefig(f'{OUT_DIR}/fig3_error_heatmap.png'); plt.close()
    print(f'  [fig] {OUT_DIR}/fig3_error_heatmap.png')


def fig_training_curve(hist):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(hist['train'], lw=1.5, label='Training')
    ax.semilogy(hist['val'], lw=1.5, label='Validation')
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('MSE Loss', fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_title('Training History', fontsize=13, fontweight='bold')
    fig.tight_layout(); fig.savefig(f'{OUT_DIR}/fig0_training.png', dpi=250); plt.close()


# ===================================================================
# 7. EXPORT: C++ compatible header
# ===================================================================
def export_cpp(model, norm, path=f'{OUT_DIR}/midplane_nn_model.h'):
    """Export NN weights as a C++ header for DEM integration."""
    lin_weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lin_weights.append({
                'weight': module.weight.detach().cpu().numpy(),
                'bias': module.bias.detach().cpu().numpy(),
            })

    n_layers = len(lin_weights)
    layer_sizes = [lin_weights[0]['weight'].shape[1]] + [w['weight'].shape[0] for w in lin_weights]
    arch_str = " -> ".join(map(str, layer_sizes))

    lines = [
        '// Auto-generated NN model for mid-plane potential prediction',
        f'// Architecture: {arch_str} (SiLU activation)',
        '// Input: float[3] = {D, boundary_value, is_cc}',
        '// Output: float u (dimensionless mid-plane potential)',
        '#ifndef MIDPLANE_NN_MODEL_H',
        '#define MIDPLANE_NN_MODEL_H',
        '',
        '#include <cmath>',
        '#include <algorithm>',
        '',
        'namespace midplane_nn {',
        '',
        f'// Normalization constants',
        f'constexpr float D_MEAN = {norm.D_mean:.8f}f;',
        f'constexpr float D_STD  = {norm.D_std:.8f}f;',
        f'constexpr float V_MEAN_CP = {norm.val_mean_cp:.8f}f;',
        f'constexpr float V_STD_CP  = {norm.val_std_cp:.8f}f;',
        f'constexpr float V_MEAN_CC = {norm.val_mean_cc:.8f}f;',
        f'constexpr float V_STD_CC  = {norm.val_std_cc:.8f}f;',
        f'constexpr float Y_MEAN = {norm.y_mean:.8f}f;',
        f'constexpr float Y_STD  = {norm.y_std:.8f}f;',
        '',
        'inline float silu(float x) { return x / (1.0f + std::exp(-x)); }',
        '',
        'inline float predict(float D, float boundary_val, float is_cc) {',
        '    float x0 = (std::log10(std::max(D, 1e-10f)) - D_MEAN) / D_STD;',
        '    float x1;',
        '    if (is_cc > 0.5f)',
        '        x1 = (std::log10(std::max(boundary_val, 1e-10f)) - V_MEAN_CC) / V_STD_CC;',
        '    else',
        '        x1 = (boundary_val - V_MEAN_CP) / V_STD_CP;',
        '    float x2 = is_cc;',
        '',
    ]

    h_names = ['in']
    for i in range(1, n_layers):
        h_names.append(f'h{i-1}')

    for li, W in enumerate(lin_weights):
        w_mat = W['weight']
        b_vec = W['bias']
        out_dim = w_mat.shape[0]
        in_dim = w_mat.shape[1]
        src = h_names[li]
        dst = h_names[li + 1] if li < n_layers - 1 else 'out'
        is_last = (li == n_layers - 1)

        if not is_last:
            lines.append(f'    // Layer {li}: {in_dim} -> {out_dim} (SiLU)')
            lines.append(f'    float {dst}[{out_dim}];')
        else:
            lines.append(f'    // Layer {li}: {in_dim} -> {out_dim} (linear)')

        if src == 'in':
            in_refs = ['x0', 'x1', 'x2']
        else:
            in_refs = [f'{src}[{j}]' for j in range(in_dim)]

        for i in range(out_dim):
            parts = []
            for j in range(in_dim):
                if abs(w_mat[i, j]) > 1e-8:
                    parts.append(f'{w_mat[i, j]:.10f}f*{in_refs[j]}')
            expr = ' + '.join(parts) if parts else '0.0f'
            if is_last:
                lines.append(f'    float u_norm = {expr} + {b_vec[i]:.10f}f;')
            else:
                lines.append(f'    {dst}[{i}] = silu({expr} + {b_vec[i]:.10f}f);')
        if not is_last:
            lines.append('')

    lines.extend([
        '',
        '    return std::exp(u_norm * Y_STD + Y_MEAN);',
        '}',
        '',
        '}  // namespace midplane_nn',
        '',
        '#endif  // MIDPLANE_NN_MODEL_H',
    ])

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    size = os.path.getsize(path) / 1024
    print(f'  [export] C++ header: {path} ({size:.1f} KB)')


# ===================================================================
# 8. REPORT
# ===================================================================
def print_report(model, norm, X, y, hist):
    """Print a comprehensive report with all metrics."""
    model.eval()
    x_norm = norm.transform_X(X)
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(x_norm)).numpy()
    y_pred = norm.inverse_y(y_pred)
    y_ana = predict_analytical(X)

    is_cp = X[:, 2] == 0
    is_cc = X[:, 2] == 1

    # Full range
    m_nn_cp = metrics(y[is_cp], y_pred[is_cp])
    m_nn_cc = metrics(y[is_cc], y_pred[is_cc])
    m_an_cp = metrics(y[is_cp], y_ana[is_cp])
    m_an_cc = metrics(y[is_cc], y_ana[is_cc])

    print('\n' + '='*78)
    print('  UNIFIED NN MODEL - COMPREHENSIVE EVALUATION REPORT')
    print('='*78)
    print(f'  Dataset: CP={int(is_cp.sum())}  CC={int(is_cc.sum())}  Total={len(X)}')
    print(f'  Model  : {sum(p.numel() for p in model.parameters()):,} parameters')
    print(f'  Epochs : {len(hist["train"])}')
    print(f'  Val    : {hist["val"][-1]:.3e}')
    print()
    print(f'  {"Method":<16} {"MAE":>10} {"RMSE":>10} {"R2":>10} {"MAPE(%)":>10} {"P95(%)":>10} {"Max(%)":>10}')
    print(f'  {"-"*16} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
    for m, name in [(m_nn_cp, 'NN (CP)'), (m_nn_cc, 'NN (CC)'),
                     (m_an_cp, 'Analytical (CP)'), (m_an_cc, 'Analytical (CC)')]:
        print(f'  {name:<16} {m["MAE"]:>10.5f} {m["RMSE"]:>10.5f} {m["R2"]:>10.6f} '
              f'{m["MAPE(%)"]:>9.3f} {m["P95(%)"]:>9.3f} {m["Max(%)"]:>9.3f}')

    # Improvement ratios
    print()
    print(f'  NN improvement over Analytical formulas:')
    for tag, nn_m, an_m in [('CP', m_nn_cp, m_an_cp), ('CC', m_nn_cc, m_an_cc)]:
        if an_m['MAPE(%)'] > 0:
            ratio = an_m['MAPE(%)'] / nn_m['MAPE(%)']
            print(f'    {tag}: MAPE reduced by {ratio:.0f}x ({an_m["MAPE(%)"]:.2f}% -> {nn_m["MAPE(%)"]:.2f}%)')

    # Close-range (D < 3) metrics (DEM-relevant region)
    print()
    print(f'  Close-range metrics (D < 3, DEM-relevant):')
    for tag, mask in [('CP', is_cp), ('CC', is_cc)]:
        dm = X[:, 0] < 3
        idx = mask & dm
        if idx.sum() > 0:
            mn = metrics(y[idx], y_pred[idx])
            ma = metrics(y[idx], y_ana[idx])
            print(f'    {tag}: NN MAPE={mn["MAPE(%)"]:.3f}%  Analytical MAPE={ma["MAPE(%)"]:.3f}%')

    print('='*78)


# ===================================================================
# 9. MAIN
# ===================================================================
def main():
    print('='*55)
    print('  Unified NN Model - Final Version')
    print('='*55)

    # 1. Load
    print('\n[1] Loading data...')
    X, y, meta = load_data()
    print(f'    CP={meta["n_cp"]}  CC={meta["n_cc"]}  Total={meta["total"]}')

    # 2. Split
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n = len(X)
    X_train = X[idx[:int(0.8*n)]]
    y_train = y[idx[:int(0.8*n)]]
    X_val   = X[idx[int(0.8*n):int(0.9*n)]]
    y_val   = y[idx[int(0.8*n):int(0.9*n)]]
    X_test  = X[idx[int(0.9*n):]]
    y_test  = y[idx[int(0.9*n):]]

    # 3. Normalize
    print('[2] Normalizing...')
    norm = Normalizer()
    norm.fit(X_train, y_train)

    # 4. DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(norm.transform_X(X_train)),
                      torch.FloatTensor(norm.transform_y(y_train))),
        batch_size=256, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(norm.transform_X(X_val)),
                      torch.FloatTensor(norm.transform_y(y_val))),
        batch_size=256)

    # 5. Train
    from copy import deepcopy
    best_model = None; best_hist = None; best_val = float('inf')

    print('[3] Training (trying 3 architectures)...')
    archs = [[64, 32, 16], [96, 48, 24], [128, 64, 32]]
    for arch in archs:
        arch_str = " -> ".join(map(str, arch))
        print(f'  Architecture: 3 -> {arch_str} -> 1')
        model = MidplaneNN(widths=arch)
        model, hist = train_model(model, train_loader, val_loader, epochs=500)
        final_vl = hist['val'][-1]
        best_vl = min(hist['val'])
        print(f'    Best val loss: {best_vl:.3e}')
        if best_vl < best_val:
            best_val = best_vl
            best_model = deepcopy(model)
            best_hist = hist

    model = best_model; hist = best_hist
    print(f'\n  Selected: val loss = {best_val:.3e}')

    # 6. Evaluate
    print('[4] Evaluating on test set...')
    print_report(model, norm, X_test, y_test, hist)

    # 7. Figures
    print('[5] Generating figures...')
    fig_training_curve(hist)

    is_cp_t = X_test[:, 2] == 0
    fig_u_curves(model, norm, X_test[is_cp_t], y_test[is_cp_t],
                 X_test[~is_cp_t], y_test[~is_cp_t])
    fig_error_analysis(model, norm, X_test, y_test)
    fig_error_heatmap(model, norm)

    # 8. Export
    print('[6] Exporting...')
    export_cpp(model, norm)

    # Save PyTorch model
    torch.save({
        'state': model.state_dict(),
        'arch': [128, 64, 32],  # actual best architecture selected above
        'norm_stats': {k: float(v) for k, v in norm.__dict__.items()},
    }, f'{OUT_DIR}/unified_nn_model.pt')
    print(f'  [export] PyTorch: {OUT_DIR}/unified_nn_model.pt')

    # Save metrics JSON
    with torch.no_grad():
        x_te = norm.transform_X(X_test)
        y_te_pred = norm.inverse_y(model(torch.FloatTensor(x_te)).numpy())
    y_te_ana = predict_analytical(X_test)
    is_cp = X_test[:, 2] == 0
    results = {
        'nn_cp': metrics(y_test[is_cp], y_te_pred[is_cp]),
        'nn_cc': metrics(y_test[~is_cp], y_te_pred[~is_cp]),
        'analytical_cp': metrics(y_test[is_cp], y_te_ana[is_cp]),
        'analytical_cc': metrics(y_test[~is_cp], y_te_ana[~is_cp]),
    }
    with open(f'{OUT_DIR}/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n  All results saved to {OUT_DIR}/')
    print('='*55)
    return model, norm, results


if __name__ == '__main__':
    model, norm, results = main()
