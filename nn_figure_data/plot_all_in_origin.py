"""
============================================================================
 Origin 批量绘图脚本 (出版质量优化版)
 功能: 将 NN画图数据/ 中所有 CSV 导入 Origin，绘制10张出版级对比图
 输出: 画图对比图.opju (10 个 worksheet + 10 个 graph)
============================================================================
改进:
  - 出版级配色 (RGB), 非 Origin 默认颜色
  - 所有曲线带图例
  - 字体 Times New Roman (中文字体保留宋体)
  - 刻度朝内
  - 浅灰色虚线网格
  - 坐标轴线宽提升到 1.5 pt
  - 子图标签 (a), (b) 等
  - 适当的图层尺寸比例
============================================================================
"""

import originpro as op
import os
import time
import math

DATA_DIR = r'E:\小论文\拟合公式\NN画图数据'
OPJU_FILE = os.path.join(DATA_DIR, '画图对比图.opju')

# ================================================================
# 出版级配色 (MATLAB parula-like, 高区分度)
# ================================================================
COLORS_5 = [
    (0.000, 0.447, 0.741),  # 蓝
    (0.850, 0.325, 0.098),  # 红橙
    (0.929, 0.694, 0.125),  # 金
    (0.494, 0.184, 0.556),  # 紫
    (0.466, 0.674, 0.188),  # 绿
]
COLORS_6 = COLORS_5 + [(0.301, 0.745, 0.933)]  # 青

# ================================================================
# 辅助函数
# ================================================================

def import_csv(ws_name, csv_name):
    ws = op.new_sheet('w', lname=ws_name)
    ws.from_file(os.path.join(DATA_DIR, csv_name))
    print(f'  Import: {csv_name} -> {ws_name} ({ws.rows} rows)')
    return ws


def _safe_lt(cmds):
    """安全执行 LabTalk 命令列表."""
    for cmd in cmds:
        try:
            op.lt_exec(cmd)
        except Exception:
            pass


def create_line_plot(ws, graph_name, y_start, y_end,
                     line_configs, legend_labels=None,
                     log_y=True, y_min=0.001, y_max=None,
                     x_min=0, x_max=10,
                     x_label='D', y_label='u',
                     subplot_label=None):
    """
    创建一条线图并应用完整出版级格式。

    Parameters
    ----------
    line_configs : list of (lstyle, lwidth, color)
        lstyle: 0=Solid, 1=Dash, 2=Dot, 3=DashDot
    legend_labels : list of str, optional
        每条曲线的图例文本 (长度必须等于曲线数)
    subplot_label : str, optional
        子图编号标签, 如 '(a)'
    """
    # ---- 1. 创建图和图层 ----
    gp = op.new_graph(lname=graph_name)
    gl = gp[0]

    # ---- 2. 添加所有曲线 ----
    for c in range(y_start, y_end + 1):
        gl.add_plot(ws, c, 0, type='l')

    # ---- 3. 曲线线型/线宽/颜色 ----
    style_map = {0: 'Solid', 1: 'Dash', 2: 'Dot', 3: 'DashDot'}
    plots = gl.plot_list()
    for i, (style, width, color) in enumerate(line_configs):
        if i < len(plots):
            p = plots[i]
            p.color = color
            p.set_float('linewidth', width)
            p.set_str('lineStyle', style_map.get(style, 'Solid'))

    # ---- 4. 轴标签 / 缩放 / 范围 (originpro API, 稳定) ----
    gl.xlabel = x_label
    gl.ylabel = y_label
    if log_y:
        gl.axis('y').scale = 'log10'
    gl.set_xlim(x_min, x_max)
    if y_min is not None:
        gl.set_ylim(y_min, y_max or 1e6)

    # ---- 5. 图例 ----
    if legend_labels:
        try:
            leg = gl.label('Legend')
            if leg is not None:
                # 用 \L(n) 占位符, 每行一条曲线
                lines = [f'\\L({i+1}) {lbl}' for i, lbl in enumerate(legend_labels)]
                leg.text = '\n'.join(lines)
        except Exception:
            pass  # 如果 API 失败, 降级用 LabTalk

    # ---- 6. LabTalk 出版级格式 ----
    gname = gp.name  # 短名如 "Graph1"
    lt_cmds = [
        # 坐标轴线宽
        f'set {gname}!x.line.width = 1.5;',
        f'set {gname}!y.line.width = 1.5;',
        # 刻度朝内
        f'set {gname}!x.ticks.in = 1;',
        f'set {gname}!y.ticks.in = 1;',
        # 轴标题字体 & 字号
        f'set {gname}!x.axistitle.font = "Times New Roman";',
        f'set {gname}!y.axistitle.font = "Times New Roman";',
        f'set {gname}!x.axistitle.size = 14;',
        f'set {gname}!y.axistitle.size = 14;',
        # 刻度标签字体 & 字号
        f'set {gname}!x.ticklabels.font = "Times New Roman";',
        f'set {gname}!y.ticklabels.font = "Times New Roman";',
        f'set {gname}!x.ticklabels.size = 12;',
        f'set {gname}!y.ticklabels.size = 12;',
        # 浅灰虚线网格
        f'set {gname}!x.grid = 1;',
        f'set {gname}!y.grid = 1;',
        f'set {gname}!x.grid.color = 0xCCCCCC;',
        f'set {gname}!y.grid.color = 0xCCCCCC;',
        f'set {gname}!x.grid.style = 2;',
        f'set {gname}!y.grid.style = 2;',
        # 图层尺寸 (cm)
        f'set {gname}!layer.width = 14;',
        f'set {gname}!layer.height = 5.5;',
    ]
    _safe_lt(lt_cmds)

    # 图例字体 (LabTalk 从 legend 对象设置)
    if legend_labels:
        _safe_lt([
            f'set {gname}!.legend.font = "Times New Roman";',
            f'set {gname}!.legend.size = 10;',
        ])

    # ---- 7. 子图标签 ----
    if subplot_label:
        # 放置在数据左上角 (根据坐标范围计算)
        lx = x_min + (x_max - x_min) * 0.04
        if log_y and y_min > 0:
            ly = 10 ** (math.log10(y_min) + (math.log10(y_max or 1e6) - math.log10(y_min)) * 0.92)
        else:
            ly = (y_min or 0) + ((y_max or 10) - (y_min or 0)) * 0.92
        gl.add_label(subplot_label, lx, ly)

    print(f'  Graph: {graph_name} ({len(gl.plot_list())} lines, legend={len(legend_labels) if legend_labels else 0})')
    return gp


# ---- 线型&图例 配置生成器 ----

def nn_pbe_configs(n_params, param_strs):
    """
    NN (实线 2.5pt) + PBE (虚线 1.2pt), 同色对应.
    param_strs : [str] 每个参数的图例标签 (如 'z=1.93')
    """
    colors = COLORS_5 if n_params <= 5 else COLORS_6[:n_params]
    configs = [(0, 2.5, colors[i]) for i in range(n_params)]
    configs += [(1, 1.2, colors[i]) for i in range(n_params)]
    labels = [f'NN {param_strs[i]}' for i in range(n_params)]
    labels += [f'PBE {param_strs[i]}' for i in range(n_params)]
    return configs, labels


def nn_pbe_configs_interleaved(n_params, param_strs):
    """
    For CSV columns arranged as NN, PBE, NN, PBE (interleaved).
    """
    colors = COLORS_5 if n_params <= 5 else COLORS_6[:n_params]
    configs = []
    labels = []
    for i in range(n_params):
        configs.append((0, 2.5, colors[i]))    # NN solid
        configs.append((1, 1.2, colors[i]))    # PBE dashed
        labels.append(f'NN {param_strs[i]}')
        labels.append(f'PBE {param_strs[i]}')
    return configs, labels


def nn_num_configs(param_strs):
    """
    NN (实线 2.0pt) vs Num (虚线 1.5pt), 同色对应.
    """
    n = len(param_strs)
    colors = COLORS_5[:n]
    configs = []
    labels = []
    for i in range(n):
        configs.append((0, 2.0, colors[i]))  # NN 实线
        configs.append((1, 1.5, colors[i]))  # Num 虚线
        labels.append(f'NN {param_strs[i]}')
        labels.append(f'Num {param_strs[i]}')
    return configs, labels


def matlab_configs(param_strs):
    """
    MATLAB 双线对比: Num(实线1.5) / New(点划线1.5), 同色.
    """
    n = len(param_strs)
    colors = COLORS_5[:n]
    configs = []
    labels = []
    for i in range(n):
        configs.append((0, 1.5, colors[i]))   # Num 实线
        configs.append((3, 1.5, colors[i]))   # New 点划线
        labels.append(f'Num {param_strs[i]}')
        labels.append(f'New {param_strs[i]}')
    return configs, labels


# ================================================================
# 主流程
# ================================================================

def main():
    print('=' * 60)
    print('Origin 批量绘图 - 出版质量优化版')
    print('=' * 60)

    # ---- 清空项目 ----
    if os.path.exists(OPJU_FILE):
        try:
            op.open(OPJU_FILE)
            op.lt_exec('doc -n;')
            print('已打开并清空现有项目')
        except Exception:
            print('新建项目')
    time.sleep(1)

    # ========================= PART 1: Fig.14 =========================
    print('\n=== PART 1: Fig.14 (Potential u vs D) ===')

    # Param strings
    z14 = ['z=1.93', 'z=3.87', 'z=5.80', 'z=7.74', 'z=9.66']
    p14 = ['p=0.266', 'p=0.532', 'p=0.799', 'p=1.065', 'p=1.331']

    ws1 = import_csv('Fig14_CP_Data', 'fig14_CP.csv')
    cfg1, lab1 = nn_pbe_configs(5, z14)
    create_line_plot(ws1, 'Fig14_CP', 1, 10,
                     cfg1, lab1,
                     log_y=True, y_min=0.001, y_max=12,
                     subplot_label='(a) Constant Potential (CP)')

    ws2 = import_csv('Fig14_CC_Data', 'fig14_CC.csv')
    cfg2, lab2 = nn_pbe_configs(5, p14)
    create_line_plot(ws2, 'Fig14_CC', 1, 10,
                     cfg2, lab2,
                     log_y=True, y_min=0.001, y_max=12,
                     subplot_label='(b) Constant Charge (CC)')

    # ========================= PART 2: Fig.17 Energy =========================
    print('\n=== PART 2: Fig.17 (Energy E vs D) ===')

    z17 = ['z=0.5', 'z=1.0', 'z=2.0', 'z=4.0', 'z=7.0', 'z=11.0']
    p17 = ['p=1', 'p=5', 'p=20', 'p=100', 'p=500', 'p=3000']

    y_lab_e = 'E (dimensionless energy)'

    ws3 = import_csv('Fig17_E_CP_Data', 'fig17_E_CP.csv')
    cfg3, lab3 = nn_pbe_configs(6, z17)
    create_line_plot(ws3, 'Fig17_E_CP', 1, 12,
                     cfg3, lab3,
                     log_y=True, y_label=y_lab_e,
                     subplot_label='(a) CP (z values)')

    ws4 = import_csv('Fig17_E_CC_Data', 'fig17_E_CC.csv')
    cfg4, lab4 = nn_pbe_configs(6, p17)
    create_line_plot(ws4, 'Fig17_E_CC', 1, 12,
                     cfg4, lab4,
                     log_y=True, y_label=y_lab_e,
                     subplot_label='(b) CC (p values)')

    # ---- Fig.17 psi/sigma aligned (NN vs PBE, 5物理值) ----
    psi_lab = ['(0.05V, z=1.93)', '(0.10V, z=3.87)', '(0.15V, z=5.80)',
               '(0.20V, z=7.74)', '(0.25V, z=9.66)']
    sig_lab = ['(0.05C/m², p=0.266)', '(0.10C/m², p=0.532)', '(0.15C/m², p=0.799)',
               '(0.20C/m², p=1.065)', '(0.25C/m², p=1.331)']

    ws5 = import_csv('Fig17_E_CP_psi_Data', 'fig17_E_CP_psi.csv')
    cfg5, lab5 = nn_pbe_configs_interleaved(5, psi_lab)
    create_line_plot(ws5, 'Fig17_E_CP_psi', 1, 10,
                     cfg5, lab5,
                     log_y=True, y_label=y_lab_e,
                     subplot_label='(c) CP (ψ-aligned)')

    ws6 = import_csv('Fig17_E_CC_sigma_Data', 'fig17_E_CC_sigma.csv')
    cfg6, lab6 = nn_pbe_configs_interleaved(5, sig_lab)
    create_line_plot(ws6, 'Fig17_E_CC_sigma', 1, 10,
                     cfg6, lab6,
                     log_y=True, y_label=y_lab_e,
                     subplot_label='(d) CC (σ-aligned)')

    # ========================= PART 3: MATLAB 拟合公式对比 =========================
    print('\n=== PART 3: MATLAB Fitting Formula Comparison ===')

    mz = ['z=1.93', 'z=3.87', 'z=5.80', 'z=7.74', 'z=9.67']
    ms = ['σ=0.05', 'σ=0.10', 'σ=0.15', 'σ=0.20', 'σ=0.25']

    ws7 = import_csv('MATLAB_CP_Energy_Data', 'MATLAB_CP_energy.csv')
    cfg7, lab7 = matlab_configs(mz)
    create_line_plot(ws7, 'MATLAB_CP_Energy', 1, 10,
                     cfg7, lab7,
                     log_y=True, y_label=y_lab_e,
                     subplot_label='(a) CP Energy')

    ws8 = import_csv('MATLAB_CP_Potential_Data', 'MATLAB_CP_potential.csv')
    cfg8, lab8 = matlab_configs(mz)
    create_line_plot(ws8, 'MATLAB_CP_Potential', 1, 10,
                     cfg8, lab8,
                     log_y=True, y_label='u',
                     subplot_label='(b) CP Potential')

    ws9 = import_csv('MATLAB_CC_Energy_Data', 'MATLAB_CC_energy.csv')
    cfg9, lab9 = matlab_configs(ms)
    create_line_plot(ws9, 'MATLAB_CC_Energy', 1, 10,
                     cfg9, lab9,
                     log_y=True, y_label=y_lab_e,
                     subplot_label='(c) CC Energy')

    ws10 = import_csv('MATLAB_CC_Potential_Data', 'MATLAB_CC_potential.csv')
    cfg10, lab10 = matlab_configs(ms)
    create_line_plot(ws10, 'MATLAB_CC_Potential', 1, 10,
                     cfg10, lab10,
                     log_y=True, y_label='u',
                     subplot_label='(d) CC Potential')

    # ========================= 保存 =========================
    print('\n' + '=' * 60)
    print('保存项目...')
    ok = op.save(OPJU_FILE)
    time.sleep(2)

    if ok:
        sz = os.path.getsize(OPJU_FILE) / 1024
        print(f'OK! 保存成功 ({sz:.1f} KB)')
    else:
        print('首次保存失败，尝试另存...')
        try:
            op.save(OPJU_FILE)
            sz = os.path.getsize(OPJU_FILE) / 1024
            print(f'重试后: {sz:.1f} KB')
        except Exception as e:
            print(f'保存出错: {e}')

    print('=' * 60)
    print('全部完成！请在 Origin 中打开查看图形效果。')
    print('=' * 60)
    op.set_show(True)


if __name__ == '__main__':
    main()
