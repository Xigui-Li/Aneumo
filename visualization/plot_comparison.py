"""
WSS 预测对比可视化脚本 - 多模型版

功能：
1. 对比所有模型 (DeepONet, FNO, UNet, MGN, Transolver) 的 WSS 预测
2. 从各模型的 results/ 目录读取 VTP 文件和 summary.txt
3. 生成：
   - 3D WSS 可视化对比图 (GT + 各模型)
   - 指标柱状图 (L2, MNAE)
   - 每个时间步的误差折线图
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pyvista as pv
from pathlib import Path
import warnings
import re
import json

warnings.filterwarnings('ignore')

# --- 服务器无头模式配置 ---
pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass

# ==========================================
# --- 配置区域 ---
# ==========================================

RENDER_VIEWS = ["front", "back"]
DENSITY = 1050.0
WSS_CLIM_KINEMATIC = [0, 0.03]
WSS_CLIM_PA = [c * DENSITY for c in WSS_CLIM_KINEMATIC]

RENDER_WINDOW_SIZE = [4000, 4000]
FOCAL_POINT = (0.0285766, 0.0364379, 0.0245677)
NORMAL = (-0.0176361, -0.817784, -0.575255)
DIST = 0.15
CAMERA_ZOOM = 1.2

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 0.5,
    'mathtext.fontset': 'stix',
})

# 路径
RESULTS_DIR = Path("results")
MODEL_DIRS = {
    "DeepONet":    RESULTS_DIR / "deeponet",
    "FNO":         RESULTS_DIR / "fno",
    "UNet":        RESULTS_DIR / "unet",
    "MGN":         RESULTS_DIR / "mgn",
}

# 模型颜色
MODEL_COLORS = {
    "DeepONet":   "#FF6B6B",
    "FNO":        "#4ECDC4",
    "UNet":       "#45B7D1",
    "MGN":        "#96CEB4",
}

OUTPUT_DIR = Path("figures_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# --- 工具函数 ---
# ==========================================

def parse_summary_file(file_path):
    """解析 summary.txt"""
    if not file_path.exists():
        return None
    content = file_path.read_text(encoding='utf-8')
    data = {}
    num = r"[\d\.eE\+\-]+"  # match both 0.000008 and 8.00e-06
    patterns = {
        'MSE': rf"Mean MSE:\s+({num})\s+±\s+({num})",
        'L2': rf"Mean L2 Error \(Full Mesh\):\s+({num})\s+±\s+({num})",
        'L2_sampled': rf"Mean L2 Error \(Sampled\):\s+({num})\s+±\s+({num})",
        'MAE': rf"Mean MAE:\s+({num})\s+±\s+({num})",
        'MNAE': rf"Mean MNAE:\s+({num})\s+±\s+({num})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            data[key] = (float(match.group(1)), float(match.group(2)))
        else:
            alt = re.search(r"Mean L2 Error:\s+([\d\.]+)\s+±\s+([\d\.]+)", content)
            if key == 'L2' and alt:
                data[key] = (float(alt.group(1)), float(alt.group(2)))
            else:
                data[key] = (0.0, 0.0)

    # Parse per-timestep errors
    per_t = []
    for m in re.finditer(r"t=(\d+):.*?L2_full=([\d\.]+)", content):
        per_t.append({'t': int(m.group(1)), 'l2': float(m.group(2))})
    data['per_timestep'] = per_t

    return data


def parse_results_json(file_path):
    """Parse results.json for per-timestep data."""
    if not file_path.exists():
        return None
    with open(file_path) as f:
        return json.load(f)


def get_camera_params(view_mode):
    fp = np.array(FOCAL_POINT)
    nm = np.array(NORMAL)
    if view_mode == "front":
        pos = fp - nm * DIST
        roll = 20
    elif view_mode == "back":
        pos = fp + nm * DIST
        roll = -20
    else:
        raise ValueError(f"Unknown view: {view_mode}")
    return pos.tolist(), fp.tolist(), (0, 1, 0), roll


def get_available_timesteps(model_dir):
    vtp_files = sorted(model_dir.glob("wss_t*.vtp"))
    return [int(f.stem.split("_t")[1]) for f in vtp_files]


def crop_whitespace(img, threshold=250, margin=5):
    gray = np.mean(img, axis=2)
    rows = np.any(gray < threshold, axis=1)
    cols = np.any(gray < threshold, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin)
    rmax = min(img.shape[0], rmax + margin)
    cmin = max(0, cmin - margin)
    cmax = min(img.shape[1], cmax + margin)
    return img[rmin:rmax, cmin:cmax]


def smooth_point_data(mesh, field_name, iterations=10):
    if field_name not in mesh.point_data:
        return None
    if isinstance(mesh, pv.PolyData):
        try:
            return mesh.smooth(n_iter=iterations, relaxation_factor=0.5).point_data[field_name]
        except Exception:
            pass
    try:
        return mesh.extract_surface().smooth(n_iter=iterations, relaxation_factor=0.5).point_data[field_name]
    except Exception:
        return np.array(mesh.point_data[field_name])


def render_mesh_to_image(mesh, scalar_name, clim, cmap, cam_pos, cam_roll, smooth_iter=0):
    smooth_field = f"{scalar_name}_smooth"
    if smooth_field in mesh.point_data:
        data = mesh.point_data[smooth_field]
    elif smooth_iter > 0:
        smoothed = smooth_point_data(mesh, scalar_name, iterations=smooth_iter)
        data = smoothed if smoothed is not None else mesh.point_data[scalar_name]
    else:
        data = mesh.point_data[scalar_name]

    data_pa = data.copy()
    if np.max(np.abs(data_pa)) < 1.0:
        data_pa = data_pa * DENSITY

    plotter = pv.Plotter(off_screen=True, window_size=RENDER_WINDOW_SIZE)
    plotter.set_background('white')
    plotter.add_mesh(mesh, scalars=data_pa, cmap=cmap, clim=clim, show_scalar_bar=False, smooth_shading=True)
    plotter.camera_position = cam_pos
    plotter.camera.zoom(CAMERA_ZOOM)
    plotter.camera.roll += cam_roll
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return crop_whitespace(img, margin=10)


# ==========================================
# --- 3D 可视化对比 ---
# ==========================================

def plot_3d_comparison(view_mode, timesteps, smooth_iter=10):
    """绘制所有模型的 3D WSS 对比图: GT + 5 models"""
    print(f"\n>>> Generating WSS COMPARISON ({view_mode.upper()})")
    cam_setup = get_camera_params(view_mode)[:3]
    _, _, _, cam_roll = get_camera_params(view_mode)

    # Find which models have VTP files
    available_models = {}
    for name, mdir in MODEL_DIRS.items():
        if mdir.exists():
            ts = set(get_available_timesteps(mdir))
            if ts:
                available_models[name] = (mdir, ts)

    if not available_models:
        print("No VTP files found for any model.")
        return

    # Common timesteps across all models
    common_ts = None
    for name, (mdir, ts) in available_models.items():
        common_ts = ts if common_ts is None else common_ts & ts
    if not common_ts:
        print("No common timesteps across models.")
        return
    common_ts = sorted(common_ts)

    # Select timesteps from those requested that are available
    selected = [t for t in timesteps if t in common_ts]
    if not selected:
        indices = np.linspace(0, len(common_ts) - 1, min(5, len(common_ts)), dtype=int)
        selected = [common_ts[i] for i in indices]

    n_times = len(selected)
    n_rows = 1 + len(available_models)  # GT + models
    model_names = list(available_models.keys())

    image_cache = {}
    for col, t in enumerate(selected):
        print(f"  Rendering t={t}...")
        image_cache[col] = {}

        # Use first available model's VTP for GT
        first_model = model_names[0]
        first_dir = available_models[first_model][0]
        vtp_path = first_dir / f"wss_t{t:04d}.vtp"
        if not vtp_path.exists():
            continue
        mesh_gt = pv.read(str(vtp_path))

        # GT row (no smooth)
        if "WSS_ground_truth" in mesh_gt.point_data:
            image_cache[col][0] = render_mesh_to_image(
                mesh_gt, "WSS_ground_truth", WSS_CLIM_PA, "turbo", cam_setup, cam_roll, smooth_iter=0)

        # Model rows
        for row_idx, name in enumerate(model_names):
            mdir = available_models[name][0]
            vtp_path = mdir / f"wss_t{t:04d}.vtp"
            if not vtp_path.exists():
                continue
            mesh = pv.read(str(vtp_path))
            if "WSS_predicted" in mesh.point_data:
                image_cache[col][row_idx + 1] = render_mesh_to_image(
                    mesh, "WSS_predicted", WSS_CLIM_PA, "turbo", cam_setup, cam_roll, smooth_iter=smooth_iter)

    fig, axes = plt.subplots(n_rows, n_times, figsize=(2.0 * n_times + 0.8, 2.0 * n_rows + 0.5))
    if n_times == 1:
        axes = axes.reshape(-1, 1)
    plt.subplots_adjust(left=0.10, right=0.88, top=0.94, bottom=0.02, wspace=0.02, hspace=0.02)

    for col, t in enumerate(selected):
        for row in range(n_rows):
            if row in image_cache.get(col, {}):
                axes[row, col].imshow(image_cache[col][row], interpolation='antialiased')
            axes[row, col].axis("off")
        axes[0, col].set_title(f"t = {t}", fontsize=12, pad=4)

    # Row labels
    row_labels = ["Ground Truth"] + model_names
    for row, label in enumerate(row_labels):
        axes[row, 0].text(-0.15, 0.5, label, transform=axes[row, 0].transAxes,
                          fontsize=9, fontweight='bold', va='center', ha='right', rotation=90)

    # Colorbar
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=WSS_CLIM_PA[0], vmax=WSS_CLIM_PA[1])
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("WSS (Pa)", fontsize=9)

    out_path = OUTPUT_DIR / f"comparison_wss_{view_mode}.pdf"
    plt.savefig(out_path, bbox_inches="tight", facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ==========================================
# --- 指标柱状图 ---
# ==========================================

def plot_metrics_bar():
    """绘制 L2, MSE, MAE, MNAE 四指标柱状图"""
    print("\n>>> Reading metrics from summary.txt ...")
    metrics = {}
    for name, mdir in MODEL_DIRS.items():
        summary = mdir / "summary.txt"
        data = parse_summary_file(summary)
        if data:
            metrics[name] = data
            print(f"  {name}: L2={data['L2'][0]:.4f}, MSE={data['MSE'][0]:.2e}, "
                  f"MAE={data['MAE'][0]:.2e}, MNAE={data['MNAE'][0]:.4f}")

    if not metrics:
        print("No metrics loaded.")
        return

    models = list(metrics.keys())
    x = np.arange(len(models))
    colors = [MODEL_COLORS.get(m, '#999999') for m in models]

    metric_specs = [
        ("L2", "Relative L2 Error", False),
        ("MSE", "MSE", True),
        ("MAE", "MAE", True),
        ("MNAE", "MNAE", False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    for i, (key, ylabel, use_sci) in enumerate(metric_specs):
        vals = [metrics[m][key][0] for m in models]
        stds = [metrics[m][key][1] for m in models]

        axes[i].grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        bars = axes[i].bar(x, vals, 0.6, yerr=stds, capsize=3,
                           color=colors, edgecolor='black', linewidth=0.5, zorder=3)

        max_y = 0
        for bar, val, std in zip(bars, vals, stds):
            label_y = bar.get_height() + std + bar.get_height() * 0.05
            fmt = f"{val:.2e}" if use_sci else f"{val:.4f}"
            axes[i].text(bar.get_x() + bar.get_width() / 2, label_y,
                         fmt, ha='center', va='bottom', fontsize=7, fontweight='bold')
            if label_y > max_y:
                max_y = label_y

        axes[i].set_ylim(top=max_y * 1.25)
        if use_sci:
            axes[i].ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))
        axes[i].set_ylabel(ylabel, fontsize=10)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, fontsize=8, rotation=15, ha='right')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'metrics_comparison.pdf'
    plt.savefig(out_path, bbox_inches="tight", facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ==========================================
# --- 每时间步误差折线图 ---
# ==========================================

def plot_per_timestep_l2():
    """绘制每个时间步的 L2 误差折线图"""
    print("\n>>> Per-timestep L2 error curves...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for name, mdir in MODEL_DIRS.items():
        # Try results.json first
        json_path = mdir / "results.json"
        summary_path = mdir / "summary.txt"
        ts, l2s = [], []

        if json_path.exists():
            data = parse_results_json(json_path)
            if data and 'per_timestep' in data:
                for pt in data['per_timestep']:
                    ts.append(pt.get('t_idx', pt.get('t', 0)))
                    l2s.append(pt.get('l2_full', pt.get('l2', 0)))
        elif summary_path.exists():
            data = parse_summary_file(summary_path)
            if data and data['per_timestep']:
                for pt in data['per_timestep']:
                    ts.append(pt['t'])
                    l2s.append(pt['l2'])

        if ts:
            color = MODEL_COLORS.get(name, '#999999')
            ax.plot(ts, l2s, 'o-', label=name, color=color, markersize=3, linewidth=1.5)

    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Relative L2 Error", fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / 'per_timestep_l2.pdf'
    plt.savefig(out_path, bbox_inches="tight", facecolor='white')
    print(f"Saved: {out_path}")
    plt.close()


# ==========================================
# --- 汇总表 ---
# ==========================================

def print_comparison_table():
    """Print a unified comparison table for all models."""
    print("\n" + "=" * 80)
    print("Model Comparison Table (Single-Geometry case_201)")
    print("=" * 80)
    header = f"{'Model':<12} {'L2':>14} {'MSE':>16} {'MAE':>16} {'MNAE':>12}"
    print(header)
    print("-" * 80)

    for name, mdir in MODEL_DIRS.items():
        summary = mdir / "summary.txt"
        data = parse_summary_file(summary)
        if data:
            l2_str = f"{data['L2'][0]:.4f}±{data['L2'][1]:.4f}"
            mse_str = f"{data['MSE'][0]:.2e}±{data['MSE'][1]:.2e}"
            mae_str = f"{data['MAE'][0]:.2e}±{data['MAE'][1]:.2e}"
            mnae_str = f"{data['MNAE'][0]:.4f}±{data['MNAE'][1]:.4f}"
            print(f"{name:<12} {l2_str:>14} {mse_str:>16} {mae_str:>16} {mnae_str:>12}")
        else:
            print(f"{name:<12} {'N/A':>14} {'N/A':>16} {'N/A':>16} {'N/A':>12}")

    print("=" * 80)


def main():
    print("=" * 60)
    print("WSS Multi-Model Comparison")
    print("=" * 60)

    # 0. Print comparison table
    print_comparison_table()

    # 1. 3D visualization
    common_ts = None
    for name, mdir in MODEL_DIRS.items():
        if mdir.exists():
            ts = set(get_available_timesteps(mdir))
            if ts:
                common_ts = ts if common_ts is None else common_ts & ts

    if common_ts:
        common_ts = sorted(common_ts)
        indices = np.linspace(0, len(common_ts) - 1, min(5, len(common_ts)), dtype=int)
        selected = [common_ts[i] for i in indices]
        print(f"\nSelected timesteps for 3D: {selected}")

        for view in RENDER_VIEWS:
            plot_3d_comparison(view, selected, smooth_iter=10)
    else:
        print("\nWarning: No common VTP timesteps found for 3D viz.")

    # 2. Metrics bar chart
    plot_metrics_bar()

    # 3. Per-timestep L2 curves
    plot_per_timestep_l2()

    print(f"\nAll figures saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()
