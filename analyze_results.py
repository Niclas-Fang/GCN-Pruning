import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

import os
from matplotlib import font_manager

FONT_SIZE = 10.5   # 五号字；如果图太挤，就统一改成 9

font_path_cn = os.path.expanduser("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")

# 注册字体（若用 addfont，每次运行都要执行）
font_manager.fontManager.addfont(font_path_cn)

cn_prop = font_manager.FontProperties(fname=font_path_cn, size=FONT_SIZE)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams['axes.unicode_minus'] = False        # 正常显示负号
# ======================
# User config (EDIT HERE)
# ======================

# 你的实验根目录（可改）
ROOT = Path("/root/Desktop/我的网盘/GCN_pruning/outputs/large_scale/exp1_baseline_methods")

# 输出图保存目录（可改）
OUT_DIR = ROOT / "_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 输入方式 A：手动列出要读取的目录（最直观，推荐） ---
# 你可以只保留你关心的目录，比如只读 seed42_* 或只读某几个方法组合
INPUT_DIRS = [
    ROOT / "seed42_feature_l2_activation_l2",
    ROOT / "seed42_grad_grad",
    ROOT / "seed42_random_random",
    ROOT / "seed42_sensitivity_sensitivity",
    ROOT / "seed42_taylor_taylor",

    ROOT / "seed123_feature_l2_activation_l2",
    ROOT / "seed123_grad_grad",
    ROOT / "seed123_random_random",
    ROOT / "seed123_sensitivity_sensitivity",
    ROOT / "seed123_taylor_taylor",

    ROOT / "seed456_feature_l2_activation_l2",
    ROOT / "seed456_grad_grad",
    ROOT / "seed456_random_random",
    ROOT / "seed456_sensitivity_sensitivity",
    ROOT / "seed456_taylor_taylor",

    # ROOT / "seed42_owen_source_node_ch32",
    # ROOT / "seed123_owen_source_node_ch32",
    # ROOT / "seed456_owen_source_node_ch32",

    ROOT / "seed42_owen_community_ch64",
    ROOT / "seed123_owen_community_ch64",
    ROOT / "seed456_owen_community_ch64",

    # ROOT / "seed42_owen_target_node_ch32",
    # ROOT / "seed123_owen_target_node_ch32",
    # ROOT / "seed456_owen_target_node_ch32",

    # ROOT / "seed42_owen_motif_clustering_ch64",
    # ROOT / "seed42_owen_motif_bridge_ch64",
    # ROOT / "seed42_owen_motif_triangle_ch64",
    # ROOT / "seed42_owen_target_node_ch32"
]

# --- 输入方式 B：手动写 glob 模式（想自动覆盖更多目录时用） ---
# 如果你启用 patterns，请把 USE_PATTERNS=True，并清空/忽略 INPUT_DIRS
USE_PATTERNS = False
INPUT_PATTERNS = [
    "seed42_*",
    "seed123_*",
    "seed456_*",
    # 或者更宽松： "seed*_*_*"
]

# 每个目录里默认读取这个文件名
RESULT_FILENAME = "results.json"

# 用于把 float sparsity 聚合到“同一个点”
# 不想做任何处理就设为 None（但可能会聚合失败）
ROUND_DIGITS = 6

# 排行榜：指定要比较的 keep_ratio 点（都为 None 表示自动取全局最小 keep_ratio）
TARGET_GRAPH_KEEP = None  # e.g. 0.5
TARGET_MODEL_KEEP = None  # e.g. 0.5


# ======================
# Helpers
# ======================

def _round_key(x: float):
    if ROUND_DIGITS is None:
        return float(x)
    return round(float(x), ROUND_DIGITS)

def mean(values):
    return sum(values) / max(1, len(values))

def method_combo(row):
    # 不清洗方法名：原样拼接
    return f"{row.get('graph_scoring_method')} + {row.get('model_scoring_method')}"

def resolve_inputs():
    """
    返回要读取的 results.json 文件列表（由用户手动配置决定）
    """
    result_files = []

    if USE_PATTERNS:
        for pat in INPUT_PATTERNS:
            for d in sorted(ROOT.glob(pat)):
                if d.is_dir():
                    f = d / RESULT_FILENAME
                    if f.exists():
                        result_files.append(f)
    else:
        for d in INPUT_DIRS:
            f = Path(d) / RESULT_FILENAME
            if f.exists():
                result_files.append(f)

    return sorted(set(result_files))

def load_records(files):
    all_rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for row in data:
                    row["_src_file"] = str(f)
                    all_rows.append(row)
            else:
                print(f"[Warn] {f} is not a list; skipped.")
        except Exception as e:
            print(f"[Warn] Failed to read {f}: {e}")
    return all_rows

def plot_curve_mean(points_by_method, xlabel, ylabel, title, save_path):
    plt.figure()
    for m, pts in sorted(points_by_method.items(), key=lambda x: x[0]):
        pts_sorted = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        plt.plot(xs, ys, marker="o", label=m)

    plt.xlabel(xlabel,fontproperties=cn_prop)
    plt.ylabel(ylabel,fontproperties=cn_prop)
    plt.title(title,fontproperties=cn_prop)
    leg = plt.legend(prop=cn_prop)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] {save_path}")

def plot_leaderboard(items, title, xlabel, save_path, top_k=20):
    items_sorted = sorted(items, key=lambda x: x[1], reverse=True)[:top_k]
    methods = [x[0] for x in items_sorted]
    scores = [x[1] for x in items_sorted]

    plt.figure(figsize=(10, max(4, 0.35 * len(methods))))
    y_pos = list(range(len(methods)))
    plt.barh(y_pos, scores)
    plt.yticks(y_pos, methods, fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Saved] {save_path}")


# ======================
# Main
# ======================

print("\n[Input config]")
print("  ROOT =", ROOT)
print("  USE_PATTERNS =", USE_PATTERNS)
if USE_PATTERNS:
    print("  INPUT_PATTERNS =")
    for p in INPUT_PATTERNS:
        print("   -", p)
else:
    print("  INPUT_DIRS =")
    for d in INPUT_DIRS:
        print("   -", d)

files = resolve_inputs()
if not files:
    raise SystemExit("No results.json found under the configured inputs. Please edit INPUT_DIRS/INPUT_PATTERNS.")

print(f"\n[Info] Will read {len(files)} files:")
for f in files:
    print("  -", f)

rows = load_records(files)
if not rows:
    raise SystemExit("No records loaded from the selected files.")

# --------
# (A) Test Acc vs Graph Sparsity (mean over same (method, graph_sparsity))
# --------
bucket_A = defaultdict(list)  # (method, x_key) -> [test_acc]
xval_A = defaultdict(list)    # (method, x_key) -> [x]

for r in rows:
    try:
        m = method_combo(r)
        x = float(r["graph_sparsity"])
        y = float(r["test_acc"])
        kx = _round_key(x)
        bucket_A[(m, kx)].append(y)
        xval_A[(m, kx)].append(x)
    except Exception:
        continue

points_A = defaultdict(list)  # method -> [(x_mean, y_mean)]
for (m, kx), ys in bucket_A.items():
    points_A[m].append((mean(xval_A[(m, kx)]), mean(ys)))

plot_curve_mean(
    points_A,
    xlabel="稀疏率",
    ylabel="测试准确率",
    title="不同稀疏率下的测试准确率",
    save_path=OUT_DIR / "A_test_acc_vs_graph_sparsity.png",
)


# --------
# (B) Test Acc vs Model Sparsity (mean over same (method, model_sparsity))
# --------
bucket_B = defaultdict(list)
xval_B = defaultdict(list)

for r in rows:
    try:
        m = method_combo(r)
        x = float(r["model_sparsity"])
        y = float(r["test_acc"])
        kx = _round_key(x)
        bucket_B[(m, kx)].append(y)
        xval_B[(m, kx)].append(x)
    except Exception:
        continue

points_B = defaultdict(list)
for (m, kx), ys in bucket_B.items():
    points_B[m].append((mean(xval_B[(m, kx)]), mean(ys)))

plot_curve_mean(
    points_B,
    xlabel="Model Sparsity",
    ylabel="Test Accuracy",
    title="(B) Test Accuracy vs Model Sparsity (mean over same method+sparsity)",
    save_path=OUT_DIR / "B_test_acc_vs_model_sparsity.png"
)

# --------
# (C) Leaderboard: Mean Test Acc at target keep-ratio point
# Default: choose global min keep ratios in loaded data (not all ROOT)
# --------
graph_keep_values = sorted(set(float(r["graph_keep_ratio"]) for r in rows if "graph_keep_ratio" in r))
model_keep_values = sorted(set(float(r["model_keep_ratio"]) for r in rows if "model_keep_ratio" in r))

g_target = TARGET_GRAPH_KEEP if TARGET_GRAPH_KEEP is not None else (min(graph_keep_values) if graph_keep_values else None)
m_target = TARGET_MODEL_KEEP if TARGET_MODEL_KEEP is not None else (min(model_keep_values) if model_keep_values else None)

if g_target is None or m_target is None:
    print("[Warn] Cannot determine target keep ratios for leaderboard; skipping.")
else:
    bucket_L = defaultdict(list)  # method -> [test_acc]
    for r in rows:
        try:
            if float(r["graph_keep_ratio"]) == float(g_target) and float(r["model_keep_ratio"]) == float(m_target):
                bucket_L[method_combo(r)].append(float(r["test_acc"]))
        except Exception:
            continue

    leaderboard = [(m, mean(accs)) for m, accs in bucket_L.items() if accs]
    title = f"(C) Leaderboard: Mean Test Acc @ (graph_keep={g_target}, model_keep={m_target})"
    plot_leaderboard(
        leaderboard,
        title=title,
        xlabel="Mean Test Accuracy",
        save_path=OUT_DIR / "C_leaderboard_test_acc.png",
        top_k=20
    )

print("\nDone. Plots saved in:", OUT_DIR)