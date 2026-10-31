import json
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# ======================
# User config (EDIT HERE)
# ======================

# 你的实验根目录（可改）
ROOT = Path("/root/Desktop/我的网盘/GCN_pruning/outputs/large_scale/exp1_baseline_methods")

# 输出图保存目录（可改）
OUT_DIR = ROOT / "_plots_owen"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 输入方式 A：手动列出要读取的目录（推荐） ---
# 这里只保留 Owen 相关目录
INPUT_DIRS = [
    ROOT / "seed42_owen_target_node_ch32",
    ROOT / "seed123_owen_target_node_ch32",
    ROOT / "seed456_owen_target_node_ch32",

    ROOT / "seed42_owen_source_node_ch32",
    ROOT / "seed123_owen_source_node_ch32",
    ROOT / "seed456_owen_source_node_ch32",

    ROOT / "seed42_owen_community_ch64",
    ROOT / "seed123_owen_community_ch64",
    ROOT / "seed456_owen_community_ch64",

    # ROOT / "seed42_owen_motif_triangle_ch64",
    # ROOT / "seed123_owen_motif_triangle_ch64",
    # ROOT / "seed456_owen_motif_triangle_ch64",

    # ROOT / "seed42_owen_motif_bridge_ch64",
    # ROOT / "seed123_owen_motif_bridge_ch64",
    # ROOT / "seed456_owen_motif_bridge_ch64",

    # ROOT / "seed42_owen_motif_clustering_ch64",
    # ROOT / "seed123_owen_motif_clustering_ch64",
    # ROOT / "seed456_owen_motif_clustering_ch64",
]

# --- 输入方式 B：手动写 glob 模式 ---
USE_PATTERNS = False
INPUT_PATTERNS = [
    "seed42_owen_*",
    "seed123_owen_*",
    "seed456_owen_*",
]

# 每个目录里默认读取这个文件名
RESULT_FILENAME = "results.json"

# 用于把 float sparsity 聚合到“同一个点”
ROUND_DIGITS = 6

# 排行榜：指定要比较的 keep_ratio 点（都为 None 表示自动取全局最小 keep_ratio）
TARGET_GRAPH_KEEP = None   # e.g. 0.5
TARGET_MODEL_KEEP = None   # e.g. 0.5


# ======================
# Plot style (extracted from original code)
# ======================
# 说明：
# 你的原代码没有显式修改 rcParams，
# 所以这里保留默认 matplotlib 风格，
# 只延续原脚本中的“作图习惯”：
# - marker="o"
# - grid alpha=0.3
# - legend fontsize=8
# - tight_layout
# - savefig dpi=200


# ======================
# Helpers
# ======================

def _round_key(x: float):
    if ROUND_DIGITS is None:
        return float(x)
    return round(float(x), ROUND_DIGITS)

def mean(values):
    return sum(values) / max(1, len(values))

def parse_owen_group_from_dirname(dirname: str):
    """
    从目录名中提取 Owen 分组方式：
    例如：
      seed42_owen_target_node_ch32      -> target_node
      seed42_owen_motif_triangle_ch64   -> motif_triangle
      seed42_owen_source_node_ch32      -> source_node
      seed42_owen_community_ch64        -> community
    """
    # 去掉 seed 前缀
    # pattern: seed\d+_owen_(.*?)(_ch\d+)?$
    m = re.match(r"seed\d+_owen_(.*?)(?:_ch\d+)?$", dirname)
    if m:
        return m.group(1)
    return dirname

def method_group(row):
    """
    这里不再使用 graph_scoring_method + model_scoring_method，
    而是改为 Owen 的“分组方式”。
    分组方式优先从源目录名提取。
    """
    src_file = row.get("_src_file", "")
    src_dirname = Path(src_file).parent.name
    return parse_owen_group_from_dirname(src_dirname)

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
                    # 只保留 Owen 记录
                    if row.get("graph_scoring_method") == "owen" and row.get("model_scoring_method") == "owen":
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

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
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
    raise SystemExit("No Owen records loaded from the selected files.")

# 打印识别到的 Owen 分组方式
groups_found = sorted(set(method_group(r) for r in rows))
print("\n[Info] Owen grouping methods found:")
for g in groups_found:
    print("  -", g)

# --------
# (A) Test Acc vs Graph Sparsity
# 按 “Owen 分组方式 + graph_sparsity” 聚合
# --------
bucket_A = defaultdict(list)  # (group, x_key) -> [test_acc]
xval_A = defaultdict(list)    # (group, x_key) -> [x]

for r in rows:
    try:
        g = method_group(r)
        x = float(r["graph_sparsity"])
        y = float(r["test_acc"])
        kx = _round_key(x)
        bucket_A[(g, kx)].append(y)
        xval_A[(g, kx)].append(x)
    except Exception:
        continue

points_A = defaultdict(list)  # group -> [(x_mean, y_mean)]
for (g, kx), ys in bucket_A.items():
    points_A[g].append((mean(xval_A[(g, kx)]), mean(ys)))

plot_curve_mean(
    points_A,
    xlabel="Graph Sparsity",
    ylabel="Test Accuracy",
    title="(A) Owen: Test Accuracy vs Graph Sparsity",
    save_path=OUT_DIR / "A_owen_test_acc_vs_graph_sparsity.png"
)

# --------
# (B) Test Acc vs Model Sparsity
# 按 “Owen 分组方式 + model_sparsity” 聚合
# --------
bucket_B = defaultdict(list)
xval_B = defaultdict(list)

for r in rows:
    try:
        g = method_group(r)
        x = float(r["model_sparsity"])
        y = float(r["test_acc"])
        kx = _round_key(x)
        bucket_B[(g, kx)].append(y)
        xval_B[(g, kx)].append(x)
    except Exception:
        continue

points_B = defaultdict(list)
for (g, kx), ys in bucket_B.items():
    points_B[g].append((mean(xval_B[(g, kx)]), mean(ys)))

plot_curve_mean(
    points_B,
    xlabel="Model Sparsity",
    ylabel="Test Accuracy",
    title="(B) Owen: Test Accuracy vs Model Sparsity",
    save_path=OUT_DIR / "B_owen_test_acc_vs_model_sparsity.png"
)

# --------
# (C) Leaderboard: Mean Test Acc at target keep-ratio point
# Default: choose global min keep ratios in loaded data
# --------
graph_keep_values = sorted(set(float(r["graph_keep_ratio"]) for r in rows if "graph_keep_ratio" in r))
model_keep_values = sorted(set(float(r["model_keep_ratio"]) for r in rows if "model_keep_ratio" in r))

g_target = TARGET_GRAPH_KEEP if TARGET_GRAPH_KEEP is not None else (min(graph_keep_values) if graph_keep_values else None)
m_target = TARGET_MODEL_KEEP if TARGET_MODEL_KEEP is not None else (min(model_keep_values) if model_keep_values else None)

if g_target is None or m_target is None:
    print("[Warn] Cannot determine target keep ratios for leaderboard; skipping.")
else:
    bucket_L = defaultdict(list)  # group -> [test_acc]
    for r in rows:
        try:
            if float(r["graph_keep_ratio"]) == float(g_target) and float(r["model_keep_ratio"]) == float(m_target):
                bucket_L[method_group(r)].append(float(r["test_acc"]))
        except Exception:
            continue

    leaderboard = [(g, mean(accs)) for g, accs in bucket_L.items() if accs]
    title = f"(C) Owen Leaderboard: Mean Test Acc @ (graph_keep={g_target}, model_keep={m_target})"
    plot_leaderboard(
        leaderboard,
        title=title,
        xlabel="Mean Test Accuracy",
        save_path=OUT_DIR / "C_owen_leaderboard_test_acc.png",
        top_k=20
    )

print("\nDone. Owen plots saved in:", OUT_DIR)