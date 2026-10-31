import json
from pathlib import Path
from collections import defaultdict
import csv

# ======================
# User config (EDIT HERE)
# ======================

# 你的实验4目录
ROOT = Path("/root/Desktop/我的网盘/GCN_pruning/outputs/large_scale/exp4_joint_grid_owen")

# 优先读取聚合文件；如果没有，可以改成某个 seed 目录下的 results.json
INPUT_JSON = ROOT / "all_results.json"

# 是否同时打印 val_acc 矩阵
PRINT_VAL_MATRIX = True

# 是否保存 csv
SAVE_CSV = True

# 浮点 keep ratio 统一 round，避免 0.6000000001 这类问题
ROUND_DIGITS = 6

# 聚合方式：mean / max
AGG = "mean"


# ======================
# Helpers
# ======================

def _round_key(x):
    return round(float(x), ROUND_DIGITS)

def mean(xs):
    return sum(xs) / max(1, len(xs))

def aggregate(xs, mode="mean"):
    if not xs:
        return None
    if mode == "mean":
        return mean(xs)
    elif mode == "max":
        return max(xs)
    else:
        raise ValueError(f"Unsupported AGG mode: {mode}")

def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a list json.")
    return data

def build_metric_matrix(rows, metric_name="test_acc"):
    """
    返回：
      graph_keeps: sorted list
      model_keeps: sorted list
      matrix: 2D list, shape = [len(model_keeps)][len(graph_keeps)]
              行对应 model_keep_ratio
              列对应 graph_keep_ratio
    """
    bucket = defaultdict(list)

    graph_keeps = sorted({
        _round_key(r["graph_keep_ratio"])
        for r in rows if "graph_keep_ratio" in r
    }, reverse=True)

    model_keeps = sorted({
        _round_key(r["model_keep_ratio"])
        for r in rows if "model_keep_ratio" in r
    }, reverse=True)

    for r in rows:
        try:
            gk = _round_key(r["graph_keep_ratio"])
            mk = _round_key(r["model_keep_ratio"])
            val = float(r[metric_name])
            bucket[(mk, gk)].append(val)
        except Exception:
            continue

    matrix = []
    for mk in model_keeps:
        row_vals = []
        for gk in graph_keeps:
            vals = bucket.get((mk, gk), [])
            row_vals.append(aggregate(vals, AGG) if vals else None)
        matrix.append(row_vals)

    return graph_keeps, model_keeps, matrix

def format_cell(x, width=8, digits=4):
    if x is None:
        return " " * (width - 2) + "--"
    s = f"{x:.{digits}f}"
    return s.rjust(width)

def print_matrix(title, graph_keeps, model_keeps, matrix):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    header = "model\\graph".rjust(12)
    for gk in graph_keeps:
        header += f"{gk:>8.2f}"
    print(header)

    for mk, row in zip(model_keeps, matrix):
        line = f"{mk:>12.2f}"
        for x in row:
            line += format_cell(x, width=8, digits=4)
        print(line)

def save_matrix_csv(save_path: Path, graph_keeps, model_keeps, matrix):
    with save_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_keep \\ graph_keep"] + graph_keeps)
        for mk, row in zip(model_keeps, matrix):
            writer.writerow([mk] + row)
    print(f"[Saved] {save_path}")


# ======================
# Main
# ======================

print("[Config]")
print("  ROOT =", ROOT)
print("  INPUT_JSON =", INPUT_JSON)
print("  PRINT_VAL_MATRIX =", PRINT_VAL_MATRIX)
print("  SAVE_CSV =", SAVE_CSV)
print("  AGG =", AGG)

if not INPUT_JSON.exists():
    raise SystemExit(f"Input json not found: {INPUT_JSON}")

rows = load_rows(INPUT_JSON)
if not rows:
    raise SystemExit("No rows loaded.")

print(f"\n[Info] Loaded {len(rows)} rows from {INPUT_JSON}")

# --------
# Test accuracy matrix
# --------
gks_test, mks_test, mat_test = build_metric_matrix(rows, metric_name="test_acc")
print_matrix(
    title="Joint Matrix of Test Accuracy",
    graph_keeps=gks_test,
    model_keeps=mks_test,
    matrix=mat_test
)

if SAVE_CSV:
    save_matrix_csv(ROOT / "joint_matrix_test_acc.csv", gks_test, mks_test, mat_test)

# --------
# Validation accuracy matrix
# --------
if PRINT_VAL_MATRIX:
    has_val = any("val_acc" in r for r in rows)
    if has_val:
        gks_val, mks_val, mat_val = build_metric_matrix(rows, metric_name="val_acc")
        print_matrix(
            title="Joint Matrix of Validation Accuracy",
            graph_keeps=gks_val,
            model_keeps=mks_val,
            matrix=mat_val
        )
        if SAVE_CSV:
            save_matrix_csv(ROOT / "joint_matrix_val_acc.csv", gks_val, mks_val, mat_val)
    else:
        print("\n[Warn] No 'val_acc' field found in json. Skip val matrix.")

print("\nDone.")