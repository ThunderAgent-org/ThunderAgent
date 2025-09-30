# plot_profile_percentages.py
# -*- coding: utf-8 -*-
# plot_profile_percentages.py
# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")  # 无需 Qt/Wayland
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import json, math
from pathlib import Path

# ===== Config =====
JSON_PATH = "/home/ergt/SWE-agent/trajectories/ergt/default__deepseek-chat__t-0.00__p-1.00__c-0.00___swe_bench_lite_test/profiling_summary.json"
OUTDIR = Path("figures"); OUTDIR.mkdir(exist_ok=True)

FIG_DPI = 180
LABEL_MIN_PCT = 0.02   # 贴文字阈值（相对父级）
ARROW_MIN_PCT = 0.05   # 画“See breakdown”箭头阈值
EPS = 1e-9             # 显示层面的极小夹紧，杜绝 0/100 文本

# 注解与文字“柔和”风格
ANNOT_COLOR = "#555555"
ANNOT_ALPHA = 0.9
ARROW_COLOR = "#6b6b6b"
ARROW_ALPHA = 0.85
ARROW_STYLE = dict(
    arrowstyle="->",
    lw=0.9,
    color=ARROW_COLOR,
    alpha=ARROW_ALPHA,
    shrinkA=0, shrinkB=0,
    connectionstyle="arc3,rad=0.12",
)

TOP_ORDER = ["agent_run", "other", "env_preparation", "env_shutdown"]

# ===== Helpers =====
def save_fig(fig, path):
    # 统一避免裁剪；略留白保证边缘文本完整
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

def nice_name(raw: str) -> str:
    if "." in raw:
        raw = raw.split(".")[-1]
    raw = raw.replace("_", " ").strip()
    mapping = {
        "agent run": "Agent Run",
        "env preparation": "Environment Prep",
        "env shutdown": "Environment Shutdown",
        "lm reasoning": "LM Reasoning",
        "tool parse validate": "Tool Parse/Validate",
        "env execution": "Env Execution",
        "retry handling": "Retry Handling",
        "observation packaging": "Observation Packaging",
        "observation get state": "Observation Get State",
        "observation postprocess": "Observation Postprocess",
        "finalization": "Finalization",
        "agent setup": "Agent Setup",
        "other time": "Other Time",
        "other": "Other",
    }
    return mapping.get(raw.lower(), raw.title())

def get_pct_and_breakdown(node):
    if isinstance(node, dict):
        pct = float(node.get("percentage", 0.0))
        br = node.get("breakdown")
        return pct, (br if isinstance(br, dict) else None)
    return float(node), None

def to_shades(base_rgba, n):
    r, g, b = to_rgb(base_rgba)
    if n <= 1: return [(r, g, b, 1.0)]
    shades = []
    for i in range(n):
        f = 0.85 - 0.50 * (i / (n - 1))  # 浅->深
        rr = 1 - f + f * r
        gg = 1 - f + f * g
        bb = 1 - f + f * b
        shades.append((rr, gg, bb, 1.0))
    return shades

def fmt_pct(p):
    # 文本层面夹紧，避免显示恰好 0%/100%
    p = max(min(p, 1.0 - EPS), EPS)
    val = p * 100.0
    if val >= 1:
        s = f"{val:.4f}%"
    elif val >= 0.01:
        s = f"{val:.6f}%"
    else:
        s = f"{val:.8f}%"
    return s.rstrip("0").rstrip(".")

# ===== Load =====
data = json.loads(Path(JSON_PATH).read_text(encoding="utf-8"))
pcts = data["aggregate"]["percentages"]

# 顶层顺序
top_items = []
for k in TOP_ORDER:
    if k in pcts: top_items.append((k, pcts[k]))
for k in pcts:
    if k not in dict(top_items): top_items.append((k, pcts[k]))

top_keys = [k for k, _ in top_items]
top_vals = [get_pct_and_breakdown(v)[0] for _, v in top_items]

tab10 = plt.get_cmap("tab10")
top_cols = [tab10(i % 10) for i in range(len(top_items))]

# ===== 1) Bar: Overall =====
fig = plt.figure(figsize=(9.2, 5.0), dpi=FIG_DPI)
ax = fig.add_subplot(111)
bars = ax.bar([nice_name(k) for k in top_keys], top_vals, color=top_cols)
ax.set_title("Overall Time Distribution", fontsize=18, fontweight="bold")
ax.set_ylabel("Share of Total Time", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.margins(y=0.12)  # 给顶部文字留空间，避免被裁
for b, v in zip(bars, top_vals):
    ax.text(
        b.get_x()+b.get_width()/2, b.get_height()+0.003,
        fmt_pct(v), ha="center", va="bottom",
        fontsize=10.5, color=ANNOT_COLOR, alpha=ANNOT_ALPHA
    )
fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.18)
save_fig(fig, OUTDIR / "overall_bar.png")

# ===== 2) Donut: Overall (no slice dropped) =====
labels_disp = [nice_name(k) for k in top_keys]

fig = plt.figure(figsize=(7.6, 7.6), dpi=FIG_DPI)
ax = fig.add_subplot(111)
ax.set_aspect("equal")
wedges, texts, _ = ax.pie(
    top_vals,
    labels=[lab if v >= LABEL_MIN_PCT else "" for lab, v in zip(labels_disp, top_vals)],
    autopct=lambda p: "",
    startangle=90,
    pctdistance=0.74,  # 稍里收，避免挤边
    labeldistance=0.82,
    colors=top_cols,
    wedgeprops=dict(width=0.38, edgecolor="white")
)
# 数值标签（仅阈值以上）
for w, lab, v in zip(wedges, labels_disp, top_vals):
    if v >= LABEL_MIN_PCT:
        ang = (w.theta2 + w.theta1) / 2.0
        x, y = math.cos(math.radians(ang))*0.58, math.sin(math.radians(ang))*0.58
        ax.text(x, y, fmt_pct(v), ha="center", va="center",
                fontsize=10, color=ANNOT_COLOR, alpha=ANNOT_ALPHA)
# 小切片图例
small = [(lab, v, top_cols[i]) for i,(lab,v) in enumerate(zip(labels_disp, top_vals)) if v < LABEL_MIN_PCT]
if small:
    handles = [plt.Line2D([0],[0], marker="o", linestyle="",
                          markerfacecolor=c, markeredgecolor="none") for _,_,c in small]
    ax.legend(handles, [f"{lab} ({fmt_pct(v)})" for lab,v,_ in small],
              title="Smaller Slices", loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False)
ax.set_title("Overall — Top Stages", fontsize=18, fontweight="bold")
fig.subplots_adjust(left=0.06, right=0.86, top=0.88, bottom=0.08)
save_fig(fig, OUTDIR / "overall_pie.png")

# ===== 3) Recursive pies =====
def plot_breakdown(stage_key, node, base_color):
    pct, br = get_pct_and_breakdown(node)
    if not br: return

    child_keys = list(br.keys())
    child_vals_abs = [get_pct_and_breakdown(br[k])[0] for k in child_keys]
    total = sum(child_vals_abs)
    if total <= 0: return
    shares = [v / total for v in child_vals_abs]

    disp_names = [nice_name(f"{stage_key}.{k}") for k in child_keys]
    sub_cols = to_shades(base_color, len(child_keys))

    fig = plt.figure(figsize=(8.6, 7.6), dpi=FIG_DPI)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    wedges, texts, _ = ax.pie(
        shares,
        labels=[n if s >= LABEL_MIN_PCT else "" for n, s in zip(disp_names, shares)],
        autopct=lambda p: "",
        startangle=90,
        pctdistance=0.74,
        labeldistance=0.82,
        colors=sub_cols,
        wedgeprops=dict(width=0.38, edgecolor="white")
    )
    # 数值标签（阈值以上）
    for w, name, s in zip(wedges, disp_names, shares):
        if s >= LABEL_MIN_PCT:
            ang = (w.theta2 + w.theta1) / 2.0
            x, y = math.cos(math.radians(ang))*0.58, math.sin(math.radians(ang))*0.58
            ax.text(x, y, fmt_pct(s), ha="center", va="center",
                    fontsize=9.5, color=ANNOT_COLOR, alpha=ANNOT_ALPHA)

    # 小切片图例
    small = [(n, s, sub_cols[i]) for i,(n,s) in enumerate(zip(disp_names, shares)) if s < LABEL_MIN_PCT]
    if small:
        handles = [plt.Line2D([0],[0], marker="o", linestyle="",
                              markerfacecolor=c, markeredgecolor="none") for _,_,c in small]
        ax.legend(handles, [f"{n} ({fmt_pct(s)})" for n,s,_ in small],
                  title="Smaller Slices", loc="center left",
                  bbox_to_anchor=(1.02, 0.5), frameon=False)

    # 更“柔和”的箭头注解（仅大块且可下钻）
    for i, (n, s) in enumerate(zip(disp_names, shares)):
        _, child_br = get_pct_and_breakdown(br[child_keys[i]])
        if child_br and s >= ARROW_MIN_PCT:
            ang = (wedges[i].theta2 + wedges[i].theta1) / 2.0
            x, y = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            ax.annotate(
                f"See breakdown: {n}",
                xy=(0.92*x, 0.92*y), xytext=(1.18*x, 1.18*y),
                arrowprops=ARROW_STYLE,
                ha="left" if x >= 0 else "right", va="center",
                fontsize=9.5, color=ANNOT_COLOR, alpha=ANNOT_ALPHA
            )

    ax.set_title(f"{nice_name(stage_key)} — Breakdown", fontsize=18, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.86, top=0.88, bottom=0.08)
    save_fig(fig, OUTDIR / f"{stage_key.replace('.', '_')}_pie.png")

    # 递归
    for i, key in enumerate(child_keys):
        plot_breakdown(f"{stage_key}.{key}", br[key], sub_cols[i])

# 触发
for i, (k, node) in enumerate(top_items):
    plot_breakdown(k, node, top_cols[i % 10])

print(f"All figures saved to: {OUTDIR.resolve()}")
