"""
generate_figures.py
Generates all four figures used in the paper:
  fig1_score_comparison.png   - Score comparison across all graders
  fig2_kappa_spearman.png     - Kappa and Spearman bar charts
  fig3_score_distribution.png - Score distribution histograms
  fig4_kappa_heatmap.png      - Pairwise Kappa agreement matrix

Requirements: matplotlib, numpy, scikit-learn, scipy
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

sm_claude   = [2, 2, 0, 3, 0, 3, 2, 2]
sm_llama    = [2, 2, 0, 3, 1, 3, 2, 2]
sm_deepseek = [2, 2, 0, 3, 0, 3, 0, 0]
sm_gpt      = [3, 2, 0, 0, 1, 1, 0, 0]
sm_grader2  = [2, 2, 0, 2, 2, 3, 0, 1]
sm_grader1  = [3, 3, 0, 2, 2, 3, 0, 1]

ls_claude   = [2,3,3,1,1,1,2,2,2,1,1,2,3,2,2,1,3,0,1,3,0,0,3,3]
ls_llama    = [2,3,3,1,2,1,3,2,3,1,0,2,3,2,2,2,3,1,2,3,2,0,3,3]
ls_deepseek = [2,3,3,0,1,1,2,1,2,0,1,2,3,2,2,1,3,1,1,3,0,0,2,3]
ls_gpt      = [2,3,3,0,1,1,0,0,1,0,1,3,3,3,2,2,3,1,2,3,1,0,2,3]
ls_grader2  = [2,3,3,1,2,1,3,2,3,1,0,2,3,2,2,1,3,0,1,3,1,0,3,3]
ls_grader1  = [2,3,3,1,2,1,3,1,3,1,1,2,3,3,2,2,3,2,2,3,0,0,3,3]

all_claude   = sm_claude   + ls_claude
all_llama    = sm_llama    + ls_llama
all_deepseek = sm_deepseek + ls_deepseek
all_gpt      = sm_gpt      + ls_gpt
all_grader2  = sm_grader2  + ls_grader2
all_grader1  = sm_grader1  + ls_grader1


def get_kappa(a, b):
    try:
        return round(cohen_kappa_score(a, b), 3)
    except Exception:
        return 0.0

def get_spear(a, b):
    r, _ = spearmanr(a, b)
    return round(r, 3)


remove = {2, 29}
all_labels = [f"SM-Q{i+1}" for i in range(8)] + [f"LS-Q{i+1}" for i in range(24)]
labels = [l for i, l in enumerate(all_labels) if i not in remove]

def filt(scores):
    return [v for i, v in enumerate(scores) if i not in remove]

c_f  = filt(all_claude);   l_f  = filt(all_llama)
ds_f = filt(all_deepseek); g_f  = filt(all_gpt)
sh_f = filt(all_grader1);  sb_f = filt(all_grader2)
n    = len(labels)

fig, ax = plt.subplots(figsize=(22, 6))
x = np.arange(n); w = 0.12

ax.bar(x - 2.5*w, c_f,  w, label="Claude Haiku 4.5", color="#FF9800", alpha=0.9)
ax.bar(x - 1.5*w, l_f,  w, label="Llama 3.1 70B",    color="#4CAF50", alpha=0.9)
ax.bar(x - 0.5*w, ds_f, w, label="DeepSeek V3.2",    color="#9C27B0", alpha=0.9)
ax.bar(x + 0.5*w, g_f,  w, label="GPT-OSS 120B",     color="#2196F3", alpha=0.9)
ax.bar(x + 1.5*w, sh_f, w, label="Human (Grader 1)", color="#F44336", alpha=0.9)
ax.bar(x + 2.5*w, sb_f, w, label="Human (Grader 2)", color="#795548", alpha=0.9)

sm_count = sum(1 for i in range(8) if i not in remove)
ax.axvline(x=sm_count - 0.5, color="black", linestyle="--", alpha=0.5, linewidth=1.5)
ax.text(sm_count / 2 - 0.5, 3.6, "Story Mode", ha="center", fontsize=9, style="italic")
ax.text(sm_count + (n - sm_count) / 2 - 0.5, 3.6, "Level Select", ha="center", fontsize=9, style="italic")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
ax.set_ylabel("Score (0-3)", fontsize=12)
ax.set_title(
    "Score Comparison: All Graders Across 30 Responses\n"
    "(SM-Q3 and LS-Q22 excluded: all graders scored 0)",
    fontsize=12, fontweight="bold"
)
ax.set_ylim(0, 4.3); ax.set_yticks([0, 1, 2, 3])
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("fig1_score_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 1 saved.")


models_label = ["Claude\nHaiku 4.5", "Llama 3.1\n70B", "DeepSeek\nV3.2",
                "GPT-OSS\n120B", "Human\n(Grader 2)"]

kappa_comb = [get_kappa(s, all_grader1)
              for s in [all_claude, all_llama, all_deepseek, all_gpt, all_grader2]]
spear_comb = [get_spear(s, all_grader1)
              for s in [all_claude, all_llama, all_deepseek, all_gpt, all_grader2]]
kappa_sm   = [get_kappa(s, sm_grader1)
              for s in [sm_claude, sm_llama, sm_deepseek, sm_gpt, sm_grader2]]
spear_sm   = [get_spear(s, sm_grader1)
              for s in [sm_claude, sm_llama, sm_deepseek, sm_gpt, sm_grader2]]

x = np.arange(len(models_label)); w = 0.2
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, kv, rv, title in [
    (axes[0], kappa_comb, spear_comb, "Combined Dataset (n=32)"),
    (axes[1], kappa_sm,   spear_sm,   "Story Mode Only (n=8)"),
]:
    b1 = ax.bar(x - w/2, kv, w, label="Cohen's Kappa",       color="#1976D2", alpha=0.9)
    b2 = ax.bar(x + w/2, rv, w, label="Spearman Correlation", color="#FF5722", alpha=0.9)
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(0.6, color="gray", linestyle=":",  alpha=0.5, linewidth=1)
    ax.axhline(0,   color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models_label, fontsize=9)
    ax.set_ylim(-0.2, 1.15); ax.legend(fontsize=8)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

fig.suptitle(
    "Cohen's Kappa and Spearman Correlation vs Human Baseline (Grader 1)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("fig2_kappa_spearman.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 2 saved.")


colors = ["#EF5350", "#FFA726", "#66BB6A", "#42A5F5"]
graders = {
    "Claude Haiku 4.5": all_claude,
    "Llama 3.1 70B":    all_llama,
    "DeepSeek V3.2":    all_deepseek,
    "GPT-OSS 120B":     all_gpt,
    "Human (Grader 1)": all_grader1,
    "Human (Grader 2)": all_grader2,
}

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
for ax, (name, scores) in zip(axes.flatten(), graders.items()):
    counts = [scores.count(v) for v in [0, 1, 2, 3]]
    bars = ax.bar(["0", "1", "2", "3"], counts, color=colors, alpha=0.9, edgecolor="white")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Score", fontsize=10)
    ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(count), ha="center", va="bottom",
                    fontsize=11, fontweight="bold")

for ax in [axes[0][0], axes[1][0]]:
    ax.set_ylabel("Number of Responses", fontsize=11)

fig.suptitle("Score Distribution by Grader (n=32 responses)", fontsize=14, fontweight="bold")
legend = [Patch(color=colors[i], label=f"Score {i}") for i in range(4)]
fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=11, bbox_to_anchor=(0.5, -0.04))
plt.tight_layout()
plt.savefig("fig3_score_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 3 saved.")


sets_clean = {
    "Claude":   all_claude,
    "Llama":    all_llama,
    "DeepSeek": all_deepseek,
    "GPT":      all_gpt,
    "Grader 2": all_grader2,
    "Grader 1": all_grader1,
}
names = list(sets_clean.keys())
n_g   = len(names)
matrix = np.zeros((n_g, n_g))
for i, (_, s1) in enumerate(sets_clean.items()):
    for j, (_, s2) in enumerate(sets_clean.items()):
        try:
            matrix[i][j] = cohen_kappa_score(s1, s2)
        except Exception:
            matrix[i][j] = 1.0 if i == j else 0.0

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(n_g)); ax.set_yticks(range(n_g))
ax.set_xticklabels(names, fontsize=11)
ax.set_yticklabels(names, fontsize=11)
for i in range(n_g):
    for j in range(n_g):
        ax.text(j, i, f"{matrix[i][j]:.3f}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if matrix[i][j] < 0.3 else "black")
plt.colorbar(im, ax=ax, label="Cohen's Kappa")
ax.set_title("Pairwise Cohen's Kappa Agreement Matrix (n=32)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig4_kappa_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figure 4 saved.")
