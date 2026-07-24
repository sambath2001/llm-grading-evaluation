"""
analysis.py
Computes Cohen's Kappa, Spearman correlation, MAE, and exact agreement
for all graders vs human baseline (Grader 1).
"""

import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr

sm_claude   = [2, 2, 0, 3, 0, 3, 2, 2]
sm_llama    = [2, 2, 0, 3, 1, 3, 2, 2]
sm_deepseek = [2, 2, 0, 3, 0, 3, 0, 0]
sm_gpt      = [3, 2, 0, 0, 1, 1, 0, 0]
sm_grader2  = [2, 2, 0, 2, 2, 3, 0, 1]
sm_grader1  = [3, 3, 0, 2, 2, 3, 0, 1]
sm_prod_ai  = [2, 2, 0, 3, 1, 3, 0, 1]

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


def kappa(a, b):
    try:
        return round(cohen_kappa_score(a, b), 3)
    except Exception:
        return "N/A"

def spear(a, b):
    r, _ = spearmanr(a, b)
    return round(r, 3)

def exact(a, b):
    return sum(1 for x, y in zip(a, b) if x == y)

def near(a, b):
    return sum(1 for x, y in zip(a, b) if abs(x - y) <= 1)

def mae(a, b):
    return round(np.mean([abs(x - y) for x, y in zip(a, b)]), 3)


def print_stats(label, scores, baseline, n):
    k  = kappa(scores, baseline)
    r  = spear(scores, baseline)
    ex = exact(scores, baseline)
    m  = mae(scores, baseline)
    print(f"  {label:<22} Kappa={k:>6}  Spearman={r:>6}  "
          f"Exact={ex}/{n}({round(ex/n*100)}%)  MAE={m}")


print("=" * 70)
print("COMBINED DATASET (n=32) vs Grader 1")
print("=" * 70)
for name, scores in [
    ("Human (Grader 2)", all_grader2),
    ("Llama 3.1 70B",    all_llama),
    ("Claude Haiku 4.5", all_claude),
    ("GPT-OSS 120B",     all_gpt),
    ("DeepSeek V3.2",    all_deepseek),
]:
    print_stats(name, scores, all_grader1, 32)

print()
print("=" * 70)
print("STORY MODE ONLY (n=8) vs Grader 1")
print("=" * 70)
for name, scores in [
    ("Production AI",    sm_prod_ai),
    ("Human (Grader 2)", sm_grader2),
    ("GPT-OSS 120B",     sm_gpt),
    ("DeepSeek V3.2",    sm_deepseek),
    ("Llama 3.1 70B",    sm_llama),
    ("Claude Haiku 4.5", sm_claude),
]:
    print_stats(name, scores, sm_grader1, 8)

print()
print("=" * 70)
print("HUMAN INTER-RATER (Grader 2 vs Grader 1)")
print("=" * 70)
print_stats("Combined (n=32)",  all_grader2, all_grader1, 32)
print_stats("Story Mode (n=8)", sm_grader2,  sm_grader1,  8)
