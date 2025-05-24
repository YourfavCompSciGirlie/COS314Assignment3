from scipy.stats import wilcoxon

# Load scores from both files
with open("gp_f1_scores.txt") as f:
    gp_scores = [float(line.strip()) for line in f.readlines()]

with open("mlp_f1_scores.txt") as f:
    mlp_scores = [float(line.strip()) for line in f.readlines()]

# Ensure same length
assert len(gp_scores) == len(mlp_scores), "Score lists must be of same length"

# Perform Wilcoxon signed-rank test
stat, p = wilcoxon(gp_scores, mlp_scores)

print("Wilcoxon Signed-Rank Test Results:")
print(f"Statistic = {stat:.4f}, p-value = {p:.4f}")

if p < 0.05:
    print("=> Statistically significant difference between GP and MLP (p < 0.05)")
else:
    print("=> No statistically significant difference (p >= 0.05)")
