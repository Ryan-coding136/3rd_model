import pandas as pd
import matplotlib.pyplot as plt

# 读取已有结构打分结果
input_csv = "H3_mutation_table_structural_v2.csv"
output_csv = "H3_mutation_table_labeled_v9b.csv"

df = pd.read_csv(input_csv)

# 保留 v9 打分函数
key_sites = [137, 138, 190, 193, 225, 226, 228]

def score_row(row):
    score = 0
    for pos in key_sites:
        if row.get(f"Mut{pos}", 0) == 1:
            score += 1
            rsa = row.get(f"RSA_{pos}", None)
            if pd.notna(rsa):
                if rsa > 0.5:
                    score += 1
                elif rsa > 0.25:
                    score += 0.5
            dist = row.get(f"Dist_{pos}", None)
            if pd.notna(dist):
                if dist < 8:
                    score += 1
                elif dist < 12:
                    score += 0.5
    return score

df["raw_score"] = df.apply(score_row, axis=1)

# ✅ 新标签映射（v9b）
def map_score_to_label(score):
    if score <= 5.0:
        return 0  # 弱结合
    elif score <= 10.0:
        return 1  # 中结合
    else:
        return 2  # 强结合

df["binding_score"] = df["raw_score"].apply(map_score_to_label)

# 打印统计信息
print("✅ 标签分布统计（v9b）：")
print(df["binding_score"].value_counts())

print("\n🔍 raw_score 分布：")
print(df["raw_score"].value_counts().sort_index())

# 可视化
df["raw_score"].value_counts().sort_index().plot(kind="bar", title="Raw Score Distribution (v9b)")
plt.xlabel("Raw Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("raw_score_distribution_v9b.png")
print("📊 图像已保存至 raw_score_distribution_v9b.png")

# 保存结果
df.to_csv(output_csv, index=False)
print(f"✅ 最终标签表已保存至：{output_csv}")