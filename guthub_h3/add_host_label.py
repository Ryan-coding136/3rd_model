import pandas as pd

# 输入文件
label_input = "H3_mutation_table_labeled_v9b.csv"   # 已打分文件
metadata_input = "H3_metadata.csv"                  # 含 Host 字段
output_csv = "H3_labeled_with_host.csv"             # 输出文件

# 加载数据
df = pd.read_csv(label_input)
meta = pd.read_csv(metadata_input)

# 提取主机信息（保留必要字段）
meta = meta[["Isolate_Id", "Host"]]
meta = meta.rename(columns={"Isolate_Id": "Isolated_Id"})


# 合并两个表格（按 Isolated_Id 对应）
merged = pd.merge(df, meta, on="Isolated_Id", how="left")

# 添加 host_label
def host_to_label(host):
    if pd.isna(host):
        return -1  # 缺失情况
    elif "human" in host.lower():
        return 0
    else:
        return 1

merged["host_label"] = merged["Host"].apply(host_to_label)

# 保存输出
merged.to_csv(output_csv, index=False)
print(f"✅ 添加 host_label 成功，保存至：{output_csv}")
print(merged["host_label"].value_counts(dropna=False))
